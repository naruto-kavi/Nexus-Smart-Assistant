"""
==============================================================
M1 Vision Module  Training Script
YOLOv8-pose fine-tuning for Face Detection + Body Pose
==============================================================
Usage:
  python train_m1.py                    # default settings
  python train_m1.py --epochs 25        # custom epochs
  python train_m1.py --device cpu       # CPU-only
  python train_m1.py --resume           # resume from last checkpoint
==============================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ["WANDB_MODE"] = "disabled"


# import torch (Moved to functions to prevent hang)
# import yaml (Moved to functions to prevent hang)
# from ultralytics import YOLO (Moved to main to prevent hang)


#  Paths 
ROOT        = Path(__file__).parent.parent
sys.path.append(str(ROOT))
DATA_YAML   = Path(__file__).parent / "dataset.yaml"
RUNS_DIR    = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Model
    "model"         : "yolov8n-pose.pt",   # nano-pose for CPU-friendly training
    # Training schedule
    "epochs"        : 20,
    "patience"      : 7,                   # early stopping patience
    "batch"         : 16,
    "imgsz"         : 640,
    # Optimiser
    "optimizer"     : "AdamW",
    "lr0"           : 1e-4,
    "lrf"           : 0.01,                # final lr = lr0 * lrf
    "momentum"      : 0.937,
    "weight_decay"  : 5e-4,
    "warmup_epochs" : 3,
    # Regularisation / augmentation
    "hsv_h"         : 0.015,
    "hsv_s"         : 0.7,
    "hsv_v"         : 0.4,
    "degrees"       : 10.0,
    "translate"     : 0.1,
    "scale"         : 0.5,
    "shear"         : 2.0,
    "flipud"        : 0.0,
    "fliplr"        : 0.5,
    "mosaic"        : 1.0,
    "mixup"         : 0.1,
    "copy_paste"    : 0.1,
    # Transfer learning freeze schedule
    "freeze_epochs" : 5,                   # freeze backbone for first N epochs
    "freeze_layers" : 10,                  # number of backbone layers to freeze
    # Hardware
    "device"        : "cpu", # Default to CPU to prevent hang, main() will detect CUDA
    "workers"       : 4,
    # Output
    "project"       : str(RUNS_DIR),
    "name"          : "m1_face_pose",
    "exist_ok"      : True,
    "save_period"   : 5,                   # save checkpoint every N epochs
    "plots"         : True,
    "verbose"       : True,
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M1 Vision Module")
    parser.add_argument("--model",   default=DEFAULT_CONFIG["model"],
                        help="YOLOv8 pose model variant (yolov8n/s/m-pose.pt)")
    parser.add_argument("--epochs",  type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch",   type=int,   default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--imgsz",   type=int,   default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--lr",      type=float, default=DEFAULT_CONFIG["lr0"])
    parser.add_argument("--device",  default="", # Auto-detect in main
                        help="Device to use (0, 1, cpu, etc.)")
    parser.add_argument("--workers", type=int,   default=DEFAULT_CONFIG["workers"])
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--no-freeze", action="store_true",
                        help="Disable backbone freezing (train all layers from epoch 1)")
    return parser.parse_args()


def check_dataset(yaml_path: Path):
    """Verify that image directories exist before training."""
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    data_root = yaml_path.parent / cfg["path"]
    missing   = []

    for key in ("train", "val"):
        dirs = cfg.get(key, [])
        if isinstance(dirs, str):
            dirs = [dirs]
        for d in dirs:
            full = data_root / d
            if not full.exists():
                missing.append(str(full))

    if missing:
        print("\n    Missing dataset directories:")
        for m in missing:
            print(f"     {m}")
        print("\n  Run  python download_datasets.py  first.\n")
        sys.exit(1)

    print("  DONE Dataset directories found")


def log_hardware():
    """Print hardware info."""
    import torch
    import sys
    print(f"\n  Device  : {'CUDA  ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM    : {vram:.1f} GB")
    print(f"  Torch   : {torch.__version__}")
    print(f"  Python  : {sys.version.split()[0]}")


# ------------------------------------------------------------------------------
# Two-phase training: freeze backbone -> unfreeze
# ------------------------------------------------------------------------------

def train_phase1_frozen(model, cfg: dict, args: argparse.Namespace):
    """Phase 1  freeze backbone layers, train detection heads only."""
    if args.no_freeze or cfg["freeze_epochs"] == 0:
        return None

    print(f"\n   Phase 1: Frozen backbone ({cfg['freeze_epochs']} epochs) ")
    results = model.train(
        data          = str(DATA_YAML),
        epochs        = cfg["freeze_epochs"],
        patience      = cfg["patience"],
        batch         = cfg["batch"],
        imgsz         = cfg["imgsz"],
        optimizer     = cfg["optimizer"],
        lr0           = cfg["lr0"] * 2,        # slightly higher LR when frozen
        lrf           = cfg["lrf"],
        momentum      = cfg["momentum"],
        weight_decay  = cfg["weight_decay"],
        warmup_epochs = cfg["warmup_epochs"],
        hsv_h         = cfg["hsv_h"],
        hsv_s         = cfg["hsv_s"],
        hsv_v         = cfg["hsv_v"],
        degrees       = cfg["degrees"],
        translate     = cfg["translate"],
        scale         = cfg["scale"],
        shear         = cfg["shear"],
        flipud        = cfg["flipud"],
        fliplr        = cfg["fliplr"],
        mosaic        = cfg["mosaic"],
        mixup         = cfg["mixup"],
        copy_paste    = cfg["copy_paste"],
        freeze        = cfg["freeze_layers"],  # freeze first N layers
        device        = cfg["device"],
        workers       = cfg["workers"],
        project       = cfg["project"],
        name          = cfg["name"] + "_phase1",
        exist_ok      = True,
        save_period   = cfg["save_period"],
        plots         = cfg["plots"],
        verbose       = cfg["verbose"],
    )
    return results


def train_phase2_unfrozen(model, cfg: dict, args: argparse.Namespace,
                          resume_path: str | None = None):
    """Phase 2  unfreeze all layers, fine-tune end-to-end."""
    remaining = cfg["epochs"] - cfg["freeze_epochs"]
    if remaining <= 0 and not args.no_freeze:
        remaining = cfg["epochs"]

    print(f"\n   Phase 2: Full fine-tune ({remaining} epochs) ")
    train_kwargs = dict(
        data          = str(DATA_YAML),
        epochs        = remaining,
        patience      = cfg["patience"],
        batch         = cfg["batch"],
        imgsz         = cfg["imgsz"],
        optimizer     = cfg["optimizer"],
        lr0           = cfg["lr0"],
        lrf           = cfg["lrf"],
        momentum      = cfg["momentum"],
        weight_decay  = cfg["weight_decay"],
        warmup_epochs = 0,                     # no warmup in phase 2
        hsv_h         = cfg["hsv_h"],
        hsv_s         = cfg["hsv_s"],
        hsv_v         = cfg["hsv_v"],
        degrees       = cfg["degrees"],
        translate     = cfg["translate"],
        scale         = cfg["scale"],
        shear         = cfg["shear"],
        flipud        = cfg["flipud"],
        fliplr        = cfg["fliplr"],
        mosaic        = cfg["mosaic"],
        mixup         = cfg["mixup"],
        copy_paste    = cfg["copy_paste"],
        freeze        = 0,                     # unfreeze all layers
        device        = cfg["device"],
        workers       = cfg["workers"],
        project       = cfg["project"],
        name          = cfg["name"],
        exist_ok      = cfg["exist_ok"],
        save_period   = cfg["save_period"],
        plots         = cfg["plots"],
        verbose       = cfg["verbose"],
        resume        = resume_path is not None,
    )
    results = model.train(**train_kwargs)
    return results


# ------------------------------------------------------------------------------
# Export
# ------------------------------------------------------------------------------

def export_to_onnx(best_pt: Path):
    """Export best.pt -> ONNX with INT8 quantisation for CPU deployment."""
    print("\n   Exporting to ONNX ")
    from ultralytics import YOLO
    model      = YOLO(str(best_pt))
    onnx_path  = model.export(
        format   = "onnx",
        imgsz    = DEFAULT_CONFIG["imgsz"],
        simplify = True,
        opset    = 17,
        dynamic  = False,
    )
    dest = WEIGHTS_DIR / "m1_face_pose.onnx"
    Path(onnx_path).rename(dest)
    print(f"  DONE ONNX model saved -> {dest}")
    return dest


# ------------------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------------------

def evaluate_model(model, cfg: dict) -> dict:
    """Run val and return key metrics."""
    print("\n   Validation ")
    metrics = model.val(
        data    = str(DATA_YAML),
        imgsz   = cfg["imgsz"],
        batch   = cfg["batch"],
        device  = cfg["device"],
        verbose = True,
    )
    results = {
        "mAP50"     : float(metrics.box.map50),
        "mAP50_95"  : float(metrics.box.map),
        "precision" : float(metrics.box.mp),
        "recall"    : float(metrics.box.mr),
    }
    # Keypoint metrics (if available)
    if hasattr(metrics, "pose"):
        results["pose_mAP50"]    = float(metrics.pose.map50)
        results["pose_mAP50_95"] = float(metrics.pose.map)

    print(f"\n  mAP@0.5       : {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95  : {results['mAP50_95']:.4f}")
    print(f"  Precision     : {results['precision']:.4f}")
    print(f"  Recall        : {results['recall']:.4f}")
    if "pose_mAP50" in results:
        print(f"  Pose mAP@0.5  : {results['pose_mAP50']:.4f}")
    return results


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    print("Loading libraries (this may take a minute)...", flush=True)
    from ultralytics import YOLO
    args = parse_args()

    print("+----------------------------------------------------------+")
    print("|   M1 Vision Module  Training Pipeline                  |")
    print("+----------------------------------------------------------+")

    log_hardware()
    check_dataset(DATA_YAML)

    #  Merge args into config 
    cfg = {**DEFAULT_CONFIG}

    #  Device detection 
    import torch
    if not args.device:
        cfg["device"] = 0 if torch.cuda.is_available() else "cpu"
    else:
        cfg["device"] = args.device

    #  Merge other args into config 
    cfg["model"]   = args.model
    cfg["epochs"]  = args.epochs
    cfg["batch"]   = args.batch
    cfg["imgsz"]   = args.imgsz
    cfg["lr0"]     = args.lr
    cfg["workers"] = args.workers

    #  Load model 
    if args.resume:
        last_ckpt = RUNS_DIR / cfg["name"] / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"\n  Resuming from {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print(f"  [WARN] No checkpoint found at {last_ckpt}. Starting fresh.")
            model = YOLO(cfg["model"])
    else:
        print(f"\n  Loading base model: {cfg['model']}")
        model = YOLO(cfg["model"])          # downloads pretrained weights automatically

    #  Phase 1: frozen backbone 
    start_time = time.time()
    if not args.resume:
        train_phase1_frozen(model, cfg, args)

        # Reload from phase-1 best for phase-2
        p1_best = RUNS_DIR / (cfg["name"] + "_phase1") / "weights" / "best.pt"
        if p1_best.exists():
            print(f"\n  Loading phase-1 best weights: {p1_best}")
            model = YOLO(str(p1_best))

    #  Phase 2: full fine-tune 
    results = train_phase2_unfrozen(model, cfg, args)

    elapsed = time.time() - start_time
    print(f"\n  DONE Training complete in {elapsed/60:.1f} min")

    #  Find best checkpoint 
    best_pt = RUNS_DIR / cfg["name"] / "weights" / "best.pt"
    if not best_pt.exists():
        print("  [WARN] best.pt not found. Using last.pt.")
        best_pt = RUNS_DIR / cfg["name"] / "weights" / "last.pt"

    # Copy best weights to weights/
    if best_pt.exists():
        dest = WEIGHTS_DIR / "m1_best.pt"
        import shutil
        shutil.copy(best_pt, dest)
        print(f"  DONE Best weights saved -> {dest}")

    #  Evaluate 
    from ultralytics import YOLO
    eval_model = YOLO(str(best_pt))
    metrics    = evaluate_model(eval_model, cfg)

    #  Save metrics 
    metrics_file = WEIGHTS_DIR / "m1_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"  DONE Metrics saved -> {metrics_file}")

    #  Export ONNX 
    try:
        export_to_onnx(best_pt)
    except Exception as e:
        print(f"  [WARN] ONNX export failed: {e}")

    #  Final summary 
    print("\n" + "-" * 60)
    print("  Training Summary")
    print("-" * 60)
    print(f"  Best weights : {WEIGHTS_DIR / 'm1_best.pt'}")
    print(f"  ONNX model   : {WEIGHTS_DIR / 'm1_face_pose.onnx'}")
    print(f"  Metrics file : {metrics_file}")
    print(f"  Tensorboard  : tensorboard --logdir {RUNS_DIR}")
    print("-" * 60)

    # Check target metric
    map50 = metrics.get("mAP50", 0)
    if map50 >= 0.85:
        print(f"\n   Target achieved! mAP@0.5 = {map50:.3f}  0.85")
    else:
        print(f"\n    mAP@0.5 = {map50:.3f}  below 0.85 target. "
              f"Try more epochs or a larger model (yolov8s-pose.pt).")


if __name__ == "__main__":
    main()
