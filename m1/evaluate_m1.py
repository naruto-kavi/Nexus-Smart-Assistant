"""
==============================================================
M1 Vision Module — Evaluation & Metrics Visualisation
==============================================================
Generates:
  • Detection/Pose: mAP, precision-recall curve
  • Activity: confusion matrix, per-class accuracy bar chart
  • Training curves from runs/ directory
Usage:
  python evaluate_m1.py
==============================================================
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT        = Path(__file__).parent.parent
sys.path.append(str(ROOT))
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR    = ROOT / "runs"
PLOTS_DIR   = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ACTIVITY_CLASSES = ["sitting", "waving", "standing", "approaching"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Training Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves():
    """Parse YOLOv8 results.csv and plot loss + mAP curves."""
    csv_path = RUNS_DIR / "m1_face_pose" / "results.csv"
    if not csv_path.exists():
        print(f"  [SKIP] results.csv not found: {csv_path}")
        return

    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): float(v) for k, v in row.items()
                         if v.strip() not in ("", "nan")})

    if not rows:
        return

    epochs = [r.get("epoch", i) for i, r in enumerate(rows)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("M1 Training Curves — YOLOv8-pose", fontsize=13, fontweight="bold")

    # ── Box losses ──
    ax = axes[0]
    for key, label, color in [
        ("train/box_loss", "Train box loss", "#4C72B0"),
        ("val/box_loss",   "Val box loss",   "#DD8452"),
    ]:
        vals = [r[key] for r in rows if key in r]
        ax.plot(epochs[:len(vals)], vals, label=label, color=color, linewidth=2)
    ax.set_title("Box Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # ── Pose / keypoint losses ──
    ax = axes[1]
    for key, label, color in [
        ("train/pose_loss", "Train pose loss", "#55A868"),
        ("val/pose_loss",   "Val pose loss",   "#C44E52"),
    ]:
        vals = [r[key] for r in rows if key in r]
        ax.plot(epochs[:len(vals)], vals, label=label, color=color, linewidth=2)
    ax.set_title("Pose (Keypoint) Loss"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # ── mAP ──
    ax = axes[2]
    for key, label, color in [
        ("metrics/mAP50(B)",    "mAP@0.5 (box)",     "#4C72B0"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95 (box)","#DD8452"),
        ("metrics/mAP50(P)",    "mAP@0.5 (pose)",    "#55A868"),
    ]:
        vals = [r[key] for r in rows if key in r]
        ax.plot(epochs[:len(vals)], vals, label=label, linewidth=2, color=color)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.7, label="Target 0.85")
    ax.set_title("mAP"); ax.set_xlabel("Epoch")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Training curves -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Activity Head — Training History
# ══════════════════════════════════════════════════════════════════════════════

def plot_activity_curves():
    hist_file = WEIGHTS_DIR / "activity_history.json"
    if not hist_file.exists():
        print(f"  [SKIP] Activity history not found: {hist_file}")
        return

    data    = json.loads(hist_file.read_text())
    history = data.get("history", [])
    per_cls = data.get("per_class", {})

    if not history:
        return

    epochs   = [r["epoch"]    for r in history]
    trn_acc  = [r["trn_acc"]  for r in history]
    val_acc  = [r["val_acc"]  for r in history]
    trn_loss = [r["trn_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Activity Head Training — MobileNetV3-Small", fontsize=13, fontweight="bold")

    # ── Loss ──
    ax = axes[0]
    ax.plot(epochs, trn_loss, label="Train", color="#4C72B0", linewidth=2)
    ax.plot(epochs, val_loss, label="Val",   color="#DD8452", linewidth=2)
    ax.set_title("Cross-Entropy Loss"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Accuracy ──
    ax = axes[1]
    ax.plot(epochs, trn_acc, label="Train", color="#4C72B0", linewidth=2)
    ax.plot(epochs, val_acc, label="Val",   color="#DD8452", linewidth=2)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.7, label="Target 0.85")
    ax.set_ylim(0, 1); ax.set_title("Accuracy")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # ── Per-class accuracy ──
    ax = axes[2]
    if per_cls:
        classes = list(per_cls.keys())
        accs    = list(per_cls.values())
        bars    = ax.bar(classes, accs, color=COLORS[:len(classes)], edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1)
        ax.axhline(0.85, color="red", linestyle="--", alpha=0.7, label="Target")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Per-Class Val Accuracy")
        ax.set_xlabel("Activity"); ax.legend()
        ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "activity_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Activity curves -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Final Metrics Summary Card
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_summary():
    m_file = WEIGHTS_DIR / "m1_metrics.json"
    if not m_file.exists():
        print(f"  [SKIP] Metrics file not found: {m_file}")
        return

    metrics = json.loads(m_file.read_text())

    labels  = ["mAP@0.5\n(box)", "mAP@0.5:0.95\n(box)", "Precision", "Recall"]
    keys    = ["mAP50", "mAP50_95", "precision", "recall"]
    values  = [metrics.get(k, 0.0) for k in keys]

    if "pose_mAP50" in metrics:
        labels.append("mAP@0.5\n(pose)")
        values.append(metrics["pose_mAP50"])

    fig, ax = plt.subplots(figsize=(8, 4))
    x    = np.arange(len(labels))
    bars = ax.bar(x, values, color=COLORS[:len(labels)] + ["#8172B3"],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.85, color="red", linestyle="--", linewidth=1.5,
               label="Target 0.85", alpha=0.8)
    ax.set_title("M1 Vision Module — Final Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "metrics_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Metrics summary -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Dataset Distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_dataset_distribution():
    from pathlib import Path as P

    data = ROOT / "data"
    splits = {
        "WIDER FACE train" : data / "wider_face/images/WIDER_train",
        "WIDER FACE val"   : data / "wider_face/images/WIDER_val",
        "COCO-Pose train"  : data / "coco_pose/images/train2017",
        "COCO-Pose val"    : data / "coco_pose/images/val2017",
    }

    counts = {}
    for name, path in splits.items():
        if path.exists():
            counts[name] = sum(1 for f in path.rglob("*")
                               if f.suffix.lower() in (".jpg", ".jpeg", ".png"))

    if not counts:
        print("  [SKIP] No dataset images found for distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(list(counts.keys()), list(counts.values()),
                  color=COLORS[:len(counts)], edgecolor="black", linewidth=0.5)
    ax.set_title("Dataset Distribution — M1 Vision Module", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.axhline(25_000, color="red", linestyle="--", alpha=0.5, label="25k cap")

    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val:,}", ha="center", va="bottom", fontsize=9)

    total = sum(counts.values())
    ax.set_title(f"Dataset Distribution — Total: {total:,} images", fontsize=12)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "dataset_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Dataset distribution -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("   M1 Vision Module — Evaluation & Plots")
    print("=" * 60 + "\n")

    plot_dataset_distribution()
    plot_training_curves()
    plot_activity_curves()
    plot_metrics_summary()

    print(f"\n  All plots saved to: {PLOTS_DIR}/")
