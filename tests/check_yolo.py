import sys
from pathlib import Path
from ultralytics import YOLO

log_path = Path(__file__).parent / "check_out.txt"

with open(log_path, "w", encoding="utf-8") as f:
    f.write("Starting YOLO check...\n")
    try:
        model_path = Path(__file__).parent.parent / "weights" / "m1_face_pose.onnx"
        if not model_path.exists():
            f.write(f"File not found: {model_path}\n")
            sys.exit(1)

        model = YOLO(str(model_path))
        f.write("Model loaded successfully.\n")
        f.write(f"ckpt_path: {getattr(model, 'ckpt_path', None)}\n")
        f.write(f"model type/class: {type(model.model)}\n")
        f.write(f"model name/path string: {getattr(model, 'model', None)}\n")
        f.write(f"overrides: {model.overrides}\n")
    except Exception as e:
        f.write(f"ERROR: {e}\n")
