"""
=============================================================
YOLO Diagnostic — Tests BOTH PyTorch (.pt) and ONNX models
=============================================================
Opens webcam and shows detections from each model.
Press:  P = switch to PyTorch model
        O = switch to ONNX model
        Q = quit

Prints conf score of each detection so you can see what
threshold is needed.
=============================================================
"""

import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from ultralytics import YOLO

WEIGHTS_DIR = ROOT / "weights"
POSE_PT   = WEIGHTS_DIR / "m1_best.pt"
POSE_ONNX = WEIGHTS_DIR / "m1_face_pose.onnx"

print("=" * 60)
print("  YOLO Diagnostic — PT vs ONNX")
print("=" * 60)

# Load both models
print("\n[1] Loading PyTorch model (.pt)...")
if POSE_PT.exists():
    model_pt = YOLO(str(POSE_PT))
    print(f"    Loaded: {POSE_PT.name} ({POSE_PT.stat().st_size // (1024*1024)} MB)")
else:
    model_pt = None
    print(f"    Not found: {POSE_PT}")

print("\n[2] Loading ONNX model...")
if POSE_ONNX.exists():
    model_onnx = YOLO(str(POSE_ONNX))
    print(f"    Loaded: {POSE_ONNX.name} ({POSE_ONNX.stat().st_size // (1024*1024)} MB)")
else:
    model_onnx = None
    print(f"    Not found: {POSE_ONNX}")

print("\n[3] Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("    Cannot open webcam!")
    sys.exit(1)
print("    Webcam open")

# Which model to use
MODE = "onnx" if model_onnx else "pt"
print(f"\nStarting with: {MODE.upper()} model")
print("Press P=PyTorch  O=ONNX  Q=quit\n")

fps_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # Pick active model
    if MODE == "pt" and model_pt:
        active_model = model_pt
        imgsz = 256
        label_prefix = "PT"
        header_color = (0, 200, 0)
    elif MODE == "onnx" and model_onnx:
        active_model = model_onnx
        imgsz = 640
        label_prefix = "ONNX"
        header_color = (200, 100, 0)
    else:
        cv2.putText(frame, "Model not available!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("YOLO Diagnostic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Run inference with low conf to see ALL detections
    try:
        results = active_model(frame, verbose=False, conf=0.25, iou=0.45, imgsz=imgsz)[0]
    except Exception as e:
        cv2.putText(frame, f"ERROR: {str(e)[:60]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("YOLO Diagnostic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    elapsed = time.time() - t0
    fps_times.append(elapsed)
    if len(fps_times) > 20:
        fps_times.pop(0)
    avg_fps = 1.0 / (sum(fps_times) / len(fps_times))

    n_det = len(results.boxes) if results.boxes else 0

    # Draw detections
    if results.boxes:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_prefix} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw keypoints if available
        if results.keypoints is not None:
            for kps in results.keypoints:
                pts = kps.xy[0].cpu().numpy()
                for px, py in pts:
                    if px > 0 and py > 0:
                        cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 255), -1)

    # Header
    header = f"[{label_prefix}] Det:{n_det}  FPS:{avg_fps:.1f}  conf>=0.25 iou=0.45 imgsz={imgsz}"
    cv2.putText(frame, header, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, header_color, 2)
    cv2.putText(frame, "P=PyTorch  O=ONNX  Q=quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Terminal output
    if n_det > 0:
        confs = [f"{float(b.conf[0]):.2f}" for b in results.boxes]
        print(f"\r[{label_prefix}] {n_det} person(s) | conf={confs} | fps={avg_fps:.1f}  ", end="", flush=True)
    else:
        print(f"\r[{label_prefix}] NO DETECTION | fps={avg_fps:.1f}                      ", end="", flush=True)

    cv2.imshow("YOLO Diagnostic", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p') and model_pt:
        MODE = "pt"
        print(f"\nSwitched to PyTorch model")
    elif key == ord('o') and model_onnx:
        MODE = "onnx"
        print(f"\nSwitched to ONNX model")

cap.release()
cv2.destroyAllWindows()
print("\n\nDone.")
