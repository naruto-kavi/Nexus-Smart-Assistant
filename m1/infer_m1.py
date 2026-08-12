"""
==============================================================
M1 Vision Module — Real-Time Inference  (FPS-Optimized v2)
==============================================================
Loads:
  • YOLOv8-pose  → face bounding boxes + body keypoints
  • Activity head → sitting / waving / standing / approaching

Optimizations v2:
  • Threaded camera capture (eliminates read() blocking)
  • Face recognition every N frames with IoU-based caching
  • Proper face cropping using YOLO pose keypoints (nose/eyes/ears)
  • Reduced YOLO input size (320 default)
  • YOLO frame-skip option for ultra-low-power devices

Usage:
  python infer_m1.py                        # webcam
  python infer_m1.py --source video.mp4     # video file
  python infer_m1.py --fast                 # max speed mode
  python infer_m1.py --face-every 10        # face recog every 10 frames
  python infer_m1.py --yolo-size 256        # smaller YOLO input
==============================================================
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

# --- Phase 2: Smart Assistant Additions ---
# NOTE: Phase 2 heads (hsemotion + mediapipe) disabled for now
# from hsemotion.facial_emotions import HSEmotionRecognizer
# import mediapipe as mp

import sys
ROOT        = Path(__file__).parent.parent
sys.path.append(str(ROOT))

# --- M2: Face Recognition ---
from m2_face_recognition import load_facenet, recognize, load_db


# ─── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = ROOT / "weights"

POSE_PT       = WEIGHTS_DIR / "m1_best.pt"
POSE_ONNX     = WEIGHTS_DIR / "m1_face_pose.onnx"
ACTIVITY_PT   = WEIGHTS_DIR / "activity_head_best.pt"
ACTIVITY_TS   = WEIGHTS_DIR / "activity_head.torchscript.pt"

ACTIVITY_CLASSES = [
    "baseball_pitch", "baseball_swing", "bench_press", "bowl", 
    "clean_and_jerk", "golf_swing", "jumping_jacks", "jump_rope", 
    "pull_ups", "push_ups", "sit_ups", "squats", 
    "strum_guitar", "tennis_serve", "tennis_swing"
]
ACTIVITY_COLORS  = {cls: (0, 255, 0) for cls in ACTIVITY_CLASSES}

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# COCO keypoint indices for face cropping
KP_NOSE     = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR  = 3
KP_RIGHT_EAR = 4

IMG_SIZE = 224
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"   # FP16 only on GPU


# ══════════════════════════════════════════════════════════════════════════════
# Threaded Camera Capture  (eliminates cap.read() blocking latency)
# ══════════════════════════════════════════════════════════════════════════════

class CameraStream:
    """
    Reads frames from the camera in a background thread.
    The main loop always gets the latest frame instantly — no blocking.
    This alone can add 3-5 FPS on CPU by removing I/O wait.
    """

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

        # Read first frame
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise RuntimeError(f"Cannot open source: {source}")

        # Start background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                self.stopped = True
                break

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def is_opened(self):
        return self.cap.isOpened() and not self.stopped

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_pose_model(use_onnx: bool = False) -> YOLO:
    if use_onnx and POSE_ONNX.exists():
        print(f"  Loading ONNX pose model: {POSE_ONNX}")
        return YOLO(str(POSE_ONNX))
    if POSE_PT.exists():
        print(f"  Loading PyTorch pose model: {POSE_PT}")
        return YOLO(str(POSE_PT))
    print("  [WARN] No trained weights found. Using pretrained yolov8n-pose.")
    return YOLO("yolov8n-pose.pt")


def load_activity_model():
    if ACTIVITY_TS.exists():
        print(f"  Loading TorchScript activity model: {ACTIVITY_TS}")
        model = torch.jit.load(str(ACTIVITY_TS), map_location=DEVICE)
    elif ACTIVITY_PT.exists():
        print(f"  Loading PyTorch activity model: {ACTIVITY_PT}")
        from train_activity import build_activity_head
        model = build_activity_head(pretrained=False)
        ckpt  = torch.load(ACTIVITY_PT, map_location=DEVICE)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("  [WARN] No activity model found. Activity classification disabled.")
        return None
    model.eval().to(DEVICE)
    if USE_HALF:
        model.half()
    return model


def load_smart_heads():
    """Initialize Phase 2 Emotion and Gesture recognition heads."""
    # Disabled for now — hsemotion model download returns 404
    print("  [SKIP] Phase 2 heads disabled (emotion + gestures)")
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Face cropping from YOLO keypoints  (FIX for recognition accuracy)
# ══════════════════════════════════════════════════════════════════════════════

def extract_face_from_keypoints(frame, keypoints, person_box, padding=0.6):
    """
    Extract a proper face crop using YOLO pose keypoints (nose, eyes, ears).
    
    The old method took the "upper half of the body box" which includes
    shoulders, background, etc. and produces BAD embeddings.
    
    This method:
      1. Finds the nose, eyes, and ears from YOLO keypoints
      2. Computes a tight bounding box around them
      3. Adds padding and returns a clean face-only crop
    
    Falls back to the upper portion of the person box if keypoints are weak.
    """
    h, w = frame.shape[:2]
    x1_p, y1_p, x2_p, y2_p = map(int, person_box)
    
    if keypoints is None:
        # Fallback: top 30% of person box (tighter than old 50%)
        face_h = int((y2_p - y1_p) * 0.30)
        return frame[max(0, y1_p):min(y1_p + face_h, h),
                     max(0, x1_p):min(x2_p, w)]

    kps = keypoints.xy[0].cpu().numpy() if hasattr(keypoints, "xy") else keypoints
    confs = (keypoints.conf[0].cpu().numpy()
             if hasattr(keypoints, "conf") and keypoints.conf is not None
             else np.ones(len(kps)))

    # Collect face keypoints (nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4)
    face_kp_indices = [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]
    face_points = []
    
    for idx in face_kp_indices:
        if idx < len(kps) and confs[idx] > 0.3:
            px, py = kps[idx]
            if px > 0 and py > 0:  # valid point
                face_points.append((int(px), int(py)))

    if len(face_points) < 2:
        # Not enough face keypoints — fallback to top 30% of person box
        face_h = int((y2_p - y1_p) * 0.30)
        return frame[max(0, y1_p):min(y1_p + face_h, h),
                     max(0, x1_p):min(x2_p, w)]

    # Compute tight bounding box around face keypoints
    xs = [p[0] for p in face_points]
    ys = [p[1] for p in face_points]
    fx1 = min(xs)
    fy1 = min(ys)
    fx2 = max(xs)
    fy2 = max(ys)

    # Add padding (face keypoints are just eyes/nose/ears — face is bigger)
    face_w = max(fx2 - fx1, 20)
    face_h = max(fy2 - fy1, 20)
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)

    # Make it more square (faces are roughly square)
    cx = (fx1 + fx2) // 2
    cy = (fy1 + fy2) // 2
    half_size = max(face_w + pad_x, face_h + pad_y) // 2

    crop_x1 = max(0, cx - half_size)
    crop_y1 = max(0, cy - half_size)
    crop_x2 = min(w, cx + half_size)
    crop_y2 = min(h, cy + half_size)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        # Final fallback
        face_h = int((y2_p - y1_p) * 0.30)
        return frame[max(0, y1_p):min(y1_p + face_h, h),
                     max(0, x1_p):min(x2_p, w)]

    return crop


# ══════════════════════════════════════════════════════════════════════════════
# Activity inference
# ══════════════════════════════════════════════════════════════════════════════

_ACTIVITY_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict_activity(model, frame_bgr: np.ndarray, box_xyxy) -> tuple[str, float]:
    """
    Crop the person region from frame, run activity head.
    Returns (activity_label, confidence).
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return "unknown", 0.0

    crop  = frame_bgr[y1:y2, x1:x2]
    pil   = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inp   = _ACTIVITY_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    if USE_HALF:
        inp = inp.half()
    logit = model(inp)
    prob  = torch.softmax(logit, dim=1)[0]
    idx   = prob.argmax().item()
    return ACTIVITY_CLASSES[idx], float(prob[idx])


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Analysis Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_gesture_label(hand_landmarks) -> str:
    """Basic heuristic to map MediaPipe landmarks to common assistant gestures."""
    if not hand_landmarks: return "none"
    
    l = hand_landmarks.landmark
    # Thumb up (Thumb tip above other thumb joints)
    is_thumb_up = l[4].y < l[3].y < l[2].y
    # Hand raised (Wrist below middle finger tip)
    is_raised = l[0].y > l[12].y
    # Basic mapping
    if is_thumb_up: return "thumbs_up"
    if is_raised: return "hand_raised"
    return "active"


def draw_box(frame, x1, y1, x2, y2, label, color=(0, 255, 0), thickness=2, status=""):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Smart Status at bottom
    if status:
        cv2.putText(frame, f"[Assistant]: {status}", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_keypoints(frame, keypoints, threshold=0.3):
    if keypoints is None:
        return
    kps = keypoints.xy[0].cpu().numpy() if hasattr(keypoints, "xy") else keypoints
    confs = (keypoints.conf[0].cpu().numpy()
             if hasattr(keypoints, "conf") and keypoints.conf is not None
             else np.ones(len(kps)))

    # Joints
    for i, (x, y) in enumerate(kps):
        if confs[i] > threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

    # Skeleton
    for (i, j) in COCO_SKELETON:
        if i < len(kps) and j < len(kps):
            if confs[i] > threshold and confs[j] > threshold:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(frame, pt1, pt2, (0, 200, 255), 2)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


# ══════════════════════════════════════════════════════════════════════════════
# Face Recognition Cache  (KEY OPTIMIZATION)
# ══════════════════════════════════════════════════════════════════════════════

class FaceCache:
    """
    Caches face recognition results and only re-runs FaceNet every N frames.
    Uses bounding-box IoU to match cached identities to current detections.
    """

    def __init__(self, recog_interval: int = 5):
        self.interval = recog_interval
        self.frame_count = 0
        self.cached: dict[int, tuple[str, float, tuple]] = {}

    def _iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def should_recognize(self) -> bool:
        return self.frame_count % self.interval == 0

    def tick(self):
        self.frame_count += 1

    def get_cached(self, box_xyxy) -> tuple[str, float] | None:
        for idx, (name, score, cached_box) in self.cached.items():
            if self._iou(box_xyxy, cached_box) > 0.3:
                return name, score
        return None

    def update(self, person_idx: int, name: str, score: float, box_xyxy: tuple):
        self.cached[person_idx] = (name, score, box_xyxy)

    def clear(self):
        self.cached.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Main inference loop  (OPTIMIZED v2)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(args):
    pose_model     = load_pose_model(use_onnx=args.onnx)
    activity_model = None  # Disabled for now
    print("  [SKIP] Activity model disabled.")

    # Phase 2 heads (disabled — emotion model 404)
    emotion_head, gesture_head = None, None

    # M2: Face Recognition
    print("  [M2] Loading FaceNet for face recognition...")
    facenet_model = load_facenet()
    face_db       = load_db()
    if face_db:
        print(f"  [M2] Database loaded: {list(face_db.keys())}")
    else:
        print("  [M2] No faces registered yet. Run: python m2_face_recognition.py --register 'YourName'")

    # FP16 for FaceNet on GPU
    if USE_HALF:
        facenet_model.half()
        print("  [OPT] FP16 half-precision enabled.")

    # ── Face recognition cache ──
    face_cache = FaceCache(recog_interval=args.face_every)
    print(f"  [OPT] Face recognition interval: every {args.face_every} frames")

    # ── YOLO input size ──
    yolo_imgsz = args.yolo_size
    print(f"  [OPT] YOLO input size: {yolo_imgsz}px")

    # ── Frame scaling ──
    scale = args.scale
    if scale < 1.0:
        print(f"  [OPT] Frame downscale factor: {scale}")

    # ── Open source with threaded capture ──
    source = args.source
    if source is None or source == "0":
        source = 0    # webcam

    try:
        cam = CameraStream(source)
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return

    print(f"\n  Running M1 inference (optimized v2). Press 'q' to quit.")
    print(f"  Device: {DEVICE} | Half: {USE_HALF} | Threaded capture: ON")
    print(f"  [FIX] Face cropping now uses YOLO keypoints (nose/eyes/ears)")
    fps_buffer = []

    # For YOLO frame-skipping: reuse last results on skipped frames
    skip_yolo = args.skip_yolo
    yolo_frame_count = 0
    last_results = None
    last_frame_for_yolo = None

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            break

        t0 = time.time()

        # ── Optional: downscale frame ──
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)

        # ── YOLOv8-pose inference (with optional frame-skipping) ──
        run_yolo = (yolo_frame_count % (skip_yolo + 1) == 0)
        yolo_frame_count += 1

        if run_yolo:
            # ONNX models require their exported size (640x640). PyTorch models can use dynamic size.
            is_onnx = str(getattr(pose_model, 'ckpt_path', '')).lower().endswith('.onnx')
            actual_imgsz = 640 if is_onnx else yolo_imgsz
            results = pose_model(frame, verbose=False, conf=0.4, iou=0.5,
                                 imgsz=actual_imgsz)[0]
            last_results = results
            last_frame_for_yolo = frame
        else:
            results = last_results  # Reuse previous YOLO results

        if results is None:
            draw_fps(frame, 0)
            cv2.imshow("M1+M2 Smart Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── Decide if we run face recognition this frame ──
        run_face = face_cache.should_recognize() and bool(face_db)
        face_cache.tick()

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf            = float(box.conf[0])
            box_tuple       = (x1, y1, x2, y2)

            # ── M2: Face Recognition (keypoint-based crop + caching) ──
            person_name, face_score = "Unknown", 0.0

            if run_face:
                # Use YOLO keypoints to get a proper face crop
                kps_for_person = None
                if results.keypoints is not None and i < len(results.keypoints):
                    kps_for_person = results.keypoints[i]

                face_crop = extract_face_from_keypoints(
                    frame, kps_for_person, (x1, y1, x2, y2)
                )

                if face_crop is not None and face_crop.size > 0:
                    person_name, face_score = recognize(facenet_model, face_crop, face_db)
                    face_cache.update(i, person_name, face_score, box_tuple)
            else:
                # Use cached result
                cached = face_cache.get_cached(box_tuple)
                if cached is not None:
                    person_name, face_score = cached

            # ── Phase 2: Emotion (disabled) ──
            emotion = "neutral"
            if emotion_head is not None:
                kps_for_person = None
                if results.keypoints is not None and i < len(results.keypoints):
                    kps_for_person = results.keypoints[i]
                face_crop = extract_face_from_keypoints(
                    frame, kps_for_person, (x1, y1, x2, y2)
                )
                if face_crop is not None and face_crop.size > 0:
                    try:
                        emot_label, _, _ = emotion_head.predict_emotions(face_crop, logits=False)
                        emotion = emot_label
                    except:
                        pass

            # ── Phase 2: Gestures (disabled) ──
            gesture = "idle"
            if gesture_head is not None:
                person_crop = frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])]
                if person_crop.size > 0:
                    hand_results = gesture_head.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                    if hand_results.multi_hand_landmarks:
                        gesture = get_gesture_label(hand_results.multi_hand_landmarks[0])

            # ── Activity classification ──
            activity, act_conf = "unknown", 0.0
            if activity_model is not None:
                activity, act_conf = predict_activity(activity_model, frame,
                                                       (x1, y1, x2, y2))

            # ── Logic Layer ──
            if person_name != "Unknown":
                id_text = f"Hello, {person_name}! ({face_score:.2f})"
                id_color = (0, 200, 0)
            else:
                id_text = f"Unknown ({face_score:.2f})"
                id_color = (0, 0, 200)

            status = f"{id_text} | Mood: {emotion} | Gesture: {gesture}"
            if person_name != "Unknown" and emotion == "happy":
                status = f"Hello {person_name}! You look happy today!"
            elif emotion in ["angry", "sad"]:
                status = f"{person_name if person_name != 'Unknown' else 'User'} seems distressed."
            elif gesture == "thumbs_up":
                status = f"{person_name if person_name != 'Unknown' else 'User'} agrees!"

            color = id_color
            label = f"{person_name} | {activity} | {conf:.2f}"
            draw_box(frame, x1, y1, x2, y2, label, color, status=status)

            # ── Keypoints ──
            if results.keypoints is not None and i < len(results.keypoints):
                draw_keypoints(frame, results.keypoints[i])

        # ── FPS ──
        elapsed = time.time() - t0
        fps_buffer.append(1.0 / max(elapsed, 1e-6))
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)
        draw_fps(frame, sum(fps_buffer) / len(fps_buffer))

        cv2.imshow("M1+M2 Smart Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 Vision Module — Inference (Optimized v2)")
    parser.add_argument("--source", default=None,
                        help="Webcam index (default 0) or video/image path")
    parser.add_argument("--onnx", action="store_true",
                        help="Use ONNX model for CPU inference")
    parser.add_argument("--face-every", type=int, default=5,
                        help="Run face recognition every N frames (default: 5)")
    parser.add_argument("--yolo-size", type=int, default=320,
                        help="YOLO input resolution (default: 320, lower=faster)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Downscale factor for input frame (e.g. 0.5 = half res)")
    parser.add_argument("--skip-yolo", type=int, default=0,
                        help="Skip N frames between YOLO runs (0=run every frame)")
    parser.add_argument("--fast", action="store_true",
                        help="Max speed preset (face-every=10, yolo-size=256, scale=0.75, skip-yolo=1)")
    args = parser.parse_args()

    # --fast preset overrides
    if args.fast:
        args.face_every = 10
        args.yolo_size  = 256
        args.scale      = 0.75
        args.skip_yolo  = 1

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   M1 Vision Module — Real-Time Inference (Optimized v2)║")
    print("╚══════════════════════════════════════════════════════════╝")
    run_inference(args)
