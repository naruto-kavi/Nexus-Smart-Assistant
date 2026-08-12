"""
==============================================================
M2 — Face Recognition Module (Runtime Registration)
==============================================================
No training required.
Uses FaceNet (InceptionResnetV1 pretrained on VGGFace2).

How it works:
  • Registration : Capture 5 face crops → average embeddings → save to face_db.pkl
  • Recognition  : Live face crop → embedding → cosine similarity → name or "Unknown"

Usage (standalone registration):
  python m2_face_recognition.py --register "Mani"
  python m2_face_recognition.py --register "Ravi"
  python m2_face_recognition.py --list
  python m2_face_recognition.py --delete "Ravi"
==============================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# ─── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DB_PATH    = ROOT / "weights" / "face_db.pkl"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD  = 0.60          # Cosine similarity threshold (0.6 = strict, 0.5 = relaxed)
N_CAPTURES = 5             # Number of face crops to average per registration
FACE_SIZE  = 160           # FaceNet input size


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_facenet():
    """Load FaceNet — ONNX (CPU-optimized) if exported, PyTorch otherwise."""
    onnx_path = ROOT / "weights" / "facenet.onnx"

    if onnx_path.exists() and not torch.cuda.is_available():
        try:
            import onnxruntime as ort
            print("  [M2] Loading FaceNet (ONNX — CPU-optimized)...")
            session = ort.InferenceSession(
                str(onnx_path), providers=['CPUExecutionProvider']
            )
            print(f"  [M2] FaceNet ONNX loaded  ({onnx_path.stat().st_size // (1024*1024)} MB).")
            return session
        except ImportError:
            print("  [M2] onnxruntime not found, falling back to PyTorch.")
        except Exception as e:
            print(f"  [M2] ONNX load failed ({e}), falling back to PyTorch.")

    print("  [M2] Loading FaceNet (PyTorch — VGGFace2 pretrained)...")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    print(f"  [M2] FaceNet loaded on {DEVICE}.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Helpers
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR face crop to a FaceNet-ready tensor. Uses cv2 (faster than PIL)."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (FACE_SIZE, FACE_SIZE),
                              interpolation=cv2.INTER_LINEAR)
    face_np  = face_resized.astype(np.float32) / 255.0
    face_np  = (face_np - 0.5) / 0.5   # Normalize to [-1, 1]
    tensor   = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor


@torch.no_grad()
def get_embedding(model, face_bgr: np.ndarray) -> np.ndarray:
    """Generate a 512-dim L2-normalized embedding for a face crop.
    Works with both PyTorch model and onnxruntime InferenceSession.
    """
    tensor = preprocess_face(face_bgr)

    # Detect model type: onnxruntime session has a .run() method
    if hasattr(model, 'run'):
        # ONNX inference
        inp = tensor.cpu().numpy()
        embedding = model.run(None, {'input': inp})[0][0]       # (512,)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    else:
        # PyTorch inference
        embedding = model(tensor)
        embedding = F.normalize(embedding, p=2, dim=1)          # L2 normalize
        embedding = embedding.cpu().numpy()[0]                  # (512,)

    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ══════════════════════════════════════════════════════════════════════════════
# Database Operations
# ══════════════════════════════════════════════════════════════════════════════

def load_db() -> dict:
    """Load face database from disk. Returns {} if not found."""
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db: dict):
    """Persist face database to disk."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"  [M2] Database saved → {DB_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# Recognition
# ══════════════════════════════════════════════════════════════════════════════

def recognize(model, face_bgr: np.ndarray, db: dict) -> tuple[str, float]:
    """
    Match a face crop against the registered database.
    Returns (name, similarity_score). 
    Returns ("Unknown", score) if no match above threshold.
    """
    if not db or face_bgr is None or face_bgr.size == 0:
        return "Unknown", 0.0

    live_emb = get_embedding(model, face_bgr)

    best_name  = "Unknown"
    best_score = 0.0

    for name, stored_emb in db.items():
        score = cosine_similarity(live_emb, stored_emb)
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score >= THRESHOLD:
        return best_name, best_score
    return "Unknown", best_score


# ══════════════════════════════════════════════════════════════════════════════
# Registration (Standalone Webcam)
# ══════════════════════════════════════════════════════════════════════════════

def register_user(name: str):
    """
    Open webcam, capture N_CAPTURES face crops, average their embeddings,
    and save to face_db.pkl.
    """
    model = load_facenet()
    db    = load_db()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open webcam.")
        return

    print(f"\n  Registering: '{name}'")
    print(f"  Look at the camera. Press SPACE to capture ({N_CAPTURES} needed).")
    print(f"  Press Q to cancel.\n")

    embeddings = []
    count      = 0

    while count < N_CAPTURES:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw guide box in center
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        box_size = min(h, w) // 3
        x1, y1 = cx - box_size, cy - box_size
        x2, y2 = cx + box_size, cy + box_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Status text
        cv2.putText(frame, f"Registering: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {count}/{N_CAPTURES}  [SPACE=capture  Q=cancel]",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("M2 — Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("  Cancelled.")
            break

        if key == ord(' '):
            # Crop and embed the center region
            face_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if face_crop.size > 0:
                emb = get_embedding(model, face_crop)
                embeddings.append(emb)
                count += 1
                print(f"  ✓ Captured {count}/{N_CAPTURES}")
                # Flash feedback
                flash = frame.copy()
                cv2.rectangle(flash, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.imshow("M2 — Register Face", flash)
                cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("  No embeddings captured. Aborted.")
        return

    # Average all captured embeddings → single robust embedding
    mean_emb   = np.mean(embeddings, axis=0)
    mean_emb  /= np.linalg.norm(mean_emb)   # Re-normalize after averaging
    db[name]   = mean_emb
    save_db(db)

    print(f"\n  ✅ '{name}' registered successfully with {len(embeddings)} captures!")
    print(f"  Database now has {len(db)} registered user(s): {list(db.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M2 Face Recognition — Registration Tool")
    parser.add_argument("--register", type=str, help="Register a new user by name")
    parser.add_argument("--list",     action="store_true", help="List all registered users")
    parser.add_argument("--delete",   type=str, help="Delete a registered user by name")
    args = parser.parse_args()

    if args.register:
        register_user(args.register)

    elif args.list:
        db = load_db()
        if not db:
            print("  No users registered yet.")
        else:
            print(f"\n  Registered Users ({len(db)}):")
            for name in db.keys():
                print(f"    • {name}")

    elif args.delete:
        db = load_db()
        if args.delete in db:
            del db[args.delete]
            save_db(db)
            print(f"  ✓ '{args.delete}' deleted from database.")
        else:
            print(f"  '{args.delete}' not found in database.")

    else:
        parser.print_help()


if __name__ == "__main__":
    print("=" * 58)
    print("  M2 -- Face Recognition Module (Runtime Registration)")
    print("=" * 58)
    main()
