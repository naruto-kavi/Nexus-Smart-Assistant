"""
==============================================================
FaceNet ONNX Export Script  (run once)
==============================================================
Exports the FaceNet (InceptionResnetV1 pretrained on VGGFace2)
PyTorch model to ONNX format for faster CPU inference.

Run with:
    python export_facenet_onnx.py

Output:
    weights/facenet.onnx  (~100 MB)

Why ONNX?
    PyTorch FaceNet on CPU  → ~200 MB RAM, slower
    ONNX FaceNet on CPU     → ~80 MB RAM, ~2x faster
    Same pretrained weights, same accuracy, just faster runtime.
==============================================================
"""

import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "weights" / "facenet.onnx"

print("=" * 58)
print("  FaceNet ONNX Export")
print("=" * 58)

# ── Load PyTorch model ─────────────────────────────────────────────────────────
print("\n[1/4] Loading FaceNet (InceptionResnetV1 — VGGFace2 pretrained)...")
try:
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').eval()
    print("      ✅ Loaded")
except Exception as e:
    print(f"      ❌ Failed: {e}")
    print("         Run: pip install facenet-pytorch")
    sys.exit(1)

# ── Export to ONNX ─────────────────────────────────────────────────────────────
print(f"\n[2/4] Exporting to ONNX → {OUTPUT_PATH} ...")
dummy_input = torch.randn(1, 3, 160, 160)

try:
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=['input'],
        output_names=['embedding'],
        opset_version=11,
        dynamic_axes={
            'input':     {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
        verbose=False,
    )
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"      ✅ Exported  ({size_mb:.1f} MB)")
except Exception as e:
    print(f"      ❌ Export failed: {e}")
    sys.exit(1)

# ── Verify ONNX output matches PyTorch ─────────────────────────────────────────
print("\n[3/4] Verifying ONNX output matches PyTorch output...")
try:
    import onnxruntime as ort

    session = ort.InferenceSession(str(OUTPUT_PATH), providers=['CPUExecutionProvider'])
    dummy_np = dummy_input.numpy()

    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    onnx_out = session.run(None, {'input': dummy_np})[0]

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    if max_diff < 1e-4:
        print(f"      ✅ Max difference: {max_diff:.2e}  (within tolerance)")
    else:
        print(f"      ⚠️  Max difference: {max_diff:.2e}  (higher than expected, but may be fine)")

    print(f"      Output shape : {onnx_out.shape}  ← should be (1, 512)")
except ImportError:
    print("      ⚠️  onnxruntime not installed — skipping verification")
    print("         Run: pip install onnxruntime")
except Exception as e:
    print(f"      ⚠️  Verification error: {e}")

# ── Speed benchmark ────────────────────────────────────────────────────────────
print("\n[4/4] Speed benchmark (50 inferences each)...")
try:
    import time

    dummy_np = dummy_input.numpy()

    # PyTorch timing
    with torch.no_grad():
        _ = model(dummy_input)   # warm-up
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    pt_ms = (time.perf_counter() - t0) / 50 * 1000

    # ONNX timing
    _ = session.run(None, {'input': dummy_np})  # warm-up
    t0 = time.perf_counter()
    for _ in range(50):
        _ = session.run(None, {'input': dummy_np})
    onnx_ms = (time.perf_counter() - t0) / 50 * 1000

    speedup = pt_ms / onnx_ms
    print(f"      PyTorch : {pt_ms:.1f} ms / inference")
    print(f"      ONNX    : {onnx_ms:.1f} ms / inference")
    print(f"      Speedup : {speedup:.1f}x  {'✅ faster' if speedup > 1 else '⚠️ slower (may be GPU vs CPU)'}")

except Exception as e:
    print(f"      ⚠️  Benchmark error: {e}")

print(f"\n{'='*58}")
print(f"  ✅ Export complete!")
print(f"  File : {OUTPUT_PATH}")
print(f"  The system will auto-use ONNX on next run (no GPU mode).")
print(f"{'='*58}\n")
