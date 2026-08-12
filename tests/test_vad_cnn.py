"""
==============================================================
Standalone VAD CNN Test Script
==============================================================
Tests your trained VADCNN model vs the current RMS approach
side by side in real-time from your microphone.

How to run:
    python test_vad_cnn.py

What it shows:
    CNN:  SPEECH (92%)  |  RMS: NOISE  |  Agree: ❌ NO
    ↑ trained model     ↑ current system   ↑ do they agree?

Use this to decide whether the CNN is better than RMS VAD
before integrating it into the main system.

Press Ctrl+C to stop. A summary will print at the end.
==============================================================
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import pyaudio
except ImportError:
    print("[ERROR] pyaudio not installed. Run: pip install pyaudio")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("[ERROR] librosa not installed. Run: pip install librosa")
    sys.exit(1)

from m3_vad.train_vad import VADCNN, N_MELS, TIME_STEPS, SR, CHUNK_DURATION

# ── Thresholds (same as main_integration.py) ──────────────────────────────────
SPEECH_THRESHOLD_RMS  = 0.008
SILENCE_THRESHOLD_RMS = 0.004
CNN_SPEECH_THRESHOLD  = 0.5     # CNN probability above this = SPEECH

# ── Load VADCNN ───────────────────────────────────────────────────────────────
WEIGHTS_PATH = ROOT / "m3_vad" / "weights" / "vad_best.pt"

print("=" * 65)
print("   VAD CNN Standalone Test — CNN vs RMS Side-by-Side")
print("=" * 65)

if not WEIGHTS_PATH.exists():
    print(f"\n[ERROR] Model not found at: {WEIGHTS_PATH}")
    print("        Train it first with: python m3_vad/train_vad.py")
    sys.exit(1)

model = VADCNN()
model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location="cpu"))
model.eval()
print(f"\n✅ VADCNN loaded  ({WEIGHTS_PATH.stat().st_size / 1024:.0f} KB)")
print(f"   Architecture  : 3x Conv2D → GlobalAvgPool → FC → Sigmoid")
print(f"   Input shape   : (1, {N_MELS}, {TIME_STEPS}) mel-spectrogram")
print(f"   CNN threshold : {CNN_SPEECH_THRESHOLD}")
print(f"   RMS threshold : speech > {SPEECH_THRESHOLD_RMS}, silence < {SILENCE_THRESHOLD_RMS}")


# ── Feature Extraction (matches train_vad.py exactly) ─────────────────────────
def extract_features(audio_chunk: np.ndarray):
    """Convert raw float32 audio to VADCNN input tensor."""
    try:
        target_len = int(SR * CHUNK_DURATION)
        if len(audio_chunk) < target_len:
            audio_chunk = np.pad(audio_chunk, (0, target_len - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[:target_len]

        spec = librosa.feature.melspectrogram(
            y=audio_chunk, sr=SR, n_mels=N_MELS, n_fft=512, hop_length=256
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)

        if spec.shape[1] > TIME_STEPS:
            spec = spec[:, :TIME_STEPS]
        elif spec.shape[1] < TIME_STEPS:
            spec = np.pad(spec, ((0, 0), (0, TIME_STEPS - spec.shape[1])))

        return torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, N_MELS, TIME_STEPS)
    except Exception as e:
        print(f"\n[FEATURE ERROR] {e}")
        return None


# ── Open Microphone ───────────────────────────────────────────────────────────
p = pyaudio.PyAudio()
chunk_samples = int(SR * CHUNK_DURATION)

stream = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=SR,
    input=True,
    frames_per_buffer=chunk_samples
)

print(f"\n🎙️  Microphone open — listening at {SR}Hz, {CHUNK_DURATION}s chunks")
print(f"\n{'THINGS TO TRY':}")
print(f"  • Speak clearly      → both should say SPEECH")
print(f"  • Stay silent        → both should say NOISE")
print(f"  • Play music/TV      → see if CNN handles it better than RMS")
print(f"  • Clap/knock on desk → see which triggers incorrectly")
print(f"\nPress Ctrl+C to stop.\n")

# ── Header ────────────────────────────────────────────────────────────────────
print(f"{'CNN Result':<16} {'Confidence':>10}  {'RMS Result':<16} {'RMS':>8}  {'Match':>6}  {'Rate':>6}")
print("─" * 70)

# ── Stats ─────────────────────────────────────────────────────────────────────
total     = 0
agree     = 0
cnn_only_speech  = 0   # CNN says speech but RMS says noise
rms_only_speech  = 0   # RMS says speech but CNN says noise

# ── Main Loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        # Read audio chunk from mic
        raw = stream.read(chunk_samples, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=np.float32)

        # ── RMS VAD decision (current system) ──
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > SPEECH_THRESHOLD_RMS:
            rms_label  = "SPEECH"
            rms_icon   = "🗣️ "
        elif rms < SILENCE_THRESHOLD_RMS:
            rms_label  = "NOISE "
            rms_icon   = "🔇 "
        else:
            rms_label  = "unsure"
            rms_icon   = "❓ "

        # ── CNN VAD decision ──
        features = extract_features(chunk)
        if features is not None:
            with torch.no_grad():
                cnn_prob = float(model(features).item())
            cnn_label = "SPEECH" if cnn_prob > CNN_SPEECH_THRESHOLD else "NOISE "
            cnn_icon  = "🗣️ " if cnn_label.strip() == "SPEECH" else "🔇 "
            cnn_conf  = cnn_prob if cnn_label.strip() == "SPEECH" else (1.0 - cnn_prob)
        else:
            cnn_label = "ERROR "
            cnn_icon  = "⚠️  "
            cnn_conf  = 0.0

        # ── Statistics ──
        total += 1
        match = cnn_label.strip() == rms_label.strip()
        if match:
            agree += 1
            match_str = "✅"
        else:
            match_str = "❌"
            if cnn_label.strip() == "SPEECH" and rms_label.strip() == "NOISE":
                cnn_only_speech += 1
            elif rms_label.strip() == "SPEECH" and cnn_label.strip() == "NOISE":
                rms_only_speech += 1

        rate = (agree / total) * 100

        # ── Print line ──
        print(
            f"\r{cnn_icon}{cnn_label:<12} {cnn_conf:>9.1%}  "
            f"{rms_icon}{rms_label:<12} {rms:>8.4f}  "
            f"{match_str:>6}  {rate:>5.0f}%",
            end="", flush=True
        )

except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

# ── Summary ───────────────────────────────────────────────────────────────────
agree_rate = (agree / total * 100) if total > 0 else 0
print(f"\n\n{'='*65}")
print(f"  TEST SUMMARY  ({total} chunks × {CHUNK_DURATION}s = {total*CHUNK_DURATION:.0f}s of audio)")
print(f"{'='*65}")
print(f"  Overall agreement   : {agree_rate:.1f}%")
print(f"  CNN said SPEECH, RMS said NOISE  : {cnn_only_speech} times")
print(f"  RMS said SPEECH, CNN said NOISE  : {rms_only_speech} times")
print()

# Recommendation
if agree_rate >= 92:
    print("  ✅ VERDICT: CNN and RMS agree strongly.")
    print("     Both approaches perform similarly in your environment.")
    print("     Keeping RMS (simpler) is fine. CNN adds no significant benefit.")

elif agree_rate >= 75:
    print("  ⚠️  VERDICT: Moderate disagreement detected.")
    if cnn_only_speech > rms_only_speech:
        print("     CNN is MORE sensitive — catches sounds RMS misses.")
        print("     If those were real speech → CNN is better.")
        print("     If those were noise → CNN needs threshold adjustment.")
    else:
        print("     RMS is MORE sensitive — catches sounds CNN misses.")
        print("     If those were real speech → CNN might be missing detections.")
        print("     If those were noise → CNN is filtering better.")
    print("     Recommendation: Review cases with ❌ manually, then decide.")

else:
    print("  ❌ VERDICT: High disagreement. Investigate before integrating CNN.")
    print("     Possible causes:")
    print("       - CNN threshold needs tuning (try 0.3 or 0.7 instead of 0.5)")
    print("       - Model trained on different noise profile than your environment")
    print("       - Mic input level/gain is different from training data")

print(f"\n  To tune CNN threshold: edit CNN_SPEECH_THRESHOLD at top of file.")
print(f"{'='*65}")
