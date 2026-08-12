from sklearn import pipeline   # Keep ONLY if your project actually uses sklearn.pipeline elsewhere

import cv2
import threading
import queue
import time
import numpy as np
import torch
import librosa
import pyaudio
from pathlib import Path
from faster_whisper import WhisperModel
import sys
import warnings
import os
import argparse

# Fix Windows console encoding issues with emojis/surrogates
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    sys.stdin.reconfigure(encoding='utf-8')
except Exception:
    pass

class _SafeWriter:
    """Wraps stdout/stderr — silently replaces unencodable surrogate chars."""
    def __init__(self, wrapped):
        self._wrapped = wrapped
    def write(self, text):
        try:
            self._wrapped.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                self._wrapped.write(text.encode('utf-8', 'replace').decode('utf-8'))
            except Exception:
                pass
    def flush(self):
        try:
            self._wrapped.flush()
        except Exception:
            pass
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

sys.stdout = _SafeWriter(sys.stdout)
sys.stderr = _SafeWriter(sys.stderr)

def clean_surrogates(text):
    if not isinstance(text, str):
        return text
    try:
        # surrogatepass handles lone surrogate code points (common in Whisper Telugu output)
        return text.encode('utf-8', 'surrogatepass').decode('utf-8', 'replace')
    except Exception:
        try:
            return text.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            return ""

# Ignore warnings for cleaner console output
warnings.filterwarnings("ignore")

# Add current directory to path so we can import from modules
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

# Load .env manually to populate environment variables
def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env()

# Parse arguments at startup
parser = argparse.ArgumentParser(description="Nexus Multimodal Assistant Integration")
parser.add_argument("--lang", default="auto", choices=["auto", "en", "te"],
                    help="Default language mode (auto, en, or te)")
args, unknown = parser.parse_known_args()

# ==============================================================================
# M1 & M2 Imports
# ==============================================================================
from m1.infer_m1 import load_pose_model, CameraStream, FaceCache, extract_face_from_keypoints, draw_box
from m2_face_recognition import load_facenet, recognize, load_db

# ==============================================================================
# M3 Imports
# ==============================================================================
from m3_vad.train_vad import VADCNN, N_MELS, TIME_STEPS, SR, CHUNK_DURATION

# ==============================================================================
# M5 Imports
# ==============================================================================
from m5_llm_chatbot import SmartAssistant

WEIGHTS_DIR = ROOT / "weights"

# ==============================================================================
# Shared State & Buffers
# ==============================================================================
class SharedState:
    def __init__(self):
        self.current_user = "Unknown"
        self.is_speaking = False
        self.speech_models_loaded = False
        self.tts_active = False
        self.chat_queue = queue.Queue()
        self.register_name = None  # Set when /reg command is used
        self.reg_embeddings = []   # Buffer for face captures
        self.lang_mode = args.lang  # "auto", "en", "te"
        self.deaf_until = 0.0      # Ignore mic input until this timestamp (echo suppression)
        self.suppress_speak = False  # When True, block nexus.speak() — set by stop, cleared by new command

state = SharedState()
audio_queue = queue.Queue()

# ==============================================================================
# Audio Recording Thread (M3 - Mic Input)
# ==============================================================================
def audio_recording_thread():
    print("🎙️  [MIC] Audio Thread Started. Listening...")
    p = pyaudio.PyAudio()
    
    # 0.5s chunk = 8000 samples at 16kHz
    chunk_samples = int(SR * CHUNK_DURATION) 
    
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SR,
                    input=True,
                    frames_per_buffer=chunk_samples)
    
    while True:
        try:
            data = stream.read(chunk_samples, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_queue.put(audio_chunk)
        except Exception as e:
            pass

# ==============================================================================
# Terminal Input Thread (Chat & Commands)
# ==============================================================================
def terminal_input_thread():
    while not state.speech_models_loaded:
        time.sleep(1.0)
    
    while True:
        try:
            user_input = clean_surrogates(input().strip())
            if not user_input:
                continue
            
            # INSTANT STOP: Stop speaking the moment user types anything
            if nexus.is_speaking():
                nexus.stop_speaking()
                state.suppress_speak = True   # Block any response currently in the LLM pipeline
                state.deaf_until = time.time() + 2.0
                time.sleep(0.1)

            if user_input.lower() in ['stop', 'quiet', 'shh', 'hush']:
                nexus.stop_speaking()
                state.suppress_speak = True
                state.deaf_until = time.time() + 2.0
                print("\U0001F6D1 [CHAT] Nexus silenced.")
                continue
            
            if user_input.startswith("/lang"):
                parts = user_input.split(" ", 1)
                if len(parts) > 1:
                    mode = clean_surrogates(parts[1].strip().lower())
                    if mode in ["en", "english"]:
                        state.lang_mode = "en"
                        print("🌐 [LANG] Language locked to ENGLISH.")
                    elif mode in ["te", "telugu"]:
                        state.lang_mode = "te"
                        print("🌐 [LANG] Language locked to TELUGU.")
                    elif mode in ["auto"]:
                        state.lang_mode = "auto"
                        print("🌐 [LANG] Language auto-detection enabled.")
                    else:
                        print("⚠️ [LANG] Invalid language. Use: /lang [en|te|auto]")
                else:
                    print(f"🌐 [LANG] Current mode: {state.lang_mode.upper()}. Usage: /lang [en|te|auto]")
                continue

            if user_input.startswith("/reg") or user_input.startswith("/register"):
                parts = user_input.split(" ", 1)
                if len(parts) > 1:
                    name = clean_surrogates(parts[1].strip())
                    print(f"📸 [REG] Starting registration for '{name}'. Please look at the camera.")
                    state.reg_embeddings = []
                    state.register_name = name
                else:
                    print("⚠️ [REG] Usage: /reg [Name]")
            elif user_input.lower() in ['/quit', '/exit', 'quit', 'exit', 'q']:
                print("👋 Goodbye!")
                os._exit(0)
            else:
                # Regular text chat
                state.chat_queue.put(user_input)
        except EOFError:
            break
        except Exception as e:
            print(f"Terminal Error: {e}")

# ==============================================================================
# VAD & STT Processing Thread (M3 & M4)
# ==============================================================================
# Global Assistant for cross-thread interruption
print("🤖 [LLM] Initializing Nexus (M5)...")
nexus = SmartAssistant("Nexus")

def speech_processing_thread():
    print("🧠 [VAD] Voice Activity Detection Thread Started.")
    print("📝 Loading Faster-Whisper...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = (
        "float16"
        if device == "cuda"
        else "int8"
        )
    whisper_model = WhisperModel(
        "small",
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
        num_workers=1
        )
    print("Faster-Whisper loaded.")
    
    
    # --- Warmup: Pre-warm both models to eliminate PyTorch cold-start delay ---
    # First inference is slow because PyTorch JIT-compiles the graph.
    # Running a tiny dummy pass here during load time (expected wait) avoids
    # a surprise 40-second freeze on the user's first Telugu query.

    # --- M5 LLM Assistant is now global for interruption support ---
    state.speech_models_loaded = True

    # --- Simple, Reliable Energy-Based VAD ---
    SPEECH_THRESHOLD = 0.008   # Reverted to 0.008 to be more stable
    SILENCE_THRESHOLD = 0.004
    MAX_SILENCE_CHUNKS = 4     # 4 × 0.5s = 2 seconds of silence before transcription (was 6 = 3s)
    INTERRUPT_THRESHOLD = 0.12  # Significantly higher to avoid self-interruption from speakers
    
    speech_buffer = []
    silence_chunks = 0
    in_speech = False

    while True:
        # Check for Text Chat first (Priority)
        if not state.chat_queue.empty():
            msg = state.chat_queue.get()
            print(f"\n\U0001F464 {state.current_user} (Text): {msg}")
            state.suppress_speak = False  # User is actively chatting — re-enable speech
            response = nexus.generate_response(msg, user_name=state.current_user)
            if not state.suppress_speak:  # Check again in case stop was hit during LLM
                print(f"\U0001F916 Nexus: {response}\n")
                nexus.speak(response)
            else:
                print("[CHAT] Response discarded — stop was pressed during LLM call.")

        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Check if Nexus is currently speaking
        is_nexus_speaking = nexus.is_speaking()
        state.tts_active = is_nexus_speaking

        rms = np.sqrt(np.mean(chunk ** 2))

        # ── Deaf period: ignore mic after stop to suppress speaker bleed echo ──
        if time.time() < state.deaf_until:
            in_speech = False
            speech_buffer = []
            silence_chunks = 0
            continue

        # Interruption Logic: If Nexus is speaking and we hear a LOUD voice, STOP.
        if is_nexus_speaking:
            if rms > INTERRUPT_THRESHOLD:
                print("\n🛑 [INTERRUPT] User spoke over Nexus.")
                nexus.stop_speaking()
                state.deaf_until = time.time() + 2.0  # 2s deaf after interrupt
                in_speech = False
                speech_buffer = []
                silence_chunks = 0
                time.sleep(0.3)
            else:
                # IMPORTANT: If Nexus is speaking and it's NOT an interruption,
                # we IGNORE the sound entirely. This prevents Nexus from hearing itself.
                print(f"\r[MIC] {'Nexus Talking '} | RMS: {rms:.4f} | TTS: ON ", end="", flush=True)
                in_speech = False
                speech_buffer = []
                silence_chunks = 0
                continue
        
        print(f"\r[MIC] {'🗣️  SPEAKING' if in_speech else 'silent  '} | RMS: {rms:.4f} | TTS: OFF", end="", flush=True)

        if rms > SPEECH_THRESHOLD:
            # --- SPEECH ---
            in_speech = True
            silence_chunks = 0
            speech_buffer.extend(chunk.tolist())
            state.is_speaking = True
            
        elif rms < SILENCE_THRESHOLD:
            # --- SILENCE ---
            state.is_speaking = False
            if in_speech:
                silence_chunks += 1
                # Keep adding audio during short pauses (natural breathing gaps)
                speech_buffer.extend(chunk.tolist())
                
                if silence_chunks >= MAX_SILENCE_CHUNKS:
                    # --- End of speech detected! Transcribe! ---
                    in_speech = False
                    audio_data = np.array(speech_buffer, dtype=np.float32)
                    speech_buffer = []
                    silence_chunks = 0
                    
                    # Only transcribe if meaningful audio (> 0.5 seconds)
                    if len(audio_data) > SR * 0.5:
                        try:
                            # ── Step 1: Compute mel spectrogram ONCE (shared across all paths) ──
                            # -----------------------------
                            # # Faster-Whisper Transcription
                            # -----------------------------
                            language = None
                            # Respect manual language selection
                            if state.lang_mode == "en":
                                language = "en"
                            elif state.lang_mode == "te":
                                language = "te"
                            print("\n[STT] 🎙️ Transcribing...")
                            print(f"[DEBUG] Audio duration: {len(audio_data)/SR:.2f} sec")
                            segments, info = whisper_model.transcribe(
                                audio_data,
                                language=language,
                                beam_size=5,
                                vad_filter=False,
                                vad_parameters=dict(
                                    min_silence_duration_ms=500,
                                    speech_pad_ms=200,
                                ),
                                condition_on_previous_text=False,
                                )
                            user_transcription = clean_surrogates(
                                "".join(segment.text for segment in segments).strip()
                                )
                            print(f"[DEBUG] Raw transcription: '{user_transcription}'")
                            print(f"[DEBUG] Language: {info.language}")
                            print(f"[DEBUG] Probability: {info.language_probability}")
                            prob = info.language_probability or 0.0
                            if state.lang_mode == "en":
                                lang_code = "EN"
                            elif state.lang_mode == "te":
                                lang_code = "TE"
                            else:
                                if prob < 0.70:
                                    lang_code = "EN"   # default when confidence is low
                                elif info.language.startswith("te"):
                                    lang_code = "TE"
                                else:
                                    lang_code = "EN"
                                print(f"[STT] Language: {info.language} ({prob:.2f}) -> {lang_code}")

                            # ── Flush stale audio chunks that piled up during Whisper processing ──
                            flushed = 0
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                    flushed += 1
                                except queue.Empty:
                                    break
                            if flushed > 0:
                                print(f"[VAD] 🗑️  Flushed {flushed} stale audio chunks.")

                            if user_transcription:
                                print(f"\n{'─'*50}")
                                print(f"🌐 [Language Detected]: {lang_code}")
                                print(f"👤 {state.current_user}: {user_transcription}")

                                # ── Voice stop commands — intercept BEFORE sending to LLM ──
                                # If user says "stop", "quiet", "shh", etc. while Nexus is speaking,
                                # we silence immediately instead of generating a new response.
                                _clean = user_transcription.strip().lower()
                                VOICE_STOP_WORDS = {
                                    "stop", "quiet", "shh", "hush", "silence",
                                    "shut up", "be quiet", "stop talking", "stop speaking",
                                    "stop it", "ok stop", "okay stop", "nexus stop",
                                    # Telugu equivalents
                                    "ఆపు", "ఆగు", "చాలు", "సైలెంట్"
                                }
                                if any(_clean == w or _clean.endswith(w) for w in VOICE_STOP_WORDS):
                                    if nexus.is_speaking():
                                        nexus.stop_speaking()
                                        print("🛑 [VOICE] Speech stopped by voice command.")
                                    else:
                                        print("\U0001F6D1 [VOICE] Stop command received (nothing playing).")
                                    nexus.stop_speaking()
                                    state.suppress_speak = True  # Block in-flight LLM response
                                    state.deaf_until = time.time() + 2.0
                                    in_speech = False
                                    speech_buffer = []
                                    silence_chunks = 0
                                    print(f"{'─'*50}\n")
                                    continue  # Do NOT send to LLM

                                # Clear suppress flag — user gave a real new command
                                state.suppress_speak = False

                                # Route audio inputs intelligently by tagging the prompt
                                prompt_with_lang = f"[Language: {lang_code}] {user_transcription}"
                                response = nexus.generate_response(prompt_with_lang, user_name=state.current_user)

                                # Check again after LLM — stop may have been pressed while LLM was thinking
                                if state.suppress_speak:
                                    print("[VAD] Response discarded — stop was pressed during LLM call.")
                                    print(f"{'─'*50}\n")
                                    continue

                                print(f"\U0001F916 Nexus: {response}")
                                print(f"{'─'*50}\n")
                                nexus.speak(response)
                        except Exception as e:
                            print(f"\n[STT ERROR] {e}")
                            # Always flush stale audio after any error to keep mic responsive
                            _flushed = 0
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                    _flushed += 1
                                except queue.Empty:
                                    break
                            if _flushed:
                                print(f"[VAD] Flushed {_flushed} stale chunks after STT error.")

# ==============================================================================
# Vision Thread (M1 & M2)
# ==============================================================================
def vision_thread():
    print("👁️  [VISION] Camera Thread Started.")
    from m2_face_recognition import get_embedding, save_db
    # Auto-select YOLO format: ONNX is faster on CPU (Pi/no-GPU), PyTorch is faster with CUDA (GPU)
    # Force PyTorch model — the ONNX export is broken (outputs raw logits ~640
    # instead of probabilities 0-1, making NMS useless → 300 ghost boxes).
    # Use _use_onnx = False until the ONNX is re-exported correctly.
    _use_onnx = False
    print(f"   [VISION] YOLO mode: PyTorch .pt (ONNX disabled — export issue)")
    pose_model = load_pose_model(use_onnx=_use_onnx)
    facenet_model = load_facenet()
    face_db = load_db()
    # Run face recognition every 20 frames (was 10) to reduce CPU load
    face_cache = FaceCache(recog_interval=5)  # Recognize every 5 frames for faster ID
    frame_count = 0
    last_results = None
    
    try:
        cam = CameraStream(0)
    except:
        print("[ERROR] Camera not found. Please connect a webcam.")
        return
        
    print("\n⏳ Waiting for Voice & Brain models to finish loading...")
    while not state.speech_models_loaded:
        time.sleep(0.5)
        
    print("\n✅ SYSTEM READY! Look at the camera and start speaking.")
    print("\n" + "="*55)
    print(" ✅ SYSTEM READY!")
    print(" 💬 CHAT: Just type in this terminal and press ENTER")
    print(" 👤 REGISTER: Type '/reg Name' to add a new person")
    print(" 🎤 VOICE: Just start speaking")
    print("="*55 + "\n")
        
    while True:
        t0 = time.time()
        ret, frame = cam.read()
        if not ret or frame is None:
            break
        
        frame_count += 1
        
        # --- PERFORMANCE FIX: Only run heavy YOLO every 2nd frame ---
        # On skipped frames, just redraw the last result. This halves CPU usage!
        if frame_count % 2 == 0 or last_results is None:
            # ONNX models require their exported size (640x640). PyTorch models can use dynamic size (256x256).
            is_onnx = str(getattr(pose_model, 'ckpt_path', '')).lower().endswith('.onnx')
            yolo_imgsz = 640 if is_onnx else 256
            # iou=0.3 (was 0.5) → tighter NMS suppresses ghost/duplicate ONNX boxes
            # max_det=5 → cap detections to avoid flooding the frame
            results = pose_model(frame, verbose=False, conf=0.4, iou=0.3, imgsz=yolo_imgsz, max_det=5)[0]
            last_results = results
        else:
            results = last_results
            
        run_face = face_cache.should_recognize() and bool(face_db)
        face_cache.tick()
        
        if results.boxes:
            # Find the biggest box (most likely the person we are interacting with)
            main_idx = -1
            max_area = 0
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2-x1)*(y2-y1)
                if area > max_area:
                    max_area = area
                    main_idx = i

            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_name, face_score = "Unknown", 0.0
                
                # --- REGISTRATION LOGIC ---
                if state.register_name and i == main_idx:
                    kps = results.keypoints[i] if results.keypoints is not None else None
                    face_crop = extract_face_from_keypoints(frame, kps, (x1, y1, x2, y2))
                    if face_crop is not None and face_crop.size > 0:
                        emb = get_embedding(facenet_model, face_crop)
                        state.reg_embeddings.append(emb)
                        # UI Feedback
                        cv2.putText(frame, f"Capturing {state.register_name}: {len(state.reg_embeddings)}/5", 
                                   (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        if len(state.reg_embeddings) >= 5:
                            # Finalize registration
                            mean_emb = np.mean(state.reg_embeddings, axis=0)
                            mean_emb /= np.linalg.norm(mean_emb)
                            face_db[state.register_name] = mean_emb
                            save_db(face_db)
                            print(f"\n✅ Registered {state.register_name}!")
                            state.current_user = state.register_name
                            state.register_name = None
                            state.reg_embeddings = []
                    
                # --- RECOGNITION LOGIC ---
                # Only run face recognition on the main (biggest) box to save CPU
                elif run_face and i == main_idx:
                    kps = results.keypoints[i] if results.keypoints is not None else None
                    face_crop = extract_face_from_keypoints(frame, kps, (x1, y1, x2, y2))
                    if face_crop is not None and face_crop.size > 0:
                        person_name, face_score = recognize(facenet_model, face_crop, face_db)
                        # recognize() already filters below THRESHOLD (0.60) — no extra check needed
                        face_cache.update(i, person_name, face_score, (x1,y1,x2,y2))
                else:
                    cached = face_cache.get_cached((x1,y1,x2,y2))
                    if cached:
                        person_name, face_score = cached
                
                # Update shared state with the biggest face / last face seen
                if i == main_idx and person_name != "Unknown":
                    state.current_user = person_name

                color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                # Flash cyan box if registering
                if state.register_name and i == main_idx:
                    color = (255, 255, 0)
                # Only show VAD status on the main box — prevents overlapping text spam
                status = f"VAD: {'SPEECH' if state.is_speaking else 'silence'}" if i == main_idx else ""
                draw_box(frame, x1, y1, x2, y2, f"{person_name} ({face_score:.2f})", color, status=status)
                
        # --- FPS CALCULATION ---
        if 'fps_buffer' not in locals():
            fps_buffer = []
        
        elapsed = time.time() - t0 if 't0' in locals() else 0.033 # Default to ~30fps if t0 not set
        t0 = time.time()
        
        fps_buffer.append(1.0 / max(elapsed, 1e-6))
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        
        # Draw FPS on frame
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show STOP hint when Nexus is speaking
        if nexus.is_speaking():
            h = frame.shape[0]
            cv2.rectangle(frame, (0, h-40), (frame.shape[1], h), (0, 0, 180), -1)
            cv2.putText(frame, "SPEAKING  |  Press SPACE or S to stop",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("M1-M4 Assistant Integration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # SPACE or S = stop Nexus from speaking instantly (camera window has focus)
        elif key == ord(' ') or key == ord('s'):
            if nexus.is_speaking():
                nexus.stop_speaking()
                state.suppress_speak = True   # Block any in-flight LLM response
                state.deaf_until = time.time() + 2.0
                print("\n[CAM] Speech stopped by user (Space/S key)")
            
    cam.release()
    cv2.destroyAllWindows()
    # Force kill threads
    import os
    os._exit(0)

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    t_audio = threading.Thread(target=audio_recording_thread, daemon=True)
    t_speech = threading.Thread(target=speech_processing_thread, daemon=True)
    t_terminal = threading.Thread(target=terminal_input_thread, daemon=True)
    
    t_audio.start()
    t_speech.start()
    t_terminal.start()
    
    # Run vision thread in main loop for OpenCV GUI compatibility
    vision_thread()
