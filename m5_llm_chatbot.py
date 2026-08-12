import os
import time
import sys
import threading
import ctypes
import asyncio
from pathlib import Path

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
        return text.encode('utf-8', 'surrogatepass').decode('utf-8', 'replace')
    except Exception:
        try:
            return text.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            return ""

# Add current directory to path
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

# ==============================================================================
# Module 5: LLM Chatbot
# ==============================================================================

import requests
import json
import tools
from memory_manager import memory_manager

# ==============================================================================
# CONFIGURATION: Hybrid Mode (Cloud first, Fallback to Local)
# ==============================================================================

# Load .env manually to populate os.environ before accessing keys
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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


class SmartAssistant:
    def __init__(self, assistant_name="Nexus"):
        self.assistant_name = assistant_name
        self.mode = "HYBRID"

        # Local (Ollama) Config
        self.ollama_url = "http://localhost:11434/api/generate"
        self.local_model = "phi3"

        # Groq Config
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_model = "llama-3.3-70b-versatile"

        print(f"[*] Initializing Nexus Hybrid Brain...")
        print(f"    - Primary: Groq Cloud ({self.groq_model})")
        print(f"    - Fallback: Local Ollama ({self.local_model})")

        # System Prompt for Bilingual Intelligence + Agentic Tool Calling
        self.system_prompt = (
            f"You are {self.assistant_name}, a highly intelligent bilingual AI assistant. "
            "You provide clear and accurate answers in English and Telugu. "
            "You MUST respond in the SAME language as the user's query (English for English, Telugu for Telugu). "
            "Keep your responses concise and professional.\n\n"

            "PHONETIC TECHNICAL TERMS — CRITICAL RULE:\n"
            "The user's speech is transcribed by an imperfect Whisper STT model. When the user speaks English "
            "technical/programming terms in a Telugu accent, Whisper often transcribes them phonetically into "
            "Telugu script. You MUST decode these phonetic spellings and respond to the ACTUAL English concept.\n"
            "Common phonetic mappings (Telugu script → English meaning):\n"
            "  పైతా / పైథాన్ / పైతాన్ → Python (programming language)\n"
            "  నెక్స్ప్పైన్ / ఎక్స్ప్లెయిన్ / ఎక్స్పైన్ → explain\n"
            "  మిసిన్ లాన్నింగ్ / మెషిన్ లెర్నింగ్ / మిషన్ లని → Machine Learning\n"
            "  డీప్ లెర్నింగ్ / డీప్ లాన్నింగ్ → Deep Learning\n"
            "  న్యూరల్ నెట్వర్క్ / న్యూరల్ నెట్ → Neural Network\n"
            "  ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ / ఏఐ → Artificial Intelligence / AI\n"
            "  అల్గారిథమ్ / అల్గొరిథమ్ → Algorithm\n"
            "  డేటా సైన్స్ / డేటా సైన్సు → Data Science\n"
            "  ప్రోగ్రామింగ్ / ప్రోగ్రామ్ → Programming\n"
            "  వేరియబుల్ / వేరియబల్ → Variable\n"
            "  ఫంక్షన్ / ఫంక్షను → Function\n"
            "  క్లాస్ / క్లాసు → Class (OOP)\n"
            "  అర్రే / అరే → Array\n"
            "  లూప్ / లూపు → Loop\n"
            "  డేటాబేస్ / డేటాబేసు → Database\n"
            "  ఏపీఐ / API → API\n"
            "  ఫ్రేమ్వర్క్ / ఫ్రేమ్వర్కు → Framework\n"
            "RULE: Never interpret these as Telugu words with literal meanings. Always map them to the English technical concept and answer that question accurately.\n\n"

            "SYSTEM TOOLS ACCESS:\n"
            "You have access to the following tools to interact with the system or search for information. When a tool is needed, you MUST invoke it by returning ONLY a single valid JSON block containing the keys \"tool\" and \"args\". Do not output any other text or markdown formatting when calling a tool.\n\n"
            "Available Tools:\n"
            "1. weather\n"
            "   Use to retrieve the current weather conditions. Arguments: {\"city\": \"<city_name_in_english>\"}\n"
            "2. web_search\n"
            "   Use when you need recent, live, or real-time information, news, current events, sports scores, election updates, facts, or statistics. Arguments: {\"query\": \"<search_query_in_english>\"}\n"
            "3. timer\n"
            "   Use when the user requests to set a timer or countdown. Arguments: {\"seconds\": <int_duration>, \"label\": \"<timer_label_or_name>\"}\n"
            "4. reminder\n"
            "   Use when the user requests to schedule a reminder. Arguments: {\"time_text\": \"<time_specifier_like_'in_5_minutes'_or_'at_3:00_PM'>\", \"message\": \"<reminder_content>\"}\n"
            "5. set_volume\n"
            "   Use when the user requests to set the volume to a specific percentage. Arguments: {\"percent\": <int_percentage_0_to_100>}\n"
            "6. change_volume\n"
            "   Use when the user requests to increase, decrease, or mute the volume. Arguments: {\"direction\": \"<'up'/'down'/'mute'>\"}\n\n"
            "INSTRUCTIONS FOR TOOL CALLING:\n"
            "- If a query requires search, weather, timer, reminder, or volume, output ONLY the JSON, e.g.: {\"tool\": \"web_search\", \"args\": {\"query\": \"latest news about Tamil politics\"}}\n"
            "- If a tool has already been executed (you will see a 'Tool result:' message in the history), use that result to answer the user's question directly. Do NOT output a tool JSON again."
        )

        # TTS — Thread-based (no subprocess startup overhead per response)
        self._tts_thread = None
        self._tts_stop_event = threading.Event()
        self._tts_generation = 0   # Incremented on every stop/new-speak — old threads abort if stale

        # Logging setup
        self.log_file = ROOT / "conversation_log.txt"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- New Session Started (Hybrid Brain): {time.ctime()} ---\n")

        # Initialize conversation memory with system prompt
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.last_user_name = "User"

        # Load pending reminders at startup
        print("[*] Scheduling pending reminders...")
        try:
            tools.schedule_pending_reminders(self.speak)
        except Exception as e:
            print(f"[ERROR] Failed to schedule pending reminders: {e}")

        print(f"\n[+] Module 5 (Hybrid) Initialization Complete!\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────────

    def log_interaction(self, user_text, response_text, user_name="User"):
        """Save the interaction to a local log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {user_name}: {user_text}\n")
            f.write(f"[{timestamp}] {self.assistant_name}: {response_text}\n")
            f.write("-" * 50 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # TTS — Thread-based (replaces subprocess.Popen per response)
    # ──────────────────────────────────────────────────────────────────────────

    def _is_telugu(self, text):
        """Check if text contains Telugu Unicode characters (U+0C00–U+0C7F)."""
        return any('\u0c00' <= c <= '\u0c7f' for c in text)

    def stop_speaking(self):
        """Stop TTS instantly from any thread."""
        import sounddevice as sd
        if self._tts_thread and self._tts_thread.is_alive():
            print("\n\U0001F6D1 [TTS] Interrupting speech...")
        # Increment generation — every in-flight thread will see itself as stale
        self._tts_generation += 1
        self._tts_stop_event.set()
        # sd.stop() is thread-safe and stops instantly — no MCI alias needed
        try:
            sd.stop()
        except Exception:
            pass
        self._tts_thread = None

    def speak(self, text):
        """Start TTS in a background daemon thread — zero subprocess startup cost."""
        if not text:
            return
        text = clean_surrogates(text)
        # Stop any in-progress speech and increment generation
        self._tts_generation += 1
        my_generation = self._tts_generation
        self._tts_stop_event.set()
        try:
            ctypes.windll.winmm.mciSendStringW('stop tts_audio', None, 0, 0)
            ctypes.windll.winmm.mciSendStringW('close tts_audio', None, 0, 0)
        except Exception:
            pass
        self._tts_stop_event.clear()   # Safe to clear now — old threads check generation
        print(f"\U0001F5E3\uFE0F  [TTS Active] Speaking response...")
        self._tts_thread = threading.Thread(
            target=self._tts_worker, args=(text, my_generation), daemon=True
        )
        self._tts_thread.start()

    def _tts_worker(self, text, my_generation):
        """Worker running in the TTS thread — aborts if generation is stale."""
        def _is_stale():
            """True if stop_speaking() or a newer speak() was called after us."""
            return self._tts_generation != my_generation or self._tts_stop_event.is_set()

        import re
        clean_text = clean_surrogates(text.replace('"', "'").replace('\n', ' ').strip())

        # If the entire response is a raw JSON tool call
        try:
            import json
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict) and "tool" in parsed:
                tool_name = parsed.get("tool", "tool")
                clean_text = f"Running {tool_name.replace('_', ' ')}, please wait."
        except Exception:
            pass

        clean_text = re.sub(r'\{[^{}]*\}', '', clean_text).strip()
        if not clean_text:
            return

        if _is_stale():
            return

        try:
            asyncio.run(self._speak_online(clean_text, my_generation))
        except Exception as e:
            print(f"Online TTS failed, using offline fallback. Error: {e}")
            if not _is_stale():
                self._speak_offline(clean_text)

    async def _speak_online(self, text, my_generation):
        """High-quality online neural TTS via edge-tts. Aborts if generation becomes stale."""
        def _is_stale():
            return self._tts_generation != my_generation or self._tts_stop_event.is_set()

        import edge_tts
        voice = "te-IN-MohanNeural" if self._is_telugu(text) else "en-US-AvaNeural"
        temp_file = ROOT / f"temp_tts_{abs(hash(text))}.mp3"

        if _is_stale():
            return

        communicate = edge_tts.Communicate(text, voice)
        audio_chunks = bytearray()
        try:
            async for chunk in communicate.stream():
                if _is_stale():
                    return
                if chunk["type"] == "audio":
                    audio_chunks.extend(chunk["data"])
        except Exception as e:
            if _is_stale():
                return
            await communicate.save(str(temp_file))
            if _is_stale():
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception:
                    pass
                return
        else:
            if _is_stale():
                return
            with open(temp_file, 'wb') as f:
                f.write(audio_chunks)

        # Final stale-check before playing
        if _is_stale():
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass
            return

        # Play using sounddevice (thread-safe stop via sd.stop())
        try:
            import sounddevice as sd
            import soundfile as sf
            data, fs = sf.read(str(temp_file), dtype='float32')
            if not _is_stale():
                sd.play(data, fs)
                # Poll while playing so we can abort mid-playback
                while sd.get_stream().active:
                    if _is_stale():
                        sd.stop()
                        break
                    time.sleep(0.05)
        except Exception as e:
            print(f"[TTS] sounddevice playback error: {e}")
        finally:
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass

    def _speak_offline(self, text):
        """Fallback offline TTS using pyttsx3 (works without internet)."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            telugu_voice = None
            for voice in voices:
                if 'telugu' in voice.name.lower() or 'te-in' in voice.id.lower():
                    telugu_voice = voice.id
                    break
            if telugu_voice:
                engine.setProperty('voice', telugu_voice)
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Offline TTS Error: {e}")

    def is_speaking(self):
        """Check if the TTS thread is currently active."""
        return self._tts_thread is not None and self._tts_thread.is_alive()

    # ──────────────────────────────────────────────────────────────────────────
    # LLM Response Generation & Tool Execution
    # ──────────────────────────────────────────────────────────────────────────

    def _call_llm_hybrid(self):
        # Trim history to save context tokens (keep system prompt at index 0)
        max_history = 12
        if len(self.messages) > max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-max_history:]

        response_text = ""
        if GROQ_API_KEY:
            print("🧠 Thinking (Groq Cloud)...")
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            # Use larger token budget for synthesis calls (tool results can be verbose)
            is_synthesis_call = any(
                msg.get("role") == "system" and "Tool result" in msg.get("content", "")
                for msg in self.messages
            )
            payload = {
                "model": self.groq_model,
                "messages": self.messages,
                "temperature": 0.2,
                "max_tokens": 500 if is_synthesis_call else 250
            }
            try:
                response = requests.post(self.groq_url, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                response_text = clean_surrogates(response.json()['choices'][0]['message']['content'].strip())
                return response_text
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_str = str(e)
                print(f"⚠️  [GROQ FAILED] {e}")
                if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                    print(f"   → Groq timed out after 60s. Falling back to Local Ollama...")
                elif "401" in err_str:
                    print(f"   → Invalid or expired Groq API key! Check your .env file.")
                elif "429" in err_str:
                    print(f"   → Groq rate limit hit. Falling back to Local Ollama...")
                else:
                    print(f"   → Falling back to Local Ollama...")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"   Response: {e.response.text[:200]}")

        # Fallback to Local Ollama
        print(f"🧠 Thinking (Local Ollama: {self.local_model})...")
        history_prompt = ""
        for msg in self.messages:
            role = "System" if msg["role"] == "system" else ("User" if msg["role"] == "user" else "Assistant")
            history_prompt += f"{role}: {msg['content']}\n"
        history_prompt += "Assistant:"

        payload = {
            "model": self.local_model,
            "prompt": history_prompt,
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            response_text = clean_surrogates(response.json().get("response", "").strip())
        except Exception as e:
            err_str = str(e)
            print(f"❌ [Local AI Error] {e}")
            if "Connection refused" in err_str or "Failed to establish" in err_str or "HTTPConnectionPool" in err_str:
                print(f"")
                print(f"   ┌─ OLLAMA NOT RUNNING ──────────────────────────────────┐")
                print(f"   │  Ollama server is not running on this machine.        │")
                print(f"   │  To fix:                                              │")
                print(f"   │    1. Open a new terminal window                      │")
                print(f"   │    2. Run: ollama serve                               │")
                print(f"   │    3. If model missing: ollama pull {self.local_model:<20}│")
                print(f"   └───────────────────────────────────────────────────────┘")
            response_text = "I'm currently offline — both Groq and Ollama are unavailable."

            # Remove last user message so user can retry
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()

        return response_text

    def _parse_tool_call(self, text):
        import json
        import re
        text = text.strip()

        # Strategy 1: The whole response is a pure JSON object — try direct parse first
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data:
                return data
        except Exception:
            pass

        # Strategy 2: Extract a JSON block that spans across nested braces (greedy from first { to last })
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                if isinstance(data, dict) and "tool" in data:
                    return data
            except Exception:
                pass

        return None

    def _execute_tool(self, name, args):
        try:
            def _get_arg(d, *preferred_keys):
                """Get arg by preferred keys, fall back to any string-value key if all miss (handles LLM typos like 'ciity')."""
                for k in preferred_keys:
                    if k in d:
                        return d[k]
                # Fuzzy: return first string value in the dict
                for v in d.values():
                    if isinstance(v, str) and v.strip():
                        return v
                return None

            if name == "weather":
                city = _get_arg(args, "city", "location", "place") or "Hyderabad"
                return tools.get_weather(str(city).strip())
            elif name == "web_search":
                query = _get_arg(args, "query", "q", "search", "text") or ""
                return tools.web_search(str(query).strip())
            elif name == "timer":
                seconds = int(args.get("seconds", args.get("duration", args.get("time", 60))))
                label = args.get("label", args.get("name", f"{seconds}s timer"))
                return tools.set_timer(seconds, label, self.speak)
            elif name == "reminder":
                time_text = _get_arg(args, "time_text", "time", "when", "at") or ""
                message = _get_arg(args, "message", "text", "content", "reminder") or "Reminder!"
                return tools.add_reminder(str(time_text), str(message), self.speak)
            elif name == "set_volume":
                percent = int(args.get("percent", args.get("level", args.get("volume", 50))))
                return tools.set_volume(percent)
            elif name == "change_volume":
                direction = _get_arg(args, "direction", "action", "type") or "up"
                return tools.change_volume(str(direction))
            else:
                return f"Error: Tool '{name}' is not recognized."
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"


    def generate_response(self, user_text, user_name="User"):
        """Hybrid generation: Decide tool call and execute via LLM dynamic context."""
        user_text = clean_surrogates(user_text)
        user_name = clean_surrogates(user_name)

        # 1. Reset conversation history if user identity changes, and load persistent memory
        if self.last_user_name != user_name or len(self.messages) <= 1:
            print(f"🔄 [CHAT] Loading persistent memory context for '{user_name}'.")
            self.last_user_name = user_name
            self.messages = [{"role": "system", "content": self.system_prompt}]
            memory_messages = memory_manager.get_context(user_name)
            self.messages.extend(memory_messages)

        # 2. Extract clean text and language settings
        import re
        clean_text = user_text
        lang_code = None
        lang_match = re.match(r'^\[Language:\s*([A-Z]{2})\]\s*(.*)', user_text, re.IGNORECASE)
        if lang_match:
            lang_code = lang_match.group(1).upper()
            clean_text = lang_match.group(2).strip()

        # Check if the query is in Telugu (either from lang tag or direct characters)
        is_query_telugu = self._is_telugu(clean_text)
        if lang_match:
            is_query_telugu = (lang_code == 'TE')

        lang_instruction = "Respond in fluent Telugu." if is_query_telugu else "Respond in English."
        system_content = self.system_prompt + f"\n\nCRITICAL: The user has queried in {'Telugu' if is_query_telugu else 'English'}. You MUST {lang_instruction}"
        
        # Enforce target language constraint dynamically in the system prompt
        self.messages[0] = {"role": "system", "content": system_content}

        # 3. Append user message to history
        self.messages.append({"role": "user", "content": user_text})

        # 4. LLM decides if it needs to call a tool
        response_text = self._call_llm_hybrid()

        # Parse tool call from response
        tool_call = self._parse_tool_call(response_text)
        if tool_call:
            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})
            print(f"🛠️  [TOOL CALL] LLM invoked: {tool_name} with arguments {args}")

            # Execute tool
            tool_result = self._execute_tool(tool_name, args)

            # Append tool-call turn as assistant
            self.messages.append({"role": "assistant", "content": response_text})

            # --- FAST PATH: Simple tools already return complete natural-language strings ---
            # weather  → "Current weather — Kadapa: 🌤️ +36°C"
            # timer    → "Timer set for 2 minutes. I'll alert you when it's done."
            # reminder → "Reminder set for 03:42 PM — I'll alert you: '...'"
            # volume   → "Volume set to 40%." / "Volume down by 15%."
            # These DON'T need a 2nd LLM call — use the result directly.
            # web_search returns raw bullet points → still needs LLM to synthesize into a sentence.
            DIRECT_RESULT_TOOLS = {"weather", "timer", "reminder", "set_volume", "change_volume"}

            if tool_name in DIRECT_RESULT_TOOLS:
                # Skip 2nd LLM call — tool result IS the response
                response_text = tool_result
                print(f"⚡ [FAST PATH] Using direct tool result (skipped 2nd LLM call)")
            else:
                # web_search (and any future complex tools) — need LLM synthesis
                synthesis_prompt = (
                    f"[TOOL RESULT from {tool_name}]:\n{tool_result}\n\n"
                    "IMPORTANT: Using the tool result above, give the user a helpful, natural-language answer. "
                    "Do NOT output JSON. Do NOT call another tool. Write a plain conversational response only."
                )
                self.messages.append({"role": "user", "content": synthesis_prompt})

                # Second LLM call to synthesize final response
                print("🧠 Thinking (Synthesizing response from tool)...")
                response_text = self._call_llm_hybrid()

                # Safety net: if LLM still outputs raw JSON, use the tool result directly
                if self._parse_tool_call(response_text):
                    response_text = f"Here's what I found: {tool_result}"

                # Remove the synthesis prompt from history (keep conversation clean)
                if self.messages and self.messages[-1]["role"] == "user":
                    self.messages.pop()

        # 5. Append final response to history, log and save to persistent memory
        self.messages.append({"role": "assistant", "content": response_text})
        self.log_interaction(user_text, response_text, user_name=user_name)

        # Save exchange to persistent memory
        memory_manager.save_exchange(user_name, clean_text, response_text)

        return response_text


if __name__ == "__main__":
    print("=" * 60)
    print("   MODULE 5: Standalone LLM Chatbot Testing")
    print("=" * 60)

    try:
        assistant = SmartAssistant("Nexus")
    except Exception as e:
        print(f"[ERROR] Failed to initialize M5: {e}")
        exit(1)

    print("You can now chat with the assistant! (Type 'quit' or 'exit' to stop)")

    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ['quit', 'exit']:
                print(f"\n👋 {assistant.assistant_name}: Goodbye!")
                assistant.speak("Goodbye! Have a great day!")
                break

            if not user_input.strip():
                continue

            # Generate the text response
            response = assistant.generate_response(user_input, user_name="Mani")
            print(f"\n🤖 {assistant.assistant_name}: {response}\n")

            # Speak it out loud
            assistant.speak(response)

        except KeyboardInterrupt:
            break
