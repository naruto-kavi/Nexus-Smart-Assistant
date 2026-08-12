import os
import json
from pathlib import Path

ROOT = Path(__file__).parent
MEMORY_DIR = ROOT / "memory"
CONVERSATIONS_FILE = MEMORY_DIR / "conversations.json"

class MemoryManager:
    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        
    def _load_conversations(self):
        if not CONVERSATIONS_FILE.exists():
            return {}
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
            
    def _save_conversations(self, data):
        try:
            with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[MEMORY] Error saving conversation memory: {e}")
            
    def get_context(self, user_name, max_exchanges=5):
        """
        Returns a list of conversation messages (dicts with role/content) for the user.
        """
        user_name = user_name or "User"
        all_convs = self._load_conversations()
        history = all_convs.get(user_name, [])
        # Each exchange is a dict: {"user": "...", "assistant": "..."}
        last_exchanges = history[-max_exchanges:]
        
        messages = []
        for ex in last_exchanges:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        return messages
        
    def save_exchange(self, user_name, user_input, assistant_response):
        user_name = user_name or "User"
        all_convs = self._load_conversations()
        if user_name not in all_convs:
            all_convs[user_name] = []
            
        all_convs[user_name].append({
            "user": user_input,
            "assistant": assistant_response
        })
        
        # Keep only the last 30 exchanges to avoid token bloat
        all_convs[user_name] = all_convs[user_name][-30:]
        self._save_conversations(all_convs)

memory_manager = MemoryManager()
