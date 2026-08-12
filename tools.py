import os
import sys
import json
import time
import re
import threading
import subprocess
import platform
import requests
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent
REMINDERS_FILE = ROOT / "memory" / "reminders.json"

# ==============================================================================
# 1. WEATHER (wttr.in - Free, no API key)
# ==============================================================================
def get_weather(city="Hyderabad"):
    try:
        # Sanitize: strip extra whitespace, collapse multiple spaces
        city = " ".join(city.strip().split()).title()

        # Translate common Telugu city names to English for wttr.in
        telugu_to_english_cities = {
            "హైదరాబాద్": "Hyderabad",
            "విజయవాడ": "Vijayawada",
            "విశాఖపట్నం": "Visakhapatnam",
            "తిరుపతి": "Tirupati",
            "వరంగల్": "Warangal",
            "గుంటూరు": "Guntur",
            "నెల్లూరు": "Nellore",
            "ఢిల్లీ": "Delhi",
            "ముంబై": "Mumbai",
            "బెంగళూరు": "Bengaluru",
            "చెన్నై": "Chennai",
            "కడప": "Kadapa",
            "ప్రొద్దుటూరు": "Proddatur",
            "నంద్యాల": "Nandyal",
            "కర్నూలు": "Kurnool",
        }
        for te_city, en_city in telugu_to_english_cities.items():
            if te_city in city:
                city = en_city
                break

        # Use format=j1 (JSON) for structured data including forecast info
        url = f"https://wttr.in/{requests.utils.quote(city)}?format=3"
        response = requests.get(url, timeout=6)
        if response.status_code == 200:
            result = response.text.strip()
            if "<html" in result.lower() or "not found" in result.lower() or "error" in result.lower():
                raise Exception("wttr.in returned invalid output")
            return f"Current weather — {result}"
        else:
            raise Exception(f"HTTP Status {response.status_code}")
    except Exception as e:
        return f"Could not fetch weather for {city} at the moment. Error: {str(e)}"


def web_search(query):
    query = query.strip()
    if not query:
        return "Search query is empty."

    # ── Layer 0: Google News RSS — real-time, no API key needed ──
    try:
        import html as html_lib
        news_url = (
            "https://news.google.com/rss/search"
            f"?q={requests.utils.quote(query)}"
            "&hl=en-IN&gl=IN&ceid=IN:en"
        )
        news_resp = requests.get(news_url, timeout=7,
                                 headers={"User-Agent": "Mozilla/5.0 Nexus-Assistant/1.0"})
        if news_resp.status_code == 200:
            items = re.findall(r'<item>(.*?)</item>', news_resp.text, re.DOTALL)
            parts = []
            for item in items[:5]:
                title_m = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
                title_raw = title_m.group(1) if title_m else ''
                # Decode CDATA + HTML entities, then strip any remaining tags
                title = html_lib.unescape(re.sub(r'<!\[CDATA\[|\]\]>|<[^>]+>', '', title_raw)).strip()
                # Remove source suffix " - The Hindu", " - NDTV" etc.
                title = re.sub(r'\s+-\s+\S.*$', '', title).strip()
                if title:
                    parts.append(f"• {title}")
            if parts:
                return "[Live News]\n" + "\n".join(parts)
    except Exception as e:
        print(f"[SEARCH] Google News RSS error: {e}. Trying Wikipedia...")

    try:
        wiki_url = (
            "https://en.wikipedia.org/w/api.php"
            "?action=query&list=search"
            f"&srsearch={requests.utils.quote(query)}"
            "&format=json&utf8=1&srlimit=4"
        )
        w_resp = requests.get(wiki_url, timeout=6,
                              headers={"User-Agent": "Nexus-Assistant/1.0"})
        if w_resp.status_code == 200:
            w_data = w_resp.json()
            search_hits = w_data.get("query", {}).get("search", [])
            parts = []
            for hit in search_hits[:4]:
                title = hit.get("title", "")
                snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", "")).strip()
                # Fetch full intro paragraph from REST API
                try:
                    intro_url = (
                        "https://en.wikipedia.org/api/rest_v1/page/summary/"
                        + requests.utils.quote(title.replace(" ", "_"))
                    )
                    intro_resp = requests.get(intro_url, timeout=4,
                                              headers={"User-Agent": "Nexus-Assistant/1.0"})
                    if intro_resp.status_code == 200:
                        intro = intro_resp.json().get("extract", "").strip()
                        if intro:
                            sentences = re.split(r'(?<=[.!?])\s+', intro)
                            parts.append(f"• {title}: {' '.join(sentences[:2])}")
                            continue
                except Exception:
                    pass
                # Fallback: use the search snippet
                if snippet:
                    parts.append(f"• {title}: {snippet}")

            if parts:
                return "\n".join(parts)
    except Exception as e:
        print(f"[SEARCH] Wikipedia API error: {e}. Trying DDG JSON...")

    # ── Layer 2: DuckDuckGo Instant Answer JSON API ──
    try:
        ddg_url = (
            "https://api.duckduckgo.com/"
            f"?q={requests.utils.quote(query)}"
            "&format=json&no_html=1&skip_disambig=1&no_redirect=1"
        )
        resp = requests.get(ddg_url, timeout=6,
                            headers={"User-Agent": "Mozilla/5.0 Nexus-Assistant/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            parts = []
            abstract = (data.get("AbstractText") or "").strip()
            abstract_src = (data.get("AbstractSource") or "").strip()
            if abstract:
                parts.append(f"• {(abstract_src + ': ') if abstract_src else ''}{abstract}")
            answer = (data.get("Answer") or "").strip()
            if answer:
                parts.append(f"• {answer}")
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict):
                    text = (topic.get("Text") or "").strip()
                    if text:
                        parts.append(f"• {text}")
            if parts:
                return "\n".join(parts)
    except Exception as e:
        print(f"[SEARCH] DDG JSON API error: {e}. Trying library...")

    # ── Layer 3: DDGS / duckduckgo_search library ──
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        ddgs_client = DDGS()
        results = list(ddgs_client.text(query, max_results=4))
        if results:
            summary = []
            for r in results:
                title = r.get('title', 'Result')
                body = r.get('body', '')
                if title and body:
                    summary.append(f"• {title}: {body}")
            if summary:
                return "\n".join(summary)
    except Exception as e:
        print(f"[SEARCH] DDGS library error: {e}. Trying HTML scraper...")

    # ── Layer 4: DuckDuckGo HTML scraper ──
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>',
                                  response.text, re.DOTALL)
            titles = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>',
                                response.text, re.DOTALL)
            results = []
            for i, snip in enumerate(snippets[:4]):
                clean_snip = re.sub(r'<[^>]+>', '', snip).strip()
                clean_title = re.sub(r'<[^>]+>', '', titles[i]).strip() if i < len(titles) else "Result"
                if clean_snip:
                    results.append(f"• {clean_title}: {clean_snip}")
            if results:
                return "\n".join(results)
    except Exception as ex:
        print(f"[SEARCH] HTML scraper failed: {ex}")

    return f"I couldn't retrieve search results for '{query}' right now. Please check your internet connection."


# ==============================================================================
# 3. TIMERS (Background Threading Timer)
# ==============================================================================
def set_timer(seconds, label, speak_callback):
    if seconds <= 0:
        return "Timer duration must be positive."

    def _timer_done():
        speak_callback(f"Timer done! {label}")

    t = threading.Timer(seconds, _timer_done)
    t.daemon = True
    t.start()

    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    time_desc = ""
    if minutes > 0:
        time_desc += f"{minutes} minute" + ("s" if minutes > 1 else "")
    if secs > 0:
        if time_desc:
            time_desc += " and "
        time_desc += f"{secs} second" + ("s" if secs > 1 else "")
    if not time_desc:
        time_desc = f"{int(seconds)} seconds"

    return f"Timer set for {time_desc}. I'll alert you when it's done."


# ==============================================================================
# 4. REMINDERS (Persistent & Background Scheduled)
# ==============================================================================
def _load_reminders():
    if not REMINDERS_FILE.exists():
        return []
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_reminders(reminders):
    try:
        REMINDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(reminders, f, indent=4)
    except Exception as e:
        print(f"[REMINDER] Error saving reminders: {e}")


# Keep track of active reminder timers to manage them
active_reminders = []


def schedule_pending_reminders(speak_callback):
    global active_reminders
    for t in active_reminders:
        try:
            t.cancel()
        except Exception:
            pass
    active_reminders = []

    reminders = _load_reminders()
    now = datetime.now()
    updated = False

    for r in reminders:
        if r.get("completed"):
            continue
        try:
            target_time = datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S")
            if target_time > now:
                delay = (target_time - now).total_seconds()

                def make_trigger(rem):
                    def _trigger():
                        speak_callback(f"Reminder: {rem['message']}")
                        all_rem = _load_reminders()
                        for x in all_rem:
                            if x["time"] == rem["time"] and x["message"] == rem["message"]:
                                x["completed"] = True
                        _save_reminders(all_rem)
                    return _trigger

                t = threading.Timer(delay, make_trigger(r))
                t.daemon = True
                t.start()
                active_reminders.append(t)
            else:
                r["completed"] = True
                updated = True
        except Exception as e:
            print(f"[REMINDER] Error scheduling: {e}")

    if updated:
        _save_reminders(reminders)


def add_reminder(time_text, message, speak_callback):
    now = datetime.now()
    target_time = None
    time_text_lower = time_text.lower()

    # 1. Relative: "in 5 minutes", "in 1 hour", "in 30 seconds"
    rel_match = re.search(
        r'(?:in\s+)?(\d+)\s*(second|minute|hour|sec|min|hr|సెకన్|నిమిష|గంట)',
        time_text_lower
    )
    if rel_match:
        val = int(rel_match.group(1))
        unit = rel_match.group(2)
        if 'sec' in unit or 'సెకన్' in unit:
            target_time = now + timedelta(seconds=val)
        elif 'min' in unit or 'నిమిష' in unit:
            target_time = now + timedelta(minutes=val)
        elif 'hour' in unit or 'hr' in unit or 'గంట' in unit:
            target_time = now + timedelta(hours=val)

    # 2. Absolute: "at 3:30 PM", "at 15:00"
    if not target_time:
        abs_match = re.search(r'(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?', time_text_lower)
        if abs_match:
            hr = int(abs_match.group(1))
            mn = int(abs_match.group(2))
            ampm = abs_match.group(3)
            if ampm:
                if ampm == 'pm' and hr < 12:
                    hr += 12
                elif ampm == 'am' and hr == 12:
                    hr = 0
            target_time = now.replace(hour=hr, minute=mn, second=0, microsecond=0)
            if target_time < now:
                target_time += timedelta(days=1)

    # 3. Fallback: dateparser
    if not target_time:
        try:
            import dateparser
            parsed = dateparser.parse(time_text, settings={'PREFER_DATES_FROM': 'future'})
            if parsed:
                target_time = parsed
        except Exception:
            pass

    if not target_time:
        return ("Could not understand the time for the reminder. "
                "Please say something like 'in 10 minutes' or 'at 3:00 PM'.")

    formatted_time = target_time.strftime("%Y-%m-%d %H:%M:%S")

    reminders = _load_reminders()
    reminder_record = {"time": formatted_time, "message": message, "completed": False}
    reminders.append(reminder_record)
    _save_reminders(reminders)

    delay = (target_time - now).total_seconds()

    def _trigger():
        speak_callback(f"Reminder: {message}")
        all_rem = _load_reminders()
        for x in all_rem:
            if x["time"] == formatted_time and x["message"] == message:
                x["completed"] = True
        _save_reminders(all_rem)

    t = threading.Timer(delay, _trigger)
    t.daemon = True
    t.start()
    global active_reminders
    active_reminders.append(t)

    time_label = target_time.strftime("%I:%M %p")
    return f"Reminder set for {time_label} — I'll alert you: '{message}'."


# ==============================================================================
# 5. VOLUME CONTROL (Windows pycaw / Linux amixer fallback)
# ==============================================================================
def set_volume(percent):
    percent = max(0, min(100, int(percent)))
    try:
        if platform.system() == "Windows":
            from pycaw.pycaw import AudioUtilities
            speakers = AudioUtilities.GetSpeakers()
            # pycaw ≥20251023: AudioDevice exposes .EndpointVolume directly
            volume_ctrl = speakers.EndpointVolume
            volume_ctrl.SetMasterVolumeLevelScalar(percent / 100.0, None)
            return f"Volume set to {percent}%."
        else:
            subprocess.run(f"amixer sset 'Master' {percent}%", shell=True, capture_output=True)
            return f"Volume set to {percent}%."
    except Exception as e:
        return f"Failed to set volume: {str(e)}"


def change_volume(direction, amount=15):
    direction = direction.lower().strip()
    try:
        if platform.system() == "Windows":
            from pycaw.pycaw import AudioUtilities
            speakers = AudioUtilities.GetSpeakers()
            volume_ctrl = speakers.EndpointVolume
            current = int(round(volume_ctrl.GetMasterVolumeLevelScalar() * 100))

            if direction == "up":
                new_vol = min(100, current + amount)
            elif direction == "down":
                new_vol = max(0, current - amount)
            elif direction in ("mute", "off", "silence"):
                new_vol = 0
            else:
                return f"Unknown volume direction '{direction}'. Use 'up', 'down', or 'mute'."

            volume_ctrl.SetMasterVolumeLevelScalar(new_vol / 100.0, None)
            label = "Muted" if new_vol == 0 else f"Volume set to {new_vol}%"
            return f"{label}."
        else:
            if direction in ("mute", "off", "silence"):
                subprocess.run("amixer sset 'Master' mute", shell=True)
                return "Muted."
            sign = "+" if direction == "up" else "-"
            subprocess.run(f"amixer sset 'Master' {amount}%{sign}", shell=True)
            return f"Volume {direction} by {amount}%."
    except Exception as e:
        return f"Failed to adjust volume: {str(e)}"
