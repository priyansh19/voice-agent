"""Speculative tool prefetch: cheap keyword rules on the transcript start a data fetch *before* the LLM runs,
so common questions (weather) are answered in one LLM round instead of tool-call -> fetch -> second round."""
import re, threading, time
from concurrent.futures import ThreadPoolExecutor
from . import tools

_pool = ThreadPoolExecutor(max_workers=2)
_WEATHER = re.compile(r"\b(weather|rain|raining|umbrella|temperature|forecast|sunny|snow|hot|cold|windy|humid"
                      r"|mausam|mosam|baarish|barish|chhata|chata|garmi|sardi|thand)\b"
                      r"|(मौसम|बारिश|बरसात|छाता|तापमान|गर्मी|सर्दी|ठंड|धूप|बर्फ)", re.I)
_LOC = re.compile(r"\b(?:in|at|around|mein)\s+([a-z][a-z\-]+(?:\s+[a-z][a-z\-]+)?)", re.I)
_LOC_HI = re.compile(r"([ऀ-ॿ]+)\s+(?:में|मे|का|की|के)")
_STOP = {"the", "there", "here", "today", "tomorrow", "morning", "afternoon", "evening", "tonight", "week", "weekend",
         "this", "that", "my", "our", "a", "an", "next", "few", "couple", "hours", "days", "case", "general", "particular",
         "what", "how", "is", "it", "was", "will", "be", "like", "and", "or", "if", "kal", "aaj", "mausam", "mosam", "weather"}


_STOP_HI = {"कल", "आज", "परसों", "सुबह", "शाम", "रात", "यहाँ", "वहाँ", "इस", "उस", "मौसम"}


def _location(text, default):
    for m in _LOC.finditer(text):
        words = [w for w in m.group(1).split() if w.lower() not in _STOP]
        if words and len(words[0]) > 2:
            return " ".join(words[:2])
    for m in _LOC_HI.finditer(text):
        if m.group(1) not in _STOP_HI:
            return m.group(1)
    return default


def plan(text: str, cfg):
    """Return (label, future) if a prefetch applies, else None."""
    if _WEATHER.search(text):
        loc = _location(text, cfg.get("default_location", ""))
        if loc:
            return f"weather for {loc}", _pool.submit(tools.get_weather, loc)
    return None


def collect(planned, timeout_s, log=print):
    """Wait up to timeout_s for the prefetch; returns context text or ''."""
    if not planned:
        return ""
    label, fut = planned
    t = time.perf_counter()
    try:
        data = fut.result(timeout=timeout_s)
        log(f"[prefetch] {label} ready in {(time.perf_counter()-t)*1000:.0f}ms")
        return f"\n\n[Live data ({label}, fetched just now): {data}]"
    except Exception as e:
        log(f"[prefetch] {label} not ready ({e.__class__.__name__}); LLM will use tools")
        return ""
