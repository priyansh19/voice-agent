"""Tools the LLM can call. Everything is keyless: Open-Meteo (weather + geocoding), Wikipedia REST, DuckDuckGo."""
import datetime, json, time
import httpx

_http = httpx.Client(timeout=8, headers={"User-Agent": "voice-agent/0.1"})

WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       66: "freezing rain", 67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
       80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
       95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}

# Keep this schema SHORT: every token here is re-saved/restored by the Ollama runner on every turn (~0.3 ms/token).
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather now + 3-day forecast for a city.",
                                      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "wikipedia", "description": "Summary of a topic/person/place.",
                                      "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Web search (news, prices, events).",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "Local date and time.",
                                      "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "calculate", "description": "Arithmetic, e.g. 12*7+3.",
                                      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]


_geo_cache: dict = {}
# Devanagari -> English for cities the geocoder's Hindi index misses
_CITY_HI = {"मिलान": "Milan", "मुंबई": "Mumbai", "बंबई": "Mumbai", "बेंगलुरु": "Bengaluru", "बैंगलोर": "Bangalore",
            "कोलकाता": "Kolkata", "चेन्नई": "Chennai", "हैदराबाद": "Hyderabad", "पुणे": "Pune", "जयपुर": "Jaipur",
            "लखनऊ": "Lucknow", "अहमदाबाद": "Ahmedabad", "चंडीगढ़": "Chandigarh", "गुड़गांव": "Gurgaon", "नोएडा": "Noida",
            "रोम": "Rome", "पेरिस": "Paris", "बर्लिन": "Berlin", "टोक्यो": "Tokyo", "न्यूयॉर्क": "New York",
            "दुबई": "Dubai", "सिंगापुर": "Singapore", "सिडनी": "Sydney", "टोरंटो": "Toronto", "वेनिस": "Venice", "फ्लोरेंस": "Florence"}


def geocode(location: str):
    key = location.strip().lower()
    if key not in _geo_cache:
        hit = None
        location = _CITY_HI.get(location.strip(), location)
        langs = ["hi", "en"] if any(ord(c) > 0x24F for c in location) else ["en"]
        for lang in langs:
            g = _http.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": location, "count": 1, "language": lang}).json()
            if g.get("results"):
                hit = g["results"][0]; break
        _geo_cache[key] = hit
    return _geo_cache[key]


def prewarm(repeat_s: float = 45.0):
    """Keep TLS connections open (the APIs close idle ones) so a real call does not pay ~1 s of handshakes."""
    import threading
    def once():
        for url in ("https://geocoding-api.open-meteo.com/v1/search?name=Rome&count=1",
                    "https://api.open-meteo.com/v1/forecast?latitude=41.9&longitude=12.5&current=temperature_2m",
                    "https://en.wikipedia.org/api/rest_v1/page/summary/Rome"):
            try: _http.get(url)
            except Exception: pass
    def loop():
        while True:
            once(); time.sleep(repeat_s)
    threading.Thread(target=loop, daemon=True).start()


_wx_cache: dict = {}   # location -> (time, text); forecasts are reused for 10 minutes


def get_weather(location: str) -> str:
    key = location.strip().lower()
    hit = _wx_cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    out = _get_weather(location)
    _wx_cache[key] = (time.time(), out)
    return out


def _get_weather(location: str) -> str:
    r = geocode(location)
    if not r:
        return f"Could not find a place called {location}."
    w = _http.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": r["latitude"], "longitude": r["longitude"], "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum"}).json()
    c, d = w["current"], w["daily"]
    out = [f"{r['name']}, {r.get('country', '')}: now {c['temperature_2m']:.0f}°C, {WMO.get(c['weather_code'], 'unknown')}, wind {c['wind_speed_10m']:.0f} km/h."]
    names = ["today", "tomorrow", "day after tomorrow"]
    for i, day in enumerate(d["time"][:3]):
        out.append(f"{names[i]} ({day}): {WMO.get(d['weather_code'][i], '')}, {d['temperature_2m_min'][i]:.0f} to {d['temperature_2m_max'][i]:.0f}°C, "
                   f"rain chance {d['precipitation_probability_max'][i] or 0}%, {d['precipitation_sum'][i] or 0} mm.")
    return " ".join(out)


def wikipedia(topic: str) -> str:
    t = topic.strip().replace(" ", "_")
    r = _http.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{t}", headers={"accept": "application/json"})
    if r.status_code != 200:
        s = _http.get("https://en.wikipedia.org/w/api.php", params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"}).json()
        if not s[1]:
            return f"No Wikipedia article found for {topic}."
        r = _http.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{s[1][0].replace(' ', '_')}", headers={"accept": "application/json"})
    j = r.json()
    return f"{j.get('title', topic)}: {j.get('extract', '')[:700]}"


def web_search(query: str) -> str:
    from ddgs import DDGS
    res = DDGS(timeout=6).text(query, max_results=3)
    if not res:
        return "No results."
    return " | ".join(f"{x['title']}: {x['body'][:200]}" for x in res)


def run_tool(name, args, log=print):
    t = time.perf_counter()
    try:
        if name == "get_weather":
            out = get_weather(str(args.get("location", "")))
        elif name == "wikipedia":
            out = wikipedia(str(args.get("topic", "")))
        elif name == "web_search":
            out = web_search(str(args.get("query", "")))
        elif name == "get_current_time":
            out = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M")
        elif name == "calculate":
            expr = str(args.get("expression", ""))
            out = str(eval(expr, {"__builtins__": {}}, {})) if all(c in "0123456789+-*/(). %" for c in expr) else "invalid expression"
        else:
            out = f"unknown tool {name}"
    except Exception as e:
        out = f"tool error: {e.__class__.__name__}: {e}"
    log(f"[tool] {name}({json.dumps(args)}) -> {out[:120]!r} ({(time.perf_counter()-t)*1000:.0f}ms)")
    return out
