import json
import pytest
from agent import tools


class _Resp:
    def __init__(self, payload, status=200): self._p, self.status_code = payload, status
    def json(self): return self._p


@pytest.fixture
def fake_http(monkeypatch):
    calls = []
    def get(url, params=None, headers=None):
        calls.append((url, params))
        if "geocoding" in url:
            name = params["name"]
            if name in ("Milan", "Rome"):
                return _Resp({"results": [{"name": name, "country": "Italy", "latitude": 45.4, "longitude": 9.2}]})
            if name == "दिल्ली" and params.get("language") == "hi":
                return _Resp({"results": [{"name": "दिल्ली", "country": "भारत", "latitude": 28.6, "longitude": 77.2}]})
            return _Resp({"results": []})
        if "forecast" in url:
            return _Resp({"current": {"temperature_2m": 30.4, "weather_code": 0, "wind_speed_10m": 3.0},
                          "daily": {"time": ["2026-09-04", "2026-09-05", "2026-09-06"], "weather_code": [3, 2, 61],
                                    "temperature_2m_max": [34, 33, 25], "temperature_2m_min": [21, 21, 18],
                                    "precipitation_probability_max": [0, 5, 60], "precipitation_sum": [0, 0, 4.2]}})
        raise AssertionError(url)
    monkeypatch.setattr(tools._http, "get", get)
    tools._geo_cache.clear(); tools._wx_cache.clear()
    return calls


def test_weather_text_and_caching(fake_http):
    out = tools.get_weather("Milan")
    assert out.startswith("Milan, Italy: now 30°C, clear sky")
    assert "tomorrow (2026-09-05): partly cloudy, 21 to 33°C, rain chance 5%" in out
    n = len(fake_http)
    assert tools.get_weather("milan") == out and len(fake_http) == n     # 10-minute cache, case-insensitive


def test_geocode_devanagari_table_and_hindi_fallback(fake_http):
    assert tools.geocode("मिलान")["name"] == "Milan"                     # transliteration table
    assert tools.geocode("दिल्ली")["name"] == "दिल्ली"                    # language=hi fallback
    assert tools.geocode("Nowhereville") is None


def test_run_tool_calculate_and_time_and_unknown():
    assert tools.run_tool("calculate", {"expression": "17*23"}, log=lambda *a: None) == "391"
    assert tools.run_tool("calculate", {"expression": "__import__('os')"}, log=lambda *a: None) == "invalid expression"
    assert len(tools.run_tool("get_current_time", {}, log=lambda *a: None)) > 10
    assert "unknown tool" in tools.run_tool("nope", {}, log=lambda *a: None)


def test_tool_schema_is_compact():
    # every schema token is re-saved/restored by the LLM runner on each turn: keep it small
    assert len(json.dumps(tools.TOOLS)) < 1200
    assert {t["function"]["name"] for t in tools.TOOLS} == {"get_weather", "wikipedia", "web_search", "get_current_time", "calculate"}
