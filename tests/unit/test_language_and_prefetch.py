import concurrent.futures as cf
from agent.stt_whisper import english_score, same_utterance, is_non_latin
from agent import prefetch


def test_english_score_separates_english_from_romanized_hindi():
    assert english_score("hey can you tell me what the weather is like in milan tomorrow") == 1.0
    assert english_score("kal milan me mosam kesa reiga kuhi chata likar jana chahi") < 0.7
    assert english_score("") == 0.0


def test_same_utterance_tolerates_case_punctuation_and_partial_overlap():
    assert same_utterance("Hey, can you tell me what the weather is like?", "hey can you tell me what the weather is like")
    assert same_utterance("bring an umbrella", "Bring an umbrella.")
    assert not same_utterance("the capital of italy", "kal milan me mosam")


def test_is_non_latin():
    assert is_non_latin("कल मिलान") and not is_non_latin("kal milan")


class _Cfg(dict):
    def get(self, k, d=None): return super().get(k, d)


def test_prefetch_plans_weather_for_english_hinglish_and_hindi(monkeypatch):
    calls = []
    monkeypatch.setattr(prefetch.tools, "get_weather", lambda loc: calls.append(loc) or f"wx:{loc}")
    for text, loc in [("what is the weather like in milan tomorrow", "milan"),
                      ("kal Milan mein baarish hogi kya", "Milan"),
                      ("कल मिलान में मौसम कैसा रहेगा", "मिलान"),
                      ("is it raining in new york today", "new york")]:
        planned = prefetch.plan(text, _Cfg())
        assert planned and planned[0] == f"weather for {loc}", text
        assert prefetch.collect(planned, 2.0, log=lambda *a: None).strip().startswith("[Live data")
    assert calls == ["milan", "Milan", "मिलान", "new york"]


def test_prefetch_ignores_non_weather_and_bad_locations():
    assert prefetch.plan("what is the capital of italy", _Cfg()) is None
    assert prefetch.plan("tell me what the weather is like", _Cfg()) is None       # no place, no default
    assert prefetch.plan("tell me what the weather is like", _Cfg(default_location="Rome"))[0] == "weather for Rome"


def test_collect_times_out_gracefully():
    fut = cf.Future()          # never completes
    assert prefetch.collect(("weather for x", fut), 0.05, log=lambda *a: None) == ""
