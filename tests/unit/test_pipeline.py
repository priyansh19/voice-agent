"""Orchestration tests with fake models: turn flow, speculation/cancel/merge, fillers, language tags,
multilingual override, tool filler, and a latency-budget test of the pipeline's own overhead."""
import time
import numpy as np, pytest, soundfile as sf
from agent import config as C
from tests.unit.fakes import FakeLLM, FakeSTT, FakeTTS, FakeWhisper, FakeGate, make_agent, wait_turn


def cfg():
    c = C.load()
    c["tts"]["fillers"]["after_ms"] = 60
    return c


def wav_frames(path="samples/test_utterance_16k.wav", tail_s=1.0, seconds=3.0):
    """First `seconds` of the sample (one continuous phrase) plus trailing silence, as 32 ms frames."""
    a, sr = sf.read(path, dtype="float32")
    a = np.concatenate([a[: int(16000 * seconds)], np.zeros(int(16000 * tail_s), np.float32)])
    return a[: len(a) // 512 * 512].reshape(-1, 512)


def test_text_turn_produces_transcript_response_metrics_and_audio():
    agent, sink, events = make_agent(cfg())
    t = agent.say("What is the capital of Italy?")
    assert wait_turn(agent, t)
    kinds = [k for k, _ in events]
    assert "transcript" in kinds and "response" in kinds and "metrics" in kinds
    assert sum(len(x) for x in sink) > 48000                      # some audio reached the speaker
    assert t.m.first_audio_played > t.m.speech_end
    # the LLM saw a language tag and no prefetch context for a non-weather question
    hist, text = agent.llm.prompts[-1]
    assert text.startswith("[user spoke English; reply in English] What is the capital")
    assert agent.history[-1]["role"] == "assistant"


def test_first_tts_chunk_is_short_and_uses_fewer_steps():
    agent, sink, events = make_agent(cfg())
    t = agent.say("Tell me about Rome."); assert wait_turn(agent, t)
    first_text, first_steps = agent.tts.requests[0]
    assert len(first_text.split()) <= 3 and first_steps == cfg().tts.first_chunk_steps


def test_filler_plays_only_when_answer_is_slow():
    slow = FakeTTS(delay=0.5)
    agent, sink, events = make_agent(cfg(), tts=slow)
    t = agent.say("Hi there"); assert wait_turn(agent, t)
    assert t.m.filler_played and t.m.filler_played < t.m.first_audio_played
    fast = FakeTTS(delay=0.0)
    agent2, sink2, events2 = make_agent(cfg(), tts=fast, llm=FakeLLM(ttft=0.0, tok_delay=0.0))
    t2 = agent2.say("Hi there"); assert wait_turn(agent2, t2)
    assert not t2.m.filler_played


def test_audio_path_speculative_start_then_commit():
    agent, sink, events = make_agent(cfg())
    agent.feed(wav_frames(), ignore_mute=True)
    t = agent.turn; assert wait_turn(agent, t)
    assert t.m.speculative and t.m.tentative_end < t.m.speech_end
    assert t.m.stt_done < t.m.speech_end + 0.2                        # STT overlapped with the endpoint wait
    assert [k for k, _ in events].count("response") == 1


def test_user_continues_speaking_merges_transcripts():
    stt = FakeSTT(text="first part", delay=0.02)
    agent, sink, events = make_agent(cfg(), stt=stt, llm=FakeLLM(ttft=0.4), tts=FakeTTS(delay=0.4))
    agent.feed(wav_frames(tail_s=0.4), ignore_mute=True)             # firm end fires, answer not audible yet
    t1 = agent.turn
    time.sleep(0.15)                                                  # let t1's STT finish (file is fed faster than real time)
    stt.text = "second part"
    agent.feed(wav_frames(tail_s=1.0), ignore_mute=True)             # user speaks again -> t1 cancelled, merged
    t2 = agent.turn; assert t2 is not t1 and wait_turn(agent, t2)
    assert t1.cancel.is_set()
    assert t2.transcript == "first part second part"


def test_non_english_transcript_waits_for_whisper_directly():
    stt = FakeSTT(text="kal milan me mosam kesa reiga kuhi chata likar jana chahi", delay=0.02)   # score 0.36
    wh = FakeWhisper("कल मिलान में मौसम कैसा रहेगा", delay=0.05)
    agent, sink, events = make_agent(cfg(), stt=stt, whisper=wh)
    agent.feed(wav_frames(), ignore_mute=True)
    t = agent.turn; assert wait_turn(agent, t)
    assert t.transcript == "कल मिलान में मौसम कैसा रहेगा"
    hist, text = agent.llm.prompts[-1]
    assert text.startswith("[user spoke Hindi (Devanagari script); reply in Hindi (Devanagari script)]")


def test_borderline_english_is_verified_in_parallel_and_overridden():
    stt = FakeSTT(text="what is the weather like in milan kesa reiga", delay=0.02)                # score ~0.78
    wh = FakeWhisper("मिलान में मौसम कैसा रहेगा", delay=0.15)
    agent, sink, events = make_agent(cfg(), stt=stt, whisper=wh, llm=FakeLLM(ttft=0.3), tts=FakeTTS(delay=0.3))
    agent.feed(wav_frames(), ignore_mute=True)
    time.sleep(0.5)
    t = agent.turn; assert wait_turn(agent, t)
    assert t.transcript == "मिलान में मौसम कैसा रहेगा"
    assert any(k == "transcript" and d.get("multilingual") for k, d in events)
    assert [k for k, _ in events].count("response") == 1


def test_voice_lock_rejects_other_speakers():
    agent, sink, events = make_agent(cfg(), gate=FakeGate(sim=0.2))
    agent.feed(wav_frames(), ignore_mute=True)
    time.sleep(0.5)
    assert not any(k == "response" for k, _ in events)
    assert any(k == "gate" and not d["ok"] for k, d in events)


def test_tool_call_plays_checking_filler():
    agent, sink, events = make_agent(cfg(), llm=FakeLLM(tool_call="get_weather", ttft=0.05))
    t = agent.say("weather please"); assert wait_turn(agent, t)
    assert any(k == "tool" for k, _ in events)
    assert any("(checking)" in c for c in t.m.tts_chunks)         # the cloned-voice "Let me check that." was queued
    assert t.m.first_audio_played and t.response.strip()          # and the answer still arrived


@pytest.mark.speed
def test_orchestration_overhead_is_small():
    """With fake models of known latency, the pipeline's own overhead must stay well under 250 ms."""
    llm = FakeLLM(reply="Rome is the capital of Italy and it is famous for history.", ttft=0.10, tok_delay=0.07)
    tts = FakeTTS(delay=0.30)
    agent, sink, events = make_agent(cfg(), llm=llm, tts=tts)
    lat = []
    for _ in range(3):
        t = agent.say("What is the capital of Italy?"); assert wait_turn(agent, t)
        lat.append((t.m.first_audio_played - t.m.speech_end) * 1000)
    expected = (0.10 + 3 * 0.07 + 0.30) * 1000       # ttft + first 3 words + tts
    overhead = min(lat) - expected
    assert overhead < 250, f"overhead {overhead:.0f} ms (latencies {lat})"
