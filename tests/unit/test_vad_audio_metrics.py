import numpy as np, soundfile as sf
from agent import config as C
from agent.vad import Endpointer
from agent.audio_io import Speaker, resample
from agent.metrics import TurnMetrics


def _frames(a, n=512):
    return a[: len(a) // n * n].reshape(-1, n)


def test_endpointer_emits_start_tentative_end_on_real_speech():
    cfg = C.load(); ep = Endpointer(cfg.vad)
    a, sr = sf.read("samples/test_utterance_16k.wav", dtype="float32")
    a = np.concatenate([a, np.zeros(16000, np.float32)])
    names = [ev for f in _frames(a) for ev, _ in ep.process(f)]
    assert names[0] == "speech_start"
    assert "tentative_end" in names and names[-1] == "end"
    assert names.index("tentative_end") < names.index("end")


def test_endpointer_ignores_silence_and_noise():
    cfg = C.load(); ep = Endpointer(cfg.vad)
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(16000 * 3) * 0.002).astype(np.float32)
    assert [ev for f in _frames(quiet) for ev, _ in ep.process(f)] == []


def test_adaptive_silence_from_transcript():
    cfg = C.load(); ep = Endpointer(cfg.vad)
    ep.set_uncertain("i was thinking that we could go to the")
    assert ep.required_silence_ms == cfg.vad.min_silence_uncertain_ms
    ep.set_uncertain("what is the capital of italy")
    assert ep.required_silence_ms == cfg.vad.min_silence_ms


def test_speaker_sink_and_remote_modes():
    got = []; sp = Speaker(sink=lambda x: got.append(x)); started = []
    sp.play(np.zeros(4800, np.float32), on_start=lambda: started.append(1))
    assert len(got) == 1 and started == [1] and not sp.is_busy()
    sent = []; sp2 = Speaker(); sp2.remote = lambda b: sent.append(b); sp2.remote_stop = lambda: sent.append("stop")
    sp2.play(np.ones(48000, np.float32) * 0.5)
    assert len(sent[0]) == 48000 * 2 and sp2.is_busy()
    sp2.stop(); assert sent[-1] == "stop" and not sp2.is_busy(tail_ms=0)


def test_resample_length():
    x = np.zeros(24000, np.float32)
    assert len(resample(x, 24000, 48000)) == 48000 and len(resample(x, 24000, 16000)) == 16000


def test_metrics_rows_and_summary():
    m = TurnMetrics(1); m.tentative_end = 9.7; m.speech_end = 10.0; m.stt_done = 9.9; m.llm_first_token = 10.3
    m.first_chunk_text = 10.5; m.first_audio_ready = 10.9; m.first_audio_played = 10.95; m.filler_played = 10.17
    m.llm_done = 11.5; m.audio_done = 13.0; m.llm_eval_tokens = 20; m.llm_eval_ms = 1200; m.tts_chunks = ["x"]
    rows = dict((r[0], r[1]) for r in m.rows())
    assert rows["FIRST ANSWER AUDIO OUT"] == 950 and rows["STT done"] == -100 and rows["speculative head start"] == -300
    assert round(m.summary()["filler_ms"]) == 170
