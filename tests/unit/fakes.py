"""Fake model components with configurable latencies, so the orchestration can be tested (and its latency budget
validated) on machines without a GPU/NPU/microphone, e.g. GitHub-hosted runners."""
import threading, time, types
import numpy as np


class FakeSTT:
    def __init__(self, text="what is the capital of italy", delay=0.05):
        self.text, self.delay, self.calls = text, delay, []
    def warmup(self): pass
    def transcribe(self, audio):
        time.sleep(self.delay); self.calls.append(len(audio)); return self.text


class FakeWhisper:
    def __init__(self, text, delay=0.2):
        self.text, self.delay = text, delay
    def warmup(self): pass
    def transcribe(self, audio):
        time.sleep(self.delay); return self.text


class FakeLLM:
    def __init__(self, reply="Rome is the capital of Italy, famous for its ancient history.", ttft=0.05, tok_delay=0.01, tool_call=None):
        self.reply, self.ttft, self.tok_delay, self.tool_call = reply, ttft, tok_delay, tool_call
        self.prompts = []
    def warmup(self): pass
    @staticmethod
    def abort(client): client["aborted"] = True
    def stream(self, history, text, cancel, on_meta=None, on_client=None, on_tool=None, max_rounds=4):
        self.prompts.append((list(history), text))
        client = {"aborted": False}
        if on_client: on_client(client)
        time.sleep(self.ttft)
        if self.tool_call and on_tool:
            on_tool(self.tool_call); time.sleep(0.05)
        for w in self.reply.split(" "):
            if cancel.is_set(): return
            yield w + " "
            time.sleep(self.tok_delay)
        if on_meta:
            on_meta(types.SimpleNamespace(prompt_eval_count=100, eval_count=len(self.reply.split()),
                                          prompt_eval_duration=50e6, eval_duration=len(self.reply.split()) * self.tok_delay * 1e9))


class FakeTTS:
    def __init__(self, delay=0.1, seconds_per_word=0.25):
        self.delay, self.spw, self.requests, self.fillers = delay, seconds_per_word, [], {}
    def wait_ready(self, timeout=None): return True
    def make_fillers(self, texts, timeout=60):
        self.fillers = {t: np.zeros(int(0.3 * 48000), np.float32) for t in texts}; return self.fillers
    def synth(self, text, on_done, steps=None, **kw):
        self.requests.append((text, steps))
        def run():
            time.sleep(self.delay)
            on_done(np.zeros(int(max(1, len(text.split())) * self.spw * 48000), np.float32), {"total_ms": self.delay * 1000})
        threading.Thread(target=run, daemon=True).start()
    def close(self): pass


class FakeGate:
    def __init__(self, sim=0.9, threshold=0.45):
        self.sim, self.threshold, self.enabled = sim, threshold, True
    def enroll(self, path): pass
    def accept(self, audio):
        return self.sim >= self.threshold, self.sim, 1.0


def make_agent(cfg, **overrides):
    """VoiceAgent with fakes and an audio sink (no devices). Returns (agent, sink_chunks, events)."""
    from agent.pipeline import VoiceAgent
    comps = {"stt": FakeSTT(), "llm": FakeLLM(), "tts": FakeTTS(), "whisper": None, "gate": None}
    comps.update(overrides)
    sink, events = [], []
    agent = VoiceAgent(cfg, log=lambda *a: None, sink=lambda x: sink.append(x), components=comps)
    agent.on_event = lambda kind, data: events.append((kind, data))
    agent.warmup()
    return agent, sink, events


def wait_turn(agent, t, timeout=10):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if getattr(t, "finished", False) or t.cancel.is_set():
            return True
        time.sleep(0.01)
    return False
