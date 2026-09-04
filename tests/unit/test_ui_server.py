"""The console server with fake models: HTTP pages, the control WebSocket, and the /audio PCM protocol."""
import json, threading, time
import numpy as np, soundfile as sf
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import ui.server as srv
    from tests.unit.fakes import FakeSTT, FakeLLM, FakeTTS
    from agent.pipeline import VoiceAgent
    srv.cfg["tts"]["fillers"]["after_ms"] = 50
    agent = VoiceAgent(srv.cfg, log=srv.log, components={"stt": FakeSTT(delay=0.02), "llm": FakeLLM(ttft=0.02, tok_delay=0.005),
                                                          "tts": FakeTTS(delay=0.02), "whisper": None, "gate": None})
    agent.on_event = srv.broadcast; agent.muted = True
    srv.agent = agent
    orig = srv.start_agent
    srv.start_agent = lambda: None            # don't build real models on startup
    with TestClient(srv.app) as c:
        agent.warmup()
        yield c
    srv.start_agent = orig


def test_pages_and_voice_endpoint(client):
    assert "Voice Agent Console" in client.get("/").text
    assert "Record your voice" in client.get("/record").text


def test_control_ws_text_turn_and_settings(client):
    with client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text()); assert hello["kind"] == "hello"
        ws.send_text(json.dumps({"cmd": "set", "key": "steps", "value": 2}))
        ws.send_text(json.dumps({"cmd": "say", "text": "hello there"}))
        seen = set(); t0 = time.time()
        while time.time() - t0 < 10 and not {"transcript", "response", "metrics"} <= seen:
            seen.add(json.loads(ws.receive_text())["kind"])
        assert {"transcript", "response", "metrics"} <= seen


def test_audio_ws_streams_pcm_in_and_out(client):
    a, sr = sf.read("samples/test_utterance_16k.wav", dtype="float32")
    pcm = (np.concatenate([a, np.zeros(16000, np.float32)]) * 32767).astype(np.int16)
    with client.websocket_connect("/ws") as ctl, client.websocket_connect("/audio") as aud:
        ctl.receive_text()
        ctl.send_text(json.dumps({"cmd": "mute", "value": False}))
        got = bytearray(); done = threading.Event()
        def reader():
            try:
                while True:
                    m = aud.receive()
                    if "bytes" in m and m["bytes"]: got.extend(m["bytes"])
                    if len(got) > 48000 * 2: done.set()
            except Exception: pass
        threading.Thread(target=reader, daemon=True).start()
        for i in range(0, len(pcm) - 511, 512):
            aud.send_bytes(pcm[i:i + 512].tobytes()); time.sleep(0.004)   # ~8x real time is enough for the VAD
        assert done.wait(15), "no reply audio came back over /audio"
        ctl.send_text(json.dumps({"cmd": "mute", "value": True}))
