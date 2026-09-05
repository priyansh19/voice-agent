"""Temporary local test UI:  uv run python ui/server.py   ->  http://127.0.0.1:8765

Mute / unmute the microphone, send text turns (skips STT), replay a test wav, toggle fillers / speculation /
TTS steps, and watch the transcript, streamed reply and the per-turn latency table live.
"""
import asyncio, json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):                       # Windows console is cp1252: never crash on Devanagari
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from agent import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
clients: set[WebSocket] = set()
loop: asyncio.AbstractEventLoop | None = None
agent = None
cfg = C.load()
history_events: list = []


def broadcast(kind, data):
    msg = {"kind": kind, "data": data, "t": time.time()}
    if kind in ("transcript", "response", "metrics"):
        history_events.append(msg); del history_events[:-200]
    if loop is None:
        return
    def _send():
        dead = []
        for ws in list(clients):
            try:
                asyncio.ensure_future(ws.send_text(json.dumps(msg)))
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
    loop.call_soon_threadsafe(_send)


def log(*a):
    line = " ".join(str(x) for x in a)
    try: print(line, flush=True)
    except Exception: pass
    broadcast("log", line)


def start_agent():
    global agent
    from agent.pipeline import VoiceAgent
    try:
        agent = VoiceAgent(cfg, log)
        agent.muted = True
        agent.on_event = broadcast
        agent.warmup()
        broadcast("ready", {"muted": True, "settings": current_settings()})
    except Exception as e:
        log(f"[ui] agent failed: {e!r}")
        broadcast("state", "error"); return
    try:
        agent.run_mic()                       # returns immediately on machines without a microphone (headless server)
    except Exception as e:
        log(f"[ui] local microphone unavailable ({e.__class__.__name__}); browser microphone mode only")


def current_settings():
    return {"fillers": bool(agent.fillers) if agent else cfg.tts.fillers.enabled,
            "speculative": cfg.vad.speculative, "barge_in": cfg.vad.barge_in,
            "steps": cfg.tts.steps, "min_silence_ms": cfg.vad.min_silence_ms, "model": cfg.llm.model,
            "backend": cfg.tts.backend,
            "voice_lock": bool(agent.gate.enabled) if agent and agent.gate else False,
            "gate_threshold": agent.gate.threshold if agent and agent.gate else 0.45,
            "voice_ref": os.path.basename(cfg.tts.voice_ref)}


@app.get("/")
async def index():
    return HTMLResponse(open(os.path.join(HERE, "index.html"), encoding="utf-8").read())


@app.get("/record")
async def record_page():
    return HTMLResponse(open(os.path.join(HERE, "record.html"), encoding="utf-8").read())


@app.get("/voice.wav")
async def voice_wav(candidate: int = 0):
    from fastapi.responses import FileResponse
    p = agent.candidate_path() if (candidate and agent) else C.abspath(cfg.tts.voice_ref)
    return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept(); clients.add(ws)
    await ws.send_text(json.dumps({"kind": "hello", "data": {"state": agent.state if agent else "loading",
                                                             "muted": agent.muted if agent else True,
                                                             "settings": current_settings(), "history": history_events}}))
    try:
        while True:
            req = json.loads(await ws.receive_text())
            cmd = req.get("cmd")
            if agent is None or agent.state == "loading":
                await ws.send_text(json.dumps({"kind": "log", "data": "[ui] still loading..."})); continue
            if cmd == "mute":
                agent.muted = bool(req.get("value", True)); broadcast("muted", agent.muted)
                log(f"[ui] mic {'muted' if agent.muted else 'LIVE'}")
            elif cmd == "say" and req.get("text", "").strip():
                threading.Thread(target=agent.say, args=(req["text"].strip(),), daemon=True).start()
            elif cmd == "test_wav":
                path = C.abspath(req.get("path") or "samples/test_utterance_16k.wav")
                threading.Thread(target=agent.run_wav, args=(path,), daemon=True).start()
            elif cmd == "stop":
                t = agent.turn
                if t: agent._cancel(t)
                agent.speaker.stop(); agent.set_state("idle")
            elif cmd == "clear":
                agent.history.clear(); log("[ui] history cleared")
            elif cmd in ("record_voice", "record_start"):
                threading.Thread(target=agent.record_voice, args=(float(req.get("seconds", 20)),), daemon=True).start()
            elif cmd == "record_use":
                threading.Thread(target=agent.use_recording, daemon=True).start()
            elif cmd == "record_test":
                threading.Thread(target=agent.test_voice, daemon=True).start()
            elif cmd == "devices":
                import sounddevice as sd
                devs = sd.query_devices(); default = sd.default.device[0]
                apis = sd.query_hostapis()
                items = [{"index": i, "name": f"{d['name']} ({apis[d['hostapi']]['name']})", "default": i == default}
                         for i, d in enumerate(devs) if d["max_input_channels"] > 0]
                await ws.send_text(json.dumps({"kind": "devices", "data": {"items": items, "current": agent.cfg.audio.input_device}}))
            elif cmd == "set":
                k, v = req.get("key"), req.get("value")
                if k == "fillers":
                    if v and not agent.fillers and cfg.tts.fillers.texts:
                        agent.fillers = agent.tts.make_fillers(cfg.tts.fillers.texts)
                    elif not v:
                        agent.fillers = {}
                elif k == "speculative": cfg["vad"]["speculative"] = bool(v)
                elif k == "barge_in": cfg["vad"]["barge_in"] = bool(v)
                elif k == "min_silence_ms": cfg["vad"]["min_silence_ms"] = int(v); agent.ep.cfg = cfg.vad
                elif k == "steps": cfg["tts"]["steps"] = int(v); agent.tts_steps = int(v)
                elif k == "input_device": agent.set_input_device(None if v in (None, "", "default") else int(v))
                elif k == "voice_lock" and agent.gate: agent.gate.enabled = bool(v); log(f"[gate] voice lock {'ON' if v else 'off'}")
                elif k == "gate_threshold" and agent.gate: agent.gate.threshold = float(v)
                broadcast("settings", current_settings())
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


@app.websocket("/audio")
async def audio_ws(ws: WebSocket):
    """Browser audio: receives int16 16 kHz mono PCM frames, sends int16 48 kHz PCM for playback ('stop' = flush)."""
    await ws.accept()
    if agent is None or agent.state == "loading":
        await ws.close(code=1013); return
    import queue as _q
    q: "_q.Queue" = _q.Queue(maxsize=400)
    lp = asyncio.get_running_loop()
    def send_pcm(b: bytes):
        lp.call_soon_threadsafe(lambda: asyncio.ensure_future(ws.send_bytes(b)))
    def send_stop():
        lp.call_soon_threadsafe(lambda: asyncio.ensure_future(ws.send_text("stop")))
    agent.attach_browser_audio(send_pcm, send_stop)
    n = cfg.audio.frame_samples
    def frames():
        buf = np.zeros(0, np.float32)
        while True:
            item = q.get()
            if item is None: return
            buf = np.concatenate([buf, item])
            while len(buf) >= n:
                yield buf[:n].copy(); buf = buf[n:]
    threading.Thread(target=agent.feed, args=(frames(),), kwargs={"source": "browser"}, daemon=True).start()
    try:
        while True:
            data = await ws.receive_bytes()
            try: q.put_nowait(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
            except _q.Full: pass
    except WebSocketDisconnect:
        pass
    finally:
        q.put(None); agent.detach_browser_audio()


@app.on_event("startup")
async def _startup():
    global loop
    loop = asyncio.get_running_loop()
    threading.Thread(target=start_agent, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8765")), log_level="warning")
