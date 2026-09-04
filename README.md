# Voice Agent — sub-second, local, open-weight only, in your own voice

A real-time spoken assistant that runs entirely on this laptop (Intel Core Ultra 7 155H: CPU + Arc iGPU + NPU),
uses only open-weight models, and answers in a clone of your voice.

## Stack

| Layer | Model (license) | Runs on | Measured |
|---|---|---|---|
| VAD / endpointing | Silero VAD v6 | CPU | 32 ms frames |
| STT (English, fast) | Granite Speech 5.0 470M TurboCTC (Apache-2.0) | **NPU** via OpenVINO, static 6 s / 15 s buckets | 115–150 ms per utterance |
| STT (any language) | Whisper large-v3-turbo int4 (MIT) | **Arc iGPU** via OpenVINO GenAI | 0.85–1.1 s; used directly when the fast transcript doesn't look English, otherwise verified in parallel |
| TTS (Hindi replies) | Kokoro-82M Hindi voice | CPU | Devanagari text is routed here (the clone is English-only) |
| LLM | Granite 4.2 3B via Ollama (Apache-2.0) | **Arc iGPU** (Vulkan) | first token ~100–250 ms, 12–15 tok/s |
| TTS (your voice) | LuxTTS / ZipVoice-distill (Apache-2.0), zero-shot clone from ~10 s | flow decoder on **Arc iGPU** via OpenVINO (static buckets, f32), vocoder on CPU | 300–500 ms per sentence |
| TTS fallback | Kokoro-82M (Apache-2.0) | CPU | generic voice |
| Voice lock | WeSpeaker CAM++ (Apache-2.0) speaker embedding | CPU (onnxruntime) | ~45 ms |

## Latency design (what makes it fast)

1. **Speculative start**: STT + LLM begin on the first 64 ms of silence, ~250–320 ms before the firm end of turn.
   Playback is gated on the firm end so it never interrupts you; if you keep talking the work is cancelled
   (Ollama is aborted by closing the socket) and the partial transcript is merged into the next turn.
2. **Adaptive endpoint**: the speculative transcript decides whether your pause is a real end of turn
   ("... and" → wait longer).
3. **STT on the NPU, LLM on the GPU, TTS split GPU/CPU** — no stage waits for another stage's hardware.
4. **Clause streaming**: the first TTS chunk is cut after ~5 words (2 flow steps), later chunks are full clauses
   (4 steps) generated while the previous chunk plays.
5. **Cloned-voice filler** ("Hmm.", "Right.") pre-synthesized at startup and played ~170 ms after you stop,
   only if the real answer is not ready yet.
6. `127.0.0.1` not `localhost` for Ollama (Windows IPv6 fallback cost 1.7 s per request).

Typical turn (measured from the firm end of your speech): filler audio 170 ms, STT done −130 ms (before the end),
LLM first token 230 ms, first answer audio **~1.3 s**.

## Languages

Every user message is tagged with the detected language (English / Hindi / Hinglish) so the LLM answers in that
language regardless of earlier turns. Weather questions in Hindi are prefetched too (Devanagari place names are
geocoded, with a transliteration table for common cities). Hindi turns cost ~4–6 s today: Whisper (~1 s) plus
Devanagari being token-expensive for Granite; Qwen3-4B-Instruct would be faster in Hindi but its Ollama tag leaks
reasoning into the answer on this build, so it is not used.

The agent starts its own private `ollama serve` on port 11436 with `OLLAMA_NUM_PARALLEL=1` (config `llm.private_instance`)
so consecutive turns reuse the KV-cache prefix; the system-wide Ollama (4 slots) re-evaluated the whole prompt on
most turns (0.5–0.8 s). Even so, Ollama's Vulkan runner saves/restores the slot state around every changed prompt,
which costs ~250–400 ms per turn on this iGPU; the tool schema and system prompt are kept short for that reason.

## Where the time goes (steady state, English, text turn)

| stage | ms after end of speech |
|---|---|
| cloned-voice filler audible | ~170 |
| LLM first token (incl. ~250 ms slot-state overhead) | 330–550 |
| first 3 words generated | +180 |
| TTS of that chunk (2 flow steps GPU + vocoder CPU) | +400 |
| **first answer audio** | **950–1200** |

Getting to ~500 ms needs runtime changes, not tuning: the LLM off Ollama onto OpenVINO GenAI with a persistent
KV cache (−250 ms), a GPU vocoder or 1-step flow (−200 ms), and a smaller LLM for faster first words (−100 ms).

## Run

```bash
uv run python ui/server.py          # test console at http://127.0.0.1:8765 (mute/unmute, text test, voice recording)
uv run python main.py               # plain terminal mode (mic)
uv run python main.py --wav samples/test_utterance_16k.wav --no-audio-out   # offline test, saves out/reply_0.wav
```

First run: open **🎙 Record my voice…** in the console (a `/record` window with a reading script, tips, countdown,
level meter, take quality check, playback, "Use this voice", "Test the clone"), then enable **Voice lock** so the
agent ignores every other voice. `voices/me.wav` currently contains a placeholder voice. Tools: Open-Meteo weather,
Wikipedia, DuckDuckGo search, clock, calculator (all keyless); weather is prefetched speculatively from the transcript.

Requirements already set up here: Ollama with `granite4.2:3b`, two uv environments (`.venv` for the agent,
`tts_worker/.venv` for LuxTTS — they need incompatible `transformers` versions, so TTS runs as a worker process).
Model compilation is cached under `models/`; a cold first start compiles STT (NPU) and TTS buckets (GPU) in ~5 min.

## Hosting: what runs where

The models need a machine with real compute (this laptop, or a GPU server). Vercel can only host the console page:

- **Backend** = `uv run python ui/server.py` on the machine with the models. Expose it with a tunnel, e.g.
  `cloudflared tunnel --url http://127.0.0.1:8765` (prints an `https://….trycloudflare.com` URL).
- **Frontend** = `web/` (generated by `python scripts/build_web.py`, plain static files). Deploy it on Vercel
  (import the GitHub repo, root directory `web`, framework "Other"). In the hosted page set the **backend** field to
  the tunnel URL and tick **Use THIS browser's microphone** — audio then streams over the `/audio` WebSocket
  (16 kHz PCM up, 48 kHz PCM down; the browser's echo cancellation replaces the local echo guard).

Sizing: a 2 vCPU / 8 GB VPS without GPU (typical Hostinger plan) gives 10–30 s per turn with this stack — CPU-only
STT/LLM/TTS measured 0.5–5 s *each* even on this 22-thread laptop. A cut-down CPU variant (whisper-tiny, 0.5B LLM,
Kokoro, no cloning) reaches ~3–6 s. A GPU instance (RTX 3090/4090 class) makes sub-500 ms realistic because every
model here has native CUDA paths (LuxTTS 150× real-time, Whisper ~150 ms, LLM 80+ tok/s).

## Tuning knobs (`config.yaml`)

- `vad.min_silence_ms` (320): lower = snappier, more mid-sentence cut-offs. `min_rms_ratio`, `threshold`: noise rejection.
- `tts.steps` / `first_chunk_steps`: quality vs speed. `tts.prompt_duration_s`: shorter reference = faster TTS.
- `speaker_gate.threshold` (0.45): voice lock strictness.
- `llm.model`: any Ollama model with tool support; smaller = faster first clause.

## Layout

```
agent/        pipeline.py (orchestrator), vad.py, stt.py, llm.py, chunker.py, tts_client.py, speaker_gate.py, audio_io.py, metrics.py
tts_worker/   server.py (worker process), lux_fast.py (LuxTTS fixes + OpenVINO buckets)
ui/           server.py + index.html (test console)
scripts/      benchmarks used to pick the stack
```
