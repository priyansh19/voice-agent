# Resume update note: Voice Agent (Sep 2026)

Paste this into the resume chat. Everything below is measured or verifiable in the public repo.

## Project

**Local Voice Agent, open source (Apache-2.0), Sep 2026, solo project**
https://github.com/priyansh19/voice-agent

One-line description: a real-time spoken assistant that runs entirely on a laptop with no cloud, no API keys and open-weight models only, answers in a clone of the user's own voice, and starts responding in about 170 ms.

## What it does

- Listens continuously, ignores other speakers (speaker-verification "voice lock"), understands English, Hindi and Hinglish, and replies in the language spoken.
- Starts transcribing and generating during the user's pause (speculative turn start, cancelled or merged if the user keeps talking), plays a short acknowledgement in the cloned voice while the answer is produced, then streams the answer clause by clause.
- Keyless internet tools (weather, Wikipedia, web search) with prefetch from the transcript.
- Browser console with per-turn latency instrumentation, mute/unmute, voice-recording window with reading script and quality check.

## Measured results (Intel Core Ultra 7 155H laptop, no discrete GPU, 2026-09-08)

- Transcript ready 79 to 90 ms before the user finishes speaking.
- Spoken reaction in the cloned voice at 166 to 174 ms.
- First LLM token at 189 to 477 ms.
- First words of the answer at 786 to 994 ms across five unscripted questions (one 1.8 s outlier).
- Hindi/Hinglish turns 4 to 5 s (multilingual STT path).
- Same code on a Mac mini M4 (Metal, mlx-whisper): first answer audio about 1.2 s.

## Technical highlights (use for bullets)

- Split the pipeline across three compute units on one chip: speech-to-text on the Intel NPU (Granite Speech 5.0 TurboCTC via OpenVINO with static shape buckets), LLM (Granite 4.2 3B via a private Ollama instance) and voice-clone TTS decoder (LuxTTS on OpenVINO GPU) on the Arc iGPU, VAD, vocoder and speaker verification (Silero VAD v6, WeSpeaker CAM++ on ONNX Runtime) on the CPU.
- Designed the low-latency orchestration: adaptive silence endpointing, speculative STT+LLM start, cancel/merge on resume, playback gated on firm end-of-turn, clause chunker with a three-word first chunk, pre-synthesised cloned-voice fillers, tool prefetch, per-stage latency metrics on every turn.
- Multilingual routing: fast English CTC model with a language-likelihood gate that escalates to Whisper large-v3-turbo (OpenVINO int4 on Intel, mlx-whisper on Apple Silicon) only when needed; Devanagari replies routed to a Hindi TTS voice.
- Ported the whole stack to Apple Silicon (torch/MPS, mlx, LuxTTS on MPS) behind one config override; TTS runs in a separate worker process with a stdio JSON+PCM protocol to isolate dependency conflicts.
- Found and fixed real latency bugs by measurement: 1.7 s IPv6 fallback from "localhost" on Windows, Ollama multi-slot prompt re-evaluation, zero-frame TTS output for short text, OpenVINO dynamic-shape recompiles, fp16 NaNs.
- Engineering hygiene: 35 unit tests with fake models that assert the orchestration adds under 250 ms, end-to-end browser-audio test, GitHub Actions CI on Linux and Windows plus a self-hosted benchmark workflow, contributing/security docs, architecture diagram.

## Deployment / infrastructure (separate bullet or "Infra" line)

- Self-hosted the agent and a portfolio site on a Mac mini M4: k3s (Colima) with Traefik ingress, cert-manager with Let's Encrypt, EndpointSlices to a native (Metal) backend, launchd services, root LaunchDaemon port passthrough, Cloudflare tunnel for public HTTPS, Tailscale for remote administration from a Windows laptop, browser microphone streamed over WebSocket to the backend.

## Stack keywords

Python 3.12, FastAPI, WebSockets, PyTorch, OpenVINO / OpenVINO GenAI, ONNX Runtime, Ollama, MLX, NumPy, Silero VAD, Whisper, Granite (IBM), LuxTTS, Kokoro, WeSpeaker, Kubernetes (k3s), Traefik, cert-manager, Let's Encrypt, Cloudflare Tunnel, Tailscale, launchd, GitHub Actions, pytest.

## Suggested resume bullets (pick two or three)

- Built an open-source, fully local voice assistant that answers in the user's cloned voice with a 170 ms spoken reaction and sub-second first answer on a laptop with no discrete GPU, by splitting STT, LLM and TTS across the Intel NPU, Arc iGPU and CPU (OpenVINO, Ollama, PyTorch).
- Designed the latency architecture (speculative turn start during the user's pause, cloned-voice fillers, clause-level streaming, tool prefetch) and instrumented every stage per turn; measured 786 to 994 ms to first answer audio across unscripted questions.
- Ported the stack to Apple Silicon (Metal/MLX) and self-hosted it on a Mac mini behind k3s, Traefik and cert-manager with Let's Encrypt, Cloudflare tunnel and Tailscale, with CI on Linux and Windows and a GPU-free test suite.
