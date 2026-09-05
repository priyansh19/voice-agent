# Benchmarks

The scripts that produced the numbers in the main README. They need the real models and the hardware (they are
not part of CI); run them from the repo root with `uv run python benchmarks/<area>/<script>.py`
(TTS scripts use the worker environment: `tts_worker/.venv/Scripts/python.exe benchmarks/tts/<script>.py`).

| script | question it answers |
|---|---|
| `stt/bench_stt2.py` | Granite TurboCTC on CPU: where the time goes, thread sweep, int8; Kokoro fp32 vs int8 |
| `stt/export_stt_openvino_static.py` | static-shape OpenVINO export; CPU vs GPU vs NPU, masked vs padded input |
| `stt/bench_whisper.py` | Whisper turbo / small on GPU and CPU, English vs Hindi vs Hinglish |
| `llm/test_stream.py` | which request features delay the first streamed token (system prompt, think flag, tools) |
| `llm/test_cancel.py` | does Ollama free the slot promptly when a streaming request is cancelled |
| `tts/bench_lux2.py` | LuxTTS with the duration fix: ORT thread sweep vs OpenVINO GPU / CPU, 2 vs 4 steps |
| `tts/test_buckets.py` | static-bucket text padding vs exact dynamic synthesis; writes wavs for listening |
| `tts/check_tts_quality.py` | transcribes those wavs with the STT to verify intelligibility objectively |
| `tts/bench_vocoder_ov.py` | can the vocoder run on the GPU via OpenVINO (it cannot: unsupported op) |
