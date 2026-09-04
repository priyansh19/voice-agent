# Contributing

Thanks for helping make local, open-weight voice agents faster. This project values **measured** latency wins,
small focused pull requests, and tests that run without special hardware.

## Development setup

```bash
uv sync --group dev                      # main environment + pytest/ruff
uv run pytest tests/unit -q              # ~1 min, no GPU/NPU/microphone needed (models are faked)
uv run ruff check .
```

For the real thing (Ollama + models), follow the README *Setup* section, then `uv run python ui/server.py`.

## What the test layers cover

| layer | how to run | what it proves |
|---|---|---|
| unit | `pytest tests/unit` | chunker, prefetch rules, language scoring, tools (HTTP mocked), VAD endpointer on a real sample, audio queue, metrics |
| orchestration | `pytest tests/unit/test_pipeline.py` | turn flow with fake STT/LLM/TTS: speculation, cancel + merge, fillers, voice lock, language tags, multilingual override, tool filler |
| latency budget | `pytest -m speed` | with fake models of known latency, the pipeline's own overhead must stay under 250 ms |
| console + protocol | `pytest tests/unit/test_ui_server.py` | FastAPI pages, control WebSocket, `/audio` PCM streaming end to end |
| real hardware | `.github/workflows/benchmark.yml` on a self-hosted runner, or `benchmarks/` scripts | actual per-stage latencies on your machine |

CI runs lint, the unit/orchestration suite on Ubuntu and Windows, the latency budget, and checks that `web/` is in
sync with `ui/index.html`. The TTS worker environment is checked weekly (it has its own venv because LuxTTS pins an
older `transformers`).

## Latency changes: bring numbers

Every latency-related PR should include the per-turn table from the console (or `main.py --wav ...`) before and
after, on the same machine, and say which stage moved. The `benchmarks/` folder holds the scripts that produced
the numbers in the README; add yours there if you explore a new backend (a new STT/TTS runtime, another GPU, a
different LLM server).

## Adding a backend

- STT: implement `transcribe(audio16k) -> str` and `warmup()`; wire it in `agent/pipeline.py` like `GraniteSTT`.
- LLM: implement `stream(history, text, cancel, on_meta, on_client, on_tool)` yielding text deltas and `abort(client)`.
- TTS: implement `synth(text, on_done, steps=None)` calling `on_done(float32_48k, meta)` in submission order,
  `wait_ready()`, `make_fillers(texts)`, `close()`.
- Keep hardware-specific code behind those interfaces so the orchestration tests keep running everywhere.

## Style

`ruff` (config in `pyproject.toml`), 130-column lines, comments that explain *why* (most of the code exists
because of a measured problem; say which one). Prefer a config knob in `config.yaml` over a hard-coded constant.

## Pull requests

- One topic per PR; include the test that would have caught the bug.
- Do not commit model weights, compiled caches, or recordings (see `.gitignore`); never commit a real person's
  voice sample.
- Fill in the PR template; a maintainer will run the real-hardware benchmark when the change touches latency.
