## What and why

<!-- one paragraph: the change, and the measured problem it addresses -->

## Latency impact (if any)

| stage | before (ms) | after (ms) |
|---|---|---|
| STT done | | |
| LLM first token | | |
| first TTS chunk text | | |
| FIRST ANSWER AUDIO OUT | | |

Machine: <!-- CPU / GPU / NPU, OS -->

## Checklist

- [ ] `uv run pytest tests/unit -q` passes locally
- [ ] `uv run ruff check .` passes
- [ ] `web/` regenerated if `ui/index.html` changed (`python tools/build_web.py`)
- [ ] No model weights, caches, or voice recordings committed
- [ ] Docs/README updated where behaviour or config changed
