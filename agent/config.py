import os, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Cfg(dict):
    """dict with attribute access, recursive."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k) from None
        return Cfg(v) if isinstance(v, dict) else v


def _merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        base[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load(path=None) -> Cfg:
    """config.yaml, optionally deep-merged with an override file (VOICE_AGENT_CONFIG env var or `path`),
    e.g. config.macmini.yaml for Apple Silicon backends."""
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    override = path or os.environ.get("VOICE_AGENT_CONFIG")
    if override and os.path.abspath(override) != os.path.join(ROOT, "config.yaml"):
        with open(abspath(override), "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return Cfg(cfg)


def abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
