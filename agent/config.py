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


def load(path=None) -> Cfg:
    with open(path or os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return Cfg(yaml.safe_load(f))


def abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
