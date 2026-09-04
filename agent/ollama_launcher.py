"""Start a private Ollama instance tuned for one low-latency conversation (no system-wide settings touched).

OLLAMA_NUM_PARALLEL=1 keeps every turn on the same slot so the KV-cache prefix is reused (the default 4 slots put
consecutive turns on different, empty slots and re-evaluate the whole prompt: ~0.5-0.8 s per turn on the iGPU).
"""
import os, subprocess, sys, time, urllib.request


def ensure_ollama(host: str, log=print, timeout_s: float = 30.0):
    """If nothing answers at `host`, spawn `ollama serve` bound to it with latency-oriented env. Returns the Popen or None."""
    def alive():
        try:
            with urllib.request.urlopen(f"{host}/api/version", timeout=1) as r:
                return r.status == 200
        except Exception:
            return False
    if alive():
        return None
    bind = host.replace("http://", "").replace("https://", "")
    env = dict(os.environ, OLLAMA_HOST=bind, OLLAMA_NUM_PARALLEL="1", OLLAMA_KEEP_ALIVE="-1",
               OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "out"), exist_ok=True)
    logf = open(os.path.join(root, "out", "ollama_private.log"), "ab")
    creation = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    p = subprocess.Popen(["ollama", "serve"], env=env, stdout=logf, stderr=logf, creationflags=creation)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if alive():
            log(f"[ollama] private instance on {bind} (NUM_PARALLEL=1) ready in {time.perf_counter()-t0:.1f}s")
            return p
        time.sleep(0.25)
    log(f"[ollama] private instance on {bind} did not come up; see out/ollama_private.log")
    return p
