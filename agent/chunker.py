"""Streaming text -> speakable chunks.  Emits the first chunk as early as possible (clause boundary after a few
words) so TTS can start while the LLM is still generating; later chunks are full clauses/sentences."""
import re

_MD = re.compile(r"[*_`#>]+")
_WS = re.compile(r"\s+")


def clean(s: str) -> str:
    s = _MD.sub("", s)
    s = s.replace("—", ", ").replace("–", ", ")
    return _WS.sub(" ", s).strip()


class ClauseChunker:
    SENT_END = ".!?।"      # includes the Devanagari danda
    CLAUSE = ",;:"

    def __init__(self, first_chunk_min_words=2, min_words=6, max_chars=180, first_chunk_max_words=0):
        self.first_min, self.min_words, self.max_chars = first_chunk_min_words, min_words, max_chars
        self.first_max = first_chunk_max_words     # >0: cut the first chunk at a word boundary after this many words
        self.buf = ""
        self.emitted = 0

    def _words(self, s):
        return len(s.split())

    def feed(self, delta: str):
        self.buf += delta
        out = []
        while True:
            chunk = self._next_chunk()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def _next_chunk(self):
        b = self.buf
        min_w = self.first_min if self.emitted == 0 else self.min_words
        # newline is always a boundary
        nl = b.find("\n")
        if nl != -1 and clean(b[:nl]):
            return self._emit(nl + 1)
        for i, ch in enumerate(b[:-1]):            # need one following char to confirm the boundary
            nxt = b[i + 1]
            if ch in self.SENT_END and nxt in " \"')]":
                if i > 0 and b[i - 1].isdigit() and nxt.isdigit():
                    continue
                if self._words(b[: i + 1]) >= 1 and clean(b[: i + 1]):
                    return self._emit(i + 1)
            elif ch in self.CLAUSE and nxt == " " and self._words(b[: i + 1]) >= min_w:
                return self._emit(i + 1)
        first_max = self.first_max
        if first_max and re.search(r"[ऀ-ॿ]", b):
            first_max = min(first_max, 3)          # Devanagari words are many tokens each: cut even earlier
        if self.emitted == 0 and first_max and self._words(b) > first_max:
            # first chunk: don't wait for punctuation, cut after first_max words (TTS starts ~300 ms sooner)
            idx, count = 0, 0
            for mm in re.finditer(r"\S+\s", b):
                count += 1; idx = mm.end()
                if count >= first_max: break
            if idx > 0:
                return self._emit(idx)
        if len(b) > self.max_chars:
            cut = b.rfind(" ", 0, self.max_chars)
            if cut > 0:
                return self._emit(cut)
        return None

    def _emit(self, n):
        chunk, self.buf = self.buf[:n], self.buf[n:]
        chunk = clean(chunk)
        if chunk and re.search(r"[^\W_]", chunk):
            self.emitted += 1
            return chunk
        return ""

    def flush(self):
        chunk = clean(self.buf)
        self.buf = ""
        if chunk and re.search(r"[^\W_]", chunk):
            self.emitted += 1
            return [chunk]
        return []
