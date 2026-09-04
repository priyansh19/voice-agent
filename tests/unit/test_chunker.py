from agent.chunker import ClauseChunker, clean


def feed_all(ch, text, step=4):
    out = []
    for i in range(0, len(text), step):
        out += ch.feed(text[i:i + step])
    return out + ch.flush()


def test_first_chunk_is_cut_after_max_words_without_punctuation():
    ch = ClauseChunker(first_chunk_min_words=2, min_words=6, max_chars=180, first_chunk_max_words=3)
    chunks = feed_all(ch, "Rome is the capital of Italy and it is famous for its ancient history. ")
    assert chunks[0] == "Rome is the"
    assert " ".join(chunks).replace("  ", " ").startswith("Rome is the capital of Italy")


def test_sentence_boundaries_and_decimal_guard():
    ch = ClauseChunker(first_chunk_min_words=1, min_words=6, max_chars=180)
    chunks = feed_all(ch, "It costs 3.5 euros. Really! Is that ok? Yes. ")
    assert chunks == ["It costs 3.5 euros.", "Really!", "Is that ok?", "Yes."]


def test_clause_split_after_min_words():
    ch = ClauseChunker(first_chunk_min_words=3, min_words=4, max_chars=180)
    chunks = feed_all(ch, "Tomorrow in Milan it will rain, so bring an umbrella, and a coat. ")
    assert chunks[0] == "Tomorrow in Milan it will rain,"
    assert chunks[-1] == "and a coat."


def test_devanagari_danda_and_short_first_chunk():
    ch = ClauseChunker(first_chunk_min_words=2, min_words=6, max_chars=180, first_chunk_max_words=5)
    chunks = feed_all(ch, "कल मिलान में मौसम साफ रहेगा। छाता नहीं चाहिए। ")
    assert chunks[0].split()[:3] == ["कल", "मिलान", "में"]
    assert len(chunks[0].split()) <= 3          # Devanagari first chunk is capped at 3 words
    assert chunks[-1] == "छाता नहीं चाहिए।"


def test_markdown_is_stripped_and_empty_chunks_dropped():
    assert clean("**Bold** _text_ — `code`") == "Bold text , code"
    ch = ClauseChunker()
    assert ch.feed("*** ") == [] and ch.flush() == []


def test_max_chars_forces_a_cut():
    ch = ClauseChunker(first_chunk_min_words=50, min_words=50, max_chars=40, first_chunk_max_words=0)
    chunks = feed_all(ch, "word " * 30)
    assert all(len(c) <= 40 for c in chunks) and len(chunks) >= 3
