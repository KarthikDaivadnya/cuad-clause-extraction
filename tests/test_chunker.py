import pytest

from src.chunker import chunk_text


def test_short_text_produces_one_chunk():
    text = "A short contract clause."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_long_text_produces_multiple_overlapping_chunks():
    text = ("This is a sentence about termination. " * 50).strip()
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    # every chunk should be non-empty and within a reasonable size bound
    for c in chunks:
        assert c.text
        assert len(c.text) <= 250  # allow a little slack for boundary snapping


def test_chunks_cover_the_full_text_with_overlap():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    # the last chunk should reach (approximately) the end of the text
    assert chunks[-1].start_char + len(chunks[-1].text) >= len(text.strip()) - 10


def test_rejects_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=200)


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []
