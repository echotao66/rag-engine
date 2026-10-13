import pytest

from rag import FixedChunker


def test_short_text_single_chunk():
    c = FixedChunker(size=100, overlap=20)
    assert c.chunk("hello") == ["hello"]


def test_long_text_splits():
    c = FixedChunker(size=10, overlap=0)
    chunks = c.chunk("a" * 22)
    assert len(chunks) == 3
    assert chunks[0] == "a" * 10
    assert chunks[-1] == "a" * 2


def test_overlap_preserves_tail():
    c = FixedChunker(size=10, overlap=5)
    chunks = c.chunk("a" * 15)
    # 相邻 chunk 重叠 5 个字符
    assert chunks[1][:5] == chunks[0][-5:]


def test_empty_text():
    assert FixedChunker().chunk("   ") == []


def test_invalid_args():
    with pytest.raises(ValueError):
        FixedChunker(size=0)
    with pytest.raises(ValueError):
        FixedChunker(size=10, overlap=10)
