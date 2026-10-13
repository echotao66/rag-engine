"""分块策略（可插拔）。

chunk 是 RAG 的最小检索单元：太大会稀释语义、太小会丢失上下文。
这里用固定长度 + 重叠作为默认实现，重叠是为了避免把一句话从中间切断。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Chunker(ABC):
    """把一篇长文本切成若干 chunk。"""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """返回切分后的 chunk 列表。"""


class FixedChunker(Chunker):
    """固定长度（按字符）+ 重叠切分。"""

    def __init__(self, size: int = 512, overlap: int = 64) -> None:
        if size <= 0:
            raise ValueError("size 必须 > 0")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap 必须满足 0 <= overlap < size")
        self.size = int(size)
        self.overlap = int(overlap)

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.size:
            return [text]

        chunks: list[str] = []
        step = self.size - self.overlap
        start = 0
        while start < len(text):
            chunks.append(text[start : start + self.size])
            start += step
        return chunks
