"""向量检索：把 query 向量化后，交给存储层做最近邻。

底层「怎么算最近邻」对上层透明：DocumentStore 用 numpy 手写余弦，
ChromaStore 用 chromadb 的 HNSW 近似索引。DenseRetriever 只负责
「query → 向量」这一步，再把检索委托出去。
"""

from __future__ import annotations

from .embedder import Embedder
from .store import VectorStore


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """返回 (chunk_id, 相似度) 列表，按相似度降序。"""
        if k <= 0:
            return []
        return self.store.search(self.embedder.embed(query), k)
