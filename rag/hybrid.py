"""混合检索：dense（向量）+ sparse（BM25）双路，RRF 融合。

核心亮点。为什么用 RRF（Reciprocal Rank Fusion，倒数排名融合）：
两路得分不在同一量纲（余弦在 [-1,1]，BM25 是正实数），直接加权相加
没有意义；RRF 只关心「排名」，对两路排名做倒数求和，量纲无关、稳定好调。

    RRFscore(d) = Σ 1 / (k + rank_i(d))     （k 通常取 60）
"""

from __future__ import annotations

from .dense import DenseRetriever
from .rerank import IdentityReranker, Reranker
from .sparse import BM25Retriever
from .store import DocumentStore


def rrf_fuse(
    dense_ranks: list[tuple[str, float]],
    sparse_ranks: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """融合两路排名，返回 (chunk_id, rrf 得分) 按得分降序。"""
    scores: dict[str, float] = {}
    for rank, (cid, _) in enumerate(dense_ranks):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(sparse_ranks):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(
        self,
        store: DocumentStore,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        reranker: Reranker | None = None,
        *,
        rrf_k: int = 60,
        dense_k: int = 10,
        sparse_k: int = 10,
    ) -> None:
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker or IdentityReranker()
        self.rrf_k = int(rrf_k)
        self.dense_k = int(dense_k)
        self.sparse_k = int(sparse_k)

    def search(self, query: str, k: int = 5) -> list[tuple[str, str, float]]:
        """返回 (chunk_id, text, score) 列表，融合 + 重排后的 top-k。"""
        d = self.dense.search(query, k=self.dense_k)
        s = self.sparse.search(query, k=self.sparse_k)
        fused = rrf_fuse(d, s, self.rrf_k)

        items = [(cid, self.store.get(cid).text, score) for cid, score in fused]
        items = self.reranker.rerank(query, items)
        return items[:k]
