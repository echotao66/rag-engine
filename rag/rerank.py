"""重排器：检索出候选后再做一次精排。

默认 IdentityReranker（透传、不重排），把重排抽象成可插拔的一层——
想要更强效果时换成 cross-encoder 即可，主流程不用改。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Reranker(ABC):
    """对候选片段精排。items 形如 [(chunk_id, text, score), ...]。"""

    @abstractmethod
    def rerank(self, query: str, items: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        """返回重排后的 items（保持三元组结构）。"""


class IdentityReranker(Reranker):
    """不重排，直接返回原顺序。"""

    def rerank(self, query: str, items: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        return items


class CrossEncoderReranker(Reranker):
    """可选：用 sentence-transformers 的 cross-encoder 精排。

    安装：pip install sentence-transformers
    cross-encoder 把 (query, chunk) 成对打分，比双塔向量更准，但更慢，
    所以只对检索出的少量候选做，不做全库。
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要 sentence-transformers：pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, items: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        if not items:
            return items
        pairs = [(query, text) for _, text, _ in items]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(items, scores), key=lambda x: -x[1])
        return [(cid, text, float(s)) for (cid, text, _), s in ranked]
