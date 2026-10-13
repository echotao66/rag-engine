"""手写 BM25 稀疏检索。

为什么需要它：纯向量检索对「专有名词、精确关键词」的召回往往不如传统
关键词检索——向量是语义近似，关键词是精确匹配。BM25 是经典的关键词
打分模型（词频 + 逆文档频率 + 长度归一化），和向量检索正好互补。

纯 Python 手写，无 rank_bm25 依赖，公式：
    score(q, d) = Σ IDF(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))
    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""

from __future__ import annotations

import math
from collections import defaultdict

from .store import DocumentStore
from .tokenize import tokenize


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._ids: list[str] = []
        self._docs: list[list[str]] = []
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def build(self, store: DocumentStore) -> None:
        """从 store 重建索引。ingest 新内容后需重新调用。"""
        self._ids = [c.id for c in store.all()]
        self._docs = [tokenize(c.text) for c in store.all()]
        self._doc_len = [len(d) for d in self._docs]

        df: dict[str, int] = defaultdict(int)
        for doc in self._docs:
            for term in set(doc):
                df[term] += 1

        n = len(self._docs)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        self._idf = {
            term: math.log(1 + (n - c + 0.5) / (c + 0.5))
            for term, c in df.items()
        }

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """返回 (chunk_id, bm25 得分) 列表，按得分降序。"""
        if not self._docs or k <= 0:
            return []

        qtokens = tokenize(query)
        scores: list[tuple[str, float]] = []
        for i, doc in enumerate(self._docs):
            tf: dict[str, int] = defaultdict(int)
            for t in doc:
                tf[t] += 1

            score = 0.0
            for t in qtokens:
                idf = self._idf.get(t)
                f = tf.get(t, 0)
                if idf is None or f == 0:
                    continue
                norm = f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avgdl)
                )
                score += idf * norm
            if score > 0:
                scores.append((self._ids[i], score))

        scores.sort(key=lambda x: -x[1])
        return scores[:k]
