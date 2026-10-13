"""RAGEngine：统一入口，把分块 / 向量化 / 混合检索 / 生成串起来。

对外只暴露两个动词：ingest（建库）和 query（问答）。
CLI、demo、测试、benchmark 都通过它操作，不直接摸底层模块。
"""

from __future__ import annotations

from typing import Any

from .chunking import Chunker, FixedChunker
from .dense import DenseRetriever
from .embedder import Embedder, HashEmbedder
from .generator import Generator
from .hybrid import HybridRetriever
from .llm import LLMClient
from .rerank import IdentityReranker, Reranker
from .sparse import BM25Retriever
from .store import DocumentStore, VectorStore


class RAGEngine:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        chunker: Chunker | None = None,
        reranker: Reranker | None = None,
        store: VectorStore | None = None,
    ) -> None:
        # 允许传入自定义实现，未传则用默认值（可插拔）。
        self.llm = llm
        self.embedder = embedder or HashEmbedder()
        self.chunker = chunker or FixedChunker()
        # 注意：不能写 `store or DocumentStore()`——VectorStore 定义了 __len__，
        # 空的 ChromaStore 是 falsy，会错误回退到 DocumentStore。
        self.store = store if store is not None else DocumentStore()
        self.dense = DenseRetriever(self.store, self.embedder)
        self.sparse = BM25Retriever()
        self.reranker = reranker or IdentityReranker()
        self.hybrid = HybridRetriever(self.store, self.dense, self.sparse, self.reranker)
        self.generator = Generator(llm) if llm is not None else None

    # ---- 建库 ----

    def ingest(
        self,
        text: str,
        *,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """切分 → 向量化 → 写入 store，返回写入的 chunk 数。"""
        n = 0
        for chunk in self.chunker.chunk(text):
            self.store.add(chunk, self.embedder.embed(chunk), metadata=metadata, doc_id=doc_id)
            n += 1
        self.sparse.build(self.store)
        return n

    # ---- 检索 ----

    def retrieve(self, query: str, *, k: int = 3) -> list[tuple[str, str, float]]:
        """混合检索，返回 [(chunk_id, text, score)]。"""
        return self.hybrid.search(query, k=k)

    def retrieve_sources(self, query: str, *, k: int = 3) -> list[dict]:
        """检索并带回来源信息（doc_id / metadata），供前端展示引用出处。"""
        out: list[dict] = []
        for cid, text, score in self.hybrid.search(query, k=k):
            chunk = self.store.get(cid)
            out.append(
                {
                    "chunk_id": cid,
                    "text": text,
                    "score": score,
                    "doc_id": chunk.doc_id,
                    "metadata": chunk.metadata,
                }
            )
        return out

    def clear(self) -> None:
        """清空知识库（供前端「重置」）。"""
        self.store.clear()
        self.sparse.build(self.store)

    # ---- 问答 ----

    def query(self, query: str, *, k: int = 3) -> dict:
        """检索 + 生成。未配置 LLM 时只返回检索结果。"""
        chunks = self.retrieve(query, k=k)
        if self.generator is None:
            return {"chunks": chunks, "answer": None, "citations": []}
        return self.generator.generate(query, chunks)
