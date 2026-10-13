"""rag-engine：轻量混合检索 RAG 引擎。

对外最常用：
- RAGEngine         ：统一入口（ingest / query）
- HybridRetriever   ：dense + sparse 双路 RRF 融合（核心亮点）
- Generator         ：引用可控生成
- Embedder / Chunker / Reranker：三处可插拔抽象
"""

from __future__ import annotations

from .chunking import Chunker, FixedChunker
from .dense import DenseRetriever
from .embedder import Embedder, HashEmbedder, SentenceTransformerEmbedder
from .engine import RAGEngine
from .generator import Generator
from .hybrid import HybridRetriever, rrf_fuse
from .llm import LLMClient, LLMConfig
from .rerank import CrossEncoderReranker, IdentityReranker, Reranker
from .sparse import BM25Retriever
from .store import Chunk, DocumentStore, VectorStore
from .vectorstore import ChromaStore

__all__ = [
    "RAGEngine",
    "HybridRetriever",
    "rrf_fuse",
    "Generator",
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "Chunker",
    "FixedChunker",
    "Reranker",
    "IdentityReranker",
    "CrossEncoderReranker",
    "DenseRetriever",
    "BM25Retriever",
    "VectorStore",
    "DocumentStore",
    "ChromaStore",
    "Chunk",
    "LLMClient",
    "LLMConfig",
]
