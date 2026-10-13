"""DocumentStore：chunk 与向量的存储，JSON 落盘 + numpy 矩阵。

不引入向量库依赖：内存里存 chunk 列表，检索时用 numpy 拼矩阵算余弦，
落盘用 JSON（向量转 list），足够演示「建库 → 检索」完整闭环。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Chunk:
    """一条被切分并向量化的文本片段。"""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


class VectorStore(ABC):
    """向量存储协议：把「chunk + 向量怎么存、怎么算最近邻」抽象出来。

    DocumentStore（本文件）是零依赖实现：内存 list + numpy 暴力余弦 + JSON 落盘；
    ChromaStore（vectorstore.py）是 chromadb 实现：HNSW 近似最近邻 + 本地持久化。
    两者同一接口，RAGEngine 通过 store= 参数切换，其余代码不用改。
    """

    @abstractmethod
    def add(
        self,
        text: str,
        embedding: list[float] | np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
        doc_id: str = "",
    ) -> str:
        """写入一个 chunk，返回其 id。"""

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk:
        """按 id 取 chunk（含 text / metadata / doc_id）。"""

    @abstractmethod
    def all(self) -> list[Chunk]:
        """返回全部 chunk。"""

    @abstractmethod
    def clear(self) -> None:
        """清空全部 chunk。"""

    @abstractmethod
    def search(
        self, query_embedding: list[float] | np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """按向量检索，返回 (chunk_id, 相似度) 列表，降序。"""

    @abstractmethod
    def __len__(self) -> int:
        """chunk 总数。"""

    def save(self, path: str | Path) -> None:
        """落盘。默认无操作：Chroma 自动持久化；DocumentStore 覆写为 JSON。"""

    def load(self, path: str | Path) -> int:
        """从磁盘恢复，返回加载条数。默认 0；DocumentStore 覆写。"""
        return 0


class DocumentStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def add(
        self,
        text: str,
        embedding: list[float] | np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
        doc_id: str = "",
    ) -> str:
        """写入一个 chunk，返回其 id。"""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        cid = f"c{self._next_id}"
        self._next_id += 1
        self._chunks.append(
            Chunk(
                id=cid,
                text=text,
                embedding=vec.tolist(),
                metadata=dict(metadata or {}),
                doc_id=doc_id,
            )
        )
        return cid

    def get(self, chunk_id: str) -> Chunk:
        for c in self._chunks:
            if c.id == chunk_id:
                return c
        raise KeyError(f"未知 chunk：{chunk_id}")

    def all(self) -> list[Chunk]:
        return list(self._chunks)

    def clear(self) -> None:
        """清空所有 chunk，重置 id 计数器。"""
        self._chunks = []
        self._next_id = 0

    def embedding_matrix(self) -> np.ndarray:
        """所有 chunk 的向量堆成一个 (n, dim) 矩阵，供向量检索。"""
        if not self._chunks:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([np.asarray(c.embedding, dtype=np.float32) for c in self._chunks])

    def search(
        self, query_embedding: list[float] | np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """按向量检索，返回 (chunk_id, 余弦相似度) 列表，降序。"""
        if not self._chunks or k <= 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        matrix = self.embedding_matrix()

        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        m_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        denom = np.maximum(q_norm * m_norm, 1e-12)
        cosine = ((matrix @ q.T).ravel()) / denom.ravel()

        top = min(k, len(self._chunks))
        indices = np.argsort(cosine)[::-1][:top]
        return [(self._chunks[i].id, float(cosine[i])) for i in indices]

    # ---- 持久化 ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self._chunks], f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = [Chunk(**item) for item in data]
        # 恢复 id 计数器，避免后续 add 撞 id
        nums = [int(c.id[1:]) for c in self._chunks if c.id[1:].isdigit()]
        self._next_id = (max(nums) + 1) if nums else 0
        return len(self._chunks)
