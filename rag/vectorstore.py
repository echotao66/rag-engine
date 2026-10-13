"""ChromaStore：chromadb 持久化向量存储（可插拔的第二种实现）。

和 DocumentStore（内存 + JSON）实现同一 VectorStore 接口，差别在：
- 最近邻交给 chromadb 的 HNSW 近似索引，数据量大时不退化成 O(n) 全量扫；
- 数据落盘在本地目录，重启自动恢复，无需手动 save/load。

chromadb 在这里只承担「向量索引 + 持久化」，向量仍由上层 Embedder 产出，
并显式关闭 chromadb 自带的 embedding 模型，避免额外下载 ONNX 模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .store import Chunk, VectorStore


class ChromaStore(VectorStore):
    """chromadb 后端。chromadb 惰性导入：不装也不影响其它功能。"""

    def __init__(
        self,
        path: str | Path,
        *,
        collection_name: str = "rag_chunks",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要 chromadb：pip install chromadb（requirements.txt 已包含）"
            ) from exc

        self._chromadb = chromadb
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        # 从已有 id 恢复计数器，避免重启后 add 撞 id
        existing = self._col.get(include=["metadatas"])
        nums = [int(i[1:]) for i in (existing["ids"] or []) if i[1:].isdigit()]
        self._next_id = (max(nums) + 1) if nums else 0

    # ---- 写入 / 读取 ----

    def add(
        self,
        text: str,
        embedding: list[float] | np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
        doc_id: str = "",
    ) -> str:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        cid = f"c{self._next_id}"
        self._next_id += 1
        meta = dict(metadata or {})
        meta["doc_id"] = doc_id  # doc_id 走保留字段，读回时还原成 Chunk.doc_id
        self._col.add(
            ids=[cid],
            embeddings=[vec.tolist()],
            documents=[text],
            metadatas=[meta],
        )
        return cid

    def get(self, chunk_id: str) -> Chunk:
        res = self._col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res["ids"]:
            raise KeyError(f"未知 chunk：{chunk_id}")
        return self._to_chunk(
            res["ids"][0],
            (res["documents"] or [""])[0],
            (res["metadatas"] or [{}])[0],
        )

    def all(self) -> list[Chunk]:
        res = self._col.get(include=["documents", "metadatas"])
        ids = res["ids"] or []
        docs = res["documents"] or [""] * len(ids)
        metas = res["metadatas"] or [{}] * len(ids)
        return [self._to_chunk(cid, text, meta) for cid, text, meta in zip(ids, docs, metas)]

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._col = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self._next_id = 0

    def search(
        self, query_embedding: list[float] | np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        if k <= 0 or self._col.count() == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1).tolist()
        res = self._col.query(
            query_embeddings=q,
            n_results=min(k, self._col.count()),
            include=["distances"],
        )
        ids = res["ids"][0] if res["ids"] else []
        dists = res["distances"][0] if res["distances"] else []
        # cosine 空间下 chromadb 返回「余弦距离」= 1 - 余弦相似度，反解成相似度
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists)]

    def __len__(self) -> int:
        return self._col.count()

    @staticmethod
    def _to_chunk(cid: str, text: str, meta: dict[str, Any] | None) -> Chunk:
        meta = dict(meta or {})
        doc_id = meta.pop("doc_id", "")
        return Chunk(id=cid, text=text, embedding=[], metadata=meta, doc_id=doc_id)
