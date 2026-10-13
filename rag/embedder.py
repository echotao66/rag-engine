"""可插拔的 Embedder 层。

为什么单独抽象这一层：DeepSeek 官方 API 目前**不提供 embedding 接口**，
所以「生成」走 DeepSeek，而「向量化」在这里解耦出去。

默认 HashEmbedder（零依赖、可复现），想要真语义时可换成
SentenceTransformerEmbedder 或任意 embedding API。
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod

import numpy as np

from .tokenize import tokenize


class Embedder(ABC):
    """把文本映射成固定维度的向量。同一文本必须产出相同向量。"""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """返回 L2 归一化后的向量（利于用余弦相似度做检索）。"""

    @abstractmethod
    def dim(self) -> int:
        """返回向量维度。"""


class HashEmbedder(Embedder):
    """零依赖的确定性 embedding：hashing trick + 词频统计。

    不是真正语义化的（"苹果手机"和"苹果水果"可能得到相近向量），
    但足够演示「混合检索」的完整流程，且不联网、跨进程可复现。
    面试时可以说：这是可插拔的一层，生产环境替换为语义模型即可。
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须 > 0")
        self._dim = int(dim)

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            # 用 crc32 而非内置 hash()：hash() 每进程随机化，
            # 会导致落盘后跨进程加载向量不一致，crc32 稳定。
            idx = zlib.crc32(token.encode("utf-8")) % self._dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def dim(self) -> int:
        return self._dim


class SentenceTransformerEmbedder(Embedder):
    """本地语义 embedding（可选，需额外依赖）。

    安装：pip install sentence-transformers
    国内下载模型建议先设置：export HF_ENDPOINT=https://hf-mirror.com
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要 sentence-transformers：pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        # 新旧版本方法名不同，用 getattr 兼容两者
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self._dim = int(get_dim())

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def dim(self) -> int:
        return self._dim
