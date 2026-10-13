"""文档摄取：从文件 / 目录加载文本，按内容 hash 去重，产出带元数据的 Document。

设计动机：RAG 的知识库不是一次性建好的，用户会不断往里加文档。
所以这里把「读取」和「去重」分开：
- load_file / load_dir：读文件、提取标题和来源路径
- dedupe：用内容 SHA-256 做 doc_id，重复导入直接跳过，实现增量摄取
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class Document:
    """一篇待入库的文档。doc_id 用内容 hash，天然可去重、可复现。"""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(text: str) -> str:
        """内容 SHA-256 前 16 位作为稳定 id：同内容同 id，不同内容必不同。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class DocumentLoader:
    """从文件或目录加载 txt/md 文档。"""

    def __init__(self, suffixes: set[str] | frozenset[str] = SUPPORTED_SUFFIXES) -> None:
        self.suffixes = suffixes

    def load_file(self, path: str | Path) -> Document:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        return Document(
            doc_id=Document.make_id(text),
            title=p.name,
            text=text,
            metadata={"source": str(p), "suffix": p.suffix},
        )

    def load_dir(self, path: str | Path) -> list[Document]:
        """递归加载目录下所有受支持后缀的文件，按路径排序保证可复现。"""
        p = Path(path)
        docs: list[Document] = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in self.suffixes:
                docs.append(self.load_file(f))
        return docs

    @staticmethod
    def dedupe(docs: list[Document], seen: set[str]) -> list[Document]:
        """过滤已见过的 doc_id，返回新文档，并把新 id 记入 seen。"""
        fresh: list[Document] = []
        for d in docs:
            if d.doc_id not in seen:
                fresh.append(d)
                seen.add(d.doc_id)
        return fresh
