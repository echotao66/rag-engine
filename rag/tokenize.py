"""共享分词：BM25（稀疏）与 HashEmbedder（向量）用同一套，保证两路词表一致。

零依赖。英文按词，中文按单字 + bigram，兼顾两种语言；
这是刻意留的替换点，想要更准的中文分词可换 jieba 等。
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """英文按词、中文按单字 + bigram 切分。"""
    tokens: list[str] = []
    tokens += re.findall(r"[a-z0-9]+", text.lower())
    cjk = _CJK.findall(text)
    tokens += cjk
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i] + cjk[i + 1])
    return tokens
