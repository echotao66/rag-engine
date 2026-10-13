"""应用层：把 rag 检索核心组装成可用的知识库问答助手。

- loader：文档摄取（txt/md、目录递归、内容 hash 去重）
- session：多轮会话（历史管理 + 查询改写 + 流式问答）
"""

from .loader import Document, DocumentLoader
from .session import ChatSession, Turn

__all__ = ["Document", "DocumentLoader", "ChatSession", "Turn"]
