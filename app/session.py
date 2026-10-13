"""多轮会话：历史管理 + 查询改写 + 带引用的 RAG 问答。

关键点是「查询改写」：RAG 拿「那第二个呢」「它有什么优点」这种残缺追问
直接去检索会召回错。这里先用 LLM 把「历史 + 当前问题」改写成独立完整的
检索 query，再走检索 → 生成。这是多轮 RAG 面试的高频考点。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag import LLMClient, RAGEngine

REWRITE_SYSTEM = (
    "你是检索查询改写器。根据对话历史，把用户的当前问题改写成一个独立、完整、"
    "自包含的检索查询，便于从知识库检索。只输出改写后的查询本身，不要加引号或解释。"
    # 必须锁定语言：实测发现不写这句时，模型会把英文追问改写成中文，
    # 而知识库以英文文档为主，跨语言召回的命中率会直接掉到 0——
    # 改写本来是为了「补全指代」，结果引入了比指代更大的新问题。
    # 注意：加了这句也只是缓解、不是根治（tests/benchmark_rewrite.py 实测
    # 10 组里仍有 4 组被改写成中文）。可靠做法是在代码里校验返回语言、
    # 不一致就退回原 query，这属于已知未修项。
    "改写后的查询必须与「当前问题」使用同一种语言，不要翻译。"
)


@dataclass
class Turn:
    """一轮对话：原始问题 + 回答 + 引用编号。"""

    question: str
    answer: str
    citations: list[int] = field(default_factory=list)


class ChatSession:
    """维护多轮历史，把「改写 → 检索 → 生成」串起来。"""

    def __init__(
        self,
        engine: RAGEngine,
        llm: LLMClient | None = None,
        *,
        history_size: int = 6,
    ) -> None:
        self.engine = engine
        self.llm = llm
        self.history_size = int(history_size)
        self.history: list[Turn] = []

    def _rewrite(self, question: str) -> str:
        """无历史或无 LLM 时直接返回原问题；否则用 LLM 改写成独立检索 query。"""
        if self.llm is None or not self.history:
            return question
        hist = "\n".join(
            f"用户：{t.question}\n助手：{t.answer}" for t in self.history[-self.history_size :]
        )
        user = f"对话历史：\n{hist}\n\n当前问题：{question}\n\n改写后的检索查询："
        try:
            rewritten = self.llm.chat(REWRITE_SYSTEM, user, temperature=0.0).strip()
        except Exception:  # noqa: BLE001 —— 改写失败不阻断，退回原问题
            return question
        return rewritten or question

    def ask(self, question: str, *, k: int = 3) -> dict:
        """非流式问答（CLI / 测试用）。返回 engine.query 结果并附改写后的 query。"""
        search_query = self._rewrite(question)
        result = self.engine.query(search_query, k=k)
        self.history.append(Turn(question, result["answer"], result.get("citations", [])))
        result["search_query"] = search_query
        return result

    def stream_ask(self, question: str, *, k: int = 3):
        """流式问答，逐条 yield (kind, payload)；kind ∈ {retrieved, delta, done}。

        retrieved：检索结果 + 改写后的 query（前端可展示来源 + 可观测）
        delta    ：生成过程中的增量文本（打字机效果）
        done     ：完整回答 + 引用编号
        """
        search_query = self._rewrite(question)
        chunks = self.engine.retrieve_sources(search_query, k=k)
        yield ("retrieved", {"chunks": chunks, "search_query": search_query})

        generator = self.engine.generator
        if generator is None or self.llm is None:
            answer = "（未配置 LLM，仅展示检索结果）"
            self.history.append(Turn(question, answer, []))
            yield ("done", {"answer": answer, "citations": [], "search_query": search_query})
            return

        refs = [(c["chunk_id"], c["text"], c["score"]) for c in chunks]
        tokens: list[str] = []
        for token in generator.stream(search_query, refs):
            tokens.append(token)
            yield ("delta", {"text": token})
        answer = "".join(tokens)
        citations = generator.extract_citations(answer)
        self.history.append(Turn(question, answer, citations))
        yield ("done", {"answer": answer, "citations": citations, "search_query": search_query})
