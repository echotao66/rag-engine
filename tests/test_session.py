from app.session import ChatSession, Turn
from rag import RAGEngine


class ScriptedLLM:
    """同时实现 chat（改写/整段生成）和 stream（流式生成）。"""

    def chat(self, system_prompt, user_prompt, *, temperature=0.0):
        return "DeepSeek 的 chat 模型能力"

    def stream(self, system_prompt, user_prompt, *, temperature=0.0):
        yield "根据资料 [1]，"
        yield "DeepSeek 支持 function calling。"


def _build():
    llm = ScriptedLLM()
    engine = RAGEngine(llm=llm)
    engine.ingest("DeepSeek 提供支持 function calling 的 chat 模型", doc_id="deepseek")
    return engine, llm


def test_rewrite_without_history_returns_question():
    engine = RAGEngine()
    s = ChatSession(engine, None)
    assert s._rewrite("什么是 RAG") == "什么是 RAG"


def test_rewrite_with_history_uses_llm():
    engine, llm = _build()
    s = ChatSession(engine, llm)
    s.history = [Turn("DeepSeek 有什么能力", "支持 function calling")]
    assert s._rewrite("那第二个呢") == "DeepSeek 的 chat 模型能力"


def test_ask_records_history():
    engine, llm = _build()
    s = ChatSession(engine, llm)
    result = s.ask("DeepSeek 的能力", k=1)
    assert result["answer"] == "DeepSeek 的 chat 模型能力"
    assert result["search_query"] == "DeepSeek 的能力"  # 无历史，不改写
    assert len(s.history) == 1
    assert s.history[0].question == "DeepSeek 的能力"


def test_stream_ask_yields_retrieved_delta_done():
    engine, llm = _build()
    s = ChatSession(engine, llm)
    events = list(s.stream_ask("DeepSeek 的能力", k=1))
    kinds = [k for k, _ in events]
    assert kinds[0] == "retrieved"
    assert "delta" in kinds
    assert kinds[-1] == "done"
    done = events[-1][1]
    assert done["citations"] == [1]
    assert "function calling" in done["answer"]


def test_stream_ask_without_llm_still_retrieves():
    engine = RAGEngine()
    engine.ingest("hello world", doc_id="doc")
    s = ChatSession(engine, None)
    events = list(s.stream_ask("hello", k=1))
    assert events[0][0] == "retrieved"
    assert events[-1][0] == "done"
    assert "未配置 LLM" in events[-1][1]["answer"]
