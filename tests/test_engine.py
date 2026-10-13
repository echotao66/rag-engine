from rag import RAGEngine


class ScriptedLLM:
    def chat(self, system_prompt, user_prompt, *, temperature=0.0):
        return "答案见 [1]。"


def test_ingest_returns_chunk_count():
    engine = RAGEngine()
    assert engine.ingest("DeepSeek 提供 chat 模型") == 1
    assert len(engine.store) == 1


def test_query_with_llm_returns_answer_and_citations():
    engine = RAGEngine(llm=ScriptedLLM())
    engine.ingest("DeepSeek 提供支持 function calling 的 chat 模型")
    result = engine.query("DeepSeek 的能力", k=1)
    assert result["answer"] == "答案见 [1]。"
    assert result["citations"] == [1]
    assert result["chunks"]


def test_query_without_llm_returns_chunks_only():
    engine = RAGEngine()
    engine.ingest("hello world")
    result = engine.query("hello", k=1)
    assert result["answer"] is None
    assert result["chunks"]


def test_retrieve_returns_ordered_chunks():
    engine = RAGEngine()
    engine.ingest("The capital of France is Paris.")
    engine.ingest("Python is a programming language.")
    results = engine.retrieve("capital of France", k=2)
    assert len(results) == 2
    assert "Paris" in results[0][1]
