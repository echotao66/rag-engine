from rag import RAGEngine, rrf_fuse


def test_rrf_combines_ranks():
    d = [("a", 0.9), ("b", 0.8)]
    s = [("b", 0.7), ("c", 0.6)]
    fused = rrf_fuse(d, s, k=60)
    # b 在两条链路都出现，融合后得分最高
    assert fused[0][0] == "b"
    assert {cid for cid, _ in fused} == {"a", "b", "c"}


def test_hybrid_retriever_returns_text_and_score():
    engine = RAGEngine()
    engine.ingest("DeepSeek 提供支持 function calling 的 chat 模型")
    engine.ingest("今天天气很好")
    results = engine.retrieve("DeepSeek 的模型", k=2)
    assert results
    assert all(len(item) == 3 for item in results)  # (id, text, score)
    assert "DeepSeek" in results[0][1]
