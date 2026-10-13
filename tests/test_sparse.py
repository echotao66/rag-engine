from rag import BM25Retriever, DocumentStore


def _build(docs):
    store = DocumentStore()
    for d in docs:
        store.add(d, [0.0] * 8)  # embedding 不参与 BM25，仅占位
    bm = BM25Retriever()
    bm.build(store)
    return bm


def test_keyword_query_ranks_most_relevant_first():
    bm = _build([
        "python is a programming language",
        "the sky is blue today",
        "python is used for machine learning",
    ])
    results = bm.search("python machine learning", k=3)
    # 第 3 条同时命中 python + machine + learning，应排第一
    assert results[0][0] == "c2"


def test_unknown_query_returns_empty():
    bm = _build(["hello world"])
    assert bm.search("zzzzqqqq") == []


def test_search_respects_k():
    bm = _build([f"doc {i} shares keyword" for i in range(5)])
    results = bm.search("keyword", k=2)
    assert len(results) == 2
