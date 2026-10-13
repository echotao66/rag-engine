"""ChromaStore 单测：验证 chromadb 后端与 DocumentStore 行为一致。

未安装 chromadb 时整个模块自动跳过（pytest.importorskip）。
"""

import pytest

pytest.importorskip("chromadb")

from rag import ChromaStore, RAGEngine  # noqa: E402


def test_add_and_search(tmp_path):
    s = ChromaStore(path=tmp_path / "chroma")
    s.add("DeepSeek 提供 chat 模型", [1.0, 0.0, 0.0])
    s.add("今天天气很好", [0.0, 1.0, 0.0])
    assert len(s) == 2
    hits = s.search([1.0, 0.0, 0.0], k=1)
    assert hits
    assert "DeepSeek" in s.get(hits[0][0]).text


def test_get_all_and_metadata(tmp_path):
    s = ChromaStore(path=tmp_path / "chroma")
    cid = s.add("hello", [1.0, 0.0], metadata={"title": "t"}, doc_id="doc1")
    chunk = s.get(cid)
    assert chunk.text == "hello"
    assert chunk.doc_id == "doc1"
    assert chunk.metadata["title"] == "t"
    assert len(s.all()) == 1


def test_clear(tmp_path):
    s = ChromaStore(path=tmp_path / "chroma")
    s.add("a", [1.0])
    s.clear()
    assert len(s) == 0
    assert s.search([1.0]) == []


def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "chroma"
    s = ChromaStore(path=path)
    s.add("persisted", [0.5, 0.5])
    del s  # 释放旧连接，模拟重启
    s2 = ChromaStore(path=path)
    assert len(s2) == 1
    assert s2.all()[0].text == "persisted"


def test_ragengine_with_chroma(tmp_path):
    store = ChromaStore(path=tmp_path / "chroma")
    engine = RAGEngine(store=store)
    # 显式断言用的是 chroma 后端，防止空 store 被 `or` 回退成 DocumentStore
    assert engine.store is store
    engine.ingest("The capital of France is Paris.", doc_id="france")
    results = engine.retrieve("capital of France", k=1)
    assert results
    assert "Paris" in results[0][1]
