from app.loader import Document, DocumentLoader


def test_make_id_is_deterministic_and_distinct():
    assert Document.make_id("hello") == Document.make_id("hello")
    assert Document.make_id("hello") != Document.make_id("world")


def test_load_file_reads_text_and_sets_title(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("你好，世界", encoding="utf-8")
    doc = DocumentLoader().load_file(p)
    assert doc.title == "a.md"
    assert doc.text == "你好，世界"
    assert doc.metadata["source"] == str(p)


def test_load_dir_recurses_and_filters_suffix(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("B", encoding="utf-8")
    (tmp_path / "skip.py").write_text("C", encoding="utf-8")
    docs = DocumentLoader().load_dir(tmp_path)
    assert {d.text for d in docs} == {"A", "B"}


def test_dedupe_skips_seen_ids():
    docs = [
        Document(doc_id="x", title="a", text="a"),
        Document(doc_id="x", title="b", text="b"),
    ]
    seen: set[str] = set()
    fresh = DocumentLoader.dedupe(docs, seen)
    assert len(fresh) == 1
    assert "x" in seen
