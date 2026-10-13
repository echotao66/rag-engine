from rag import Generator


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, system_prompt, user_prompt, *, temperature=0.0):
        self.calls.append((system_prompt, user_prompt))
        return "根据资料 [1] 和 [3]，答案是 X。"


def test_extract_citations_dedup_and_sort():
    g = Generator(FakeLLM())
    assert g._extract_citations("见 [1] 和 [3]，以及重复的 [1]") == [1, 3]


def test_generate_injects_refs_and_parses_citations():
    llm = FakeLLM()
    g = Generator(llm)
    chunks = [("c0", "资料内容A", 0.9), ("c1", "资料内容B", 0.8)]
    result = g.generate("问题？", chunks)

    assert result["answer"] == "根据资料 [1] 和 [3]，答案是 X。"
    assert result["citations"] == [1, 3]
    # prompt 里注入了编号资料
    assert "[1] 资料内容A" in llm.calls[0][1]
    assert "[2] 资料内容B" in llm.calls[0][1]


def test_generate_empty_chunks():
    g = Generator(FakeLLM())
    result = g.generate("问题？", [])
    assert result["answer"] == "未检索到相关资料。"
    assert result["citations"] == []
