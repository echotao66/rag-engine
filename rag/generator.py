"""生成器：把检索到的片段注入 prompt，强制模型带引用回答。

「引用可控」是本模块的精华：检索的片段编号 [1][2]... 注入 prompt，
要求模型回答时标注来源，再解析出引用了哪些编号回显给用户，
解决「LLM 回答无依据」的信任问题。

两种产出方式：
- generate：一次性拿到全文（CLI / 测试用）
- stream：逐 token 流式产出（后端 SSE 推送、前端打字机效果）
"""

from __future__ import annotations

import re

from .llm import LLMClient

SYSTEM_PROMPT = (
    "你是严谨的问答助手。只根据下面提供的资料回答用户问题，"
    "并在答案中用 [n] 标注引用来源（n 为资料编号）。"
    "资料中没有的信息，如实说明「资料中未提及」，不要编造。"
)


class Generator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def build_prompt(query: str, chunks: list[tuple[str, str, float]]) -> tuple[str, str]:
        """把 chunks 编号注入 prompt，返回 (system_prompt, user_prompt)。"""
        refs = "\n".join(f"[{i + 1}] {text}" for i, (_, text, _) in enumerate(chunks))
        user_prompt = f"资料：\n{refs}\n\n用户问题：{query}\n\n请回答并标注引用："
        return SYSTEM_PROMPT, user_prompt

    def generate(self, query: str, chunks: list[tuple[str, str, float]]) -> dict:
        """chunks: [(chunk_id, text, score)]。返回 {answer, chunks, citations}。"""
        if not chunks:
            return {"answer": "未检索到相关资料。", "chunks": [], "citations": []}

        system_prompt, user_prompt = self.build_prompt(query, chunks)
        answer = self.llm.chat(system_prompt, user_prompt)
        return {
            "answer": answer,
            "chunks": chunks,
            "citations": self.extract_citations(answer),
        }

    def stream(self, query: str, chunks: list[tuple[str, str, float]]):
        """流式生成，逐段 yield 增量文本（供 SSE 推送）。"""
        if not chunks:
            return
        system_prompt, user_prompt = self.build_prompt(query, chunks)
        yield from self.llm.stream(system_prompt, user_prompt)

    @staticmethod
    def extract_citations(text: str) -> list[int]:
        """从回答里提取 [n] 引用编号，去重排序。"""
        return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})

    # 兼容旧调用：早期测试用 _extract_citations
    @staticmethod
    def _extract_citations(text: str) -> list[int]:
        return Generator.extract_citations(text)
