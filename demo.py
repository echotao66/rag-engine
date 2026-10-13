"""rag-engine 离线演示：不联网、不花钱，也能看「混合检索 + 引用生成」完整链路。

用脚本化模型按预设返回带引用的回答，走一遍：
1. ingest 录入三段资料
2. 混合检索（dense + BM25 + RRF）召回相关片段
3. 生成器把片段注入 prompt，模型带 [n] 引用回答
4. 解析并回显引用编号

运行：python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rag import RAGEngine  # noqa: E402


class ScriptedLLM:
    """脚本化模型：返回固定带引用的回答，模拟真实生成。"""

    def __init__(self) -> None:
        self.last_prompt = ""

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.last_prompt = user_prompt
        return (
            "根据资料 [1]，DeepSeek 提供支持 function calling 的 chat 模型，"
            "可用于构建 Agent；资料 [2] 提到 RAG 通过检索增强降低幻觉。"
        )


def main() -> None:
    engine = RAGEngine(llm=ScriptedLLM())

    engine.ingest("DeepSeek 是一家中国 AI 公司，提供支持 function calling 的 chat 模型。", doc_id="deepseek")
    engine.ingest("RAG 通过检索增强生成，可以显著降低大模型的幻觉。", doc_id="rag")
    engine.ingest("今天天气晴朗，适合出门散步。", doc_id="weather")

    query = "DeepSeek 的模型有什么能力？"
    print("=== rag-engine 混合检索 + 引用生成（离线）===\n")
    print(f"提问：{query}\n")

    result = engine.query(query, k=3)
    print("检索到的片段：")
    for cid, text, score in result["chunks"]:
        print(f"  [{cid}] ({score:.3f}) {text}")

    print(f"\n回答：{result['answer']}")
    print(f"引用编号：{result['citations']}")
    print("\n=== 演示结束 ===")


if __name__ == "__main__":
    main()
