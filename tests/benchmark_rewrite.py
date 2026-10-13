"""多轮追问改写评测：对比「直接检索残缺追问」与「LLM 改写后再检索」的 recall@3。

运行：python tests/benchmark_rewrite.py     （会真实调用 DeepSeek，约 10 次请求）

为什么单独做这个实验：`app/session.py` 的立场是「『那第二个呢』这种残缺追问
缺上下文，直接拿去检索会召回错」，但这句话之前只是设计主张、没有量化。
这里构造 10 组「历史 + 残缺追问」，把两种做法的 recall@3 都测出来。

语料复用 tests/benchmark.py 的 50 篇文档，标注沿用同一套公式。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 模型缓存目录如需指定，请先设置 HF_HOME 环境变量（不设则用 HuggingFace 默认缓存）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.session import ChatSession, Turn  # noqa: E402
from rag import LLMClient, SentenceTransformerEmbedder, RAGEngine  # noqa: E402
from tests.benchmark import DOCS, MODEL  # noqa: E402


def load_env() -> None:
    """从 .env 读取 DEEPSEEK_API_KEY（与 cli.py 同一套极简实现）。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# 每条：(对话历史, 残缺追问, 相关文档下标)
# 历史刻意写成「刚聊完这个话题」，追问则是不带主语的指代/省略句。
#
# 语言必须与目标文档一致（这里都是英文），否则实验会被「跨语言检索」混杂：
# 中文追问配英文文档时，改写前后都会因为语种不匹配而一起召回失败，
# 测到的就不是「改写有没有用」，而是「跨语言检索行不行」。
CASES: list[tuple[list[tuple[str, str]], str, list[int]]] = [
    (
        [("What are the downsides of BM25 and of dense retrieval?",
          "BM25 misses paraphrases, while dense retrieval misses exact keywords.")],
        "What if we combine the two?",
        [7],
    ),
    (
        [("Why does RAG reduce hallucination?",
          "Because it retrieves relevant passages first and answers from them.")],
        "And how do we measure how good the retrieval is?",
        [8, 9, 10, 11],
    ),
    (
        [("How should documents be chunked?",
          "Split by fixed length and let adjacent chunks overlap a little.")],
        "Why the overlap?",
        [5],
    ),
    (
        [("What is RRF?",
          "Reciprocal rank fusion: it sums the reciprocal ranks of two retrieval lists.")],
        "Why can't we just add the scores instead?",
        [2],
    ),
    (
        [("How does an agent call outside capabilities?",
          "Through function calling, describing the tool with JSON arguments.")],
        "What if the arguments are wrong?",
        [16],
    ),
    (
        [("Can tool results be cached?",
          "Idempotent tools can be cached, tools with side effects cannot.")],
        "How do we decide which ones?",
        [17],
    ),
    (
        [("Does memory grow without bound?",
          "There is a capacity limit, the oldest entries get evicted.")],
        "And how do old memories fade out gradually?",
        [27],
    ),
    (
        [("Is it safe to expose an agent publicly?",
          "You must restrict which files and commands it can touch.")],
        "How do people usually get around that?",
        [19],
    ),
    (
        [("Which document formats does the knowledge base accept?",
          "txt and markdown, and it can walk a directory recursively.")],
        "What happens if we import the same file twice?",
        [32],
    ),
    (
        [("Where do the citation numbers in the answer come from?",
          "Retrieved passages are numbered and injected into the prompt.")],
        "And how is the streaming output done?",
        [24],
    ),
]


def recall_at(engine: RAGEngine, query: str, rel: list[int], k: int = 3) -> float:
    ids = {cid for cid, *_ in engine.retrieve(query, k=k)}
    rel_ids = {f"c{i}" for i in rel}
    return len(ids & rel_ids) / len(rel_ids)


def main() -> None:
    load_env()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY，无法评测改写（先配 .env）")
        return

    engine = RAGEngine(embedder=SentenceTransformerEmbedder(MODEL))
    for i, doc in enumerate(DOCS):
        engine.ingest(doc, doc_id=f"doc{i}")

    session = ChatSession(engine, llm=LLMClient(), history_size=6)

    raw_hits = new_hits = 0.0
    helped = hurt = same = 0

    print(f"语料 {len(DOCS)} 篇 / {len(CASES)} 组残缺追问，指标 recall@3\n")
    print(f"{'原始追问':<22} | {'改写后 query':<44} | {'raw':>5} {'new':>5}")
    print("-" * 88)

    for history, followup, rel in CASES:
        session.history = [Turn(q, a) for q, a in history]
        rewritten = session._rewrite(followup)

        raw_recall = recall_at(engine, followup, rel, k=3)
        new_recall = recall_at(engine, rewritten, rel, k=3)
        raw_hits += raw_recall
        new_hits += new_recall

        if new_recall > raw_recall:
            helped += 1
        elif new_recall < raw_recall:
            hurt += 1
        else:
            same += 1

        shown = rewritten if len(rewritten) <= 42 else rewritten[:41] + "…"
        print(f"{followup:<22} | {shown:<44} | {raw_recall:>5.0%} {new_recall:>5.0%}")

    n = len(CASES)
    print("-" * 88)
    print(f"\n直接检索残缺追问  recall@3 = {raw_hits / n:.1%}")
    print(f"LLM 改写后再检索  recall@3 = {new_hits / n:.1%}")
    print(f"改写有帮助 {helped} 组 / 无变化 {same} 组 / 变差 {hurt} 组")


if __name__ == "__main__":
    main()
