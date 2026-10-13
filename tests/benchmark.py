"""召回对比实验：dense / sparse / hybrid 三种检索的 recall@k / precision@k / MRR / nDCG。

运行：python tests/benchmark.py

要点：
- 用语义 embedder（paraphrase-multilingual-MiniLM-L12-v2，首次运行自动下载），
  模型下载走 hf-mirror 加速。
- 语料 50 篇（中英各半），query 40 条，分四类：
    en_exact  英文精确关键词（BM25 强项）
    en_para   英文近义改写（语义向量强项）
    en_multi  英文多相关文档（一个 query 命中 2~4 篇，precision 才有意义）
    zh_exact / zh_para  中文精确 / 中文改写
- 每条 query 标注 1~4 篇相关文档，且只标「确实回答了这个问题」的文档。
- 输出 recall@k / precision@k / MRR@5 / nDCG@5 与单条 query 平均耗时。
- 简历里的数字从这里取，别编。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 加速镜像（必须在加载模型前设置）
# 模型缓存目录如需指定，请先设置 HF_HOME 环境变量（不设则用 HuggingFace 默认缓存）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import HashEmbedder, RAGEngine  # noqa: E402

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 50 篇文档，每条都短于 chunk size（512 字符），所以「一篇文档 = 一个 chunk」，
# 文档下标 d{i} 与检索返回的 chunk id c{i} 一一对应，标注好写、也好核对。
DOCS = [
    # --- 检索 / RAG (d00~d12) ---
    "Retrieval-Augmented Generation (RAG) retrieves relevant passages from a knowledge base and feeds them to the model, which reduces hallucination.",
    "BM25 is a lexical ranking function combining term frequency, inverse document frequency and document length normalization.",
    "Reciprocal Rank Fusion (RRF) merges ranked lists from different retrievers by summing 1/(k+rank), so scores from different scales never need to be compared.",
    "Dense retrieval encodes queries and passages into vectors and ranks them by cosine similarity, capturing semantic similarity rather than exact wording.",
    "Chunking splits long documents into smaller passages; too large a chunk dilutes the embedding, too small a chunk loses context.",
    "An overlap between consecutive chunks prevents a sentence from being cut in half at a boundary.",
    "A cross-encoder reranker scores query and passage jointly, which is more accurate but far slower than embedding similarity.",
    "Hybrid retrieval combines lexical and semantic search so that exact terms and paraphrases are both covered.",
    "Recall@k measures the fraction of all relevant passages that appear in the top k results.",
    "Precision@k measures the fraction of the top k results that are actually relevant.",
    "MRR (mean reciprocal rank) averages 1/rank of the first relevant result across queries.",
    "nDCG discounts the relevance of a hit by the logarithm of its rank position.",
    "Query rewriting turns an underspecified follow-up question into a standalone query before retrieval.",
    # --- Agent / 工具 (d13~d24) ---
    "An AI agent perceives its environment and takes actions to reach a goal.",
    "ReAct interleaves reasoning traces with tool calls, so the model decides the next action from the observation.",
    "Function calling lets a model request an external tool with structured JSON arguments.",
    "JSON Schema validation rejects malformed tool arguments before the tool is executed.",
    "An idempotent tool can be cached safely; a tool with side effects must not be.",
    "A sandbox restricts which files and commands the agent may touch.",
    "Path traversal attacks use ../ to escape a working directory.",
    "A command allowlist permits only known-safe executables such as ls and pwd.",
    "MCP (Model Context Protocol) standardises how tools are exposed to models across processes.",
    "The Model Context Protocol uses stdio or HTTP transports to connect a client to a server.",
    "An event stream pushes structured progress events to the frontend so every agent step is visible.",
    "Server-Sent Events keep one long-lived HTTP response open and push tokens as they are generated.",
    # --- 向量 / 记忆 (d25~d33) ---
    "An embedding maps text into a dense vector so that similar meanings end up close together.",
    "Cosine similarity measures the angle between two vectors and ignores their magnitude.",
    "Exponential time decay multiplies a memory score by exp(-lambda * age), so old memories fade instead of being deleted.",
    "A vector index such as HNSW performs approximate nearest neighbour search in sublinear time.",
    "An exhaustive linear scan over all vectors is exact but costs O(n) per query.",
    "A transformer uses self-attention to process all positions of a sequence in parallel.",
    "Fine-tuning updates model weights on labelled data, while prompt engineering leaves the weights untouched.",
    "A content hash such as SHA-256 can serve as a stable document identifier so duplicate imports are skipped.",
    "Redis is an in-memory key-value store commonly used as a cache.",
    # --- 后端 / 工程 (d34~d41) ---
    "A database index speeds up lookups by keeping keys in sorted order.",
    "Docker packages an application with its runtime so the same image runs on any host.",
    "A reverse proxy can terminate TLS and enforce authentication before requests reach the application.",
    "A container should run as a non-root user to limit the blast radius of a compromise.",
    "A healthcheck lets the orchestrator detect and restart an unhealthy container.",
    "FastAPI serves an ASGI application and supports streaming responses.",
    "pytest collects test functions named test_* and reports failures with fixtures.",
    "The Python GIL prevents multiple threads from executing bytecode simultaneously.",
    # --- 中文语料 (d42~d49) ---
    "检索增强生成先从知识库中召回相关片段，再把它们拼进提示词，从而降低大模型的幻觉。",
    "BM25 是一种关键词打分算法，综合考虑词频、逆文档频率和文档长度归一化。",
    "倒数排名融合只关心两路检索结果的排名，因此不受余弦相似度和 BM25 分数不同量纲的影响。",
    "向量检索把问题和文档都编码成向量，再按余弦相似度排序，擅长语义相近但用词不同的情况。",
    "文档分块太大会稀释语义，太小又会丢失上下文，因此需要在两者之间取平衡。",
    "智能体能够感知环境并采取行动来达成目标，工具调用让它能够请求外部能力。",
    "路径穿越攻击利用点点斜杠跳出工作目录，因此沙箱必须校验解析后的真实路径。",
    "时间衰减让旧记忆的得分乘以指数衰减因子，从而逐渐淡出而不是被直接删除。",
]

# (query, 相关文档下标, 类别)
QUERIES: list[tuple[str, list[int], str]] = [
    # --- 英文精确关键词 ---
    ("What is BM25?", [1], "en_exact"),
    ("What does RRF stand for in hybrid retrieval?", [2], "en_exact"),
    ("How is recall@k defined?", [8], "en_exact"),
    ("What is MRR?", [10], "en_exact"),
    ("What transport does MCP use?", [22], "en_exact"),
    ("What is HNSW used for?", [28], "en_exact"),
    ("What is Redis?", [33], "en_exact"),
    ("What does a reverse proxy do?", [36], "en_exact"),
    ("What is a cross-encoder reranker?", [6], "en_exact"),
    ("What is path traversal?", [19], "en_exact"),
    # --- 英文近义改写 ---
    ("How can a model ask to run an outside function with structured arguments?", [15], "en_para"),
    ("Ways to stop a language model from making things up", [0], "en_para"),
    ("How to choose the size of a passage before embedding it", [4], "en_para"),
    ("Why do adjacent passages share some text?", [5], "en_para"),
    ("How to keep exact terms and paraphrases both findable", [7], "en_para"),
    ("How is a score penalised for appearing further down a list?", [11], "en_para"),
    ("Turning a vague follow-up into something you can search with", [12], "en_para"),
    ("Rejecting a tool call when the arguments are the wrong shape", [16], "en_para"),
    ("Which tool results is it safe to remember?", [17], "en_para"),
    ("Keeping the agent away from sensitive files and programs", [18], "en_para"),
    ("Letting a user watch every step the agent takes", [23], "en_para"),
    ("Pushing generated words to the browser as they appear", [24], "en_para"),
    ("How do old memories fade away instead of being deleted?", [27], "en_para"),
    ("What is the difference between fine-tuning and prompt engineering?", [31], "en_para"),
    ("Why should a container run as a non-root user?", [37], "en_para"),
    # --- 英文多相关（一个 query 命中 2~4 篇，这类 query 才让 precision 有意义）---
    ("Which metrics measure retrieval quality?", [8, 9, 10, 11], "en_multi"),
    ("How do lexical and semantic retrieval get combined?", [2, 7], "en_multi"),
    ("What makes nearest neighbour search fast on large collections?", [28, 29], "en_multi"),
    ("How does chunking affect retrieval quality?", [4, 5], "en_multi"),
    ("Which techniques make an agent safe to expose publicly?", [18, 19, 20], "en_multi"),
    # --- 中文精确 ---
    ("什么是倒数排名融合？", [44], "zh_exact"),
    ("BM25 是什么算法？", [43], "zh_exact"),
    ("路径穿越攻击是怎么实现的？", [48], "zh_exact"),
    ("时间衰减是怎么起作用的？", [49], "zh_exact"),
    ("文档分块太大或太小有什么问题？", [46], "zh_exact"),
    ("什么是检索增强生成？", [42], "zh_exact"),
    # --- 中文改写 ---
    ("怎么让大模型少胡说八道？", [42], "zh_para"),
    ("怎么把关键词检索和语义检索的优点合起来？", [44], "zh_para"),
    ("怎么给旧记忆降权而不是直接删掉？", [49], "zh_para"),
    ("把一篇文章切多长比较合适？", [46], "zh_para"),
    ("怎么防止别人用点点斜杠跳出目录？", [48], "zh_para"),
    ("用词不一样但意思相近的文档怎么找出来？", [45], "zh_para"),
]

CATEGORIES = ["en_exact", "en_para", "en_multi", "zh_exact", "zh_para"]


def _relevant_ids(rel: list[int]) -> set[str]:
    return {f"c{i}" for i in rel}


def evaluate(searcher, queries, k: int) -> dict[str, float]:
    """返回一组检索指标。

    recall@k    = 命中的相关 chunk 数 / 全部相关 chunk 数（召回有没有漏）
    precision@k = 命中的相关 chunk 数 / 返回的 chunk 数 k（返回的有没有错）
    MRR@k       = 第一条相关结果的 1/rank 的平均（第一条有多靠前）
    nDCG@k      = Σ rel_i / log2(rank_i + 1) 的平均（相关性有没有排到前面）
    """
    hits = total_rel = 0
    rr_sum = ndcg_sum = 0.0
    for q, rel, _ in queries:
        ids = [cid for cid, *_ in searcher(q, k=k)]
        rel_ids = _relevant_ids(rel)
        matched = set(ids) & rel_ids
        hits += len(matched)
        total_rel += len(rel_ids)

        rr = 0.0
        ndcg = 0.0
        for rank, cid in enumerate(ids, start=1):
            if cid in rel_ids:
                if rr == 0.0:
                    rr = 1.0 / rank
                ndcg += 1.0 / _log2(rank + 1)
        rr_sum += rr
        # 理想 DCG：相关文档全部排在前面
        ideal = sum(1.0 / _log2(i + 1) for i in range(1, min(len(rel_ids), k) + 1))
        ndcg_sum += (ndcg / ideal) if ideal else 0.0

    n = len(queries)
    return {
        "recall": hits / total_rel if total_rel else 0.0,
        "precision": hits / (k * n) if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
        "ndcg": ndcg_sum / n if n else 0.0,
    }


def _log2(x: float) -> float:
    import math

    return math.log2(x)


def latency_ms(searcher, queries, k: int) -> float:
    """单条 query 的平均检索耗时（毫秒）。"""
    start = time.perf_counter()
    for q, _, _ in queries:
        searcher(q, k=k)
    return (time.perf_counter() - start) / len(queries) * 1000.0


def build_engine() -> tuple[RAGEngine, bool]:
    """优先用语义 embedder；不可用则退回 HashEmbedder（结果无参考价值）。"""
    try:
        from rag import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(MODEL)
        print(f"使用语义 embedder：{MODEL}\n")
        return RAGEngine(embedder=embedder), True
    except Exception as exc:  # noqa: BLE001
        print(f"语义 embedder 不可用（{exc}），回退 HashEmbedder（结果无参考价值）\n")
        return RAGEngine(embedder=HashEmbedder()), False


def main() -> None:
    engine, semantic = build_engine()
    for i, doc in enumerate(DOCS):
        engine.ingest(doc, doc_id=f"doc{i}")

    n_para = sum(1 for _, _, c in QUERIES if not c.endswith("exact"))
    print(
        f"（{len(DOCS)} 篇文档 / {len(QUERIES)} 条 query："
        f"精确关键词 {sum(1 for _, _, c in QUERIES if c.endswith('exact'))} 条，"
        f"近义改写 {n_para} 条，多相关 {sum(1 for _, _, c in QUERIES if c == 'en_multi')} 条）\n"
    )

    searchers = {
        "dense": engine.dense.search,
        "sparse": engine.sparse.search,
        "hybrid": engine.hybrid.search,
    }

    # ---- 总表：各检索方式在 k=1/3/5 上的指标 ----
    print(f"{'k':>2} | {'dense':>32} | {'sparse':>32} | {'hybrid':>32}")
    print(
        f"{'':>2} | {'R@k':>7}{'P@k':>7}{'MRR':>8}{'nDCG':>8} "
        f"| {'R@k':>7}{'P@k':>7}{'MRR':>8}{'nDCG':>8} "
        f"| {'R@k':>7}{'P@k':>7}{'MRR':>8}{'nDCG':>8}"
    )
    for k in (1, 3, 5):
        cells = []
        for name in ("dense", "sparse", "hybrid"):
            m = evaluate(searchers[name], QUERIES, k)
            cells.append(
                f"{m['recall']:>6.1%} {m['precision']:>6.1%} {m['mrr']:>6.3f} {m['ndcg']:>6.3f}"
            )
        print(f"{k:>2} | " + " | ".join(cells))

    # ---- 分类别 recall（看两路各自强在哪）----
    # 注：recall@k 的天花板是 min(1, k/该 query 的相关文档数)，en_multi 里有 4 篇相关的
    # query，所以它那一列 recall@3 天然上不去，看 recall@5 才公平。
    print(
        f"\nrecall by category（括号内为 recall@5）\n"
        f"{'category':>10} | {'n':>2} | {'dense':>15} | {'sparse':>15} | {'hybrid':>15}"
    )
    for cat in CATEGORIES:
        sub = [q for q in QUERIES if q[2] == cat]
        cells = []
        for name in ("dense", "sparse", "hybrid"):
            r3 = evaluate(searchers[name], sub, 3)["recall"]
            r5 = evaluate(searchers[name], sub, 5)["recall"]
            cells.append(f"{r3:>6.1%} ({r5:>5.1%})")
        print(f"{cat:>10} | {len(sub):>2} | " + " | ".join(cells))

    # ---- 耗时 ----
    print(f"\n单条 query 平均耗时（{len(DOCS)} 篇语料，k=5，纯 Python 实现）")
    for name in ("dense", "sparse", "hybrid"):
        print(f"  {name:>6}: {latency_ms(searchers[name], QUERIES, 5):>7.2f} ms")
    if not semantic:
        print("\n[警告] 当前跑的是 HashEmbedder 回退路径，以上 dense/hybrid 数字无参考价值。")


if __name__ == "__main__":
    main()
