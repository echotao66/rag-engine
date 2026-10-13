# 私有知识库问答助手（RAG）

一个**从零手写核心**的知识库问答应用：把文档喂进去，用自然语言提问，检索相关片段并生成**带引用出处**的回答，减少大模型「一本正经地胡说」。

核心链路（分词 / 检索 / 融合 / 引用）不依赖 LangChain，手写实现；向量存储做成可插拔——默认零依赖（numpy 余弦 + JSON 落盘），可选 chromadb（HNSW 近似最近邻 + 本地持久化）。自带 FastAPI + SSE 后端和原生 Web 前端，可点开即用。

## 能力一览

| 能力 | 实现 | 位置 |
|------|------|------|
| **文档摄取** | txt/md 文件与目录递归加载，内容 SHA-256 去重、增量导入 | `app/loader.py` |
| **混合检索** | dense（numpy 余弦）+ sparse（手写 BM25）双路，RRF 倒数排名融合 | `rag/hybrid.py` `rag/sparse.py` |
| **两阶段检索** | 召回与精排解耦，预留 cross-encoder 重排 | `rag/rerank.py` |
| **引用可控生成** | 命中片段 `[n]` 编号注入 prompt，解析回显引用 | `rag/generator.py` |
| **多轮对话** | 历史管理 + LLM 查询改写（追问补全成独立检索 query） | `app/session.py` |
| **流式生成** | OpenAI 兼容 stream，SSE 逐 token 推送（打字机效果） | `rag/llm.py` `backend/main.py` |
| **可观测** | 检索来源、改写后的 query、引用编号全程以事件流推送 | `backend/main.py` |
| **可插拔** | embedder / 分块 / 重排 / 存储四处协议抽象（存储默认 JSON，可切 chromadb） | `rag/embedder.py` `rag/vectorstore.py` |
| **评测** | 自建 recall@k 召回对比（dense / sparse / hybrid） | `tests/benchmark.py` |

## 架构

```
文档（txt/md）
  ├─ DocumentLoader：读取 + hash 去重（增量摄取）
  └─ ingest：分块 → 向量化 → 写入 store（JSON 落盘 / chromadb 持久化，二选一）

用户问题（多轮）
  ├─ ChatSession：历史 + LLM 查询改写
  ├─ 双路检索：dense（余弦）+ sparse（BM25）
  ├─ RRF 融合 → 可选 cross-encoder 重排
  ├─ 引用可控生成（DeepSeek，SSE 流式）
  └─ 前端：打字机回答 + [n] 引用高亮 + 来源面板
```

## 目录结构

```
rag-engine/
├── rag/                 # 检索核心（纯 Python，可独立复用）
│   ├── engine.py        # 统一入口：ingest / retrieve / query
│   ├── hybrid.py        # RRF 融合（核心亮点）
│   ├── sparse.py        # 手写 BM25
│   ├── dense.py         # numpy 余弦向量检索
│   ├── generator.py     # 引用可控生成（generate / stream）
│   ├── chunking.py      # 分块策略（可插拔）
│   ├── embedder.py      # 可插拔 embedding
│   ├── rerank.py        # 重排器协议
│   ├── store.py         # chunk 存储协议 + 内存 JSON 实现（零依赖）
│   ├── vectorstore.py   # chromadb 后端（HNSW 持久化，可插拔第二种实现）
│   └── llm.py           # DeepSeek 客户端（chat / stream）
├── app/                 # 应用层
│   ├── loader.py        # 文档摄取 + 去重
│   └── session.py       # 多轮会话 + 查询改写 + 流式问答
├── backend/             # Web 服务
│   ├── main.py          # FastAPI + SSE
│   └── static/index.html# 原生 HTML/CSS/JS 前端（无框架）
├── cli.py               # 交互式命令行
├── demo.py              # 离线演示（不联网）
├── tests/               # 单测 + benchmark（召回对比）
└── requirements.txt
```

## 快速开始

```bash
cd rag-engine
pip install -r requirements.txt

# 1) 离线演示（无需 key，脚本化模型走完整链路）
python demo.py

# 2) 召回对比实验（无需 key）
python tests/benchmark.py

# 3) 交互式命令行（search 无需 key，ask 需配 key）
python cli.py

# 4) Web 服务（浏览器打开 http://127.0.0.1:8000）
python -m uvicorn backend.main:app --port 8000

# 5) 跑测试
python -m pytest -q
```

默认用**内存 + JSON** 存储。想换 **chromadb**（HNSW 近似最近邻 + 持久化，数据落 `.rag/chroma`），设环境变量 `RAG_STORE=chroma` 再启动即可：

```bash
RAG_STORE=chroma python -m uvicorn backend.main:app --port 8000
```

`.env` 里配置 `DEEPSEEK_API_KEY`（复制 `.env.example` 改名即可）。不配 key 也能建库、检索、跑 Web，只是生成阶段降级为「仅展示检索结果」。

## 面试可讲的点

1. **为什么混合检索**：纯向量对「专有名词、精确关键词」召回差（语义是近似、关键词是精确），BM25 正好互补；两路得分量纲不同（余弦在 [-1,1]、BM25 是正实数），直接加权相加无意义，所以用 **RRF 只关心排名** 做融合。
2. **引用可控**：检索片段按 `[n]` 编号注入 prompt，要求模型标注来源再解析回显，解决「回答无依据」的信任问题。
3. **多轮追问怎么处理**：RAG 拿「那第二个呢」这种残缺问句直接检索会召回错，所以先用 LLM 把「历史 + 当前问题」改写成独立检索 query，再走检索。
4. **可插拔**：embedder / 分块 / 重排 / 存储四处都是协议接口，默认零依赖实现（HashEmbedder + numpy + JSON），并已接入 chromadb（`RAG_STORE=chroma` 一键切换，HNSW 近似最近邻 + 持久化）；生产还可换 sentence-transformers / cross-encoder。
5. **可靠性细节**：HashEmbedder 用 `crc32` 而非内置 `hash()`（`hash()` 每进程随机化会导致落盘后跨进程向量不一致）；文档摄取用内容 SHA-256 去重，重复导入自动跳过。

## 技术栈

- **检索**：dense（numpy 手写余弦，或 chromadb HNSW）+ 手写 BM25，RRF 融合
- **LLM**：DeepSeek（`deepseek-chat`，OpenAI 兼容，支持 stream）
- **后端**：FastAPI + SSE（流式事件推送）
- **前端**：原生 HTML / CSS / JS（无框架，EventSource + fetch）
- **测试**：pytest，脚本化 LLM 离线跑通全链路
