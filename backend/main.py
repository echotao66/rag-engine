"""FastAPI 后端：知识库管理 + SSE 流式问答。

运行：
    cd rag-engine
    python -m uvicorn backend.main:app --port 8000
浏览器打开 http://127.0.0.1:8000

接口：
    GET  /                     前端页面
    GET  /api/kb               知识库统计（chunk 数、文档数、LLM 是否就绪）
    POST /api/ingest/text      导入文本（JSON: {text, title?}）
    POST /api/ingest/file      上传文件（multipart）
    POST /api/clear            清空知识库
    GET  /api/chat/stream      流式问答（SSE，?message=...）

SSE 事件流（可观测）：
    retrieved -> 检索到的片段 + 改写后的 query
    delta     -> 生成增量文本（打字机）
    done      -> 完整回答 + 引用编号
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.session import ChatSession  # noqa: E402
from rag import ChromaStore, LLMClient, RAGEngine  # noqa: E402

STORE_FILE = ROOT / ".rag" / "store.json"
CHROMA_DIR = ROOT / ".rag" / "chroma"
STATIC_DIR = Path(__file__).parent / "static"


def _build_store():
    """按环境变量 RAG_STORE 选择向量存储后端。

    默认（未设置 / 其它值）：内存 + JSON（DocumentStore，零依赖）；
    RAG_STORE=chroma：chromadb 持久化（ChromaStore，HNSW 近似最近邻）。
    """
    if os.environ.get("RAG_STORE", "").strip().lower() == "chroma":
        return ChromaStore(path=CHROMA_DIR)
    return None


def _load_env() -> None:
    """从项目根 .env 读取环境变量（不依赖 python-dotenv）。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


app = FastAPI(title="私有知识库问答助手")


class IngestText(BaseModel):
    text: str
    title: str = ""


def _build_llm() -> LLMClient | None:
    try:
        return LLMClient()
    except ValueError:
        return None


_load_env()
llm = _build_llm()
engine = RAGEngine(llm=llm, store=_build_store())
session = ChatSession(engine, llm)

# 启动时恢复上次持久化的知识库：
# DocumentStore 从 JSON 恢复，ChromaStore 由 chromadb 自动恢复（load 为无操作）。
# 无论哪种后端，都要重建 BM25 稀疏索引（它不落盘，从 store.all() 现算）。
engine.store.load(STORE_FILE)
engine.sparse.build(engine.store)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/kb")
def kb_stats() -> dict:
    docs: dict[str, int] = {}
    for c in engine.store.all():
        key = c.doc_id or "未命名"
        docs[key] = docs.get(key, 0) + 1
    return {
        "chunks": len(engine.store),
        "docs": len(docs),
        "llm_ready": llm is not None,
        "doc_list": [{"doc_id": k, "chunks": v} for k, v in docs.items()],
    }


@app.post("/api/ingest/text")
def ingest_text(body: IngestText) -> dict:
    n = engine.ingest(body.text, doc_id=body.title)
    engine.sparse.build(engine.store)
    engine.store.save(STORE_FILE)
    return {"chunks": n, "title": body.title}


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    text = raw.decode("utf-8", errors="replace")
    n = engine.ingest(text, doc_id=file.filename or "上传文件")
    engine.sparse.build(engine.store)
    engine.store.save(STORE_FILE)
    return {"chunks": n, "filename": file.filename}


@app.post("/api/clear")
def clear() -> dict:
    engine.clear()
    engine.store.save(STORE_FILE)
    return {"ok": True}


@app.get("/api/chat/stream")
def chat_stream(message: str) -> StreamingResponse:
    def gen():
        try:
            for kind, payload in session.stream_ask(message):
                yield _sse(kind, payload)
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")
