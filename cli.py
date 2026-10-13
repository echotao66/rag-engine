"""rag-engine 交互式 CLI：建库 + 混合检索 + 引用生成。

命令：
    ingest <路径> 导入文件或目录（txt/md，内容 hash 去重）
    add <文本>    直接录入一段文本
    search <q>    混合检索（无需 LLM，显示 top chunk）
    ask <q>       检索 + 生成（需 LLM，带引用）
    help / exit

运行：
    python cli.py

依赖：.env 里配 DEEPSEEK_API_KEY（复制 .env.example 改名 .env）。
不配 key 也能建库和检索，只是 ask 不可用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

STORE_FILE = Path(__file__).parent / ".rag" / "store.json"


def _force_utf8() -> None:
    """Windows 终端默认 GBK，强制 UTF-8 避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _load_env() -> None:
    """从项目根 .env 读取环境变量（不依赖 python-dotenv）。"""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _print_chunks(chunks) -> None:
    for cid, text, score in chunks:
        print(f"  [{cid}] ({score:.3f}) {text}")


def main() -> None:
    _force_utf8()
    _load_env()

    from rag import LLMClient, RAGEngine

    llm = None
    try:
        llm = LLMClient()
    except ValueError:
        print("⚠️  未配置 DEEPSEEK_API_KEY：仅支持 search，ask 不可用\n")

    engine = RAGEngine(llm=llm)
    if STORE_FILE.exists():
        n = engine.store.load(STORE_FILE)
        engine.sparse.build(engine.store)
        print(f"已加载 {n} 个历史 chunk\n")

    print("命令：ingest <路径> | add <文本> | search <q> | ask <q> | help | exit\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not line:
            continue

        cmd, _, arg = line.partition(" ")
        arg = arg.strip()

        if cmd in {"exit", "quit", "q"}:
            print("再见！")
            break
        elif cmd == "help":
            print(
                "  ingest <路径> 导入文件或目录（txt/md，自动去重）\n"
                "  add <文本>    直接录入一段文本\n"
                "  search <q>    混合检索（无需 LLM）\n"
                "  ask <q>       检索 + 生成（需 LLM，带引用）"
            )
        elif cmd in {"load", "ingest"}:
            if not arg:
                print(f"用法：{cmd} <文件或目录路径>")
                continue
            p = Path(arg)
            if not p.exists():
                print(f"路径不存在：{arg}")
                continue
            from app.loader import DocumentLoader

            loader = DocumentLoader()
            docs = loader.load_dir(p) if p.is_dir() else [loader.load_file(p)]
            seen = {c.doc_id for c in engine.store.all()}
            fresh = loader.dedupe(docs, seen)
            total = 0
            for doc in fresh:
                total += engine.ingest(
                    doc.text, doc_id=doc.doc_id, metadata={"title": doc.title}
                )
            engine.store.save(STORE_FILE)
            print(
                f"已导入 {total} 个 chunk（新增 {len(fresh)} 篇，"
                f"跳过 {len(docs) - len(fresh)} 篇重复）"
            )
        elif cmd == "add":
            if not arg:
                print("用法：add <文本>")
                continue
            n = engine.ingest(arg)
            engine.store.save(STORE_FILE)
            print(f"已录入 {n} 个 chunk")
        elif cmd == "search":
            if not arg:
                print("用法：search <query>")
                continue
            _print_chunks(engine.retrieve(arg, k=3))
        elif cmd == "ask":
            if llm is None:
                print("ask 需要 LLM：请在 .env 配置 DEEPSEEK_API_KEY")
                continue
            if not arg:
                print("用法：ask <query>")
                continue
            result = engine.query(arg, k=3)
            _print_chunks(result["chunks"])
            print(f"\n💬 {result['answer']}")
            if result.get("citations"):
                print(f"引用：{[f'[{c}]' for c in result['citations']]}")
        else:
            print(f"未知命令：{cmd}（输入 help 查看）")


if __name__ == "__main__":
    main()
