"""DeepSeek 生成客户端（精简版：只保留 chat）。

RAG 生成阶段只需要纯对话，所以这里不涉及 function calling（那是 cortex 的职责）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


@dataclass
class LLMConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL


class LLMClient:
    """DeepSeek chat 客户端。api_key 留空则从环境变量 DEEPSEEK_API_KEY 读取。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        config = config or LLMConfig()
        api_key = config.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "缺少 DEEPSEEK_API_KEY：请设置环境变量，或 LLMConfig(api_key=...) 传入"
            )
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)
        self.model = config.model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
    ):
        """流式生成，逐段 yield 增量文本（OpenAI 兼容 stream 模式）。

        供后端 SSE 逐 token 推送，前端实现打字机效果。
        """
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
