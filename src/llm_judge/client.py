"""OpenAI-compatible client wrapper with retry, cache, and length-truncation detection.

All judges (Claude family via gateway, GPT family) share this single client
because the gateway speaks OpenAI-compatible protocol. Anthropic-native protocol
is also available on the same base_url (without /v1) and supports cache_control,
but gateway-side cache_read_input_tokens stays 0 in practice (verified
empirically). We therefore prefer the simpler OpenAI route and rely on the
local ResponseCache for repeatability and cost control.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from llm_judge import config
from llm_judge.cache import ResponseCache


class TruncatedResponseError(RuntimeError):
    """模型达到 max_tokens 且响应不完整时抛出。"""


@dataclass
class CallResult:
    text: str
    cached: bool
    finish_reason: str | None
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None


class JudgeClient:
    def __init__(
        self,
        cache: ResponseCache | None = None,
        *,
        request_timeout_seconds: float | None = None,
        retry_max_attempts: int | None = None,
        retry_on_empty: bool = False,
    ):
        """输入：可选响应缓存。输出：无。作用：初始化 OpenAI 兼容的评审客户端。"""
        self._request_timeout_seconds = request_timeout_seconds or config.REQUEST_TIMEOUT_SECONDS
        self._retry_max_attempts = retry_max_attempts or config.RETRY_MAX_ATTEMPTS
        self._retry_on_empty = retry_on_empty
        self._client = OpenAI(
            base_url=config.API_BASE_URL,
            api_key=config.API_KEY,
            timeout=self._request_timeout_seconds,
        )
        self._cache = cache or ResponseCache(config.CACHE_DIR)

    def call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float | None = None,
    ) -> CallResult:
        """输入：模型、提示词、token 上限和温度。输出：CallResult。作用：调用或复用 LLM 响应。"""
        if temperature is None:
            temperature = config.TEMPERATURE

        cached = self._cache.get(model, system, user)
        if cached:
            return CallResult(text=cached, cached=True, finish_reason="cached", model=model)

        attempts = 0
        last_exc: Exception | None = None
        current_max = max_tokens
        empty_attempts = 0
        # Tokens billed by the provider on attempts that returned empty content.
        wasted_prompt_tokens = 0
        wasted_cached_prompt_tokens = 0

        while attempts < self._retry_max_attempts:
            attempts += 1
            try:
                request = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": current_max,
                    "timeout": self._request_timeout_seconds,
                }
                if config.LLM_REASONING_EFFORT:
                    request["reasoning_effort"] = config.LLM_REASONING_EFFORT
                if config.LLM_EXTRA_BODY:
                    request["extra_body"] = config.LLM_EXTRA_BODY
                resp = self._client.chat.completions.create(**request)
                choice = resp.choices[0]
                text = choice.message.content or ""
                finish = choice.finish_reason

                if finish == "length" and current_max < 4096:
                    # Response was cut off mid-generation. Double the budget and retry.
                    current_max = min(current_max * 2, 4096)
                    continue

                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                cached_pt = _cached_prompt_tokens(usage)

                if self._retry_on_empty and not text.strip():
                    empty_attempts += 1
                    wasted_prompt_tokens += pt or 0
                    wasted_cached_prompt_tokens += cached_pt or 0
                    last_exc = RuntimeError("empty response content")
                    time.sleep(config.RETRY_BACKOFF_SECONDS * empty_attempts)
                    continue

                if text.strip():
                    self._cache.put(model, system, user, text)
                return CallResult(
                    text=text,
                    cached=False,
                    finish_reason=finish,
                    model=model,
                    prompt_tokens=(pt or 0) + wasted_prompt_tokens if pt is not None or wasted_prompt_tokens else pt,
                    completion_tokens=ct,
                    cached_prompt_tokens=(cached_pt or 0) + wasted_cached_prompt_tokens
                    if cached_pt is not None or wasted_cached_prompt_tokens
                    else cached_pt,
                )

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exc = e
                time.sleep(config.RETRY_BACKOFF_SECONDS * (2 ** (attempts - 1)))
            except APIError as e:
                last_exc = e
                status = getattr(e, "status_code", None)
                if status is not None and 500 <= status < 600 and attempts < config.RETRY_MAX_ATTEMPTS:
                    time.sleep(config.RETRY_BACKOFF_SECONDS * (2 ** (attempts - 1)))
                    continue
                raise

        raise RuntimeError(
            f"LLM call failed after {self._retry_max_attempts} attempts: {last_exc}"
        ) from last_exc


def _cached_prompt_tokens(usage) -> int | None:
    """输入：OpenAI-compatible usage 对象。输出：缓存命中输入 token 数。作用：支持 DeepSeek 计费。"""
    if usage is None:
        return None
    for attr in ("prompt_cache_hit_tokens", "cache_read_input_tokens"):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            return value
    details = getattr(usage, "prompt_tokens_details", None)
    value = getattr(details, "cached_tokens", None) if details else None
    return value if isinstance(value, int) else None
