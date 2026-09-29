"""LLM calls for observe / consolidate / profile (not used on retrieve path).

**To change provider or API style, edit this file only** (or replace `_chat_completions`).

Environment:
- `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`
- `MEMORY_ONLINE_LLM_BASE_URL` (optional, default DeepSeek OpenAI-compatible endpoint)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

DEFAULT_BASE_URL = "https://api.deepseek.com"


def _load_env_file() -> None:
    path = os.environ.get("MEMORY_ONLINE_ENV_FILE", "").strip()
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


def require_api_key() -> str:
    _load_env_file()
    key = (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY required for memory extraction.")
    return key


def _base_url() -> str:
    return (os.environ.get("MEMORY_ONLINE_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _parse_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


@lru_cache(maxsize=4)
def _openai_client():
    from openai import OpenAI

    return OpenAI(api_key=require_api_key(), base_url=_base_url())


def _chat_completions(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    """Single place to swap HTTP client / vendor (OpenAI-compatible API)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    client = _openai_client()
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        for attr in ("reasoning_content", "reasoning"):
            alt = getattr(choice.message, attr, None)
            if alt and str(alt).strip():
                return str(alt).strip()
        raise RuntimeError(f"empty completion finish_reason={choice.finish_reason}")
    return text


def chat_text_for_model(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_cache: bool = True,
    cache_kind: str = "memory_online_chat",
) -> str:
    from se.core.memory.online.llm_cache import cache_key, load_cached, save_cached

    require_api_key()
    key = cache_key(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if use_cache:
        cached = load_cached(cache_kind, key)
        if cached and isinstance(cached.get("text"), str):
            return cached["text"]
    text = _chat_completions(model, messages, temperature=temperature, max_tokens=max_tokens, json_mode=False)
    if use_cache:
        save_cached(cache_kind, key, {"text": text})
    return text


def chat_json_for_model(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_cache: bool = True,
    cache_kind: str = "memory_online_json",
) -> dict[str, Any]:
    from se.core.memory.online.llm_cache import cache_key, load_cached, save_cached

    require_api_key()
    key = cache_key(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if use_cache:
        cached = load_cached(cache_kind, key)
        if cached and isinstance(cached.get("data"), dict):
            return cached["data"]
    raw = _chat_completions(model, messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
    data = _parse_json(raw)
    if not isinstance(data, dict):
        raise ValueError("model JSON must be an object")
    if use_cache:
        save_cached(cache_kind, key, {"data": data})
    return data
