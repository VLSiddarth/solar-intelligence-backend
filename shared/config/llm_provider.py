# ============================================================
# shared/config/llm_provider.py
# Unified LLM client — works with ANY provider, zero paid API required
#
# Priority order (auto-detected from .env):
#   1. Ollama   — 100% local, 100% free, recommended
#   2. Gemini   — Google free tier (60 req/min free)
#   3. Groq     — Free tier (fastest cloud inference)
#   4. LMStudio — Local alternative to Ollama
#   5. OpenAI   — Paid, only if explicitly configured
#
# Usage (identical regardless of provider):
#   from shared.config.llm_provider import get_llm_client
#   client = get_llm_client()
#   response = await client.complete(messages, max_tokens=2048)
# ============================================================

import os
import json
import time
import asyncio
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from shared.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str
    latency_ms: float
    truncated: bool = False


# ─────────────────────────────────────────────────────────────
# Provider Detection
# ─────────────────────────────────────────────────────────────

def _detect_provider() -> str:
    """
    Auto-detect which LLM provider to use based on .env config.
    Returns the provider name string.
    """
    explicit = os.getenv("LLM_PROVIDER", "").lower().strip()
    if explicit:
        return explicit

    # Auto-detect by which keys/URLs are set
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    # Default: Ollama (always free, no key needed)
    return "ollama"


# ─────────────────────────────────────────────────────────────
# Base Client Interface
# ─────────────────────────────────────────────────────────────

class BaseLLMClient:
    """All providers implement this interface."""

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError

    async def health(self) -> bool:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# Ollama Client (100% local, 100% free)
# ─────────────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """
    Calls Ollama running locally.
    Install: curl -fsSL https://ollama.ai/install.sh | sh
    Pull model: ollama pull mistral
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model    = os.getenv("OLLAMA_MODEL", "mistral")
        logger.info("llm_provider_ollama", extra={
            "base_url": self.base_url,
            "model":    self.model,
        })

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import httpx
        t_start = time.monotonic()

        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "num_predict":   max_tokens,
                "temperature":   temperature,
                "num_ctx":       4096,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content          = data["message"]["content"]
        prompt_tokens    = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        latency          = (time.monotonic() - t_start) * 1000

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.model,
            provider="ollama",
            latency_ms=round(latency, 1),
        )

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        import httpx

        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   True,
            "options":  {"num_predict": max_tokens},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break

    async def health(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# Gemini Client (Google free tier — 60 req/min)
# ─────────────────────────────────────────────────────────────

class GeminiClient(BaseLLMClient):
    """
    Calls Google Gemini free API.
    Get free key: https://aistudio.google.com/app/apikey
    Free limits: 60 requests/min, 1500 requests/day
    """

    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com/app/apikey "
                "or set LLM_PROVIDER=ollama to use local Ollama instead."
            )
        logger.info("llm_provider_gemini", extra={"model": self.model})

    def _messages_to_gemini(self, messages: list[dict]) -> tuple[str, list]:
        """Convert OpenAI-style messages to Gemini format."""
        system_parts = []
        contents     = []

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append({"text": content})
            elif role == "user":
                contents.append({"role": "user",  "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        return system_parts, contents

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import httpx
        t_start = time.monotonic()

        system_parts, contents = self._messages_to_gemini(messages)

        payload: dict = {
            "contents":         contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature":     temperature,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = self.GEMINI_URL.format(model=self.model)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                json=payload,
                params={"key": self.api_key},
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        candidate = data["candidates"][0]
        content   = candidate["content"]["parts"][0]["text"]

        usage             = data.get("usageMetadata", {})
        prompt_tokens     = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        latency           = (time.monotonic() - t_start) * 1000

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.model,
            provider="gemini",
            latency_ms=round(latency, 1),
        )

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        import httpx

        system_parts, contents = self._messages_to_gemini(messages)
        payload: dict = {
            "contents":         contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent"

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", url,
                json=payload,
                params={"key": self.api_key, "alt": "sse"},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        raw  = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        data = json.loads(raw)
                        try:
                            token = data["candidates"][0]["content"]["parts"][0]["text"]
                            if token:
                                yield token
                        except (KeyError, IndexError):
                            pass

    async def health(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}",
                    params={"key": self.api_key},
                )
                return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# Groq Client (Free tier — fastest cloud inference)
# ─────────────────────────────────────────────────────────────

class GroqClient(BaseLLMClient):
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. "
                "Get a free key at https://console.groq.com"
            )
        logger.info("llm_provider_groq", extra={"model": self.model})

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import httpx
        t_start = time.monotonic()

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/chat/completions",
                json={
                    "model":       self.model,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content           = data["choices"][0]["message"]["content"]
        usage             = data.get("usage", {})
        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        latency           = (time.monotonic() - t_start) * 1000

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.model,
            provider="groq",
            latency_ms=round(latency, 1),
        )

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                json={
                    "model":      self.model,
                    "messages":   messages,
                    "max_tokens": max_tokens,
                    "stream":     True,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        data  = json.loads(raw)
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token

    async def health(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.BASE_URL}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# LM Studio Client (local, OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────

class LMStudioClient(BaseLLMClient):
    """
    Calls LM Studio local server (OpenAI-compatible).
    Download: https://lmstudio.ai
    Start: Local Server tab → Start Server
    """

    def __init__(self):
        self.base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
        self.model    = os.getenv("LMSTUDIO_MODEL", "local-model")
        logger.info("llm_provider_lmstudio", extra={"base_url": self.base_url})

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import httpx
        t_start = time.monotonic()

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model":       self.model,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        content           = data["choices"][0]["message"]["content"]
        usage             = data.get("usage", {})
        latency           = (time.monotonic() - t_start) * 1000

        return LLMResponse(
            content=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=self.model,
            provider="lmstudio",
            latency_ms=round(latency, 1),
        )

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model":      self.model,
                    "messages":   messages,
                    "max_tokens": max_tokens,
                    "stream":     True,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        data  = json.loads(raw)
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token

    async def health(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# OpenAI Client (kept for users who want to use it)
# ─────────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    """OpenAI client — paid, only used if explicitly configured."""

    def __init__(self):
        from openai import AsyncOpenAI as _AsyncOpenAI
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = _AsyncOpenAI(api_key=self.api_key)
        logger.info("llm_provider_openai", extra={"model": self.model})

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t_start  = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content           = response.choices[0].message.content or ""
        usage             = response.usage
        latency           = (time.monotonic() - t_start) * 1000

        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=self.model,
            provider="openai",
            latency_ms=round(latency, 1),
        )

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# Provider Factory — singleton with auto-detect
# ─────────────────────────────────────────────────────────────

_client: Optional[BaseLLMClient] = None

PROVIDER_MAP = {
    "ollama":   OllamaClient,
    "gemini":   GeminiClient,
    "groq":     GroqClient,
    "lmstudio": LMStudioClient,
    "openai":   OpenAIClient,
}


def get_llm_client() -> BaseLLMClient:
    """
    Get the configured LLM client singleton.
    Provider is auto-detected from .env — see LLM_PROVIDER variable.
    """
    global _client
    if _client is not None:
        return _client

    provider = _detect_provider()
    cls      = PROVIDER_MAP.get(provider)

    if cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Valid options: {list(PROVIDER_MAP.keys())}"
        )

    try:
        _client = cls()
        logger.info("llm_client_created", extra={"provider": provider})
        return _client
    except Exception as e:
        logger.error("llm_client_creation_failed", extra={
            "provider": provider,
            "error":    str(e),
        })
        # Automatic fallback to Ollama if configured provider fails
        if provider != "ollama":
            logger.warning("llm_falling_back_to_ollama")
            _client = OllamaClient()
            return _client
        raise


def reset_client() -> None:
    """Reset singleton — used in tests and when switching providers."""
    global _client
    _client = None