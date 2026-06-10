# ============================================================
# core/llm/provider.py
# LLM provider abstraction — Groq / OpenAI / Ollama
# All three expose OpenAI-compatible API; we just swap base_url + key.
# Usage:
#   from core.llm.provider import get_llm_client, get_model_name
#   client = get_llm_client()
#   response = await client.chat.completions.create(model=get_model_name(), ...)
# ============================================================

from openai import AsyncOpenAI
from shared.config.settings import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def get_llm_client() -> AsyncOpenAI:
    """
    Return an AsyncOpenAI-compatible client for the configured LLM provider.

    Providers:
        groq   — Groq Cloud (fastest inference, OpenAI-compatible)
        openai — OpenAI API directly
        ollama — Local Ollama server (free, runs on host machine)
    """
    provider = settings.llm_provider

    if provider == "groq":
        logger.debug("llm_provider_groq_selected")
        return AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    elif provider == "openai":
        logger.debug("llm_provider_openai_selected")
        return AsyncOpenAI(api_key=settings.openai.api_key)

    elif provider == "ollama":
        logger.debug("llm_provider_ollama_selected")
        return AsyncOpenAI(
            api_key="ollama",  # Ollama ignores the key but SDK requires it
            base_url="http://host.docker.internal:11434/v1",
        )

    else:
        logger.warning("unknown_llm_provider_falling_back_to_groq",
                       extra={"provider": provider})
        return AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )


def get_model_name() -> str:
    """Return the configured model name for the current provider."""
    provider = settings.llm_provider

    if provider == "groq":
        return settings.groq_model
    elif provider == "ollama":
        return settings.ollama_model
    else:
        return settings.openai.model


def get_provider_name() -> str:
    return settings.llm_provider