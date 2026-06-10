# ============================================================
# convective/agents/token_governor.py  (UPDATED — uses LLM provider)
# Token governor — hard ceilings per query mode
# Now uses core.llm.provider to route to Groq/OpenAI/Ollama
# ============================================================

import time
from typing import Optional
from shared.config.settings import settings
from shared.models.entities import QueryMode, TokenUsage, LLMResponse
from shared.utils.correlation import get_correlation_id
from shared.utils.logging import get_logger
from core.llm.provider import get_llm_client, get_model_name, get_provider_name

logger = get_logger(__name__)

# Groq pricing (llama-3.1-8b-instant)
COST_PER_1K_INPUT  = 0.00005
COST_PER_1K_OUTPUT = 0.00008

LIMITS: dict[QueryMode, dict] = {
    QueryMode.CHAIN_OF_THOUGHT: {
        "soft": settings.token.cot_soft,    # 4096
        "hard": settings.token.cot_hard,    # 8192
    },
    QueryMode.STANDARD: {
        "soft": settings.token.standard_soft,  # 2048
        "hard": settings.token.standard_hard,  # 4096
    },
    QueryMode.EDGE: {
        "soft": settings.token.edge_soft,   # 512
        "hard": settings.token.edge_hard,   # 1024
    },
}

TRUNCATION_FALLBACK = (
    "[SI: Response truncated at token ceiling. "
    "The answer was becoming too long. "
    "Please rephrase your query to be more specific, or use a higher-capacity mode.]"
)


class TokenGovernor:
    """
    Governs LLM token spend per query mode.
    Works with any provider via the provider abstraction.
    """

    def __init__(self):
        # KEY CHANGE: use provider abstraction
        self._client          = get_llm_client()
        self._model           = get_model_name()
        self._provider        = get_provider_name()
        self._session_tokens  = 0
        self._session_cost    = 0.0

        logger.info("token_governor_initialized", extra={
            "provider": self._provider,
            "model":    self._model,
        })

    async def complete(
        self,
        messages: list[dict],
        mode: QueryMode = QueryMode.STANDARD,
        model: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Run a governed LLM completion.
        Enforces soft and hard token limits based on query mode.
        """
        cid    = correlation_id or get_correlation_id()
        limits = LIMITS[mode]
        mdl    = model or self._model
        t_start = time.monotonic()

        logger.info("llm_call_started", extra={
            "provider":         self._provider,
            "model":            mdl,
            "mode":             mode.value,
            "soft_limit":       limits["soft"],
            "hard_limit":       limits["hard"],
            "correlation_id":   cid,
        })

        try:
            response = await self._client.chat.completions.create(
                model=mdl,
                messages=messages,
                max_tokens=limits["soft"],
                temperature=0.1 if mode == QueryMode.CHAIN_OF_THOUGHT else 0.0,
            )

            content   = response.choices[0].message.content or ""
            usage     = response.usage
            truncated = response.choices[0].finish_reason == "length"

            total = usage.prompt_tokens + usage.completion_tokens

            # Hard kill: if total tokens hit hard limit, replace with fallback
            if total >= limits["hard"]:
                logger.error("hard_token_ceiling_hit", extra={
                    "total_tokens": total,
                    "hard_limit":   limits["hard"],
                    "mode":         mode.value,
                    "correlation_id": cid,
                })
                content   = TRUNCATION_FALLBACK
                truncated = True

            latency = (time.monotonic() - t_start) * 1000
            cost_usd = (
                (usage.prompt_tokens     / 1000) * COST_PER_1K_INPUT +
                (usage.completion_tokens / 1000) * COST_PER_1K_OUTPUT
            )

            self._session_tokens += total
            self._session_cost   += cost_usd

            logger.info("llm_call_complete", extra={
                "provider":         self._provider,
                "model":            mdl,
                "prompt_tokens":    usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens":     total,
                "cost_usd":         round(cost_usd, 6),
                "latency_ms":       round(latency, 1),
                "truncated":        truncated,
                "correlation_id":   cid,
            })

            return LLMResponse(
                content=content,
                usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=total,
                    cost_usd=cost_usd,
                    mode=mode,
                ),
                truncated=truncated,
                provider=self._provider,
                model=mdl,
            )

        except Exception as e:
            logger.error("llm_call_failed", extra={
                "provider":       self._provider,
                "model":          mdl,
                "error":          str(e),
                "correlation_id": cid,
            })
            raise


# Module-level singleton
_governor: Optional[TokenGovernor] = None


def get_governor() -> TokenGovernor:
    global _governor
    if _governor is None:
        _governor = TokenGovernor()
    return _governor