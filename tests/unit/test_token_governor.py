import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from convective.agents.token_governor import TokenGovernor, LIMITS, TRUNCATION_FALLBACK, GovernerResponse
from shared.models.entities import QueryMode, TokenUsage


def _make_mock_llm_response(content, prompt_tokens, completion_tokens, finish_reason="stop"):
    from shared.config.llm_provider import LLMResponse
    return LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model="test-model",
        provider="groq",
        latency_ms=300.0,
    )


@pytest.fixture
def governor():
    with patch("convective.agents.token_governor.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value = mock_client
        g = TokenGovernor()
        g._client = mock_client
        return g


class TestTokenGovernor:

    @pytest.mark.asyncio
    async def test_standard_query_returns_response(self, governor):
        governor._client.complete = AsyncMock(
            return_value=_make_mock_llm_response("The answer is 42.", 50, 10)
        )
        result = await governor.complete(
            [{"role": "user", "content": "What is 6x7?"}],
            mode=QueryMode.STANDARD,
        )
        assert result.content == "The answer is 42."
        assert result.usage.prompt_tokens == 50
        assert result.usage.completion_tokens == 10
        assert result.usage.truncated is False

    @pytest.mark.asyncio
    async def test_soft_limit_applied(self, governor):
        create_mock = AsyncMock(
            return_value=_make_mock_llm_response("Short answer.", 100, 50)
        )
        governor._client.complete = create_mock
        await governor.complete([{"role": "user", "content": "test"}], mode=QueryMode.EDGE)
        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["max_tokens"] == LIMITS[QueryMode.EDGE]["soft"]

    @pytest.mark.asyncio
    async def test_truncation_on_hard_limit(self, governor):
        hard_limit = LIMITS[QueryMode.STANDARD]["hard"]
        governor._client.complete = AsyncMock(
            return_value=_make_mock_llm_response(
                "Very long content", 3000, hard_limit - 3000 + 1
            )
        )
        result = await governor.complete(
            [{"role": "user", "content": "test"}],
            mode=QueryMode.STANDARD,
        )
        assert result.content == TRUNCATION_FALLBACK
        assert result.usage.truncated is True

    @pytest.mark.asyncio
    async def test_hard_limit_returns_fallback(self, governor):
        hard_limit = LIMITS[QueryMode.STANDARD]["hard"]
        governor._client.complete = AsyncMock(
            return_value=_make_mock_llm_response("Content", 2000, hard_limit - 2000 + 1)
        )
        result = await governor.complete(
            [{"role": "user", "content": "test"}],
            mode=QueryMode.STANDARD,
        )
        assert result.content == TRUNCATION_FALLBACK
        assert result.usage.truncated is True

    @pytest.mark.asyncio
    async def test_cot_mode_uses_cot_limits(self, governor):
        create_mock = AsyncMock(
            return_value=_make_mock_llm_response("Step 1... Step 2...", 200, 400)
        )
        governor._client.complete = create_mock
        await governor.complete(
            [{"role": "user", "content": "think step by step"}],
            mode=QueryMode.CHAIN_OF_THOUGHT,
        )
        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["max_tokens"] == LIMITS[QueryMode.CHAIN_OF_THOUGHT]["soft"]
        assert call_kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_cost_is_zero_for_free_providers(self, governor):
        governor._client.complete = AsyncMock(
            return_value=_make_mock_llm_response("test", 1000, 1000)
        )
        result = await governor.complete(
            [{"role": "user", "content": "test"}],
            mode=QueryMode.STANDARD,
        )
        assert result.usage.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_session_stats_accumulate(self, governor):
        governor._client.complete = AsyncMock(
            return_value=_make_mock_llm_response("answer", 100, 50)
        )
        for _ in range(3):
            await governor.complete([{"role": "user", "content": "q"}], mode=QueryMode.EDGE)
        stats = governor.session_stats()
        assert stats["session_tokens"] == 450
        assert stats["provider"] == "MagicMock"

    @pytest.mark.asyncio
    async def test_llm_exception_propagates(self, governor):
        governor._client.complete = AsyncMock(
            side_effect=Exception("Groq API timeout")
        )
        with pytest.raises(Exception, match="Groq API timeout"):
            await governor.complete(
                [{"role": "user", "content": "test"}],
                mode=QueryMode.STANDARD,
            )
