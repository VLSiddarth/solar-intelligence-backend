# ============================================================
# tests/integration/test_webhook_delivery.py
# Integration tests — Webhook delivery engine
# Uses aiohttp mock server to simulate endpoints
# ============================================================

import asyncio
import json
import pytest
from aiohttp import web
from unittest.mock import AsyncMock, MagicMock, patch
from corona.webhooks.delivery import WebhookDeliveryEngine
from shared.models.entities import WebhookEvent


def make_event(url: str, tenant: str = "t1", payload: dict = None) -> WebhookEvent:
    return WebhookEvent(
        url=url,
        payload=payload or {"event": "test", "data": "value"},
        tenant_id=tenant,
        correlation_id="test-cid-123",
        max_attempts=3,
    )


@pytest.fixture
def engine():
    with patch("corona.webhooks.delivery.get_producer", MagicMock()):
        return WebhookDeliveryEngine()


class TestWebhookDelivery:

    @pytest.mark.asyncio
    async def test_successful_delivery(self, engine):
        """Mock an endpoint that always returns 200."""
        async def handler(request):
            return web.Response(status=200, text="OK")

        app    = web.Application()
        runner = web.AppRunner(app)
        await runner.setup()
        site   = web.TCPSite(runner, "127.0.0.1", 18765)
        app.router.add_post("/webhook", handler)
        await site.start()

        try:
            event = make_event("http://127.0.0.1:18765/webhook")
            ok    = await engine.deliver(event)
            assert ok is True
            assert engine.stats()["delivered"] == 1
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_retry_on_500(self, engine):
        """Endpoint returns 500 twice then 200 — should succeed after 3rd attempt."""
        attempt_count = [0]

        async def handler(request):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                return web.Response(status=500, text="Server Error")
            return web.Response(status=200, text="OK")

        app    = web.Application()
        runner = web.AppRunner(app)
        await runner.setup()
        site   = web.TCPSite(runner, "127.0.0.1", 18766)
        app.router.add_post("/webhook", handler)
        await site.start()

        try:
            # Speed up backoff for testing
            import corona.webhooks.delivery as wd
            original_backoff = wd.BACKOFF_SECONDS
            wd.BACKOFF_SECONDS = [0.01, 0.01, 0.01]

            event = make_event("http://127.0.0.1:18766/webhook")
            ok    = await engine.deliver(event)

            wd.BACKOFF_SECONDS = original_backoff
            assert ok is True
            assert attempt_count[0] == 3
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_dlq_on_all_retries_exhausted(self, engine):
        """Endpoint always returns 500 → all retries exhaust → goes to DLQ."""
        async def handler(request):
            return web.Response(status=500, text="Always fails")

        app    = web.Application()
        runner = web.AppRunner(app)
        await runner.setup()
        site   = web.TCPSite(runner, "127.0.0.1", 18767)
        app.router.add_post("/webhook", handler)
        await site.start()

        try:
            import corona.webhooks.delivery as wd
            wd.BACKOFF_SECONDS = [0.01, 0.01, 0.01]

            event = make_event("http://127.0.0.1:18767/webhook")
            with patch("corona.webhooks.delivery._alert_pagerduty", AsyncMock()):
                ok = await engine.deliver(event)

            assert ok is False
            assert engine.stats()["dlq_size"] >= 1
            assert engine.stats()["failed"] >= 1
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_unreachable_host_goes_to_dlq(self, engine):
        """Connection refused → retries → DLQ."""
        import corona.webhooks.delivery as wd
        wd.BACKOFF_SECONDS = [0.01, 0.01, 0.01]

        event = make_event("http://127.0.0.1:19999/webhook")  # Nothing listening
        with patch("corona.webhooks.delivery._alert_pagerduty", AsyncMock()):
            ok = await engine.deliver(event)
        assert ok is False
        assert engine.stats()["dlq_size"] >= 1

    @pytest.mark.asyncio
    async def test_hmac_signature_present(self, engine):
        """Verify HMAC signature header is sent when secret provided."""
        received_headers = {}

        async def handler(request):
            received_headers.update(dict(request.headers))
            return web.Response(status=200, text="OK")

        app    = web.Application()
        runner = web.AppRunner(app)
        await runner.setup()
        site   = web.TCPSite(runner, "127.0.0.1", 18768)
        app.router.add_post("/webhook", handler)
        await site.start()

        try:
            event = make_event("http://127.0.0.1:18768/webhook")
            await engine.deliver(event, secret="my-webhook-secret")
            assert "X-Si-Signature" in received_headers or "x-si-signature" in received_headers
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_correlation_id_in_headers(self, engine):
        """Verify X-Correlation-ID is passed in webhook headers."""
        received_cid = [None]

        async def handler(request):
            received_cid[0] = request.headers.get("X-Correlation-Id") or \
                               request.headers.get("X-Correlation-ID")
            return web.Response(status=200, text="OK")

        app    = web.Application()
        runner = web.AppRunner(app)
        await runner.setup()
        site   = web.TCPSite(runner, "127.0.0.1", 18769)
        app.router.add_post("/webhook", handler)
        await site.start()

        try:
            event = make_event("http://127.0.0.1:18769/webhook")
            await engine.deliver(event)
            assert received_cid[0] == "test-cid-123"
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_dlq_retry_recovers_events(self, engine):
        """Put event in DLQ manually, then retry → recovers."""
        event = make_event("http://127.0.0.1:18770/retry-test")
        engine._dlq.append(event)
        assert len(engine._dlq) == 1

        async def handler(request):
            return web.Response(status=200, text="OK")

        app    = web.Application()
        runner = web.AppRunner(app)
        await runner.setup()
        site   = web.TCPSite(runner, "127.0.0.1", 18770)
        app.router.add_post("/retry-test", handler)
        await site.start()

        try:
            import corona.webhooks.delivery as wd
            wd.BACKOFF_SECONDS = [0.01, 0.01, 0.01]
            result = await engine.retry_dlq(limit=1)
            assert result["retried"] == 1
        finally:
            await runner.cleanup()

    def test_duplicate_event_not_delivered_twice(self, engine):
        """Same event_id should not be in-flight twice."""
        event = make_event("http://127.0.0.1:19001/test")
        engine._in_flight.add(event.event_id)

        async def run():
            return await engine.deliver(event)

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is False