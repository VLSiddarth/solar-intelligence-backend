# ============================================================
# corona/webhooks/delivery.py
# Webhook delivery with DLQ, retry, and PagerDuty alerting
# P0 fix: failed webhooks no longer cause silent data loss
# Retry: 3 attempts with backoff 1s → 4s → 16s
# ============================================================

import asyncio
import json
import time
import hmac
import hashlib
from typing import Optional

import aiohttp

from shared.config.settings import settings
from shared.models.entities import WebhookEvent
from shared.utils.correlation import get_correlation_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)

BACKOFF_SECONDS = [1, 4, 16]
DELIVERY_TIMEOUT = 10.0


# ─────────────────────────────────────────────
# PagerDuty Alerting
# ─────────────────────────────────────────────

async def _alert_pagerduty(event: WebhookEvent, error: str) -> None:
    """Send a PagerDuty alert when a webhook exhausts all retries."""
    if not settings.webhook.pagerduty_key:
        logger.warning("pagerduty_key_not_configured_skipping_alert")
        return

    payload = {
        "routing_key":  settings.webhook.pagerduty_key,
        "event_action": "trigger",
        "payload": {
            "summary":   f"SI Webhook DLQ: {event.url} failed after {event.max_attempts} attempts",
            "severity":  "error",
            "source":    "solar-intelligence-corona",
            "custom_details": {
                "event_id":       event.event_id,
                "tenant_id":      event.tenant_id,
                "url":            event.url,
                "correlation_id": event.correlation_id,
                "error":          error,
            },
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 202:
                    logger.info("pagerduty_alert_sent", extra={"event_id": event.event_id})
                else:
                    logger.warning("pagerduty_alert_failed", extra={"status": resp.status})
    except Exception as e:
        logger.error("pagerduty_alert_exception", extra={"error": str(e)})


# ─────────────────────────────────────────────
# HMAC Signature
# ─────────────────────────────────────────────

def _sign_payload(payload: dict, secret: str) -> str:
    """Sign webhook payload with HMAC-SHA256 for verification at receiver."""
    body   = json.dumps(payload, sort_keys=True).encode()
    sig    = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


# ─────────────────────────────────────────────
# Webhook Delivery Engine
# ─────────────────────────────────────────────

class WebhookDeliveryEngine:
    """
    Reliable webhook delivery with:
        - 3 retries with exponential backoff (1s → 4s → 16s)
        - Dead letter queue on exhaustion
        - HMAC-SHA256 signature on all deliveries
        - PagerDuty alert when webhook enters DLQ
        - Per-event correlation ID in headers
    """

    def __init__(self):
        self._dlq:      list[WebhookEvent]  = []
        self._delivered: int = 0
        self._failed:    int = 0
        self._in_flight: set[str] = set()

    async def deliver(self, event: WebhookEvent, secret: Optional[str] = None) -> bool:
        """
        Deliver a webhook event with retry logic.
        Returns True if delivered successfully, False if sent to DLQ.
        """
        if event.event_id in self._in_flight:
            logger.warning("webhook_already_in_flight", extra={"event_id": event.event_id})
            return False

        self._in_flight.add(event.event_id)

        try:
            return await self._deliver_with_retry(event, secret)
        finally:
            self._in_flight.discard(event.event_id)

    async def _deliver_with_retry(
        self, event: WebhookEvent, secret: Optional[str]
    ) -> bool:
        last_error = ""

        for attempt in range(event.max_attempts):
            try:
                success = await self._attempt_delivery(event, attempt, secret)
                if success:
                    self._delivered += 1
                    logger.info("webhook_delivered", extra={
                        "event_id":   event.event_id,
                        "url":        event.url,
                        "attempt":    attempt + 1,
                        "tenant_id":  event.tenant_id,
                    })
                    return True
            except Exception as e:
                last_error = str(e)
                logger.warning("webhook_attempt_failed", extra={
                    "event_id":  event.event_id,
                    "url":       event.url,
                    "attempt":   attempt + 1,
                    "error":     last_error,
                })

            # Backoff before next attempt (skip after last attempt)
            if attempt < event.max_attempts - 1:
                backoff = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                logger.debug("webhook_backoff", extra={
                    "event_id": event.event_id,
                    "backoff_s": backoff,
                })
                await asyncio.sleep(backoff)

        # All retries exhausted → DLQ
        await self._send_to_dlq(event, last_error)
        return False

    async def _attempt_delivery(
        self, event: WebhookEvent, attempt: int, secret: Optional[str]
    ) -> bool:
        headers = {
            "Content-Type":           "application/json",
            "X-SI-Event-ID":          event.event_id,
            "X-SI-Tenant-ID":         event.tenant_id,
            "X-Correlation-ID":       event.correlation_id,
            "X-SI-Attempt":           str(attempt + 1),
            "X-SI-Timestamp":         str(int(time.time())),
        }

        if secret:
            headers["X-SI-Signature"] = _sign_payload(event.payload, secret)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                event.url,
                json=event.payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=DELIVERY_TIMEOUT),
            ) as resp:
                if 200 <= resp.status < 300:
                    return True
                body = await resp.text()
                raise ValueError(f"HTTP {resp.status}: {body[:200]}")

    async def _send_to_dlq(self, event: WebhookEvent, error: str) -> None:
        """Send to in-memory DLQ and alert PagerDuty."""
        self._dlq.append(event)
        self._failed += 1

        logger.error("webhook_dlq", extra={
            "event_id":       event.event_id,
            "url":            event.url,
            "tenant_id":      event.tenant_id,
            "attempts":       event.max_attempts,
            "last_error":     error,
            "correlation_id": event.correlation_id,
        })

        # Also produce to Kafka DLQ topic for persistence
        try:
            from core.kafka.producer import get_producer
            from shared.config.settings import settings
            get_producer().produce(
                topic=settings.webhook.dlq_topic,
                value={
                    "event_id":   event.event_id,
                    "url":        event.url,
                    "payload":    event.payload,
                    "error":      error,
                    "tenant_id":  event.tenant_id,
                    "failed_at":  time.time(),
                },
                key=event.event_id,
            )
        except Exception as e:
            logger.error("webhook_dlq_kafka_failed", extra={"error": str(e)})

        # Alert PagerDuty
        asyncio.create_task(_alert_pagerduty(event, error))

    async def retry_dlq(self, limit: int = 10) -> dict:
        """
        Retry items from the dead letter queue.
        Call manually via admin endpoint or on a schedule.
        """
        retried   = 0
        recovered = 0
        items_to_retry = self._dlq[:limit]
        self._dlq       = self._dlq[limit:]

        for event in items_to_retry:
            retried += 1
            # Reset attempt count for DLQ retry
            event.attempt      = 0
            event.max_attempts = 2  # Reduced retries on DLQ replay
            success = await self.deliver(event)
            if success:
                recovered += 1

        return {"retried": retried, "recovered": recovered, "remaining_dlq": len(self._dlq)}

    def stats(self) -> dict:
        return {
            "delivered": self._delivered,
            "failed":    self._failed,
            "dlq_size":  len(self._dlq),
            "in_flight": len(self._in_flight),
        }

    @property
    def dlq(self) -> list[WebhookEvent]:
        return list(self._dlq)


# Singleton
_engine: Optional[WebhookDeliveryEngine] = None


def get_delivery_engine() -> WebhookDeliveryEngine:
    global _engine
    if _engine is None:
        _engine = WebhookDeliveryEngine()
    return _engine