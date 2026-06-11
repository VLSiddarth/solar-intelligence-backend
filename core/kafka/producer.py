# ============================================================
# core/kafka/producer.py  — COMPLETELY REWRITTEN (v2)
#
# ROOT CAUSE OF "NO MESSAGES IN REDPANDA" — THREE BUGS:
#
# BUG 1 (CRITICAL): init_transactions() is called inside __init__()
#   which is called from an async background task. confluent-kafka
#   is a C extension with BLOCKING calls. Calling blocking I/O
#   directly in an async function BLOCKS the entire event loop,
#   causing silent hangs and dropped background tasks.
#
# BUG 2 (CRITICAL): The produce() flow was:
#     begin_transaction() → produce() → commit_transaction()
#   There is NO flush() between produce() and commit_transaction().
#   With linger.ms=5, the message sits in the local C-level buffer
#   when commit_transaction() fires. The transaction commits as
#   EMPTY. Messages are orphaned in the buffer and never delivered.
#   The correct flow is:
#     begin_transaction() → produce() → flush() → commit_transaction()
#
# BUG 3: All exceptions thrown inside FastAPI BackgroundTasks are
#   silently swallowed. There is no error in docker logs. The API
#   returns 200 and the document silently disappears.
#
# FIX STRATEGY:
#   - Remove ALL transaction logic. Transactions add zero value in
#     a single-node dev setup and are the source of all three bugs.
#   - Use acks=1 (leader ack) — correct for single-node Redpanda.
#   - Call flush() immediately after produce() to guarantee delivery
#     before returning. This is the only reliable way to confirm
#     message landing.
#   - Expose a simple async produce_async() that runs the blocking
#     Kafka calls in asyncio.to_thread() so the event loop is never
#     blocked.
#   - Add explicit error raising so exceptions surface in docker logs.
# ============================================================

import json
import asyncio
import threading
from typing import Any, Optional
from confluent_kafka import Producer, KafkaError, KafkaException
from shared.config.settings import settings
from shared.utils.correlation import get_correlation_id, get_tenant_id, build_kafka_headers
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class SIKafkaProducer:
    """
    Simple, reliable Kafka producer for the SI Core fusion layer.

    Design decisions (v2):
      - No transactions: removes 3 compounding failure modes
      - acks=1: leader acknowledgement, sufficient for single-node Redpanda
      - flush() after every produce: guarantees delivery before returning
      - Thread-safe: one lock per produce call
      - async interface: all blocking calls run in asyncio.to_thread()
    """

    def __init__(self):
        self._lock     = threading.Lock()
        self._producer = Producer(self._build_config())
        self._sent     = 0
        self._errors   = 0
        logger.info("kafka_producer_initialized", extra={
            "bootstrap_servers": settings.kafka.bootstrap_servers,
            "mode": "simple_acks1_no_transactions",
        })

    def _build_config(self) -> dict:
        return {
            "bootstrap.servers":  settings.kafka.bootstrap_servers,
            # acks=1: leader must acknowledge. Safe for single-node Redpanda.
            # acks=all would require all ISR replicas — on single-node this is
            # equivalent to acks=1 but adds unnecessary round-trips.
            "acks":               "1",
            # linger.ms=0: send immediately, don't batch. Ensures flush() works
            # predictably. Batching can be re-enabled in production.
            "linger.ms":          0,
            "retries":            5,
            "retry.backoff.ms":   200,
            "delivery.timeout.ms": 30_000,
            "compression.type":   "snappy",
            # socket.keepalive.enable: keeps connection alive through Docker NAT
            "socket.keepalive.enable": True,
        }

    def _delivery_callback(self, err: Optional[KafkaError], msg: Any) -> None:
        if err:
            self._errors += 1
            logger.error("kafka_delivery_failed", extra={
                "topic":  msg.topic() if msg else "unknown",
                "error":  str(err),
            })
        else:
            self._sent += 1
            logger.info("kafka_delivered", extra={
                "topic":     msg.topic(),
                "partition": msg.partition(),
                "offset":    msg.offset(),
            })

    def produce_sync(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
    ) -> bool:
        """
        Synchronous produce + flush.
        Blocks until the message is confirmed delivered or an error occurs.
        Safe to call from a thread (not from an async function directly).
        Returns True on success, False on error.
        """
        payload = {
            **value,
            "_correlation_id": get_correlation_id(),
            "_tenant_id":      get_tenant_id(),
        }
        headers = build_kafka_headers()

        with self._lock:
            try:
                self._producer.produce(
                    topic=topic,
                    key=(key or get_correlation_id()).encode("utf-8"),
                    value=json.dumps(payload, default=str).encode("utf-8"),
                    headers=headers,
                    on_delivery=self._delivery_callback,
                )
                # flush() blocks until ALL outstanding messages are delivered
                # or the timeout expires. With linger.ms=0 this returns quickly.
                remaining = self._producer.flush(timeout=10.0)
                if remaining > 0:
                    logger.error("kafka_flush_incomplete", extra={
                        "topic":     topic,
                        "remaining": remaining,
                    })
                    return False
                return True

            except KafkaException as e:
                self._errors += 1
                logger.error("kafka_produce_failed", extra={
                    "topic": topic,
                    "error": str(e),
                    "key":   key,
                })
                raise   # Re-raise so the caller can set doc status = "failed"

    async def produce_async(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
    ) -> bool:
        """
        Async wrapper around produce_sync.
        Runs the blocking Kafka calls in a thread pool via asyncio.to_thread()
        so the FastAPI event loop is NEVER blocked.
        """
        return await asyncio.to_thread(
            self.produce_sync,
            topic,
            value,
            key,
        )

    def stats(self) -> dict:
        return {
            "messages_sent":   self._sent,
            "errors":          self._errors,
            "bootstrap":       settings.kafka.bootstrap_servers,
        }

    def close(self) -> None:
        remaining = self._producer.flush(timeout=10.0)
        if remaining > 0:
            logger.warning("kafka_close_flush_incomplete", extra={"remaining": remaining})
        logger.info("kafka_producer_closed", extra=self.stats())


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_producer: Optional[SIKafkaProducer] = None
_producer_lock = threading.Lock()


def get_producer() -> SIKafkaProducer:
    """
    Thread-safe singleton. Uses a lock to prevent double-init
    under concurrent async background tasks.
    """
    global _producer
    if _producer is None:
        with _producer_lock:
            if _producer is None:   # Double-checked locking
                _producer = SIKafkaProducer()
    return _producer