# ============================================================
# core/kafka/producer.py
# Exactly-once Kafka producer — P0 fix for idempotency gap
# enable.idempotence=true + acks=all + max.in.flight=1
# ============================================================

import json
import time
from typing import Any, Optional, Callable
from confluent_kafka import Producer, KafkaError, KafkaException
from shared.config.settings import settings
from shared.utils.correlation import get_correlation_id, get_tenant_id, build_kafka_headers
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class SIKafkaProducer:
    """
    Exactly-once Kafka producer for the Core fusion layer.

    Key config:
        enable.idempotence=true    — deduplicates retried messages at broker
        acks=all                   — all ISR replicas must ack
        max.in.flight.requests=1   — ordered delivery guarantee
        transactional.id           — enables exactly-once across partitions
    """

    def __init__(self):
        self._producer = Producer(self._build_config())
        self._producer.init_transactions()
        self._message_count = 0
        self._error_count = 0
        logger.info("kafka_producer_initialized", extra={
            "bootstrap_servers": settings.kafka.bootstrap_servers,
            "transactional_id": settings.kafka.transactional_id,
        })

    def _build_config(self) -> dict:
        return {
            "bootstrap.servers":                     settings.kafka.bootstrap_servers,
            "enable.idempotence":                    True,
            "acks":                                  "all",
            "max.in.flight.requests.per.connection": 1,
            "retries":                               2_147_483_647,
            "delivery.timeout.ms":                   120_000,
            "transactional.id":                      settings.kafka.transactional_id,
            "compression.type":                      "snappy",
            "linger.ms":                             5,
            "batch.size":                            65536,
        }

    def _delivery_callback(self, err: Optional[KafkaError], msg: Any) -> None:
        if err:
            self._error_count += 1
            logger.error("kafka_delivery_failed", extra={
                "topic":  msg.topic(),
                "error":  str(err),
                "correlation_id": get_correlation_id(),
            })
        else:
            self._message_count += 1
            logger.debug("kafka_delivered", extra={
                "topic":     msg.topic(),
                "partition": msg.partition(),
                "offset":    msg.offset(),
            })

    def produce(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ) -> None:
        """
        Produce a single message inside a transaction.
        Always includes correlation context in headers.
        """
        headers = build_kafka_headers()
        if extra_headers:
            headers += [(k, v.encode() if isinstance(v, str) else v)
                        for k, v in extra_headers.items()]

        # Inject correlation ID into payload as well (belt + suspenders)
        payload = {**value, "_correlation_id": get_correlation_id(), "_tenant_id": get_tenant_id()}

        try:
            self._producer.begin_transaction()
            self._producer.produce(
                topic=topic,
                key=(key or get_correlation_id()).encode(),
                value=json.dumps(payload, default=str).encode(),
                headers=headers,
                on_delivery=self._delivery_callback,
            )
            self._producer.commit_transaction()
        except KafkaException as e:
            logger.error("kafka_transaction_failed", extra={"error": str(e)})
            try:
                self._producer.abort_transaction()
            except KafkaException:
                pass
            raise

    def produce_batch(self, topic: str, messages: list[dict[str, Any]]) -> None:
        """
        Produce a batch of messages in a single transaction.
        All succeed or all are rolled back — true exactly-once.
        """
        if not messages:
            return

        try:
            self._producer.begin_transaction()
            for msg in messages:
                payload = {
                    **msg,
                    "_correlation_id": get_correlation_id(),
                    "_tenant_id": get_tenant_id(),
                }
                self._producer.produce(
                    topic=topic,
                    key=msg.get("doc_id", get_correlation_id()).encode(),
                    value=json.dumps(payload, default=str).encode(),
                    headers=build_kafka_headers(),
                    on_delivery=self._delivery_callback,
                )
            self._producer.flush(timeout=30)
            self._producer.commit_transaction()
            logger.info("kafka_batch_produced", extra={
                "topic": topic,
                "count": len(messages),
            })
        except KafkaException as e:
            logger.error("kafka_batch_failed", extra={"error": str(e)})
            try:
                self._producer.abort_transaction()
            except KafkaException:
                pass
            raise

    def flush(self, timeout: float = 30.0) -> None:
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning("kafka_flush_incomplete", extra={"remaining": remaining})

    def stats(self) -> dict:
        return {
            "messages_delivered": self._message_count,
            "errors": self._error_count,
        }

    def close(self) -> None:
        self.flush()
        logger.info("kafka_producer_closed", extra=self.stats())


# Module-level singleton
_producer: Optional[SIKafkaProducer] = None


def get_producer() -> SIKafkaProducer:
    global _producer
    if _producer is None:
        _producer = SIKafkaProducer()
    return _producer