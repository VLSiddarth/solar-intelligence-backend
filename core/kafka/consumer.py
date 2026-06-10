# ============================================================
# core/kafka/consumer.py
# Exactly-once Kafka consumer — reads raw docs, restores correlation context
# ============================================================

import json
import signal
import time
from typing import Callable, Optional
from confluent_kafka import Consumer, KafkaError, KafkaException, Message
from shared.config.settings import settings
from shared.utils.correlation import extract_kafka_headers, inject_correlation_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class SIKafkaConsumer:
    """
    Exactly-once consumer for the Core ingestion pipeline.
    Restores correlation context from Kafka headers on every message.
    """

    def __init__(
        self,
        topics: list[str],
        group_id: Optional[str] = None,
        auto_offset_reset: str = "earliest",
    ):
        self._topics = topics
        self._consumer = Consumer(self._build_config(group_id or settings.kafka.consumer_group, auto_offset_reset))
        self._consumer.subscribe(topics)
        self._running = False
        self._message_count = 0
        logger.info("kafka_consumer_initialized", extra={
            "topics": topics,
            "group_id": group_id or settings.kafka.consumer_group,
        })

    def _build_config(self, group_id: str, auto_offset_reset: str) -> dict:
        return {
            "bootstrap.servers":        settings.kafka.bootstrap_servers,
            "group.id":                 group_id,
            "auto.offset.reset":        auto_offset_reset,
            "enable.auto.commit":       False,   # Manual commit = exactly-once on consumer side
            "isolation.level":          "read_committed",  # Only read committed transactions
            "max.poll.interval.ms":     300_000,
            "session.timeout.ms":       30_000,
            "fetch.min.bytes":          1,
            "fetch.wait.max.ms":        500,
        }

    def consume(
        self,
        handler: Callable[[dict], None],
        poll_timeout: float = 1.0,
        max_messages: Optional[int] = None,
    ) -> None:
        """
        Main consume loop. Calls handler for each message.
        Commits offset only after handler succeeds (at-least-once processing
        combined with idempotent handler logic = effectively exactly-once).
        """
        self._running = True

        def _shutdown(sig, frame):
            logger.info("consumer_shutdown_signal_received")
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        logger.info("consumer_loop_started", extra={"topics": self._topics})

        while self._running:
            msg: Optional[Message] = self._consumer.poll(timeout=poll_timeout)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("kafka_consumer_error", extra={"error": str(msg.error())})
                continue

            try:
                # Restore correlation context from message headers
                extract_kafka_headers(msg.headers() or [])

                payload = json.loads(msg.value().decode())
                handler(payload)

                # Commit only on success
                self._consumer.commit(msg, asynchronous=False)
                self._message_count += 1

                if max_messages and self._message_count >= max_messages:
                    logger.info("consumer_max_messages_reached", extra={"count": self._message_count})
                    break

            except Exception as e:
                logger.error("kafka_message_handler_failed", extra={
                    "error": str(e),
                    "topic": msg.topic(),
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                })
                # Don't commit — message will be redelivered
                # For poison pills, implement a DLQ here

    def close(self) -> None:
        self._running = False
        self._consumer.close()
        logger.info("kafka_consumer_closed", extra={"messages_processed": self._message_count})