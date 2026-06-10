# ============================================================
# core/worker/main.py
# Fusion Worker — the missing link between Kafka and the knowledge graph.
#
# This is WHY ingest returns "queued" but nothing appeared in Redpanda:
# the API only PRODUCES to Kafka. This worker CONSUMES and processes.
#
# Flow per document:
#   Kafka (si.core.raw_documents)
#     → embed via TGI (BGE-M3)
#     → store vector in pgvector (active blue/green index)
#     → GraphRAG extraction via Groq
#     → write entities + edges to FalkorDB
#     → commit Kafka offset (exactly-once)
# ============================================================

import asyncio
import json
import logging
import signal
import sys
import uuid
from typing import Optional

from confluent_kafka import Consumer, KafkaError, Message

from shared.config.settings import settings
from shared.models.entities import RawDocument
from shared.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Async processing pipeline
# ─────────────────────────────────────────────────────────────

async def build_pipeline():
    """
    Lazily initialise heavy clients once so the worker loop stays clean.
    Returns (embedder, index, graphrag_pipeline).
    """
    from radiative.embeddings.client import get_embedding_client
    from radiative.vector.index_manager import BlueGreenIndexManager
    from core.graphrag.pipeline import GraphRAGPipeline

    embedder = get_embedding_client()
    index    = BlueGreenIndexManager()
    pipeline = GraphRAGPipeline()

    # Wait for TGI to be ready (it can take a few minutes on first boot)
    for attempt in range(30):
        healthy = await embedder.health()
        if healthy:
            logger.info("tgi_embedding_server_ready")
            break
        logger.info("waiting_for_tgi", extra={"attempt": attempt + 1})
        await asyncio.sleep(10)
    else:
        logger.error("tgi_never_became_healthy_embedding_disabled")

    return embedder, index, pipeline


async def process_document(
    payload: dict,
    embedder,
    index,
    graphrag_pipeline,
) -> None:
    """
    Full processing pipeline for a single raw document.

    Steps:
        1. Deserialise RawDocument from Kafka payload
        2. Generate 1024-dim BGE-M3 embedding
        3. Store in active pgvector index
        4. Run GraphRAG extraction (Groq → entities/relationships → FalkorDB)
    """
    # Strip internal Kafka headers that aren't part of RawDocument schema
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}

    try:
        doc = RawDocument(**clean)
    except Exception as e:
        logger.error("worker_deserialise_failed", extra={"error": str(e), "keys": list(clean.keys())})
        return

    doc_id = doc.doc_id
    logger.info("worker_processing", extra={
        "doc_id":      doc_id,
        "title":       doc.title or "untitled",
        "content_len": len(doc.content),
        "tenant_id":   doc.tenant_id,
    })

    # ── Step 1: Embed ─────────────────────────────────────────
    try:
        embedding = await embedder.embed_single(doc.content[:2000])
        logger.info("worker_embedding_done", extra={"doc_id": doc_id, "dim": len(embedding)})
    except Exception as e:
        logger.error("worker_embedding_failed", extra={"doc_id": doc_id, "error": str(e)})
        embedding = []

    # ── Step 2: Store in pgvector ────────────────────────────
    if embedding:
        try:
            import hashlib
            content_hash = hashlib.sha256(doc.content.encode()).hexdigest()[:16]
            index.insert([{
                "vector_id":    doc_id,
                "entity_id":    doc_id,
                "tenant_id":    doc.tenant_id,
                "embedding":    embedding,
                "content_hash": content_hash,
            }])
            logger.info("worker_vector_stored", extra={"doc_id": doc_id})
        except Exception as e:
            logger.error("worker_vector_store_failed", extra={"doc_id": doc_id, "error": str(e)})

    # ── Step 3: GraphRAG extraction ──────────────────────────
    try:
        fused = await graphrag_pipeline.ingest(doc)
        logger.info("worker_graphrag_done", extra={
            "doc_id":        doc_id,
            "entities":      len(fused.entities),
            "relationships": len(fused.relationships),
            "cost":          fused.token_cost,
        })
    except Exception as e:
        # GraphRAG failure is non-fatal: document is still searchable via vector
        logger.warning("worker_graphrag_failed_doc_still_searchable", extra={
            "doc_id": doc_id,
            "error":  str(e),
        })


# ─────────────────────────────────────────────────────────────
# Kafka consumer loop (async)
# ─────────────────────────────────────────────────────────────

async def consume_loop(
    embedder,
    index,
    graphrag_pipeline,
    shutdown_event: asyncio.Event,
) -> None:
    consumer = Consumer({
        "bootstrap.servers":        settings.kafka.bootstrap_servers,
        "group.id":                 f"{settings.kafka.consumer_group}-worker",
        "auto.offset.reset":        "earliest",
        "enable.auto.commit":       False,   # Manual commit after successful processing
        "isolation.level":          "read_committed",  # Only see committed txn messages
        "max.poll.interval.ms":     300_000,
        "session.timeout.ms":       30_000,
        "fetch.min.bytes":          1,
        "fetch.wait.max.ms":        500,
    })
    consumer.subscribe([settings.kafka.topic_raw_docs])

    logger.info("fusion_worker_consume_loop_started", extra={
        "topic":    settings.kafka.topic_raw_docs,
        "brokers":  settings.kafka.bootstrap_servers,
        "group_id": f"{settings.kafka.consumer_group}-worker",
    })

    processed = 0
    errors    = 0

    try:
        while not shutdown_event.is_set():
            msg: Optional[Message] = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("kafka_consumer_error", extra={"error": str(msg.error())})
                errors += 1
                continue

            try:
                payload = json.loads(msg.value().decode("utf-8"))
                await process_document(payload, embedder, index, graphrag_pipeline)
                consumer.commit(msg, asynchronous=False)  # Exactly-once: commit after success
                processed += 1

                if processed % 10 == 0:
                    logger.info("worker_heartbeat", extra={
                        "processed": processed,
                        "errors":    errors,
                    })

            except Exception as e:
                errors += 1
                logger.error("worker_message_failed", extra={
                    "error":     str(e),
                    "topic":     msg.topic(),
                    "partition": msg.partition(),
                    "offset":    msg.offset(),
                })
                # Do NOT commit — message will be redelivered

            # Yield control so other async tasks can run
            await asyncio.sleep(0)

    finally:
        consumer.close()
        logger.info("fusion_worker_stopped", extra={
            "processed": processed,
            "errors":    errors,
        })


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

async def main() -> None:
    configure_logging(level=settings.log_level, json_output=settings.is_production)

    logger.info("fusion_worker_starting", extra={
        "llm_provider": settings.llm_provider,
        "topic":        settings.kafka.topic_raw_docs,
        "vector_backend": settings.vector.backend,
    })

    shutdown_event = asyncio.Event()

    def _handle_signal(sig, frame):
        logger.info("shutdown_signal_received", extra={"signal": sig})
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        embedder, index, graphrag_pipeline = await build_pipeline()
        await consume_loop(embedder, index, graphrag_pipeline, shutdown_event)
    except Exception as e:
        logger.error("fusion_worker_fatal_error", extra={"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())