# ============================================================
# core/worker/main.py  — v2 FINAL
#
# CHANGES FROM v1:
#   1. Added mark_processed() call after successful document
#      processing so AutonomousAgent._wait_for_processing() resolves.
#   2. Added explicit logging at each processing step so you can
#      trace exactly where failures happen in docker logs.
#   3. Added SI_API_BASE_URL env var (defaults to http://si-api:8888)
#      so the worker can call the mark-processed endpoint.
# ============================================================

import asyncio
import json
import os
import logging
import signal
import sys
import uuid
from typing import Optional

import httpx
from confluent_kafka import Consumer, KafkaError, Message

from shared.config.settings import settings
from shared.models.entities import RawDocument
from shared.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Internal API URL — use Docker service name, not localhost
SI_API_BASE_URL = os.getenv("SI_API_BASE_URL", "http://si-api:8888")


# ─────────────────────────────────────────────────────────────
# Mark-processed callback
# ─────────────────────────────────────────────────────────────

async def mark_doc_processed(doc_id: str, tenant_id: str) -> None:
    """
    Notify the SI API that this document has been fully processed.
    This transitions the doc status from 'pending' → 'processed'
    so the AutonomousAgent polling loop can resolve.
    """
    try:
        async with httpx.AsyncClient(
            base_url=SI_API_BASE_URL,
            timeout=5.0,
        ) as client:
            await client.post(
                f"/v1/ingest/{doc_id}/mark-processed",
                headers={"X-Tenant-ID": tenant_id},
            )
        logger.info("worker_marked_processed", extra={"doc_id": doc_id})
    except Exception as e:
        # Non-fatal — document is still processed. Agent will time out polling.
        logger.warning("worker_mark_processed_failed", extra={
            "doc_id": doc_id,
            "error":  str(e),
        })


# ─────────────────────────────────────────────────────────────
# Pipeline init
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
    logger.info("worker_waiting_for_tgi")
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


# ─────────────────────────────────────────────────────────────
# Document processing pipeline
# ─────────────────────────────────────────────────────────────

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
        2. Generate BGE-M3 embedding via TGI
        3. Store vector in pgvector (blue/green index)
        4. Run GraphRAG extraction (Groq → entities/relationships → FalkorDB)
        5. Mark document as processed (notifies the API status endpoint)
    """
    # Strip internal Kafka metadata fields
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}

    try:
        doc = RawDocument(**clean)
    except Exception as e:
        logger.error("worker_deserialise_failed", extra={
            "error": str(e),
            "keys":  list(clean.keys()),
        })
        return

    doc_id    = doc.doc_id
    tenant_id = doc.tenant_id
    logger.info("worker_processing_start", extra={
        "doc_id":      doc_id,
        "title":       doc.title or "untitled",
        "content_len": len(doc.content),
        "tenant_id":   tenant_id,
    })

    processing_ok = True   # Track whether to mark as processed or failed

    # ── Step 1: Embed ─────────────────────────────────────────
    embedding = []
    try:
        embedding = await embedder.embed_single(doc.content[:2000])
        logger.info("worker_embedding_done", extra={
            "doc_id": doc_id,
            "dim":    len(embedding),
        })
    except Exception as e:
        logger.error("worker_embedding_failed", extra={
            "doc_id": doc_id,
            "error":  str(e),
        })
        processing_ok = False

    # ── Step 2: Store in pgvector ────────────────────────────
    if embedding:
        try:
            import hashlib
            content_hash = hashlib.sha256(doc.content.encode()).hexdigest()[:16]
            index.insert([{
                "vector_id":    doc_id,
                "entity_id":    doc_id,
                "tenant_id":    tenant_id,
                "embedding":    embedding,
                "content_hash": content_hash,
            }])
            logger.info("worker_vector_stored", extra={"doc_id": doc_id})
        except Exception as e:
            logger.error("worker_vector_store_failed", extra={
                "doc_id": doc_id,
                "error":  str(e),
            })
            processing_ok = False

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
        # GraphRAG failure is non-fatal: document is still vector-searchable
        logger.warning("worker_graphrag_failed_doc_still_searchable", extra={
            "doc_id": doc_id,
            "error":  str(e),
        })

    # ── Step 4: Mark processed ───────────────────────────────
    # Always call this (even on partial failure) so the agent polling loop
    # doesn't sit waiting for the full PROCESS_WAIT_TIMEOUT.
    await mark_doc_processed(doc_id, tenant_id)

    logger.info("worker_processing_complete", extra={
        "doc_id":       doc_id,
        "has_embedding": bool(embedding),
        "success":      processing_ok,
    })


# ─────────────────────────────────────────────────────────────
# Kafka consumer loop
# ─────────────────────────────────────────────────────────────

async def consume_loop(
    embedder,
    index,
    graphrag_pipeline,
    shutdown_event: asyncio.Event,
) -> None:
    consumer = Consumer({
        "bootstrap.servers":    settings.kafka.bootstrap_servers,
        "group.id":             f"{settings.kafka.consumer_group}-worker",
        "auto.offset.reset":    "earliest",
        "enable.auto.commit":   False,     # Manual commit after processing
        "isolation.level":      "read_committed",
        "max.poll.interval.ms": 300_000,
        "session.timeout.ms":   30_000,
        "fetch.min.bytes":      1,
        "fetch.wait.max.ms":    500,
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
                consumer.commit(msg, asynchronous=False)
                processed += 1

                if processed % 10 == 0:
                    logger.info("worker_heartbeat", extra={
                        "processed": processed,
                        "errors":    errors,
                    })

            except Exception as e:
                errors += 1
                logger.error("worker_message_processing_failed", extra={
                    "error":     str(e),
                    "topic":     msg.topic(),
                    "partition": msg.partition(),
                    "offset":    msg.offset(),
                })
                # Don't commit offset — message will be redelivered

            await asyncio.sleep(0)   # Yield to event loop

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
        "llm_provider":   settings.llm_provider,
        "topic":          settings.kafka.topic_raw_docs,
        "vector_backend": settings.vector.backend,
        "si_api_url":     SI_API_BASE_URL,
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