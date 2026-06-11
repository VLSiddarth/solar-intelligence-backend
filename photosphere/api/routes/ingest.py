# ============================================================
# photosphere/api/routes/ingest.py  — v2 FINAL FIX
#
# CHANGES FROM v1:
#
#   1. publish_to_kafka() now calls producer.produce_async() which
#      runs the blocking confluent-kafka calls in asyncio.to_thread().
#      The event loop is NEVER blocked. Background tasks complete
#      reliably and exceptions surface in docker logs.
#
#   2. All errors are now explicitly logged AND set doc status to
#      "failed" so the caller can see what happened.
#
#   3. Status tracking uses a Redis key so the AutonomousAgent's
#      _wait_for_processing() can poll and confirm completion.
# ============================================================

import uuid
import asyncio
from typing import List, Optional
from fastapi import APIRouter, Header, BackgroundTasks, HTTPException
from shared.models.entities import IngestRequest
from shared.config.settings import settings
from shared.utils.correlation import get_correlation_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────
# Redis doc-status helpers
# ─────────────────────────────────────────────────────────────

def _redis():
    import redis as r
    return r.Redis(
        host=settings.agent.redis_host,
        port=settings.agent.redis_port,
        decode_responses=True,
        socket_connect_timeout=2,
    )


def _set_status(doc_id: str, status: str, tenant_id: str) -> None:
    try:
        _redis().setex(f"si:doc:{tenant_id}:{doc_id}:status", 86400, status)
    except Exception as e:
        logger.warning("doc_status_write_failed", extra={"doc_id": doc_id, "err": str(e)})


def _get_status(doc_id: str, tenant_id: str) -> Optional[str]:
    try:
        return _redis().get(f"si:doc:{tenant_id}:{doc_id}:status")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Kafka publish — async, non-blocking, fully logged
# ─────────────────────────────────────────────────────────────

async def publish_to_kafka(topic: str, payload: dict, tenant_id: str) -> None:
    """
    Publishes a document payload to Redpanda via the fixed SIKafkaProducer.

    Uses produce_async() which runs blocking confluent-kafka calls in
    asyncio.to_thread() — the FastAPI event loop is never blocked.

    All errors are logged to docker compose logs si-api so nothing is
    silently swallowed.
    """
    doc_id = payload.get("doc_id", "unknown")
    try:
        from core.kafka.producer import get_producer
        producer = get_producer()

        success = await producer.produce_async(
            topic=topic,
            value=payload,
            key=doc_id,
        )

        if success:
            _set_status(doc_id, "pending", tenant_id)
            logger.info("document_published_to_kafka", extra={
                "doc_id": doc_id,
                "topic":  topic,
                "tenant": tenant_id,
            })
        else:
            _set_status(doc_id, "failed", tenant_id)
            logger.error("kafka_publish_incomplete", extra={
                "doc_id": doc_id,
                "topic":  topic,
            })

    except Exception as e:
        _set_status(doc_id, "failed", tenant_id)
        # This log line WILL appear in: docker compose logs si-api
        logger.error("kafka_publish_exception", extra={
            "doc_id":    doc_id,
            "topic":     topic,
            "error":     str(e),
            "error_type": type(e).__name__,
        })


# ─────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────

@router.post("/ingest")
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    x_tenant_id: str = Header(default="default"),
    x_correlation_id: str = Header(default=None),
):
    """
    Ingest a single document into the SI fusion pipeline.
    Returns immediately with a doc_id.
    Poll /v1/ingest/{doc_id}/status to track processing state.
    """
    doc_id         = str(uuid.uuid4())
    correlation_id = x_correlation_id or get_correlation_id() or str(uuid.uuid4())

    payload = {
        "doc_id":         doc_id,
        "tenant_id":      x_tenant_id,
        "title":          request.title,
        "content":        request.content,
        "source_url":     getattr(request, "source_url", "") or "",
        "metadata":       getattr(request, "metadata", {}) or {},
        "correlation_id": correlation_id,
    }

    background_tasks.add_task(
        publish_to_kafka,
        topic=settings.kafka.topic_raw_docs,   # "si.core.raw_documents"
        payload=payload,
        tenant_id=x_tenant_id,
    )

    return {
        "status":         "queued",
        "doc_id":         doc_id,
        "tenant_id":      x_tenant_id,
        "correlation_id": correlation_id,
        "document_title": request.title,
        "status_url":     f"/v1/ingest/{doc_id}/status",
    }


@router.get("/ingest/{doc_id}/status")
async def get_ingest_status(
    doc_id: str,
    x_tenant_id: str = Header(default="default"),
):
    """
    Check document processing status.

    Status values:
      pending   → message is in Redpanda, si-worker hasn't finished yet
      processed → embedding stored + GraphRAG entities written to FalkorDB
      failed    → Kafka publish or worker processing failed (check docker logs)
      unknown   → doc_id not found (expired TTL or invalid)
    """
    status = _get_status(doc_id, x_tenant_id)
    return {
        "doc_id":    doc_id,
        "status":    status or "unknown",
        "tenant_id": x_tenant_id,
    }


@router.post("/ingest/batch")
async def ingest_documents_batch(
    requests: List[IngestRequest],
    background_tasks: BackgroundTasks,
    x_tenant_id: str = Header(default="default"),
    x_correlation_id: str = Header(default=None),
):
    """Ingest multiple documents in a single request."""
    correlation_id = x_correlation_id or get_correlation_id() or str(uuid.uuid4())
    doc_ids        = []

    for req in requests:
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        payload = {
            "doc_id":         doc_id,
            "tenant_id":      x_tenant_id,
            "title":          req.title,
            "content":        req.content,
            "source_url":     getattr(req, "source_url", "") or "",
            "metadata":       getattr(req, "metadata", {}) or {},
            "correlation_id": correlation_id,
        }
        background_tasks.add_task(
            publish_to_kafka,
            topic=settings.kafka.topic_raw_docs,
            payload=payload,
            tenant_id=x_tenant_id,
        )

    return {
        "status":         "queued",
        "doc_ids":        doc_ids,
        "tenant_id":      x_tenant_id,
        "correlation_id": correlation_id,
        "count":          len(doc_ids),
        "status_urls":    [f"/v1/ingest/{did}/status" for did in doc_ids],
    }


@router.post("/ingest/{doc_id}/mark-processed")
async def mark_processed(
    doc_id: str,
    x_tenant_id: str = Header(default="default"),
):
    """
    Called by si-worker after successful document processing.
    Transitions status from 'pending' → 'processed'.
    The AutonomousAgent polls /v1/ingest/{doc_id}/status every 5s
    waiting for this transition.
    """
    _set_status(doc_id, "processed", x_tenant_id)
    return {"doc_id": doc_id, "status": "processed"}