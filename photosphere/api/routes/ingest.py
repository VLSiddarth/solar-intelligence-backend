# ============================================================
# photosphere/api/routes/ingest.py  — FIXED
#
# BUG FIXED: publish_to_queue() was a STUB that only logged.
#   It never called SIKafkaProducer.produce() so documents
#   never reached Redpanda → si-worker never processed them →
#   vector store stayed empty → queries returned generic answers.
#
# BUG FIXED: Topic was "document_ingestion" — wrong.
#   Correct topic: settings.kafka.topic_raw_docs = "si.core.raw_documents"
#
# NEW: /ingest/{doc_id}/status endpoint added.
#   AutonomousAgent polls this to know when fusion worker is done.
#   Status persisted in Redis with 24h TTL.
# ============================================================

import uuid
from typing import List, Optional
from fastapi import APIRouter, Header, BackgroundTasks, HTTPException
from shared.models.entities import IngestRequest
from shared.config.settings import settings
from shared.utils.correlation import get_correlation_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────
# Redis status tracker helpers
# ─────────────────────────────────────────────────────────────

def _get_redis():
    import redis as redis_lib
    return redis_lib.Redis(
        host=settings.agent.redis_host,
        port=settings.agent.redis_port,
        decode_responses=True,
    )


def _set_doc_status(doc_id: str, status: str, tenant_id: str) -> None:
    """Write doc processing status to Redis. TTL = 24h."""
    try:
        r = _get_redis()
        r.setex(f"si:doc:{tenant_id}:{doc_id}:status", 86400, status)
    except Exception as e:
        logger.warning("status_redis_write_failed", extra={"doc_id": doc_id, "error": str(e)})


def _get_doc_status(doc_id: str, tenant_id: str) -> Optional[str]:
    """Read doc processing status from Redis."""
    try:
        r = _get_redis()
        return r.get(f"si:doc:{tenant_id}:{doc_id}:status")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Real Kafka publish
# ─────────────────────────────────────────────────────────────

async def publish_to_kafka(topic: str, payload: dict, tenant_id: str) -> None:
    """
    FIXED: Actually produces the message to Redpanda via SIKafkaProducer.
    Old version only logged — documents never reached the fusion worker.
    """
    doc_id = payload.get("doc_id", "unknown")
    try:
        from core.kafka.producer import get_producer
        producer = get_producer()
        producer.produce(
            topic=topic,
            value=payload,
            key=doc_id,
        )
        _set_doc_status(doc_id, "pending", tenant_id)
        logger.info("document_published_to_kafka", extra={
            "doc_id": doc_id,
            "topic":  topic,
            "tenant": tenant_id,
        })
    except Exception as e:
        _set_doc_status(doc_id, "failed", tenant_id)
        logger.error("kafka_publish_failed", extra={
            "error":  str(e),
            "doc_id": doc_id,
            "topic":  topic,
        })


# ─────────────────────────────────────────────────────────────
# Routes
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
    The fusion worker processes the doc asynchronously via Kafka.
    Poll /v1/ingest/{doc_id}/status to track processing.
    """
    doc_id         = str(uuid.uuid4())
    correlation_id = x_correlation_id or get_correlation_id() or str(uuid.uuid4())

    payload = {
        "doc_id":         doc_id,
        "tenant_id":      x_tenant_id,
        "title":          request.title,
        "content":        request.content,
        "source_url":     getattr(request, "source_url", ""),
        "metadata":       getattr(request, "metadata", {}),
        "correlation_id": correlation_id,
    }

    # FIXED: Use correct Kafka topic from settings (not hardcoded "document_ingestion")
    background_tasks.add_task(
        publish_to_kafka,
        topic=settings.kafka.topic_raw_docs,  # "si.core.raw_documents"
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
    NEW ENDPOINT: Track document processing status.
    Status values:
      - pending   → in Kafka queue, fusion worker not done yet
      - processed → embedding + GraphRAG extraction complete
      - failed    → processing error (doc still searchable via keyword)
      - unknown   → doc_id not found (expired or invalid)
    
    AutonomousAgent polls this every 5 seconds after ingestion.
    """
    status = _get_doc_status(doc_id, x_tenant_id)
    if status is None:
        return {"doc_id": doc_id, "status": "unknown", "tenant_id": x_tenant_id}
    return {"doc_id": doc_id, "status": status, "tenant_id": x_tenant_id}


@router.post("/ingest/batch")
async def ingest_documents_batch(
    requests: List[IngestRequest],
    background_tasks: BackgroundTasks,
    x_tenant_id: str = Header(default="default"),
    x_correlation_id: str = Header(default=None),
):
    """Ingest multiple documents in one call."""
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
            "source_url":     getattr(req, "source_url", ""),
            "metadata":       getattr(req, "metadata", {}),
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
async def mark_document_processed(
    doc_id: str,
    x_tenant_id: str = Header(default="default"),
):
    """
    Internal endpoint called by the fusion worker after successful processing.
    This updates the status so AutonomousAgent.wait_for_processing() resolves.
    """
    _set_doc_status(doc_id, "processed", x_tenant_id)
    return {"doc_id": doc_id, "status": "processed"}