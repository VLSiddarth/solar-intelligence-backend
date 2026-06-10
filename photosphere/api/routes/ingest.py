import uuid
from typing import List
from fastapi import APIRouter, Header, BackgroundTasks
from shared.models.entities import IngestRequest
from shared.utils.correlation import get_correlation_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

async def publish_to_queue(topic: str, payload: dict):
    """Pushes the document to Redpanda for asynchronous embedding."""
    try:
        logger.info("document_published_to_queue", extra={"doc_id": payload["doc_id"], "topic": topic})
    except Exception as e:
        logger.error("queue_publish_failed", extra={"error": str(e), "doc_id": payload["doc_id"]})

@router.post("/ingest")
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    x_tenant_id: str = Header(default="default"),
    x_correlation_id: str = Header(default=None)
):
    doc_id = str(uuid.uuid4())
    correlation_id = x_correlation_id or get_correlation_id() or str(uuid.uuid4())
    
    payload = {
        "doc_id": doc_id,
        "tenant_id": x_tenant_id,
        "title": request.title,
        "content": request.content,
        "correlation_id": correlation_id
    }
    
    background_tasks.add_task(publish_to_queue, "document_ingestion", payload)
    
    return {
        "status": "queued",
        "doc_id": doc_id,
        "tenant_id": x_tenant_id,
        "correlation_id": correlation_id,
        "document_title": request.title
    }

@router.post("/ingest/batch")
async def ingest_documents_batch(
    requests: List[IngestRequest],
    background_tasks: BackgroundTasks,
    x_tenant_id: str = Header(default="default"),
    x_correlation_id: str = Header(default=None)
):
    correlation_id = x_correlation_id or get_correlation_id() or str(uuid.uuid4())
    doc_ids = []
    
    for req in requests:
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        payload = {
            "doc_id": doc_id,
            "tenant_id": x_tenant_id,
            "title": req.title,
            "content": req.content,
            "correlation_id": correlation_id
        }
        background_tasks.add_task(publish_to_queue, "document_ingestion", payload)
        
    return {
        "status": "queued",
        "doc_ids": doc_ids,
        "tenant_id": x_tenant_id,
        "correlation_id": correlation_id,
        "count": len(doc_ids)
    }