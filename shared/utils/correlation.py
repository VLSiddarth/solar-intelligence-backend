# ============================================================
# shared/utils/correlation.py
# Correlation ID — injected at Core, propagated through all 5 layers
# Every request, every Kafka message, every HTTP call carries this ID
# ============================================================

import uuid
import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

# Thread-local (asyncio-safe) correlation ID store
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="unknown")
_layer: ContextVar[str] = ContextVar("si_layer", default="unknown")

HEADER_NAME = "X-Correlation-ID"
TENANT_HEADER = "X-Tenant-ID"
LAYER_HEADER = "X-SI-Layer"


def new_correlation_id() -> str:
    """Generate a new UUID4 correlation ID."""
    return str(uuid.uuid4())


def inject_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set correlation ID in context. If none provided, generates a new one.
    Call this at the entry point of every request (API, Kafka consumer, etc.)
    """
    cid = cid or new_correlation_id()
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get current correlation ID, generating one if missing."""
    cid = _correlation_id.get()
    if not cid:
        cid = inject_correlation_id()
    return cid


def set_tenant_id(tenant_id: str) -> None:
    _tenant_id.set(tenant_id)


def get_tenant_id() -> str:
    return _tenant_id.get() or "unknown"


def set_layer(layer: str) -> None:
    _layer.set(layer)


def get_layer() -> str:
    return _layer.get() or "unknown"


def get_context() -> dict:
    """Full context dict — attach to every log line and Kafka header."""
    return {
        "correlation_id": get_correlation_id(),
        "tenant_id": get_tenant_id(),
        "si_layer": get_layer(),
    }


# ─────────────────────────────────────────────
# Kafka Header Utilities
# ─────────────────────────────────────────────

def build_kafka_headers() -> list[tuple[str, bytes]]:
    """Build Kafka message headers carrying correlation context."""
    return [
        ("correlation_id", get_correlation_id().encode()),
        ("tenant_id", get_tenant_id().encode()),
        ("si_layer", get_layer().encode()),
    ]


def extract_kafka_headers(headers: list[tuple[str, bytes]]) -> dict[str, str]:
    """Extract and restore correlation context from Kafka message headers."""
    ctx = {}
    for key, value in (headers or []):
        ctx[key] = value.decode() if isinstance(value, bytes) else value

    cid = ctx.get("correlation_id", "")
    if cid:
        inject_correlation_id(cid)
    else:
        inject_correlation_id()

    if tenant := ctx.get("tenant_id"):
        set_tenant_id(tenant)

    if layer := ctx.get("si_layer"):
        set_layer(layer)

    return ctx


# ─────────────────────────────────────────────
# FastAPI Middleware
# ─────────────────────────────────────────────

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    Injects correlation ID at the Photosphere boundary.
    Reads from incoming header if present (B2B calls), generates if missing (new requests).
    Propagates to response header so clients can track their requests.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Read from incoming request or generate new
        cid = request.headers.get(HEADER_NAME) or new_correlation_id()
        tenant = request.headers.get(TENANT_HEADER, "anonymous")
        inject_correlation_id(cid)
        set_tenant_id(tenant)
        set_layer("photosphere")

        logger.info(
            "request_received",
            extra={
                "correlation_id": cid,
                "tenant_id": tenant,
                "path": request.url.path,
                "method": request.method,
            },
        )

        response = await call_next(request)

        # Always return the correlation ID in the response
        response.headers[HEADER_NAME] = cid
        response.headers["X-SI-Version"] = "v1"
        response.headers["X-SI-Layer"] = "photosphere"

        return response