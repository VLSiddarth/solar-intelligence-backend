# ============================================================
# photosphere/api/main.py
# FastAPI application — the Photosphere boundary
# All requests enter and exit through here
# ============================================================

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from photosphere.api.routes import ingest, query, health, admin, mcp
from photosphere.middleware.correlation import CorrelationMiddleware
from photosphere.middleware.tenant import TenantMiddleware
from corona.telemetry.tracer import setup_telemetry, get_tracer
from shared.config.settings import settings
from shared.utils.logging import configure_logging, get_logger
from shared.utils.correlation import set_layer

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # ── Startup ────────────────────────────────
    configure_logging(level=settings.log_level, json_output=settings.is_production)
    set_layer("photosphere")
    setup_telemetry()

    logger.info("si_api_starting", extra={
        "env":     settings.env,
        "version": "1.0.0",
        "layer":   "photosphere",
    })

    # Warm vector index check
    try:
        from radiative.vector.index_manager import BlueGreenIndexManager
        index = BlueGreenIndexManager()
        logger.info("vector_index_ready", extra={"active": index.active_index})
    except Exception as e:
        logger.warning("vector_index_init_warning", extra={"error": str(e)})

    yield

    # ── Shutdown ───────────────────────────────
    logger.info("si_api_shutting_down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Solar Intelligence API",
        description=(
            "The Photosphere boundary of the Solar Intelligence architecture. "
            "L = 4πR²σTeff⁴"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    # ── Middleware (order matters — outermost executes first) ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(TenantMiddleware)
    app.add_middleware(CorrelationMiddleware)

    # ── Routers ────────────────────────────────
    app.include_router(health.router,  prefix="",     tags=["Health"])
    app.include_router(ingest.router,  prefix="/v1",  tags=["Ingest"])
    app.include_router(query.router,   prefix="/v1",  tags=["Query"])
    app.include_router(admin.router,   prefix="/v1/admin", tags=["Admin"])
    app.include_router(mcp.router,     prefix="",     tags=["MCP"]) # ADDED ROUTER HERE

    # V2 placeholder (N-1 backward compat maintained)
    from fastapi import APIRouter
    v2 = APIRouter(prefix="/v2")

    @v2.get("/ping")
    async def v2_ping():
        return {"version": "v2", "status": "coming_soon"}

    app.include_router(v2, tags=["V2"])

    # ── Global exception handler ───────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        from shared.utils.correlation import get_correlation_id
        from shared.models.entities import MCPError
        cid = get_correlation_id()
        logger.error("unhandled_exception", extra={
            "error":          str(exc),
            "path":           request.url.path,
            "correlation_id": cid,
        })
        error = MCPError.build(5000, str(exc), request_id=cid)
        return JSONResponse(status_code=500, content=error.model_dump())

    return app


app = create_app()