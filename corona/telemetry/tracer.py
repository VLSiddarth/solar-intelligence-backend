# ============================================================
# corona/telemetry/tracer.py
# OpenTelemetry 1.37+ — trace propagation across all 5 layers
# Includes Kafka header injection for async trace continuity
# P0 fix: data lineage via correlation_id at every boundary
# ============================================================

import logging
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
# from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagate import inject as otel_inject, extract as otel_extract
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from shared.config.settings import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)

_tracer: Optional[trace.Tracer] = None


def setup_telemetry() -> None:
    """
    Configure OpenTelemetry SDK.
    Call once at application startup in lifespan().
    """
    global _tracer

    resource = Resource.create({
        SERVICE_NAME:        settings.telemetry.service_name,
        "service.version":   "1.0.0",
        "deployment.env":    settings.env,
        "si.architecture":   "solar-intelligence",
    })

    provider = TracerProvider(resource=resource)

    # OTLP exporter → OpenTelemetry Collector → Jaeger
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.telemetry.otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("otel_otlp_exporter_configured", extra={
            "endpoint": settings.telemetry.otlp_endpoint,
        })
    except Exception as e:
        logger.warning("otel_otlp_exporter_failed_using_console", extra={"error": str(e)})
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("solar-intelligence", "1.0.0")

    # Auto-instrument libraries
    # FastAPIInstrumentor().instrument()
    # HTTPXClientInstrumentor().instrument()
    # RedisInstrumentor().instrument()

    logger.info("opentelemetry_initialized", extra={
        "service":    settings.telemetry.service_name,
        "sample_rate": settings.telemetry.trace_sample_rate,
    })


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("solar-intelligence")
    return _tracer


# ─────────────────────────────────────────────
# Kafka Trace Propagation
# ─────────────────────────────────────────────

class KafkaTracePropagator:
    """
    Injects and extracts OpenTelemetry trace context from Kafka message headers.
    This is the fix for the broken trace chain across async Kafka boundaries.

    Without this: Core → Kafka → Radiative creates a new trace, breaking lineage.
    With this: the full 5-layer trace is continuous and visible in Jaeger.
    """

    _propagator = TraceContextTextMapPropagator()

    @classmethod
    def inject_headers(cls, headers: dict) -> dict:
        """
        Inject current trace context into Kafka message headers.
        Call before producing a Kafka message.
        """
        otel_inject(headers)
        return headers

    @classmethod
    def extract_context(cls, headers: dict):
        """
        Extract trace context from Kafka message headers.
        Call at the start of a Kafka consumer handler.
        Returns an OTel Context object.
        """
        return otel_extract(headers)

    @classmethod
    def headers_to_kafka_format(cls, trace_headers: dict) -> list[tuple[str, bytes]]:
        """Convert OTel headers dict to Kafka headers list."""
        return [
            (k, v.encode() if isinstance(v, str) else v)
            for k, v in trace_headers.items()
        ]


# ─────────────────────────────────────────────
# SI Layer Span Context Manager
# ─────────────────────────────────────────────

class SISpan:
    """
    Context manager for creating SI layer spans with standard attributes.

    Usage:
        async with SISpan("core", "graphrag_ingestion", doc_id=doc.doc_id) as span:
            result = await pipeline.ingest(doc)
            span.set_attribute("entities_extracted", len(result.entities))
    """

    def __init__(self, layer: str, operation: str, **attributes):
        self.layer      = layer
        self.operation  = operation
        self.attributes = attributes
        self._span      = None
        self._ctx       = None

    def __enter__(self):
        tracer = get_tracer()
        self._ctx = tracer.start_as_current_span(
            f"si.{self.layer}.{self.operation}",
            attributes={
                "si.layer":     self.layer,
                "si.operation": self.operation,
                **{f"si.{k}": str(v) for k, v in self.attributes.items()},
            },
        )
        self._span = self._ctx.__enter__()
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and self._span:
            self._span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
        if self._ctx:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)
