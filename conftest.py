# ============================================================
# conftest.py  — root conftest.py
# Adds project root to sys.path so all imports resolve correctly
# Provides shared fixtures used across all test suites
# ============================================================

import sys
import os
import pytest

# ── Make sure the project root is on sys.path ────────────────
# This resolves all "from core.xxx import yyy" style imports
# in tests regardless of how pytest is invoked
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Force test environment before any SI module is imported ──
os.environ.setdefault("SI_ENV",       "development")
os.environ.setdefault("SI_LOG_LEVEL", "WARNING")   # Quiet logs during tests

# Kafka — use localhost defaults (will be mocked in unit/integration tests)
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("KAFKA_TRANSACTIONAL_ID",  "si-test-producer")

# FalkorDB
os.environ.setdefault("FALKORDB_HOST", "localhost")
os.environ.setdefault("FALKORDB_PORT", "6380")

# Vector / pgvector
os.environ.setdefault("VECTOR_BACKEND",   "pgvector")
os.environ.setdefault("POSTGRES_HOST",    "localhost")
os.environ.setdefault("POSTGRES_PORT",    "5432")
os.environ.setdefault("POSTGRES_DB",      "si_vectors")
os.environ.setdefault("POSTGRES_USER",    "si_user")
os.environ.setdefault("POSTGRES_PASSWORD","si_password")

# Embeddings
os.environ.setdefault("EMBEDDING_HOST", "localhost")
os.environ.setdefault("EMBEDDING_PORT", "8080")

# Redis
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")

# OpenAI — tests mock this, but pydantic-settings requires a value
os.environ.setdefault("OPENAI_API_KEY",   "sk-test-key-for-unit-tests-only")
os.environ.setdefault("OPENAI_MODEL",     "gpt-4o-mini")

# API
os.environ.setdefault("SI_SECRET_KEY",    "change-me-in-production-use-openssl-rand-hex-32")
os.environ.setdefault("API_HOST",         "0.0.0.0")
os.environ.setdefault("API_PORT",         "8888")

# Telemetry
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
os.environ.setdefault("OTEL_SERVICE_NAME",            "solar-intelligence-test")

# SLA
os.environ.setdefault("SLA_P50_MS",  "150")
os.environ.setdefault("SLA_P95_MS",  "500")
os.environ.setdefault("SLA_P99_MS",  "1500")


# ── Shared fixtures ──────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Configure minimal logging for the test session."""
    from shared.utils.logging import configure_logging
    configure_logging(level="WARNING", json_output=False)


@pytest.fixture
def sample_raw_document():
    """A valid RawDocument for use in tests."""
    from shared.models.entities import RawDocument
    return RawDocument(
        doc_id="test-doc-001",
        content=(
            "Solar Intelligence maps stellar thermodynamics to AI pipeline design. "
            "The Core layer fuses raw data using Kafka and FalkorDB GraphRAG. "
            "The Radiative Zone transports vectors via Milvus and semantic caching. "
            "The Convective Zone routes queries through the vLLM Semantic Router. "
            "The Photosphere exposes a FastAPI + Kong gateway to the outside world. "
            "The Corona manages telemetry, enforcement, and edge deployments."
        ),
        title="SI Test Document",
        tenant_id="test-tenant",
        correlation_id="test-cid-001",
    )


@pytest.fixture
def sample_embedding():
    """A deterministic 1024-dim unit-norm embedding for tests."""
    import numpy as np
    rng = np.random.default_rng(42)
    v   = rng.random(1024).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def sample_routing_request():
    """A valid RoutingRequest."""
    from shared.models.entities import RoutingRequest
    return RoutingRequest(
        query="What is the Convective Zone in Solar Intelligence?",
        tenant_id="test-tenant",
        correlation_id="test-cid-002",
    )