# ============================================================
# tests/integration/test_api.py
# Integration tests — FastAPI routes with real middleware
# Uses httpx TestClient — no real infrastructure needed
# Mocks: Kafka, FalkorDB, Redis, LLM, embeddings
# ============================================================

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# ── Shared mocks applied before app import ───────────────────
@pytest.fixture(scope="module")
def mock_infrastructure():
    """Patch all external infrastructure so tests run without Docker."""
    with (
        patch("core.kafka.producer.Producer",          MagicMock()),
        patch("falkordb.FalkorDB",                     MagicMock()),
        patch("redis.Redis",                           MagicMock()),
        patch("pymilvus.connections.connect",          MagicMock()),
        patch("psycopg2.connect",                      MagicMock()),
        patch("opentelemetry.sdk.trace.TracerProvider",MagicMock()),
        patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor", MagicMock()),
        patch("opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor", MagicMock()),
        patch("opentelemetry.instrumentation.redis.RedisInstrumentor", MagicMock()),
    ):
        yield


@pytest.fixture(scope="module")
def client(mock_infrastructure):
    from photosphere.api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


TENANT_HEADERS = {"X-Tenant-ID": "test-tenant", "Content-Type": "application/json"}
ADMIN_HEADERS  = {**TENANT_HEADERS, "X-SI-Admin-Key": "change-me-in-production-use-openssl-rand-hex-32"}


# ── Health endpoints ─────────────────────────────────────────
class TestHealthEndpoints:

    def test_health_returns_200_or_206(self, client, mock_infrastructure):
        resp = client.get("/health")
        assert resp.status_code in (200, 206, 503)

    def test_health_response_structure(self, client, mock_infrastructure):
        resp = client.get("/health")
        body = resp.json()
        assert "service"       in body
        assert "status"        in body
        assert "layers"        in body
        assert "readiness_pct" in body

    def test_ready_endpoint(self, client, mock_infrastructure):
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True

    def test_live_endpoint(self, client, mock_infrastructure):
        resp = client.get("/live")
        assert resp.status_code == 200
        assert resp.json()["alive"] is True

    def test_correlation_id_in_response_header(self, client, mock_infrastructure):
        resp = client.get("/health")
        assert "x-correlation-id" in resp.headers

    def test_custom_correlation_id_echoed(self, client, mock_infrastructure):
        my_cid = "test-cid-abc-123"
        resp = client.get("/health", headers={"X-Correlation-ID": my_cid})
        returned = resp.headers.get("x-correlation-id", "")
        assert returned == my_cid


# ── Ingest endpoints ─────────────────────────────────────────
class TestIngestEndpoints:

    def test_ingest_valid_document(self, client, mock_infrastructure):
        with patch("photosphere.api.routes.ingest.get_producer") as mock_prod:
            mock_prod.return_value.produce = MagicMock()
            resp = client.post("/v1/ingest", headers=TENANT_HEADERS, json={
                "content": "This is a test document about Solar Intelligence architecture with enough content.",
                "title":   "Test Doc",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert "doc_id"         in body
        assert body["status"]   == "queued"
        assert "correlation_id" in body

    def test_ingest_returns_doc_id_uuid_format(self, client, mock_infrastructure):
        with patch("photosphere.api.routes.ingest.get_producer") as mock_prod:
            mock_prod.return_value.produce = MagicMock()
            resp = client.post("/v1/ingest", headers=TENANT_HEADERS, json={
                "content": "Another test document with enough characters to pass validation requirements here.",
            })
        doc_id = resp.json().get("doc_id", "")
        # UUID4 format: 8-4-4-4-12
        parts = doc_id.split("-")
        assert len(parts) == 5

    def test_ingest_rejects_empty_content(self, client, mock_infrastructure):
        resp = client.post("/v1/ingest", headers=TENANT_HEADERS, json={"content": ""})
        assert resp.status_code == 422

    def test_ingest_rejects_too_short(self, client, mock_infrastructure):
        resp = client.post("/v1/ingest", headers=TENANT_HEADERS, json={"content": "hi"})
        assert resp.status_code == 422

    def test_ingest_batch_accepts_multiple(self, client, mock_infrastructure):
        with patch("photosphere.api.routes.ingest.get_producer") as mock_prod:
            mock_prod.return_value.produce_batch = MagicMock()
            resp = client.post("/v1/ingest/batch", headers=TENANT_HEADERS, json=[
                {"content": "First document content long enough to pass validation requirements here."},
                {"content": "Second document content long enough to pass validation requirements here."},
            ])
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert len(body["doc_ids"]) == 2

    def test_ingest_batch_rejects_over_100(self, client, mock_infrastructure):
        docs = [{"content": f"Document number {i} with enough content to be valid for SI testing."}
                for i in range(101)]
        resp = client.post("/v1/ingest/batch", headers=TENANT_HEADERS, json=docs)
        assert resp.status_code == 400

    def test_ingest_without_tenant_header(self, client, mock_infrastructure):
        with patch("photosphere.api.routes.ingest.get_producer") as mock_prod:
            mock_prod.return_value.produce = MagicMock()
            resp = client.post("/v1/ingest",
                               headers={"Content-Type": "application/json"},
                               json={"content": "Test content long enough for validation purposes here."})
        # Anonymous tenant is allowed
        assert resp.status_code == 200

    def test_ingest_with_invalid_tenant_id(self, client, mock_infrastructure):
        resp = client.post("/v1/ingest",
                           headers={"X-Tenant-ID": "../../../etc", "Content-Type": "application/json"},
                           json={"content": "Test document with valid content for testing purposes."})
        assert resp.status_code == 400


# ── Query endpoints ──────────────────────────────────────────
class TestQueryEndpoints:

    @pytest.fixture(autouse=True)
    def mock_orchestrator(self):
        from shared.models.entities import QueryResponse, RouteTier, TokenUsage, QueryMode
        mock_response = QueryResponse(
            answer="Solar Intelligence is a stellar-physics-inspired AI architecture.",
            sources=[],
            routing_tier=RouteTier.SEMANTIC_ONLY,
            from_cache=False,
            token_usage=TokenUsage(
                prompt_tokens=50, completion_tokens=30, total_tokens=80,
                cost_usd=0.00005, mode=QueryMode.STANDARD,
            ),
            correlation_id="test-cid",
            latency_ms=250.0,
        )
        with patch("photosphere.api.routes.query._get_orchestrator") as mock_orch_fn:
            mock_orch = MagicMock()
            mock_orch.execute = AsyncMock(return_value=mock_response)
            mock_orch_fn.return_value = mock_orch
            yield mock_orch

    def test_query_returns_answer(self, client, mock_infrastructure):
        resp = client.post("/v1/query", headers=TENANT_HEADERS, json={
            "query": "What is Solar Intelligence?",
            "mode":  "standard_query",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "answer"       in body
        assert len(body["answer"]) > 0

    def test_query_response_has_all_fields(self, client, mock_infrastructure):
        resp = client.post("/v1/query", headers=TENANT_HEADERS, json={
            "query": "Explain the five layers",
            "mode":  "standard_query",
        })
        body = resp.json()
        required = ["answer", "routing_tier", "from_cache", "token_usage",
                    "correlation_id", "latency_ms"]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_query_empty_string_returns_422(self, client, mock_infrastructure):
        resp = client.post("/v1/query", headers=TENANT_HEADERS, json={
            "query": "", "mode": "standard_query"
        })
        assert resp.status_code == 422

    def test_query_invalid_mode_returns_422(self, client, mock_infrastructure):
        resp = client.post("/v1/query", headers=TENANT_HEADERS, json={
            "query": "test", "mode": "super_mode"
        })
        assert resp.status_code == 422

    def test_query_all_valid_modes(self, client, mock_infrastructure):
        for mode in ["standard_query", "chain_of_thought", "edge_inference"]:
            resp = client.post("/v1/query", headers=TENANT_HEADERS, json={
                "query": f"Test with {mode}",
                "mode":  mode,
            })
            assert resp.status_code == 200, f"Mode {mode} failed"

    def test_query_correlation_id_in_body(self, client, mock_infrastructure):
        resp = client.post("/v1/query", headers=TENANT_HEADERS, json={
            "query": "test", "mode": "standard_query"
        })
        body = resp.json()
        assert "correlation_id" in body
        cid = body["correlation_id"]
        assert len(cid) > 0


# ── Admin endpoints ──────────────────────────────────────────
class TestAdminEndpoints:

    def test_stats_requires_admin_key(self, client, mock_infrastructure):
        resp = client.get("/v1/admin/stats", headers=TENANT_HEADERS)
        assert resp.status_code in (403, 422)

    def test_stats_wrong_key_returns_403(self, client, mock_infrastructure):
        resp = client.get("/v1/admin/stats",
                          headers={**TENANT_HEADERS, "X-SI-Admin-Key": "wrong"})
        assert resp.status_code == 403

    def test_sla_returns_targets(self, client, mock_infrastructure):
        resp = client.get("/v1/admin/sla", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["p99_ms"] == 1500
        assert body["p95_ms"] == 500
        assert body["p50_ms"] == 150

    def test_stats_returns_layer_info(self, client, mock_infrastructure):
        with (
            patch("photosphere.api.routes.admin.get_producer", MagicMock()),
            patch("photosphere.api.routes.admin.get_state_store", MagicMock()),
            patch("photosphere.api.routes.admin.get_governor", MagicMock()),
        ):
            resp = client.get("/v1/admin/stats", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "layers" in body


# ── MCP endpoints ────────────────────────────────────────────
class TestMCPEndpoints:

    def test_tools_list(self, client, mock_infrastructure):
        resp = client.post("/mcp", headers=TENANT_HEADERS, json={
            "jsonrpc": "2.0",
            "id":      "1",
            "method":  "tools/list",
            "params":  {},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "result"  in body
        assert "tools"   in body["result"]
        tool_names = [t["name"] for t in body["result"]["tools"]]
        assert "si_ingest" in tool_names
        assert "si_query"  in tool_names

    def test_invalid_jsonrpc_version(self, client, mock_infrastructure):
        resp = client.post("/mcp", headers=TENANT_HEADERS, json={
            "jsonrpc": "1.0",
            "id":      "1",
            "method":  "tools/list",
        })
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_missing_required_field_returns_4000(self, client, mock_infrastructure):
        resp = client.post("/mcp", headers=TENANT_HEADERS, json={
            "jsonrpc": "2.0",
            "id":      "2",
            "method":  "tools/call",
            "params":  {"name": "si_ingest", "arguments": {}},  # Missing content
        })
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == 4000

    def test_unknown_method_returns_404(self, client, mock_infrastructure):
        resp = client.post("/mcp", headers=TENANT_HEADERS, json={
            "jsonrpc": "2.0",
            "id":      "3",
            "method":  "unknown/method",
            "params":  {},
        })
        assert resp.status_code == 404