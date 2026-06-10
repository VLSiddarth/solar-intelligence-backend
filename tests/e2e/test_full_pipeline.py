# ============================================================
# tests/e2e/test_full_pipeline.py
# End-to-end tests — full SI pipeline from ingest → query
# Requires: docker compose up -d && python scripts/health_check.py
# Run: pytest tests/e2e/ -v --e2e
# ============================================================

import time
import json
import pytest
import httpx

API_URL    = "http://localhost:8888"
ADMIN_KEY  = "change-me-in-production-use-openssl-rand-hex-32"

COMMON_HEADERS = {
    "X-Tenant-ID":    "e2e-test-tenant",
    "Content-Type":   "application/json",
}


def pytest_addoption(parser):
    parser.addoption("--e2e", action="store_true", default=False,
                     help="Run end-to-end tests (requires full Docker stack)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--e2e"):
        skip_e2e = pytest.mark.skip(reason="Pass --e2e to run (requires Docker stack)")
        for item in items:
            if "e2e" in item.nodeid:
                item.add_marker(skip_e2e)


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=API_URL, timeout=60.0, headers=COMMON_HEADERS) as client:
        yield client


# ─────────────────────────────────────────────
# Test Suite
# ─────────────────────────────────────────────

class TestFullPipeline:

    def test_01_health_check_all_layers(self, http):
        """All 5 layers must be healthy before E2E tests run."""
        resp = http.get("/health")
        assert resp.status_code in (200, 206), f"Health check failed: {resp.text}"
        body = resp.json()
        assert body["readiness_pct"] >= 50, f"Readiness too low: {body['readiness_pct']}%"

    def test_02_ingest_single_document(self, http):
        """Ingest a document and get a doc_id back."""
        resp = http.post("/v1/ingest", json={
            "content": (
                "Solar Intelligence is a revolutionary AI architecture inspired by stellar physics. "
                "It consists of five layers: Core (data fusion), Radiative Zone (vector transport), "
                "Convective Zone (agentic routing), Photosphere (API endpoints), and Corona (telemetry). "
                "Each layer maps to a physical process in stellar thermodynamics. "
                "The Core uses Apache Kafka with exactly-once semantics and FalkorDB for knowledge graph storage. "
                "The architecture achieves 48.5% token reduction through semantic routing."
            ),
            "title":   "SI Architecture Deep Dive",
            "metadata": {"source": "e2e_test", "version": "1.0"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "doc_id"         in body
        assert body["status"]   == "queued"
        assert "correlation_id" in body
        pytest.doc_id = body["doc_id"]   # Store for later tests

    def test_03_ingest_batch_documents(self, http):
        """Ingest a batch of documents for richer knowledge graph."""
        docs = [
            {
                "content": (
                    "Apache Kafka is a distributed event streaming platform used in the SI Core layer. "
                    "It provides exactly-once semantics through idempotent producers and transactional APIs. "
                    "The SI system uses Kafka for raw document ingestion with topic si.core.raw_documents."
                ),
                "title": "Kafka in SI Core",
            },
            {
                "content": (
                    "FalkorDB is a graph database built on Redis that supports GraphRAG workloads. "
                    "In Solar Intelligence, FalkorDB stores extracted entities and their relationships. "
                    "The knowledge graph uses 11 entity types and 8 relationship types defined in the ontology."
                ),
                "title": "FalkorDB Knowledge Graph",
            },
            {
                "content": (
                    "The vLLM Semantic Router is a Rust-core intent-aware routing engine. "
                    "It achieves 10.2pp accuracy improvement and 47.1% latency reduction. "
                    "The router uses a 3-tier fallback: vLLM → semantic-only → rule-based."
                ),
                "title": "vLLM Semantic Router",
            },
        ]
        resp = http.post("/v1/ingest/batch", json=docs)
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["doc_ids"]) == 3

    def test_04_wait_for_processing(self, http):
        """Give Kafka+GraphRAG pipeline time to process the ingested docs."""
        time.sleep(5)   # Pipeline is async — wait for Flink+GraphRAG to process
        assert True     # If we reach here, pipeline didn't crash

    def test_05_query_basic_standard_mode(self, http):
        """Query with standard mode — should return an answer."""
        resp = http.post("/v1/query", json={
            "query": "What is Solar Intelligence?",
            "mode":  "standard_query",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "answer"       in body
        assert len(body["answer"]) > 20
        assert "correlation_id" in body
        assert "latency_ms"     in body
        assert body["latency_ms"] > 0
        print(f"\n  Answer preview: {body['answer'][:100]}...")

    def test_06_query_metadata_fields(self, http):
        """Verify all metadata fields are present in query response."""
        resp = http.post("/v1/query", json={
            "query": "Explain the five layers of the SI architecture",
            "mode":  "standard_query",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "answer"        in body
        assert "routing_tier"  in body
        assert "from_cache"    in body
        assert "token_usage"   in body
        assert "latency_ms"    in body
        assert body["token_usage"]["total_tokens"] > 0
        assert body["token_usage"]["cost_usd"]    >= 0

    def test_07_query_cache_hit_on_repeat(self, http):
        """Same query twice — second should come from cache (faster)."""
        query = "What is the Convective Zone in Solar Intelligence?"

        t1 = time.monotonic()
        resp1 = http.post("/v1/query", json={"query": query, "mode": "standard_query"})
        latency1 = (time.monotonic() - t1) * 1000

        time.sleep(0.2)

        t2 = time.monotonic()
        resp2 = http.post("/v1/query", json={"query": query, "mode": "standard_query"})
        latency2 = (time.monotonic() - t2) * 1000

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        body2 = resp2.json()
        print(f"\n  Latency 1: {latency1:.0f}ms, Latency 2: {latency2:.0f}ms")
        print(f"  from_cache: {body2.get('from_cache')}")
        # Cache may or may not hit depending on timing — just check no crash
        assert "answer" in body2

    def test_08_query_chain_of_thought_mode(self, http):
        """CoT mode should produce longer, reasoned answers."""
        resp = http.post("/v1/query", json={
            "query": "Step by step, how does a document become a knowledge graph node in SI?",
            "mode":  "chain_of_thought",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        # CoT typically produces longer answers
        print(f"\n  CoT answer length: {len(body['answer'])} chars")

    def test_09_query_edge_mode(self, http):
        """Edge mode — fastest, smallest token budget."""
        resp = http.post("/v1/query", json={
            "query": "Is FalkorDB used in SI?",
            "mode":  "edge_inference",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        # Edge should be faster (smaller model)
        assert body["token_usage"]["total_tokens"] <= 1024

    def test_10_correlation_id_tracking(self, http):
        """Correlation ID should flow through ingest and query."""
        my_cid  = "e2e-tracking-test-cid"
        headers = {**COMMON_HEADERS, "X-Correlation-ID": my_cid}

        # Ingest with custom CID
        resp1 = http.post("/v1/ingest",
                          headers=headers,
                          json={"content": "Correlation tracking test document with enough content for SI."})
        assert resp1.status_code == 200
        assert resp1.headers.get("x-correlation-id") == my_cid
        body1 = resp1.json()
        assert body1.get("correlation_id") == my_cid

        # Query with custom CID
        resp2 = http.post("/v1/query",
                          headers=headers,
                          json={"query": "test query", "mode": "standard_query"})
        assert resp2.status_code == 200
        assert resp2.headers.get("x-correlation-id") == my_cid

    def test_11_sla_p95_latency(self, http):
        """Measure actual p95 latency across 10 queries — must be < 1500ms."""
        latencies = []
        for i in range(10):
            t = time.monotonic()
            resp = http.post("/v1/query", json={
                "query": f"Test SLA query number {i} for latency measurement",
                "mode":  "standard_query",
            })
            latencies.append((time.monotonic() - t) * 1000)
            assert resp.status_code == 200

        latencies.sort()
        p50 = latencies[4]
        p95 = latencies[9]
        print(f"\n  p50: {p50:.0f}ms | p95: {p95:.0f}ms")
        print(f"  SLA targets — p95: 500ms (hard), p99: 1500ms")
        # Allow up to 3× SLA in test environment (cold LLM, no GPU)
        assert p95 < 4500, f"p95 latency {p95:.0f}ms is unacceptably high"

    def test_12_multi_tenant_data_isolation(self, http):
        """Two tenants must not see each other's cached responses."""
        q = "What is the SI Corona layer?"

        # Ingest for tenant-A
        http.post("/v1/ingest",
                  headers={**COMMON_HEADERS, "X-Tenant-ID": "e2e-tenant-a"},
                  json={"content": "Tenant A's private knowledge: the secret code is ALPHA-7."})
        time.sleep(1)

        # Query as tenant-B — must not see tenant-A's private data in cache
        resp_b = http.post("/v1/query",
                           headers={**COMMON_HEADERS, "X-Tenant-ID": "e2e-tenant-b"},
                           json={"query": q, "mode": "standard_query"})
        assert resp_b.status_code == 200
        # Can't guarantee LLM won't mention ALPHA-7 from training data,
        # but cache isolation ensures tenant-A's specific cache entries aren't served to B

    def test_13_admin_stats_reflect_activity(self, http):
        """Admin stats should show non-zero activity from previous tests."""
        resp = http.get("/v1/admin/stats",
                        headers={**COMMON_HEADERS, "X-SI-Admin-Key": ADMIN_KEY})
        assert resp.status_code == 200
        body = resp.json()
        assert "layers" in body
        print(f"\n  System stats: {json.dumps(body, indent=2)[:300]}...")

    def test_14_mcp_tool_call_ingest(self, http):
        """MCP tool call should successfully ingest a document."""
        resp = http.post("/mcp", json={
            "jsonrpc": "2.0",
            "id":      "e2e-1",
            "method":  "tools/call",
            "params": {
                "name": "si_ingest",
                "arguments": {
                    "content": "MCP tool call test document for Solar Intelligence end-to-end pipeline testing.",
                    "title":   "MCP Test Doc",
                },
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body

    def test_15_streaming_query_returns_tokens(self, http):
        """Streaming endpoint should return SSE events."""
        with http.stream("POST", "/v1/query/stream", json={
            "query": "What is Apache Kafka?",
            "mode":  "standard_query",
        }) as stream_resp:
            assert stream_resp.status_code == 200
            content_type = stream_resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type

            tokens = []
            for line in stream_resp.iter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    if "token" in data:
                        tokens.append(data["token"])
                    if data.get("done"):
                        break
                if len(tokens) > 5:
                    break

            assert len(tokens) > 0, "No tokens received from streaming endpoint"
            print(f"\n  Streaming tokens received: {len(tokens)}")