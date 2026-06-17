<div align="center">

# ☀ Solar Intelligence

### The Autonomous Knowledge Intelligence Platform

*Modeled on the five layers of the Sun. Built for production.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Redpanda](https://img.shields.io/badge/Redpanda-Kafka-red.svg)](https://redpanda.com)
[![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-blue.svg)](https://github.com/pgvector/pgvector)
[![FalkorDB](https://img.shields.io/badge/FalkorDB-GraphRAG-orange.svg)](https://falkordb.com)

**Solar Intelligence (SI)** is an open-source autonomous AI knowledge platform that continuously ingests, fuses, and synthesizes intelligence from any domain — running 24/7 for 72 hours per agent cycle. It solves the critical flaw in production RAG systems: **stale knowledge serving expired data with the same confidence as live data**.

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [API Reference](#-api-reference) • [72-Hour Agent](#-72-hour-autonomous-agent) • [Deployment](#-production-deployment) • [Contributing](#-contributing)

</div>

---
## ⚡ Production Telemetry & Infrastructure

Solar Intelligence is not a wrapper; it is a fully observable, event-driven microservices cluster. Here is a look at the live orchestration:

<p align="center">
  <img src="The Orchestration Boundary.png" width="800" alt="The Orchestration Boundary (FastAPI)"><br>
  <em>The Orchestration Boundary (FastAPI)</em>
</p>

<p align="center">
  <img src="High-Throughput Event Streaming.png" width="800" alt="High-Throughput Event Streaming (Redpanda)"><br>
  <em>High-Throughput Event Streaming (Redpanda)</em>
</p>

<p align="center">
  <img src="Live Telemetry & Message Activity.png" width="800" alt="Live Telemetry & Message Activity (Grafana)"><br>
  <em>Live Telemetry & Message Activity (Grafana)</em>
</p>

<p align="center">
  <img src="Distributed Tracing .png" width="800" alt="Distributed Tracing for the Retrieval Latency (Jaeger)"><br>
  <em>Distributed Tracing for Retrieval/Decay Latency (Jaeger)</em>
</p>

<p align="center">
  <img src="State & Cache Management.png" width="800" alt="State & Cache Management (Redis)"><br>
  <em>State & Cache Management (Redis)</em>
</p>

---

## The Problem SI Solves

Every vector database today has a fundamental flaw: a regulatory document from 2024 and the live 2026 version score **identical cosine similarity**. Your RAG system serves expired data with the same confidence as current data. This causes:

- Trading agents acting on 5-hour-old market sentiment
- Legal AI citing superseded regulations
- Medical assistants referencing outdated clinical guidelines
- Compliance tools unaware of new requirements

Solar Intelligence solves this with a **temporal knowledge fusion pipeline** — continuous ingestion, autonomous decay detection, and grounded synthesis. Every query answer traces back to a real, time-stamped document.

---

## ✨ Key Features

- **72-Hour Autonomous Agent** — runs 144 monitoring cycles, discovers new knowledge every 30 minutes, synthesizes intelligence reports every 12 hours
- **Full RAG Pipeline** — real vector search (not a chatbot wrapper), grounded answers from your actual documents
- **GraphRAG Knowledge Extraction** — entities and relationships extracted to FalkorDB, enabling multi-hop reasoning
- **Semantic Cache** — cosine similarity cache with drift-based invalidation, 60%+ hit rate when warm
- **Blue-Green Vector Index** — zero-downtime index promotion using pgvector
- **5-Layer Observability** — every request traced through Jaeger with correlation IDs across all layers
- **Multi-Tenant** — full tenant isolation via X-Tenant-ID header
- **MCP Server** — Model Context Protocol compatible tool server for agent integrations
- **$0 Infrastructure** — runs entirely on GitHub Student Dev Pack free tiers

---

## Architecture

Solar Intelligence maps the five layers of stellar physics to AI pipeline architecture.

```
LAYER V   CORONA        Telemetry · Tracing · Webhooks · ECDSA Signing
                        OTel Collector → Jaeger · Prometheus · Grafana
                        Equation: T_corona >> T_photosphere

LAYER IV  PHOTOSPHERE   API Gateway · Rate Limiting · Auth
                        FastAPI (Uvicorn) · Kong · mTLS
                        POST /v1/ingest · POST /v1/query · MCP Tool Server
                        Equation: L = 4πR²σTeff⁴

LAYER III CONVECTIVE    Agentic Routing · State · Token Governance
          ZONE          SemanticRouter → AgentOrchestrator → TokenGovernor
                        72h AutonomousAgent · Redis state checkpointing
                        Equation: F_conv = ρCₚvΔT

LAYER II  RADIATIVE     Vector Transport · Semantic Cache
          ZONE          TGI (BGE-small-en-v1.5) → 384-dim embeddings
                        pgvector blue/green index · IVFFlat cosine
                        SemanticCache (Redis, 0.85 threshold)
                        Equation: dP_rad/dr = -κρ(L/4πr²c)

LAYER I   CORE          Data Fusion Engine
                        Redpanda (Kafka) — exactly-once delivery
                        si-worker: embed → pgvector + GraphRAG extraction
                        FalkorDB — entities + relationships (Cypher)
                        Equation: ε = ε_nuclear + ε_gravitational
```

**Data Flow — Ingest:**
```
POST /v1/ingest
  → Redpanda (si.core.raw_documents)
  → si-worker consumes message
  → TGI generates 384-dim BGE embedding
  → pgvector stores vector (v_blue table, IVFFlat cosine index)
  → Groq LLM extracts entities + relationships (Cypher)
  → FalkorDB stores knowledge graph
  → Redis: doc status = "processed"
```

**Data Flow — Query:**
```
POST /v1/query
  → SemanticCache check (cosine similarity >= 0.85 = instant cache hit)
  → TGI embeds query (384-dim)
  → pgvector top-k search (IVFFlat cosine, default k=10)
  → RAG context assembled from retrieved documents
  → TokenGovernor → Groq LLM call with context
  → Cache result → return grounded answer + sources + trace ID
```

---

## Quick Start

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Git
- A free [Groq API key](https://console.groq.com) — 14,400 req/day free

### 1. Clone and Configure

```bash
git clone https://github.com/VLSiddarth/solar-intelligence-backend.git
cd solar-intelligence-backend
cp .env.example .env
```

Edit `.env` — minimum required values:

```bash
SI_ENV=development
SI_SECRET_KEY=<run: openssl rand -hex 32>
GROQ_API_KEY=<your-groq-key>
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
MTLS_ENABLED=false
ECDSA_SIGNING_ENABLED=false
VECTOR_DIMENSION=384
```

### 2. Fix docker-compose.yml (Required)

In `docker-compose.yml`, add these 3 lines to the `si-worker` environment block:

```yaml
  si-worker:
    environment:
      PYTHONPATH: /app
      VECTOR_DIMENSION: "384"
      SI_API_BASE_URL: "http://si-api:8888"
```

Also remove (or comment out) the `command:` line from the redis service so RedisInsight starts on port 8001.

### 3. Start the Stack

```bash
docker compose up -d
docker compose logs -f si-api
# Wait for: "Application startup complete."
```

### 4. Create Kafka Topics (first boot only)

```bash
docker exec si-redpanda rpk topic create si.core.raw_documents --partitions 6
docker exec si-redpanda rpk topic create si.core.fused_entities --partitions 6
docker exec si-redpanda rpk topic create si.core.audit --partitions 3
docker exec si-redpanda rpk topic create si.corona.webhook_dlq --partitions 3
docker compose restart si-api si-worker
```

### 5. Verify

```bash
source .venv/bin/activate
python scripts/health_check.py
```

```
── Layer CORE ────────────────────  ── Layer PHOTOSPHERE ──────────────
  Redpanda (Kafka)     ✓ UP           SI API               ✓ UP
  FalkorDB (GraphRAG)  ✓ UP           Kong Gateway         ✓ UP
── Layer RADIATIVE ──────────────   ── Layer CORONA ───────────────────
  PostgreSQL+pgvector  ✓ UP           OTel Collector       ✓ UP
  TGI (BGE-M3)         ✓ UP           Jaeger UI            ✓ UP
── Layer CONVECTIVE ─────────────    Prometheus            ✓ UP
  Redis Stack          ✓ UP           Grafana              ✓ UP
☀  All systems nominal.
```

---

## API Reference

Base URL: `http://localhost:8888`
Interactive docs: `http://localhost:8888/docs`

Headers supported on all endpoints:
- `X-Tenant-ID` — namespace isolation (default: `"default"`)
- `X-Correlation-ID` — request tracing (auto-generated if omitted)

---

### POST /v1/ingest

Ingest a document into the SI fusion pipeline.

**Request:**
```json
{
  "title": "Q1 2026 Compliance Report",
  "content": "The updated GDPR enforcement guidelines published in March 2026 require...",
  "source_url": "https://gdpr.eu/2026/q1-enforcement",
  "metadata": { "category": "compliance", "region": "EU" }
}
```

**Response:**
```json
{
  "status": "queued",
  "doc_id": "7f034fa1-c6d8-4e3a-9350-0fb71ab2d84a",
  "tenant_id": "default",
  "correlation_id": "b40c3184-7c46-4c63-966c-acd18066d015",
  "status_url": "/v1/ingest/7f034fa1-c6d8-4e3a-9350-0fb71ab2d84a/status"
}
```

---

### GET /v1/ingest/{doc_id}/status

Poll document processing status.

**Response:**
```json
{
  "doc_id": "7f034fa1-c6d8-4e3a-9350-0fb71ab2d84a",
  "status": "processed",
  "tenant_id": "default"
}
```

Status lifecycle: `queued → pending → processed | failed`

---

### POST /v1/ingest/batch

Ingest multiple documents in one call.

**Request:**
```json
[
  { "title": "Doc A", "content": "..." },
  { "title": "Doc B", "content": "..." }
]
```

---

### POST /v1/query

Query the SI knowledge base using RAG.

**Request:**
```json
{
  "query": "What are the latest GDPR enforcement requirements?",
  "mode": "standard_query",
  "top_k": 10
}
```

**Query modes:**

| Mode | Description | Token Budget |
|------|-------------|-------------|
| `standard_query` | Fast vector search + grounded LLM answer | 4,096 |
| `chain_of_thought` | Step-by-step reasoning over retrieved documents | 8,192 |
| `edge_inference` | Compressed response for latency-sensitive clients | 1,024 |

**Response:**
```json
{
  "answer": "According to the Q1 2026 GDPR enforcement guidelines...",
  "sources": [
    { "entity_id": "7f034fa1-...", "score": 0.9234, "index": 1 }
  ],
  "routing_tier": "vllm_semantic",
  "from_cache": false,
  "token_usage": {
    "prompt_tokens": 312,
    "completion_tokens": 187,
    "total_tokens": 499,
    "cost_usd": 0.000025,
    "mode": "standard_query",
    "truncated": false
  },
  "correlation_id": "7b7072d1-08eb-4120-8b4e-239ef2d05553",
  "latency_ms": 905.12
}
```

---

### Health Endpoints

```
GET /health   full layer-by-layer health with readiness_pct
GET /ready    readiness probe (200 when ready)
GET /live     liveness probe (200 always)
```

---

## 72-Hour Autonomous Agent

The SI Autonomous Agent runs for 72 hours across 144 cycles (one every 30 minutes). Each cycle:

1. **Discovers** — calls KU API for new documents on configured topics
2. **Ingests** — publishes documents to Redpanda → si-worker processes
3. **Waits** — polls `/v1/ingest/{id}/status` until `processed`
4. **Synthesizes** — chain-of-thought query over new knowledge
5. **Checkpoints** — saves full state to Redis (survives restarts)
6. **Reports** — full intelligence report every 12 hours (6 per run)

### Launch Agent (Linux/Mac)

```bash
curl -X POST http://localhost:8888/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Monitor autonomous AI agent and production RAG developments",
    "topics": [
      "autonomous AI agents 2025",
      "production RAG systems",
      "knowledge graph LLM extraction",
      "vector database temporal decay"
    ],
    "cycle_interval_minutes": 30
  }'
```

### Launch Agent (Windows PowerShell)

```powershell
$agent = Invoke-RestMethod -Uri "http://localhost:8888/v1/agent/start" `
    -Method POST -ContentType "application/json" `
    -Body (@{
        goal   = "Monitor autonomous AI agent and production RAG developments"
        topics = @(
            "autonomous AI agents 2025",
            "production RAG systems",
            "knowledge graph LLM extraction"
        )
        cycle_interval_minutes = 30
    } | ConvertTo-Json -Depth 3)

$agentId = $agent.agent_id
Write-Host "Agent launched: $agentId"
```

### Monitor Agent (PowerShell)

```powershell
# Status
Invoke-RestMethod -Uri "http://localhost:8888/v1/agent/$agentId"

# Reports (generated every 12h)
Invoke-RestMethod -Uri "http://localhost:8888/v1/agent/$agentId/report"

# All agents
Invoke-RestMethod -Uri "http://localhost:8888/v1/agent/list/all"

# Stop early
Invoke-RestMethod -Uri "http://localhost:8888/v1/agent/$agentId/stop" -Method POST
```

### Agent Status Response

```json
{
  "agent_id": "ad4b6abc-8616-4fff-b27a-67c7c36fe380",
  "status": "running",
  "goal": "Monitor autonomous AI agent developments",
  "started_at": "2026-06-11T10:14:29",
  "last_cycle_at": "2026-06-11T10:44:55",
  "cycle_count": 3,
  "total_docs": 45,
  "total_insights": 3,
  "recent_finding": "Key development: Multi-agent coordination protocols...",
  "decay_alerts": [],
  "reports_count": 0,
  "ku_calls": 9
}
```

---

## Dashboards

| Dashboard | URL | Credentials |
|-----------|-----|-------------|
| SI API Swagger | http://localhost:8888/docs | — |
| Redpanda Console | http://localhost:8085 | — |
| Grafana | http://localhost:3000 | admin / si_admin |
| Jaeger Traces | http://localhost:16686 | — |
| RedisInsight | http://localhost:8001 | — |
| Prometheus | http://localhost:9090 | — |
| Flink UI | http://localhost:8081 | — |

**What to look for in Redpanda Console:** Topics → `si.core.raw_documents` → Messages → see every ingested document with its partition, offset, key (doc_id), and full JSON payload in real time.

**What to look for in Jaeger:** Service: `solar-intelligence` → Find Traces → click any trace → see the full 5-layer span tree showing exactly how long each layer took.

**What to look for in Grafana:** Dashboards → SI → SI Overview → watch request rate, embedding latency, Kafka consumer lag, and cumulative token spend per hour.

---

## Test Suite

```bash
source .venv/bin/activate

# End-to-end smoke test (30 assertions, requires live stack)
python scripts/smoke_test.py

# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Full pipeline test
pytest tests/e2e/ -v --timeout=120
```

---

## Production Deployment (DigitalOcean)

GitHub Student Dev Pack provides $200 DigitalOcean credit. A 72-hour run costs ~$2.40.

**Minimum droplet:** 4GB RAM / 2 vCPU / 80GB SSD — $24/month (BLR1 region for India)

```bash
# On your DigitalOcean droplet

# Install Docker
curl -fsSL https://get.docker.com | sh
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 \
     -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

# Clone and deploy
cd /opt
git clone https://github.com/VLSiddarth/solar-intelligence-backend.git si
cd si
cp .env.example .env
nano .env  # set GROQ_API_KEY, SI_SECRET_KEY

docker compose up -d

# Create topics (wait 30s for Redpanda to be healthy first)
sleep 30
docker exec si-redpanda rpk topic create si.core.raw_documents --partitions 6
docker exec si-redpanda rpk topic create si.core.fused_entities --partitions 6
docker exec si-redpanda rpk topic create si.core.audit --partitions 3
docker exec si-redpanda rpk topic create si.corona.webhook_dlq --partitions 3
docker compose restart si-api si-worker

# Launch 72-hour agent
curl -X POST http://localhost:8888/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{"goal":"Monitor AI developments","topics":["autonomous AI agents 2025"],"cycle_interval_minutes":30}'
```

Access your deployment:
```
SI API:    http://<droplet-ip>:8888
Docs:      http://<droplet-ip>:8888/docs
Redpanda:  http://<droplet-ip>:8085
Grafana:   http://<droplet-ip>:3000
```

---

## Repository Structure

```
solar-intelligence/
├── core/                    # Layer I: Data Fusion Engine
│   ├── kafka/producer.py    # Kafka producer (acks=1, flush-guaranteed)
│   ├── kafka/consumer.py    # Kafka consumer
│   ├── worker/main.py       # Fusion worker: embed → pgvector → GraphRAG
│   ├── graphrag/pipeline.py # Entity/relationship extraction → FalkorDB
│   └── llm/provider.py      # LLM client factory (Groq / OpenAI)
│
├── radiative/               # Layer II: Vector Transport
│   ├── embeddings/client.py # TGI BGE embedding client
│   ├── vector/index_manager.py  # Blue-green pgvector index
│   └── cache/semantic_cache.py  # Cosine similarity cache
│
├── convective/              # Layer III: Agentic Routing
│   ├── agents/autonomous_agent.py  # 72-hour monitoring agent
│   ├── agents/orchestrator.py      # Multi-hop agent orchestrator
│   ├── agents/token_governor.py    # Cost ceiling enforcement
│   ├── agents/ku_client.py         # Knowledge Universe API client
│   └── router/semantic_router.py   # Query tier routing
│
├── photosphere/             # Layer IV: API Gateway
│   ├── api/main.py          # FastAPI app + middleware
│   └── api/routes/
│       ├── ingest.py        # POST /v1/ingest + status tracking
│       ├── query.py         # POST /v1/query (full RAG pipeline)
│       ├── agents.py        # POST /v1/agent/* endpoints
│       ├── health.py        # GET /health /ready /live
│       ├── admin.py         # GET /v1/admin/stats /sla
│       └── mcp.py           # MCP tool server
│
├── corona/                  # Layer V: Observability
│   ├── telemetry/tracer.py  # OpenTelemetry setup
│   ├── telemetry/enforcement.py  # Agent kill-switch
│   └── signing/ecdsa_sidecar.py  # Webhook ECDSA signing
│
├── shared/                  # Cross-layer utilities
│   ├── config/settings.py   # All environment variables
│   ├── models/entities.py   # All Pydantic data models
│   └── utils/               # Correlation ID, structured logging
│
├── infra/                   # Infrastructure config
│   ├── docker/              # Dockerfiles, redis.conf, redpanda config
│   ├── grafana/             # Grafana auto-provisioning
│   ├── prometheus/          # Prometheus scrape config + alerts
│   ├── otel/                # OpenTelemetry Collector config
│   └── kong/                # Kong gateway config
│
├── scripts/
│   ├── health_check.py      # Layer-by-layer health check
│   ├── smoke_test.py        # 30-assertion end-to-end test
│   └── init_postgres.sql    # pgvector schema (v_blue + v_green tables)
│
├── tests/                   # Unit + Integration + E2E tests
├── docker-compose.yml       # Full local development stack
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variable template
```

---

## Configuration Reference

Copy `.env.example` to `.env` and configure.

**Required:**

| Variable | Description |
|----------|-------------|
| `SI_SECRET_KEY` | App secret — run `openssl rand -hex 32` |
| `GROQ_API_KEY` | Groq API key from console.groq.com |
| `LLM_PROVIDER` | `groq` or `openai` |
| `VECTOR_DIMENSION` | `384` (bge-small-en-v1.5) or `1024` (bge-m3) |

**LLM:**

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `COT_HARD_TOKEN_LIMIT` | `8192` | Max tokens for chain-of-thought |
| `STANDARD_HARD_TOKEN_LIMIT` | `4096` | Max tokens for standard query |
| `OPENAI_HARD_COST_CEILING_USD` | `50.0` | Emergency cost ceiling |

**Infrastructure:**

| Variable | Default |
|----------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | `si-redpanda:29092` |
| `POSTGRES_HOST` | `si-postgres` |
| `FALKORDB_HOST` | `si-falkordb` |
| `REDIS_HOST` | `si-redis` |
| `EMBEDDING_HOST` | `si-tgi` |

---

## Troubleshooting

**si-worker: `No module named 'shared'`**
Add `PYTHONPATH: /app` to si-worker environment in docker-compose.yml.

**Redpanda Console 500**
Create topics with `rpk topic create`, then restart `redpanda-console`.
Verify `infra/docker/redpanda-console.yaml` contains `connect.enabled: false`.

**Documents stuck as `pending`**
Check si-worker is consuming: `docker compose logs --tail=30 si-worker`
Look for: `fusion_worker_consume_loop_started`

**pgvector dimension mismatch**
`VECTOR_DIMENSION` in `.env` must match `init_postgres.sql` and TGI model.
`bge-small-en-v1.5` = 384 dims. `bge-m3` = 1024 dims.
To reset: `docker compose down && docker volume rm solar-intelligence_postgres-data && docker compose up -d`

**RedisInsight not loading on port 8001**
Remove `command: redis-server ...` from redis service in docker-compose.yml.
The redis-stack image starts RedisInsight automatically when command is not overridden.

**GraphRAG `graphrag_ingestion_failed`**
Non-fatal — documents are still vector-searchable. Common causes:
- FalkorDB not healthy
- Groq API key missing or invalid
Check: `docker compose logs si-worker | grep graphrag`

---

## Free Tier Usage (GitHub Student Dev Pack)

| Service | Free Tier | SI Usage | Safe? |
|---------|-----------|---------|-------|
| Groq API | 14,400 req/day | ~150 req/day agent mode | ✅ |
| DigitalOcean | $200 credit | $2.40 per 72h run | ✅ 83 runs |
| FalkorDB | Open source | Self-hosted | ✅ |
| pgvector | Open source | Self-hosted | ✅ |
| Redpanda | Open source | Self-hosted | ✅ |
| TGI + BGE | Open source | Self-hosted | ✅ |
| KU API | 500 calls/month | 144 calls per 72h | ✅ |

**Total cost for a 72-hour autonomous run: $0 locally, ~$2.40 on DigitalOcean.**

---

## Contributing

Areas with highest impact:

1. **Full RAG context** — store raw document text in postgres alongside vectors so query answers include full document text, not just entity IDs
2. **Weaviate integration** — add as alternative vector backend
3. **LangChain integration** — expose SI as a LangChain retriever and tool
4. **Financial data connector** — Alpha Vantage / Polygon.io for XAUUSD, equity sentiment monitoring
5. **Grafana dashboards** — pre-built SI dashboard JSON (token spend, Kafka lag, cache hit rate, agent cycle metrics)
6. **arXiv paper** — co-author the Temporal Fidelity Index benchmark

```bash
git clone https://github.com/VLSiddarth/solar-intelligence-backend.git
cd solar-intelligence-backend
git checkout -b feature/your-feature
# make changes, add tests
git push origin feature/your-feature
# open a Pull Request
```

---

## License

MIT License. See [LICENSE](LICENSE).

SI Platform is free and open source. The Knowledge Universe API data network is a separate commercial service.

---

## Built With

[Redpanda](https://redpanda.com) · [FalkorDB](https://falkordb.com) · [pgvector](https://github.com/pgvector/pgvector) · [HuggingFace TGI](https://github.com/huggingface/text-embeddings-inference) · [Groq](https://groq.com) · [FastAPI](https://fastapi.tiangolo.com) · [OpenTelemetry](https://opentelemetry.io) · [Jaeger](https://jaegertracing.io) · [Grafana](https://grafana.com) · [Prometheus](https://prometheus.io)

---

<div align="center">

**☀ Solar Intelligence — The knowledge backbone for autonomous agents.**

*Every answer traces back to a real document. Every document has a timestamp. Every timestamp has meaning.*

[⭐ Star this repo](https://github.com/VLSiddarth/solar-intelligence-backend) · [🐛 Report a Bug](https://github.com/VLSiddarth/solar-intelligence-backend/issues) · [💡 Request a Feature](https://github.com/VLSiddarth/solar-intelligence-backend/issues)

</div>