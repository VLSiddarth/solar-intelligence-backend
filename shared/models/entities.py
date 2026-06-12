# ============================================================
# shared/models/entities.py — FIXED
#
# BUG FIXED: LLMResponse was missing `truncated` and `provider`
#   fields. token_governor.py creates LLMResponse(...,
#   truncated=truncated, provider=self._provider) but these
#   fields didn't exist in the model — causing ValidationError
#   every time the full orchestrator path was exercised.
#
# ALSO: TokenUsage.mode is now Optional so cache-hit responses
#   (which don't have a mode) don't fail validation.
# ============================================================

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Literal, Optional
from datetime import datetime, date
from enum import Enum
import uuid


# ─────────────────────────────────────────────────────────────
# Core Enums
# ─────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    PERSON       = "Person"
    ORGANIZATION = "Organization"
    TECHNOLOGY   = "Technology"
    PROJECT      = "Project"
    DOCUMENT     = "Document"
    CONCEPT      = "Concept"
    EVENT        = "Event"
    METRIC       = "Metric"
    LOCATION     = "Location"
    PRODUCT      = "Product"
    REGULATION   = "Regulation"


class RelationshipType(str, Enum):
    HAS_ROLE    = "HAS_ROLE"
    AUTHORED_BY = "AUTHORED_BY"
    RELATES_TO  = "RELATES_TO"
    USES        = "USES"
    BELONGS_TO  = "BELONGS_TO"
    MENTIONS    = "MENTIONS"
    GOVERNS     = "GOVERNS"
    MEASURES    = "MEASURES"


class RouteTier(str, Enum):
    VLLM_SEMANTIC = "vllm_semantic"
    SEMANTIC_ONLY = "semantic_only"
    RULE_BASED    = "rule_based"


class QueryMode(str, Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STANDARD         = "standard_query"
    EDGE             = "edge_inference"


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


# ─────────────────────────────────────────────────────────────
# Layer I: Core — Data Fusion Models
# ─────────────────────────────────────────────────────────────

class RawDocument(BaseModel):
    """Input to the Core fusion pipeline."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_url: Optional[str] = None
    content: str
    title: Optional[str] = None
    tenant_id: str
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()


class ExtractedEntity(BaseModel):
    """An entity extracted from a document by the NLP pipeline."""
    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    name: str
    fields: dict[str, Any] = Field(default_factory=dict)
    source_doc_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    correlation_id: str


class ExtractedRelationship(BaseModel):
    """A relationship between two extracted entities."""
    rel_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    relationship_type: RelationshipType
    from_entity_id: str
    to_entity_id: str
    fields: dict[str, Any] = Field(default_factory=dict)
    source_doc_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class FusedKnowledge(BaseModel):
    """Output of the Core fusion layer — goes to Radiative Zone."""
    fusion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
    source_doc_id: str
    tenant_id: str
    correlation_id: str
    fused_at: datetime = Field(default_factory=datetime.utcnow)
    token_cost: int = 0
    model_used: str = "llama-3.1-8b-instant"


class DedupCandidate(BaseModel):
    """Two entities flagged for potential merge."""
    entity_a_id: str
    entity_b_id: str
    cosine_similarity: float
    requires_review: bool


# ─────────────────────────────────────────────────────────────
# Layer II: Radiative — Vector Models
# ─────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    texts: list[str]
    model: str = "BAAI/bge-small-en-v1.5"
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    token_count: int
    latency_ms: float


class VectorRecord(BaseModel):
    """A vector stored in the Radiative Zone."""
    vector_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str
    tenant_id: str
    embedding: list[float]
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    index_version: str = "v_blue"


class SearchRequest(BaseModel):
    query: str
    tenant_id: str
    top_k: int = Field(10, ge=1, le=100)
    recall_target: float = Field(0.87, ge=0.0, le=1.0)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SearchResult(BaseModel):
    vector_id: str
    entity_id: str
    score: float
    metadata: dict[str, Any]
    from_cache: bool = False
    latency_ms: float = 0.0


class CacheEntry(BaseModel):
    query_hash: str
    query_embedding: list[float]
    response: str
    hits: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    tenant_id: str


# ─────────────────────────────────────────────────────────────
# Layer III: Convective — Routing Models
# ─────────────────────────────────────────────────────────────

class RoutingRequest(BaseModel):
    query: str
    tenant_id: str
    correlation_id: str
    mode_hint: Optional[QueryMode] = None
    session_id: Optional[str] = None


class RoutingDecision(BaseModel):
    agent: str
    tier: RouteTier
    confidence: float
    query_mode: QueryMode
    reasoning: Optional[str] = None
    latency_ms: float = 0.0


class AgentState(BaseModel):
    """Persisted state for a multi-hop agent workflow."""
    workflow_id: str
    current_hop: int = 0
    max_hops: int = 10
    state: dict[str, Any] = Field(default_factory=dict)
    checkpointed_at: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str
    tenant_id: str
    status: Literal["running", "completed", "failed", "timed_out"] = "running"


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    mode: Optional[QueryMode] = None    # Optional: cache hits don't have a mode
    truncated: bool = False


class LLMResponse(BaseModel):
    content: str
    usage: TokenUsage
    model: str
    latency_ms: float = 0.0
    correlation_id: str = ""
    # FIXED: Added truncated and provider — token_governor creates these fields
    truncated: bool = False
    provider: str = ""


# ─────────────────────────────────────────────────────────────
# Layer IV: Photosphere — API Models
# ─────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Public API: ingest a document into SI."""
    content: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Content too short — minimum 10 characters")
        return v.strip()


class QueryRequest(BaseModel):
    """Public API: query the SI knowledge graph."""
    query: str
    mode: QueryMode = QueryMode.STANDARD
    top_k: int = Field(10, ge=1, le=50)
    include_sources: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    routing_tier: RouteTier
    from_cache: bool
    token_usage: TokenUsage
    correlation_id: str
    latency_ms: float


class MCPError(BaseModel):
    """Standard MCP error response — always returned on failure."""
    jsonrpc: str = "2.0"
    error: dict[str, Any]

    @classmethod
    def build(cls, code: int, message: str, request_id: str, layer: str = "photosphere") -> "MCPError":
        ERROR_TYPES = {
            4000: "schema_validation_error",
            4001: "missing_required_field",
            4002: "invalid_field_type",
            4003: "unauthorized_tool_call",
            5000: "upstream_llm_error",
            5001: "vector_store_unavailable",
            5002: "graph_synthesis_failed",
            5003: "routing_timeout",
        }
        return cls(
            error={
                "code":       code,
                "type":       ERROR_TYPES.get(code, "unknown_error"),
                "message":    message,
                "request_id": request_id,
                "layer":      layer,
                "timestamp":  datetime.utcnow().isoformat(),
            }
        )


class HealthStatus(BaseModel):
    service: str
    status: Literal["healthy", "degraded", "critical"]
    layers: dict[str, str]
    readiness_pct: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


# ─────────────────────────────────────────────────────────────
# Layer V: Corona — Telemetry Models
# ─────────────────────────────────────────────────────────────

class TraceSpan(BaseModel):
    """A single trace span propagated through all 5 layers."""
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str
    correlation_id: str
    parent_span_id: Optional[str] = None
    layer: Literal["core", "radiative", "convective", "photosphere", "corona"]
    operation: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "unset"] = "unset"
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None


class WebhookEvent(BaseModel):
    """An event to be delivered via webhook."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    payload: dict[str, Any]
    attempt: int = 0
    max_attempts: int = 3
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str
    correlation_id: str


class EnforcementAction(BaseModel):
    """Corona enforcement — kill or redirect a misbehaving agent."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    agent_id: str
    action: Literal["kill", "redirect", "warn"]
    reason: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float
    correlation_id: str