# ============================================================
# core/graphrag/pipeline.py  (UPDATED — uses llm provider abstraction)
# GraphRAG ingestion pipeline — document → knowledge graph
# Now routes through core.llm.provider instead of OpenAI directly.
# This is what makes Groq work for entity extraction.
# ============================================================

import time
import asyncio
import json
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from falkordb import FalkorDB

from core.llm.provider import get_llm_client, get_model_name
from core.ontology.schema import validator, ENTITY_SCHEMAS
from shared.config.settings import settings
from shared.models.entities import (
    RawDocument, ExtractedEntity, ExtractedRelationship,
    FusedKnowledge, EntityType, RelationshipType, DedupCandidate,
)
from shared.utils.correlation import get_correlation_id, get_tenant_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Circuit Breaker (unchanged)
# ─────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    error_threshold_pct: float = 30.0
    window_seconds: float = 10.0
    half_open_max_calls: int = 3
    recovery_timeout_seconds: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _errors: list[float] = field(default_factory=list, init=False)
    _calls:  list[float] = field(default_factory=list, init=False)
    _open_since: float   = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)

    def record_success(self) -> None:
        now = time.monotonic()
        self._calls.append(now)
        self._prune(now)
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._half_open_calls = 0
                logger.info("circuit_breaker_closed")

    def record_failure(self) -> None:
        now = time.monotonic()
        self._errors.append(now)
        self._calls.append(now)
        self._prune(now)
        self._evaluate()

    def is_open(self) -> bool:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._open_since > self.recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("circuit_breaker_half_open")
                return False
            return True
        return False

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._errors = [t for t in self._errors if t > cutoff]
        self._calls  = [t for t in self._calls  if t > cutoff]

    def _evaluate(self) -> None:
        if len(self._calls) < 5:
            return
        rate = len(self._errors) / len(self._calls) * 100
        if rate > self.error_threshold_pct and self._state == CircuitState.CLOSED:
            self._state     = CircuitState.OPEN
            self._open_since = time.monotonic()
            logger.error("circuit_breaker_tripped", extra={
                "error_rate_pct": round(rate, 1),
                "threshold_pct":  self.error_threshold_pct,
            })


# ─────────────────────────────────────────────────────────────
# Cost Tracker (now tracks Groq token usage)
# ─────────────────────────────────────────────────────────────

class CostTracker:
    # Groq pricing is ~10x cheaper than GPT-4o-mini
    # llama-3.1-8b-instant: $0.05/M input, $0.08/M output
    COST_PER_1K_INPUT  = 0.00005
    COST_PER_1K_OUTPUT = 0.00008

    def __init__(self, ceiling_usd: float):
        self.ceiling_usd  = ceiling_usd
        self.total_cost   = 0.0
        self.total_tokens = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        cost = (
            (prompt_tokens     / 1000) * self.COST_PER_1K_INPUT +
            (completion_tokens / 1000) * self.COST_PER_1K_OUTPUT
        )
        self.total_cost   += cost
        self.total_tokens += prompt_tokens + completion_tokens

        if self.total_cost >= self.ceiling_usd:
            logger.error("llm_cost_ceiling_hit", extra={
                "total_cost_usd": round(self.total_cost, 4),
                "ceiling_usd":    self.ceiling_usd,
            })
            raise RuntimeError(
                f"LLM cost ceiling reached: ${self.total_cost:.4f} >= ${self.ceiling_usd:.2f}"
            )

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens":   self.total_tokens,
            "ceiling_usd":    self.ceiling_usd,
            "pct_used":       round(self.total_cost / max(self.ceiling_usd, 0.001) * 100, 2),
        }


# ─────────────────────────────────────────────────────────────
# Extraction prompt (unchanged)
# ─────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a knowledge graph extraction engine for the Solar Intelligence system.

Extract entities and relationships from the following document.
Return ONLY valid JSON — no preamble, no markdown, no code fences.

Entity types allowed: {entity_types}
Relationship types allowed: {rel_types}

Document:
{content}

Return this exact JSON structure:
{{
  "entities": [
    {{"name": "...", "type": "Person|Organization|...", "fields": {{...}}}}
  ],
  "relationships": [
    {{"from_name": "...", "to_name": "...", "type": "HAS_ROLE|...", "fields": {{...}}}}
  ]
}}"""

ENTITY_TYPES_STR = ", ".join([e.value for e in EntityType])
REL_TYPES_STR    = ", ".join([r.value for r in RelationshipType])


# ─────────────────────────────────────────────────────────────
# GraphRAG Pipeline
# ─────────────────────────────────────────────────────────────

class GraphRAGPipeline:
    """
    Full GraphRAG ingestion pipeline.
    Document → NLP extraction (via configured LLM) → entity resolution → FalkorDB
    """

    def __init__(self):
        # KEY CHANGE: use provider abstraction instead of hardcoded OpenAI
        self.llm      = get_llm_client()
        self.model    = get_model_name()
        self.graph_db = FalkorDB(
            host=settings.graphrag.falkordb_host,
            port=settings.graphrag.falkordb_port,
        )
        self.graph    = self.graph_db.select_graph("si_knowledge")
        self.breaker  = CircuitBreaker()
        self.cost     = CostTracker(settings.openai.hard_cost_ceiling_usd)
        self._entity_cache: dict[str, str] = {}

        logger.info("graphrag_pipeline_initialized", extra={
            "llm_model": self.model,
            "falkordb":  f"{settings.graphrag.falkordb_host}:{settings.graphrag.falkordb_port}",
        })

    async def ingest(self, doc: RawDocument) -> FusedKnowledge:
        if self.breaker.is_open():
            logger.warning("graphrag_circuit_open", extra={"doc_id": doc.doc_id})
            raise RuntimeError("GraphRAG circuit breaker is OPEN")

        t_start = time.monotonic()

        try:
            entities_raw, rels_raw, usage = await self._extract(doc)
            entities, dedup_candidates    = self._resolve_entities(entities_raw, doc)
            validated_entities            = self._validate_entities(entities, doc)
            validated_rels                = self._validate_relationships(rels_raw, validated_entities, doc)
            self._write_to_graph(validated_entities, validated_rels, doc.tenant_id)
            self.breaker.record_success()

            latency = (time.monotonic() - t_start) * 1000
            logger.info("graphrag_ingestion_complete", extra={
                "doc_id":           doc.doc_id,
                "entities":         len(validated_entities),
                "relationships":    len(validated_rels),
                "dedup_candidates": len(dedup_candidates),
                "latency_ms":       round(latency, 1),
                "cost":             self.cost.summary(),
                "model":            self.model,
            })

            return FusedKnowledge(
                entities=validated_entities,
                relationships=validated_rels,
                source_doc_id=doc.doc_id,
                tenant_id=doc.tenant_id,
                correlation_id=doc.correlation_id,
                token_cost=usage.get("total_tokens", 0),
                model_used=self.model,
            )

        except RuntimeError:
            self.breaker.record_failure()
            raise
        except Exception as e:
            self.breaker.record_failure()
            logger.error("graphrag_ingestion_failed", extra={"doc_id": doc.doc_id, "error": str(e)})
            raise

    async def _extract(self, doc: RawDocument) -> tuple[list, list, dict]:
        prompt = EXTRACTION_PROMPT.format(
            entity_types=ENTITY_TYPES_STR,
            rel_types=REL_TYPES_STR,
            content=doc.content[:3000],
        )

        # Groq supports JSON mode via response_format
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        usage = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }
        self.cost.record(usage["prompt_tokens"], usage["completion_tokens"])

        result = json.loads(response.choices[0].message.content)
        return result.get("entities", []), result.get("relationships", []), usage

    def _resolve_entities(self, entities_raw: list, doc: RawDocument) -> tuple:
        resolved = []
        dedup_candidates = []

        for raw in entities_raw:
            name = raw.get("name", "").strip()
            entity_type_str = raw.get("type", "")

            if not name:
                continue

            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                logger.warning("unknown_entity_type", extra={"type": entity_type_str})
                continue

            cache_key = f"{entity_type.value}:{name.lower()}"
            if cache_key in self._entity_cache:
                existing_id = self._entity_cache[cache_key]
                dedup_candidates.append(DedupCandidate(
                    entity_a_id=existing_id,
                    entity_b_id=f"new:{cache_key}",
                    cosine_similarity=1.0,
                    requires_review=False,
                ))
                continue

            import uuid
            entity_id = str(uuid.uuid4())
            self._entity_cache[cache_key] = entity_id

            entity = ExtractedEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                name=name,
                fields=raw.get("fields", {}),
                source_doc_id=doc.doc_id,
                confidence=raw.get("confidence", 0.8),
                correlation_id=doc.correlation_id,
            )
            resolved.append(entity)

        return resolved, dedup_candidates

    def _validate_entities(self, entities: list, doc: RawDocument) -> list:
        validated = []
        for entity in entities:
            is_valid, errors = validator.validate_entity(entity.entity_type, entity.fields)
            if is_valid:
                validated.append(entity)
            else:
                logger.warning("entity_validation_failed", extra={
                    "entity_type": entity.entity_type,
                    "name":        entity.name,
                    "errors":      errors,
                    "doc_id":      doc.doc_id,
                })
        return validated

    def _validate_relationships(self, rels_raw: list, entities: list, doc: RawDocument) -> list:
        entity_by_name = {e.name.lower(): e for e in entities}
        validated = []

        for raw in rels_raw:
            from_name    = raw.get("from_name", "").lower()
            to_name      = raw.get("to_name", "").lower()
            rel_type_str = raw.get("type", "")

            if from_name not in entity_by_name or to_name not in entity_by_name:
                continue

            try:
                rel_type = RelationshipType(rel_type_str)
            except ValueError:
                continue

            from_entity = entity_by_name[from_name]
            to_entity   = entity_by_name[to_name]

            is_valid, errors = validator.validate_relationship(
                rel_type, from_entity.entity_type, to_entity.entity_type, raw.get("fields", {})
            )

            if is_valid:
                import uuid
                validated.append(ExtractedRelationship(
                    rel_id=str(uuid.uuid4()),
                    relationship_type=rel_type,
                    from_entity_id=from_entity.entity_id,
                    to_entity_id=to_entity.entity_id,
                    fields=raw.get("fields", {}),
                    source_doc_id=doc.doc_id,
                    confidence=raw.get("confidence", 0.75),
                ))
            else:
                logger.warning("relationship_validation_failed", extra={
                    "rel_type": rel_type_str, "errors": errors,
                })

        return validated

    def _write_to_graph(self, entities: list, relationships: list, tenant_id: str) -> None:
        # FIX: FalkorDB Cypher requires SET n += $map (map param), NOT SET n += {key: $key}
        # The original generated invalid syntax: SET n += {entity_id: $entity_id, name: $name, ...}
        # FalkorDB rejects inline parameter references inside map literals.
        for entity in entities:
            props = {
                "entity_id":  entity.entity_id,
                "name":       entity.name,
                "tenant_id":  tenant_id,
                "source_doc": entity.source_doc_id,
            }
            # Merge any extra fields (sanitise keys — no spaces or special chars)
            for k, v in entity.fields.items():
                safe_key = k.replace(" ", "_").replace("-", "_")
                if safe_key.isidentifier():
                    props[safe_key] = v

            query = (
                f"MERGE (n:{entity.entity_type.value} {{entity_id: $entity_id}}) "
                f"SET n += $props"
            )
            self.graph.query(query, {"entity_id": entity.entity_id, "props": props})

        for rel in relationships:
            query = (
                f"MATCH (a {{entity_id: $from_id}}), (b {{entity_id: $to_id}}) "
                f"MERGE (a)-[r:{rel.relationship_type.value}]->(b) "
                f"SET r.rel_id = $rel_id, r.confidence = $confidence"
            )
            self.graph.query(query, {
                "from_id":    rel.from_entity_id,
                "to_id":      rel.to_entity_id,
                "rel_id":     rel.rel_id,
                "confidence": rel.confidence,
            })