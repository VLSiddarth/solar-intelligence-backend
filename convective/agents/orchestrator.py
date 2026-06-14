# ============================================================
# convective/agents/orchestrator.py
# Multi-hop agent orchestrator
# Routes → executes → checkpoints state → loops until done
# ============================================================

import uuid
import time
import asyncio
from typing import Optional, Any

from convective.router.semantic_router import get_router
from convective.state.agent_state import get_state_store
from convective.agents.token_governor import get_governor
from radiative.vector.index_manager import BlueGreenIndexManager
from radiative.embeddings.client import get_embedding_client
from radiative.cache.semantic_cache import SemanticCache
from shared.config.settings import settings
from shared.models.entities import (
    RoutingRequest, QueryMode, QueryResponse, RouteTier,
    AgentState,
)
from shared.utils.correlation import get_correlation_id, get_tenant_id, set_layer
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Multi-hop agent orchestrator for the Convective Zone.

    Flow per query:
        1. Check semantic cache (hit → return immediately)
        2. Route to appropriate agent
        3. Acquire hop lock
        4. Execute hop (LLM call with token governance)
        5. Checkpoint state in Redis
        6. Repeat until done or max_hops reached
        7. Cache result
        8. Return structured response
    """

    MAX_HOPS = 10

    def __init__(
        self,
        vector_index: BlueGreenIndexManager,
        cache: SemanticCache,
    ):
        self._router    = get_router()
        self._state     = get_state_store()
        self._governor  = get_governor()
        self._embedder  = get_embedding_client()
        self._index     = vector_index
        self._cache     = cache
        set_layer("convective")

    async def execute(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        mode_override: Optional[QueryMode] = None,
    ) -> QueryResponse:
        """
        Main entry point — execute a query through the full Convective Zone.
        """
        t_start        = time.monotonic()
        tenant         = tenant_id or get_tenant_id()
        cid            = get_correlation_id()
        workflow_id    = session_id or str(uuid.uuid4())

        # ── Step 1: Semantic cache check ──────────────────
        cached = await self._cache.get(query, tenant_id=tenant)
        if cached:
            response_text, similarity = cached
            latency = (time.monotonic() - t_start) * 1000
            from shared.models.entities import TokenUsage
            return QueryResponse(
                answer=response_text,
                sources=[],
                routing_tier=RouteTier.SEMANTIC_ONLY,
                from_cache=True,
                token_usage=TokenUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    cost_usd=0.0, mode=QueryMode.STANDARD,
                ),
                correlation_id=cid,
                latency_ms=round(latency, 1),
            )

        # ── Step 2: Route the query ────────────────────────
        routing_req = RoutingRequest(
            query=query,
            tenant_id=tenant,
            correlation_id=cid,
            mode_hint=mode_override,
        )
        decision = await self._router.route(routing_req)
        mode     = mode_override or decision.query_mode

        # ── Step 3: Resume or init workflow state ──────────
        existing_state = self._state.resume(workflow_id, tenant)
        if existing_state and existing_state.status == "running":
            hop        = existing_state.current_hop
            ctx_state  = existing_state.state
            logger.info("agent_workflow_resumed", extra={
                "workflow_id": workflow_id,
                "at_hop":      hop,
            })
        else:
            hop       = 0
            ctx_state = {"query": query, "history": [], "sources": []}

        # ── Step 4: Execute hops ───────────────────────────
        final_answer = ""
        total_usage  = None

        while hop < self.MAX_HOPS:
            # Distributed lock — one process per hop
            lock_acquired = self._state.acquire_hop_lock(workflow_id, tenant)
            if not lock_acquired:
                logger.warning("hop_lock_not_acquired_waiting", extra={
                    "workflow_id": workflow_id, "hop": hop
                })
                await asyncio.sleep(1)
                continue

            try:
                # Retrieve relevant context from vector index
                # FIXED: TGI is ready. Run real vector search.
                query_embedding = []
                try:
                    query_embedding = await self._embedder.embed_single(query)
                except Exception as e:
                    logger.warning("orchestrator_embed_failed", extra={"error": str(e)})

                search_results = []
                if query_embedding:
                    try:
                        search_results = self._index.query(
                            query_embedding, tenant_id=tenant, top_k=10
                        )
                    except Exception as e:
                        logger.warning("orchestrator_vector_search_failed", extra={"error": str(e)})

                ctx_state["sources"]  = search_results
                ctx_state["hop"]      = hop

                # Build messages for LLM
                messages = self._build_messages(
                    query=query,
                    agent=decision.agent,
                    context=search_results,
                    history=ctx_state.get("history", []),
                    mode=mode,
                )

                # Governed LLM call
                llm_response = await self._governor.complete(
                    messages=messages,
                    mode=mode,
                    correlation_id=cid,
                )

                final_answer = llm_response.content
                total_usage  = llm_response.usage

                # Update history
                ctx_state["history"].append({
                    "role":    "assistant",
                    "content": final_answer[:500],  # Truncate for state storage
                    "hop":     hop,
                })

                # Checkpoint state
                self._state.checkpoint(
                    workflow_id=workflow_id,
                    hop=hop + 1,
                    state=ctx_state,
                    tenant_id=tenant,
                    correlation_id=cid,
                )

                # Check if we have a complete answer (single-hop for most queries)
                if self._is_answer_complete(final_answer, mode):
                    break

                hop += 1

            finally:
                self._state.release_hop_lock(workflow_id, tenant)

        # ── Step 5: Mark complete ──────────────────────────
        self._state.complete(workflow_id, {"final_answer": final_answer}, tenant)

        # ── Step 6: Cache the result ───────────────────────
        if final_answer and not total_usage.truncated:
            await self._cache.put(query, final_answer, tenant_id=tenant, tier="warm")

        latency = (time.monotonic() - t_start) * 1000

        logger.info("agent_orchestration_complete", extra={
            "workflow_id":  workflow_id,
            "agent":        decision.agent,
            "hops":         hop + 1,
            "latency_ms":   round(latency, 1),
            "from_cache":   False,
            "tokens":       total_usage.total_tokens if total_usage else 0,
        })

        return QueryResponse(
            answer=final_answer,
            sources=ctx_state.get("sources", []),
            routing_tier=decision.tier,
            from_cache=False,
            token_usage=total_usage,
            correlation_id=cid,
            latency_ms=round(latency, 1),
        )

    def _build_messages(
        self,
        query: str,
        agent: str,
        context: list[dict],
        history: list[dict],
        mode: QueryMode,
    ) -> list[dict]:
        """Build the message array for the LLM call."""
        system_prompts = {
            "knowledge_graph_agent": (
                "You are a knowledge graph reasoning agent for Solar Intelligence. "
                "Traverse entity relationships to answer the query. "
                "Be precise about connections and cite the specific entities you traverse."
            ),
            "semantic_search_agent": (
                "You are a semantic search agent for Solar Intelligence. "
                "Use the provided context to answer the query accurately. "
                "If context is insufficient, say so clearly."
            ),
            "synthesis_agent": (
                "You are a knowledge synthesis agent for Solar Intelligence. "
                "Synthesize information from multiple sources into a comprehensive answer. "
                "Highlight key insights and note any contradictions."
            ),
            "analytics_agent": (
                "You are a data analytics agent for Solar Intelligence. "
                "Perform precise calculations and analysis. "
                "Show your reasoning step by step."
            ),
            "classification_agent": (
                "You are a classification agent for Solar Intelligence. "
                "Classify the query into the most appropriate category. "
                "Be concise and definitive."
            ),
        }

        system = system_prompts.get(agent, system_prompts["semantic_search_agent"])

        if mode == QueryMode.CHAIN_OF_THOUGHT:
            system += "\n\nThink step by step. Show your reasoning before giving the final answer."

        context_text = "\n\n".join([
            f"[Source {i+1}] entity_id={r.get('entity_id','?')} score={r.get('score',0):.3f}"
            for i, r in enumerate(context[:5])
        ]) or "No relevant context found in knowledge graph."

        messages = [{"role": "system", "content": system}]

        # Include recent history (last 3 hops)
        for h in history[-3:]:
            messages.append({"role": "assistant", "content": h["content"]})

        messages.append({
            "role": "user",
            "content": f"Context from knowledge graph:\n{context_text}\n\nQuery: {query}",
        })

        return messages

    def _is_answer_complete(self, answer: str, mode: QueryMode) -> bool:
        """
        Determine if the current answer is complete enough to stop.
        For edge/standard mode: always stop after 1 hop.
        For CoT: stop when answer has substance (>100 chars).
        """
        if mode in (QueryMode.EDGE, QueryMode.STANDARD):
            return True
        return len(answer) > 100 and not answer.endswith("...")