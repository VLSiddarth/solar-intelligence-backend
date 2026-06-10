# ============================================================
# convective/agents/autonomous_agent.py
# 72-Hour Autonomous Agent for Solar Intelligence + KU API
#
# Architecture:
#   Every 30 min:
#     1. Call KU API → discover new/updated documents for configured topics
#     2. Ingest discovered docs into SI pipeline (POST /v1/ingest)
#     3. Wait for fusion worker to process (poll /v1/ingest/{id}/status)
#     4. Run synthesis query (POST /v1/query) — build insight report
#     5. Check decay thresholds — trigger re-index if knowledge is stale
#     6. Checkpoint state to Redis
#   Every 12 hours: generate comprehensive report
#   At 72 hours: final report + graceful shutdown
# ============================================================

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx
import redis

from convective.agents.ku_client import KUAPIClient, KUDocument
from shared.config.settings import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Agent state model
# ─────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    IDLE      = "idle"
    RUNNING   = "running"
    PAUSED    = "paused"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class CycleResult:
    cycle:            int
    timestamp:        str
    topics_searched:  list[str]
    docs_discovered:  int
    docs_ingested:    int
    docs_processed:   int
    synthesis:        str
    decay_alerts:     list[str]
    ku_calls_used:    int
    duration_seconds: float


@dataclass
class AgentState:
    agent_id:         str
    status:           AgentStatus
    goal:             str
    topics:           list[str]
    started_at:       str
    last_cycle_at:    str
    cycle_count:      int
    total_docs:       int
    total_insights:   int
    cycles:           list[dict]            = field(default_factory=list)
    findings:         list[str]             = field(default_factory=list)
    decay_alerts:     list[str]             = field(default_factory=list)
    reports:          list[dict]            = field(default_factory=list)
    error:            Optional[str]         = None

    def to_redis(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, default=str)

    @classmethod
    def from_redis(cls, raw: str) -> "AgentState":
        d = json.loads(raw)
        d["status"] = AgentStatus(d["status"])
        d["cycles"] = d.get("cycles", [])
        return cls(**d)


# ─────────────────────────────────────────────────────────────
# Autonomous Agent
# ─────────────────────────────────────────────────────────────

class AutonomousAgent:
    """
    72-Hour Autonomous Knowledge Monitoring Agent.

    Runs a continuous cycle every 30 minutes for 72 hours.
    Uses KU API for discovery and SI pipeline for processing + synthesis.

    Usage (from API):
        agent = AutonomousAgent()
        asyncio.create_task(agent.run(goal="...", topics=[...]))
    """

    CYCLE_INTERVAL_SECONDS  = 30 * 60       # 30 minutes between cycles
    TOTAL_DURATION_SECONDS  = 72 * 60 * 60  # 72 hours total
    REPORT_INTERVAL_CYCLES  = 24            # Full report every 12 hours (24 × 30min)
    DECAY_ALERT_THRESHOLD   = 0.4           # Decay score below this triggers alert
    PROCESS_WAIT_TIMEOUT    = 60            # Seconds to wait for doc processing
    SI_API_BASE_URL         = "http://localhost:8888"   # Internal API (or si-api:8888 in Docker)

    def __init__(self, agent_id: Optional[str] = None):
        self._agent_id = agent_id or str(uuid.uuid4())
        self._redis    = redis.Redis(
            host=settings.agent.redis_host,
            port=settings.agent.redis_port,
            decode_responses=True,
        )
        self._ku       = KUAPIClient()
        self._http     = httpx.AsyncClient(
            base_url=self.SI_API_BASE_URL,
            timeout=30.0,
        )
        self._state: Optional[AgentState] = None
        self._shutdown_event = asyncio.Event()

    # ─────────────────────────────────────────
    # Public API (called from REST endpoints)
    # ─────────────────────────────────────────

    async def run(self, goal: str, topics: list[str]) -> None:
        """
        Main entry point. Runs the full 72-hour agent loop.
        Call this as an asyncio.create_task() — it runs in the background.
        """
        self._state = AgentState(
            agent_id=self._agent_id,
            status=AgentStatus.RUNNING,
            goal=goal,
            topics=topics,
            started_at=datetime.now(timezone.utc).isoformat(),
            last_cycle_at="",
            cycle_count=0,
            total_docs=0,
            total_insights=0,
        )
        self._save_state()

        logger.info("autonomous_agent_started", extra={
            "agent_id": self._agent_id,
            "goal":     goal,
            "topics":   topics,
            "duration": f"{self.TOTAL_DURATION_SECONDS // 3600}h",
        })

        deadline = time.monotonic() + self.TOTAL_DURATION_SECONDS

        try:
            while time.monotonic() < deadline and not self._shutdown_event.is_set():
                cycle_start = time.monotonic()
                cycle_num   = self._state.cycle_count + 1

                logger.info("agent_cycle_start", extra={
                    "agent_id": self._agent_id,
                    "cycle":    cycle_num,
                    "time_left_hours": round((deadline - time.monotonic()) / 3600, 1),
                })

                result = await self._run_cycle(cycle_num, topics)

                self._state.cycle_count   += 1
                self._state.total_docs    += result.docs_ingested
                self._state.last_cycle_at  = datetime.now(timezone.utc).isoformat()
                self._state.cycles.append(asdict(result))
                self._state.findings.extend(
                    [result.synthesis[:300]] if result.synthesis else []
                )
                self._state.decay_alerts.extend(result.decay_alerts)

                # Periodic comprehensive report every REPORT_INTERVAL_CYCLES cycles
                if cycle_num % self.REPORT_INTERVAL_CYCLES == 0:
                    report = await self._generate_report(cycle_num)
                    self._state.reports.append(report)
                    logger.info("agent_periodic_report_generated", extra={
                        "agent_id": self._agent_id,
                        "cycle":    cycle_num,
                    })

                self._save_state()

                # Wait for next cycle (subtract processing time already spent)
                elapsed = time.monotonic() - cycle_start
                wait    = max(0, self.CYCLE_INTERVAL_SECONDS - elapsed)

                if not self._shutdown_event.is_set() and time.monotonic() + wait < deadline:
                    logger.info("agent_sleeping_until_next_cycle", extra={
                        "agent_id":   self._agent_id,
                        "wait_min":   round(wait / 60, 1),
                        "next_cycle": cycle_num + 1,
                    })
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=wait,
                        )
                    except asyncio.TimeoutError:
                        pass  # Normal — sleep expired, run next cycle

            # 72-hour run complete or shutdown requested
            final_report = await self._generate_report(self._state.cycle_count, final=True)
            self._state.reports.append(final_report)
            self._state.status = AgentStatus.COMPLETED
            self._state.total_insights = len(self._state.findings)
            self._save_state()

            logger.info("autonomous_agent_completed", extra={
                "agent_id":    self._agent_id,
                "cycles":      self._state.cycle_count,
                "total_docs":  self._state.total_docs,
                "insights":    self._state.total_insights,
                "decay_alerts": len(self._state.decay_alerts),
            })

        except Exception as e:
            logger.error("autonomous_agent_fatal_error", extra={
                "agent_id": self._agent_id,
                "error":    str(e),
            })
            if self._state:
                self._state.status = AgentStatus.FAILED
                self._state.error  = str(e)
                self._save_state()
            raise

        finally:
            await self._ku.close()
            await self._http.aclose()

    def pause(self) -> None:
        if self._state:
            self._state.status = AgentStatus.PAUSED
            self._save_state()

    def stop(self) -> None:
        self._shutdown_event.set()

    def get_status(self) -> Optional[dict]:
        raw = self._redis.get(self._redis_key())
        if raw:
            state = AgentState.from_redis(raw)
            return {
                "agent_id":       state.agent_id,
                "status":         state.status.value,
                "goal":           state.goal,
                "topics":         state.topics,
                "started_at":     state.started_at,
                "last_cycle_at":  state.last_cycle_at,
                "cycle_count":    state.cycle_count,
                "total_docs":     state.total_docs,
                "total_insights": state.total_insights,
                "recent_finding": state.findings[-1] if state.findings else None,
                "decay_alerts":   state.decay_alerts[-5:],
                "reports_count":  len(state.reports),
                "latest_report":  state.reports[-1] if state.reports else None,
                "ku_calls":       self._ku.calls_made,
            }
        return None

    # ─────────────────────────────────────────
    # Cycle logic
    # ─────────────────────────────────────────

    async def _run_cycle(self, cycle_num: int, topics: list[str]) -> CycleResult:
        cycle_start   = time.monotonic()
        all_docs:     list[KUDocument] = []
        ingested_ids: list[str]        = []
        decay_alerts: list[str]        = []

        # Step 1 — Discover ─────────────────────────────────
        for topic in topics:
            docs = await self._ku.discover(
                topic=topic,
                since_hours=1,          # Only new content since last cycle
                max_results=5,
            )
            all_docs.extend(docs)

            # Decay threshold check
            stale = [d for d in docs if d.decay_score < self.DECAY_ALERT_THRESHOLD]
            for d in stale:
                alert = (f"[DECAY ALERT] '{d.title}' decay={d.decay_score:.2f} "
                         f"velocity={d.knowledge_velocity:.2f}")
                decay_alerts.append(alert)
                logger.warning("decay_threshold_breached", extra={
                    "doc_title":   d.title,
                    "decay_score": d.decay_score,
                    "topic":       topic,
                })

        logger.info("agent_discovery_done", extra={
            "cycle":         cycle_num,
            "docs_found":    len(all_docs),
            "decay_alerts":  len(decay_alerts),
        })

        # Step 2 — Ingest to SI pipeline ────────────────────
        for doc in all_docs:
            doc_id = await self._ingest_document(doc)
            if doc_id:
                ingested_ids.append(doc_id)

        logger.info("agent_ingest_done", extra={
            "cycle":    cycle_num,
            "ingested": len(ingested_ids),
        })

        # Step 3 — Wait for fusion worker to process ────────
        processed = 0
        if ingested_ids:
            processed = await self._wait_for_processing(ingested_ids)

        # Step 4 — Synthesise insights ──────────────────────
        synthesis = ""
        if self._state and self._state.goal:
            synthesis = await self._synthesise(
                goal=self._state.goal,
                cycle=cycle_num,
                topics=topics,
                new_docs=len(ingested_ids),
            )

        duration = time.monotonic() - cycle_start

        return CycleResult(
            cycle=cycle_num,
            timestamp=datetime.now(timezone.utc).isoformat(),
            topics_searched=topics,
            docs_discovered=len(all_docs),
            docs_ingested=len(ingested_ids),
            docs_processed=processed,
            synthesis=synthesis,
            decay_alerts=decay_alerts,
            ku_calls_used=self._ku.calls_made,
            duration_seconds=round(duration, 1),
        )

    async def _ingest_document(self, doc: KUDocument) -> Optional[str]:
        """POST a KU document to the SI ingest endpoint."""
        try:
            resp = await self._http.post("/v1/ingest", json={
                "content":    doc.content,
                "title":      doc.title,
                "source_url": doc.source_url,
                "metadata":   doc.metadata,
            }, headers={
                "X-Tenant-ID": self._agent_id,
                "Content-Type": "application/json",
            })
            if resp.status_code == 200:
                data = resp.json()
                return data.get("doc_id")
            else:
                logger.warning("agent_ingest_failed", extra={
                    "status": resp.status_code,
                    "title":  doc.title,
                })
        except Exception as e:
            logger.error("agent_ingest_error", extra={"error": str(e), "title": doc.title})
        return None

    async def _wait_for_processing(self, doc_ids: list[str]) -> int:
        """Poll ingest status until docs are processed or timeout."""
        processed = 0
        deadline  = time.monotonic() + self.PROCESS_WAIT_TIMEOUT

        while time.monotonic() < deadline:
            remaining = []
            for doc_id in doc_ids:
                try:
                    resp = await self._http.get(
                        f"/v1/ingest/{doc_id}/status",
                        headers={"X-Tenant-ID": self._agent_id},
                    )
                    if resp.status_code == 200:
                        status = resp.json().get("status", "unknown")
                        if status == "processed":
                            processed += 1
                        elif status == "pending":
                            remaining.append(doc_id)
                except Exception:
                    remaining.append(doc_id)

            if not remaining:
                break
            doc_ids = remaining
            await asyncio.sleep(5)

        return processed

    async def _synthesise(
        self,
        goal: str,
        cycle: int,
        topics: list[str],
        new_docs: int,
    ) -> str:
        """
        Query the SI pipeline to synthesise insights from the latest ingested data.
        Uses chain-of-thought mode for deeper reasoning.
        """
        query = (
            f"Based on the latest knowledge about {', '.join(topics)}, "
            f"what are the most important developments and insights relevant to: {goal}? "
            f"Focus on information from the most recent monitoring cycle (cycle {cycle}). "
            f"New documents ingested this cycle: {new_docs}."
        )

        try:
            resp = await self._http.post("/v1/query", json={
                "query":   query,
                "mode":    "chain_of_thought",
            }, headers={
                "X-Tenant-ID":  self._agent_id,
                "Content-Type": "application/json",
            })
            if resp.status_code == 200:
                data = resp.json()
                return data.get("answer", "")
        except Exception as e:
            logger.warning("agent_synthesis_failed", extra={"error": str(e)})
        return ""

    async def _generate_report(self, cycle: int, final: bool = False) -> dict:
        """Generate a comprehensive summary report."""
        if not self._state:
            return {}

        recent_cycles   = self._state.cycles[-self.REPORT_INTERVAL_CYCLES:]
        total_docs      = sum(c.get("docs_ingested", 0) for c in recent_cycles)
        total_processed = sum(c.get("docs_processed", 0) for c in recent_cycles)
        recent_findings = self._state.findings[-10:]

        report_query = (
            f"Generate a comprehensive {'final ' if final else ''}intelligence report for: "
            f"{self._state.goal}. "
            f"Synthesize all knowledge gathered about: {', '.join(self._state.topics)}. "
            f"Recent findings: {'; '.join(recent_findings[:3])}. "
            f"Highlight key trends, risks, and actionable insights."
        )

        summary = await self._synthesise(
            goal=report_query,
            cycle=cycle,
            topics=self._state.topics,
            new_docs=total_docs,
        )

        return {
            "type":           "final_report" if final else "periodic_report",
            "cycle":          cycle,
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "docs_this_period": total_docs,
            "processed":      total_processed,
            "decay_alerts":   self._state.decay_alerts[-20:],
            "summary":        summary,
            "key_findings":   recent_findings,
        }

    # ─────────────────────────────────────────
    # Redis state persistence
    # ─────────────────────────────────────────

    def _redis_key(self) -> str:
        return f"si:agent:{self._agent_id}:state"

    def _save_state(self) -> None:
        if self._state:
            try:
                # TTL = 80 hours (72h run + 8h to read results)
                self._redis.setex(
                    self._redis_key(),
                    80 * 3600,
                    self._state.to_redis(),
                )
            except Exception as e:
                logger.warning("agent_state_save_failed", extra={"error": str(e)})

    @classmethod
    def load_from_redis(cls, agent_id: str) -> Optional["AutonomousAgent"]:
        """Reload an agent from its persisted Redis state."""
        agent = cls(agent_id=agent_id)
        raw = agent._redis.get(agent._redis_key())
        if raw:
            agent._state = AgentState.from_redis(raw)
            return agent
        return None


# ─────────────────────────────────────────────────────────────
# Global agent registry (in-memory for current process)
# ─────────────────────────────────────────────────────────────

_active_agents: dict[str, AutonomousAgent] = {}


def register_agent(agent: AutonomousAgent) -> None:
    _active_agents[agent._agent_id] = agent


def get_agent(agent_id: str) -> Optional[AutonomousAgent]:
    return _active_agents.get(agent_id)


def list_agents() -> list[dict]:
    return [
        agent.get_status() or {"agent_id": aid, "status": "unknown"}
        for aid, agent in _active_agents.items()
    ]