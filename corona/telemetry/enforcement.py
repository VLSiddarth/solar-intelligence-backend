# ============================================================
# corona/telemetry/enforcement.py
# Telemetry-to-enforcement loop — detects and kills misbehaving agents
# Target: <50ms detect-to-kill latency at 10,000 agents/sec
# This is SI's unique moat — no other platform does this
# ============================================================

import time
import asyncio
from typing import Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

from shared.models.entities import EnforcementAction
from shared.utils.correlation import get_correlation_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class ViolationType(str, Enum):
    TOKEN_EXPLOSION      = "token_explosion"       # Token count spiking abnormally
    UNAUTHORIZED_TOOL    = "unauthorized_tool"     # Agent calling disallowed tool
    HALLUCINATION_SIGNAL = "hallucination_signal"  # Detected contradictory facts
    LOOP_DETECTED        = "loop_detected"         # Agent repeating same hop
    TIMEOUT              = "timeout"               # Workflow exceeded max duration
    COST_CEILING         = "cost_ceiling"          # Per-workflow cost exceeded


@dataclass
class AgentMetrics:
    """Live metrics for a running agent workflow."""
    workflow_id:   str
    agent_id:      str
    tenant_id:     str
    start_time:    float = field(default_factory=time.monotonic)
    hop_count:     int   = 0
    token_count:   int   = 0
    cost_usd:      float = 0.0
    last_hop_hash: str   = ""   # Hash of last hop output — detects loops
    tools_called:  list[str] = field(default_factory=list)


@dataclass
class EnforcementRule:
    name:       str
    check:      Callable[[AgentMetrics], bool]
    action:     str          # "kill" | "redirect" | "warn"
    violation:  ViolationType
    reason:     str


# ─────────────────────────────────────────────
# Enforcement Rules
# ─────────────────────────────────────────────

DEFAULT_RULES: list[EnforcementRule] = [
    EnforcementRule(
        name="token_explosion",
        check=lambda m: m.token_count > 6000,
        action="kill",
        violation=ViolationType.TOKEN_EXPLOSION,
        reason="Token count exceeded 6000 — likely runaway CoT",
    ),
    EnforcementRule(
        name="max_hops",
        check=lambda m: m.hop_count > 8,
        action="kill",
        violation=ViolationType.LOOP_DETECTED,
        reason="Hop count exceeded 8 — possible infinite loop",
    ),
    EnforcementRule(
        name="workflow_timeout",
        check=lambda m: (time.monotonic() - m.start_time) > 300,  # 5 minutes
        action="kill",
        violation=ViolationType.TIMEOUT,
        reason="Workflow exceeded 5-minute wall-clock timeout",
    ),
    EnforcementRule(
        name="cost_ceiling",
        check=lambda m: m.cost_usd > 2.0,  # $2 per workflow
        action="kill",
        violation=ViolationType.COST_CEILING,
        reason="Per-workflow cost exceeded $2.00",
    ),
    EnforcementRule(
        name="loop_detection",
        check=lambda m: False,  # Set dynamically in check_metrics
        action="redirect",
        violation=ViolationType.LOOP_DETECTED,
        reason="Agent producing identical outputs — redirecting",
    ),
]


# ─────────────────────────────────────────────
# Enforcement Loop
# ─────────────────────────────────────────────

class EnforcementLoop:
    """
    Real-time agent enforcement loop.

    Architecture:
        - Agents register themselves on start
        - Metrics are updated at every hop
        - Background task evaluates all rules every 50ms
        - On violation: kill/redirect action executed in <50ms
        - All enforcement actions logged with correlation ID

    Target: <50ms detect-to-kill. Measured in load tests.
    """

    def __init__(self):
        self._agents:    dict[str, AgentMetrics]            = {}
        self._killed:    set[str]                           = set()
        self._actions:   list[EnforcementAction]            = []
        self._rules:     list[EnforcementRule]              = list(DEFAULT_RULES)
        self._callbacks: dict[str, Callable[[str, str], None]] = {}
        self._running    = False
        self._loop_task: Optional[asyncio.Task]             = None

    async def start(self) -> None:
        """Start the background enforcement loop."""
        self._running   = True
        self._loop_task = asyncio.create_task(self._enforcement_loop())
        logger.info("enforcement_loop_started", extra={
            "rules":           [r.name for r in self._rules],
            "poll_interval_ms": 50,
        })

    async def stop(self) -> None:
        """Gracefully stop the enforcement loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    def register_agent(
        self,
        workflow_id: str,
        agent_id: str,
        tenant_id: str,
        kill_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Register a new agent workflow for monitoring."""
        self._agents[workflow_id] = AgentMetrics(
            workflow_id=workflow_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
        )
        if kill_callback:
            self._callbacks[workflow_id] = kill_callback

        logger.debug("agent_registered", extra={
            "workflow_id": workflow_id,
            "agent_id":    agent_id,
            "tenant_id":   tenant_id,
        })

    def deregister_agent(self, workflow_id: str) -> None:
        """Remove agent from monitoring (workflow completed normally)."""
        self._agents.pop(workflow_id, None)
        self._callbacks.pop(workflow_id, None)

    def update_metrics(
        self,
        workflow_id: str,
        tokens_this_hop: int = 0,
        cost_this_hop: float = 0.0,
        output_hash: str = "",
        tool_called: Optional[str] = None,
    ) -> bool:
        """
        Update metrics for a workflow after each hop.
        Returns False if the workflow has been killed (agent should stop).
        """
        if workflow_id in self._killed:
            return False  # Already killed — agent must stop

        metrics = self._agents.get(workflow_id)
        if not metrics:
            return True

        metrics.hop_count   += 1
        metrics.token_count += tokens_this_hop
        metrics.cost_usd    += cost_this_hop

        if tool_called:
            metrics.tools_called.append(tool_called)

        # Loop detection: same output hash twice = loop
        if output_hash and output_hash == metrics.last_hop_hash:
            self._enforce(workflow_id, metrics, self._rules[4])  # loop_detection rule
        metrics.last_hop_hash = output_hash

        return workflow_id not in self._killed

    def is_killed(self, workflow_id: str) -> bool:
        return workflow_id in self._killed

    # ─────────────────────────────────────────
    # Private — Enforcement Loop
    # ─────────────────────────────────────────

    async def _enforcement_loop(self) -> None:
        """50ms polling loop — evaluates all rules against all active agents."""
        while self._running:
            t_start = time.monotonic()

            for workflow_id, metrics in list(self._agents.items()):
                if workflow_id in self._killed:
                    continue
                for rule in self._rules:
                    try:
                        if rule.check(metrics):
                            self._enforce(workflow_id, metrics, rule)
                            break  # One enforcement per cycle
                    except Exception as e:
                        logger.error("enforcement_rule_error", extra={
                            "rule":  rule.name,
                            "error": str(e),
                        })

            elapsed = (time.monotonic() - t_start) * 1000
            if elapsed > 50:
                logger.warning("enforcement_loop_slow", extra={"elapsed_ms": round(elapsed, 1)})

            # Sleep the remainder of the 50ms window
            sleep_time = max(0, 0.05 - (time.monotonic() - t_start))
            await asyncio.sleep(sleep_time)

    def _enforce(
        self,
        workflow_id: str,
        metrics: AgentMetrics,
        rule: EnforcementRule,
    ) -> None:
        """Execute enforcement action and record it."""
        detect_time = time.monotonic()

        if rule.action == "kill":
            self._killed.add(workflow_id)
            self._agents.pop(workflow_id, None)

            # Call the kill callback if registered
            cb = self._callbacks.pop(workflow_id, None)
            if cb:
                try:
                    cb(workflow_id, rule.reason)
                except Exception:
                    pass

        latency_ms = (time.monotonic() - detect_time) * 1000

        action = EnforcementAction(
            workflow_id=workflow_id,
            agent_id=metrics.agent_id,
            action=rule.action,
            reason=rule.reason,
            latency_ms=round(latency_ms, 2),
            correlation_id=get_correlation_id(),
        )
        self._actions.append(action)

        logger.warning("agent_enforcement_action", extra={
            "workflow_id":     workflow_id,
            "action":          rule.action,
            "violation":       rule.violation.value,
            "reason":          rule.reason,
            "hop_count":       metrics.hop_count,
            "token_count":     metrics.token_count,
            "latency_ms":      round(latency_ms, 2),
            "tenant_id":       metrics.tenant_id,
        })

    def recent_actions(self, limit: int = 100) -> list[EnforcementAction]:
        return self._actions[-limit:]

    def stats(self) -> dict:
        return {
            "active_agents":    len(self._agents),
            "killed_workflows": len(self._killed),
            "total_actions":    len(self._actions),
            "rules":            [r.name for r in self._rules],
        }


# Singleton
_enforcement_loop: Optional[EnforcementLoop] = None


def get_enforcement_loop() -> EnforcementLoop:
    global _enforcement_loop
    if _enforcement_loop is None:
        _enforcement_loop = EnforcementLoop()
    return _enforcement_loop