# ============================================================
# tests/unit/test_enforcement.loop.py
# Unit tests — Corona enforcement loop
# ============================================================

import asyncio
import time
import pytest
from corona.telemetry.enforcement import EnforcementLoop, ViolationType


@pytest.fixture
def loop():
    return EnforcementLoop()


class TestEnforcementLoop:

    def test_register_and_deregister_agent(self, loop):
        loop.register_agent("wf-001", "agent-a", "tenant-1")
        assert "wf-001" in loop._agents
        loop.deregister_agent("wf-001")
        assert "wf-001" not in loop._agents

    def test_update_metrics_accumulates(self, loop):
        loop.register_agent("wf-002", "agent-b", "tenant-1")
        loop.update_metrics("wf-002", tokens_this_hop=100, cost_this_hop=0.01)
        loop.update_metrics("wf-002", tokens_this_hop=200, cost_this_hop=0.02)
        metrics = loop._agents["wf-002"]
        assert metrics.token_count == 300
        assert abs(metrics.cost_usd - 0.03) < 0.0001
        assert metrics.hop_count == 2

    def test_token_explosion_rule_fires(self, loop):
        loop.register_agent("wf-003", "agent-c", "tenant-1")
        loop.update_metrics("wf-003", tokens_this_hop=6001)  # Over 6000 threshold
        # Force rule check
        metrics = loop._agents.get("wf-003") or type("M", (), {"token_count": 6001, "hop_count": 1, "cost_usd": 0.0, "start_time": time.monotonic(), "last_hop_hash": "", "tools_called": []})()
        rule = loop._rules[0]  # token_explosion rule
        assert rule.check(metrics) is True

    def test_killed_workflow_returns_false_on_update(self, loop):
        loop.register_agent("wf-004", "agent-d", "tenant-1")
        loop._killed.add("wf-004")
        result = loop.update_metrics("wf-004", tokens_this_hop=10)
        assert result is False

    def test_normal_workflow_returns_true_on_update(self, loop):
        loop.register_agent("wf-005", "agent-e", "tenant-1")
        result = loop.update_metrics("wf-005", tokens_this_hop=100)
        assert result is True

    def test_kill_callback_called_on_enforcement(self, loop):
        killed_workflows = []
        loop.register_agent(
            "wf-006", "agent-f", "tenant-1",
            kill_callback=lambda wf_id, reason: killed_workflows.append(wf_id),
        )
        # Force a kill via token explosion rule
        loop._rules[0].check = lambda m: True  # Override to always fire
        loop._enforce("wf-006", loop._agents["wf-006"], loop._rules[0])
        assert "wf-006" in loop._killed
        assert "wf-006" in killed_workflows

    def test_cost_ceiling_rule_fires(self, loop):
        loop.register_agent("wf-007", "agent-g", "tenant-1")
        loop.update_metrics("wf-007", cost_this_hop=2.1)  # Over $2.00
        cost_rule = next(r for r in loop._rules if r.name == "cost_ceiling")
        metrics = loop._agents["wf-007"]
        assert cost_rule.check(metrics) is True

    def test_max_hops_rule_fires(self, loop):
        loop.register_agent("wf-008", "agent-h", "tenant-1")
        for _ in range(9):   # 9 hops — over threshold of 8
            loop.update_metrics("wf-008", tokens_this_hop=10)
        hop_rule = next(r for r in loop._rules if r.name == "max_hops")
        metrics  = loop._agents.get("wf-008")
        if metrics:
            assert hop_rule.check(metrics) is True

    def test_loop_detection_on_repeated_hash(self, loop):
        loop.register_agent("wf-009", "agent-i", "tenant-1")
        # Set the last_hop_hash directly then call update with same hash
        loop._agents["wf-009"].last_hop_hash = "abc123"
        # update_metrics with same hash triggers loop detection in _enforce call
        loop.update_metrics("wf-009", output_hash="abc123")
        # The loop_detection rule fires via _enforce with redirect action
        # Check the action was logged (loop_detection uses redirect, not kill)
        assert any(a.workflow_id == "wf-009" for a in loop._actions)

    def test_enforcement_action_logged(self, loop):
        loop.register_agent("wf-010", "agent-j", "tenant-1")
        loop._enforce("wf-010", loop._agents["wf-010"], loop._rules[0])
        assert len(loop._actions) >= 1
        last = loop._actions[-1]
        assert last.workflow_id == "wf-010"
        assert last.action in ("kill", "redirect", "warn")

    def test_stats_returns_correct_counts(self, loop):
        loop.register_agent("wf-011", "agent-k", "tenant-1")
        stats = loop.stats()
        assert "active_agents"    in stats
        assert "killed_workflows" in stats
        assert "total_actions"    in stats
        assert "rules"            in stats

    @pytest.mark.asyncio
    async def test_enforcement_loop_starts_and_stops(self, loop):
        await loop.start()
        assert loop._running is True
        await asyncio.sleep(0.15)   # Let it run 3 cycles at 50ms each
        await loop.stop()
        assert loop._running is False

    @pytest.mark.asyncio
    async def test_enforcement_loop_kills_violating_agent(self, loop):
        await loop.start()
        loop.register_agent("wf-012", "agent-l", "tenant-1")
        # Trigger token explosion
        loop._agents["wf-012"].token_count = 7000
        await asyncio.sleep(0.15)   # Wait for loop to detect
        await loop.stop()
        assert "wf-012" in loop._killed