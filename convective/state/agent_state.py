# ============================================================
# convective/state/agent_state.py
# Redis agent state persistence — survives node restarts
# P0 fix: multi-hop workflows no longer die on restart
# TTL = max_workflow_duration × 1.2, checkpointed at every hop
# ============================================================

import json
import time
from typing import Optional, Any
from datetime import datetime

import redis

from shared.config.settings import settings
from shared.models.entities import AgentState
from shared.utils.correlation import get_correlation_id, get_tenant_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)

KEY_PREFIX = "si:agent"
LOCK_PREFIX = "si:agent_lock"


class AgentStateStore:
    """
    Redis-backed agent state persistence.

    Design:
        - State is checkpointed at every hop
        - TTL = max_workflow_duration × 1.2 (20% buffer for slow workflows)
        - Per-workflow distributed lock prevents race conditions
        - On Redis failure: in-memory fallback (data survives process, not node)
        - Workflow IDs are scoped per tenant (tenant isolation)
    """

    def __init__(self):
        self._ttl = int(
            settings.agent.max_workflow_duration * settings.agent.state_ttl_multiplier
        )
        self._redis = redis.Redis(
            host=settings.agent.redis_host,
            port=settings.agent.redis_port,
            password=settings.agent.redis_password or None,
            db=settings.agent.redis_db,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        # In-memory fallback when Redis is unavailable
        self._fallback: dict[str, str] = {}
        self._redis_available = True
        self._test_connection()

    def _test_connection(self) -> None:
        try:
            self._redis.ping()
            logger.info("agent_state_redis_connected", extra={
                "host": settings.agent.redis_host,
                "ttl":  self._ttl,
            })
        except redis.exceptions.ConnectionError:
            self._redis_available = False
            logger.warning("agent_state_redis_unavailable_using_memory_fallback")

    # ─────────────────────────────────────────
    # Core State Operations
    # ─────────────────────────────────────────

    def checkpoint(
        self,
        workflow_id: str,
        hop: int,
        state: dict[str, Any],
        tenant_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Save agent state at the current hop.
        Called after every successful hop — never loses more than one hop on restart.
        """
        tenant = tenant_id or get_tenant_id()
        cid    = correlation_id or get_correlation_id()
        key    = self._key(workflow_id, tenant)

        agent_state = AgentState(
            workflow_id=workflow_id,
            current_hop=hop,
            state=state,
            checkpointed_at=datetime.utcnow(),
            correlation_id=cid,
            tenant_id=tenant,
            status="running",
        )

        payload = agent_state.model_dump_json()

        try:
            self._redis.setex(key, self._ttl, payload)
            self._redis_available = True
            logger.debug("agent_state_checkpointed", extra={
                "workflow_id": workflow_id,
                "hop":         hop,
                "tenant_id":   tenant,
                "ttl":         self._ttl,
            })
        except redis.exceptions.RedisError as e:
            logger.warning("agent_state_redis_write_failed_using_fallback", extra={
                "error": str(e),
            })
            self._redis_available = False
            self._fallback[key] = payload

    def resume(
        self, workflow_id: str, tenant_id: Optional[str] = None
    ) -> Optional[AgentState]:
        """
        Load agent state for a workflow.
        Returns None if workflow doesn't exist or has expired.
        Used to resume interrupted multi-hop workflows after node restart.
        """
        tenant = tenant_id or get_tenant_id()
        key    = self._key(workflow_id, tenant)

        raw = self._get(key)
        if not raw:
            return None

        try:
            return AgentState.model_validate_json(raw)
        except Exception as e:
            logger.error("agent_state_deserialize_failed", extra={
                "workflow_id": workflow_id,
                "error":       str(e),
            })
            return None

    def complete(
        self,
        workflow_id: str,
        final_state: dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> None:
        """Mark workflow as completed and persist final state."""
        tenant = tenant_id or get_tenant_id()
        key    = self._key(workflow_id, tenant)

        existing = self.resume(workflow_id, tenant_id)
        if not existing:
            return

        existing.status = "completed"
        existing.state.update(final_state)
        existing.checkpointed_at = datetime.utcnow()

        # Keep completed state for 1 hour for audit
        try:
            self._redis.setex(key, 3600, existing.model_dump_json())
        except redis.exceptions.RedisError:
            self._fallback[key] = existing.model_dump_json()

        logger.info("agent_workflow_completed", extra={
            "workflow_id": workflow_id,
            "hops":        existing.current_hop,
            "tenant_id":   tenant,
        })

    def fail(
        self,
        workflow_id: str,
        error: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Mark workflow as failed with error context."""
        tenant   = tenant_id or get_tenant_id()
        existing = self.resume(workflow_id, tenant_id)
        if not existing:
            return

        existing.status = "failed"
        existing.state["_error"] = error
        key = self._key(workflow_id, tenant)

        try:
            self._redis.setex(key, 1800, existing.model_dump_json())
        except redis.exceptions.RedisError:
            self._fallback[key] = existing.model_dump_json()

        logger.error("agent_workflow_failed", extra={
            "workflow_id": workflow_id,
            "error":       error,
            "tenant_id":   tenant,
        })

    def clear(self, workflow_id: str, tenant_id: Optional[str] = None) -> None:
        """Delete workflow state (cleanup after completion)."""
        tenant = tenant_id or get_tenant_id()
        key    = self._key(workflow_id, tenant)
        try:
            self._redis.delete(key)
        except redis.exceptions.RedisError:
            self._fallback.pop(key, None)

    def exists(self, workflow_id: str, tenant_id: Optional[str] = None) -> bool:
        """Check if a workflow state exists."""
        tenant = tenant_id or get_tenant_id()
        key    = self._key(workflow_id, tenant)
        try:
            return bool(self._redis.exists(key))
        except redis.exceptions.RedisError:
            return key in self._fallback

    def ttl(self, workflow_id: str, tenant_id: Optional[str] = None) -> int:
        """Return remaining TTL in seconds for a workflow."""
        tenant = tenant_id or get_tenant_id()
        key    = self._key(workflow_id, tenant)
        try:
            return self._redis.ttl(key)
        except redis.exceptions.RedisError:
            return -1

    # ─────────────────────────────────────────
    # Distributed Lock (prevents concurrent hops)
    # ─────────────────────────────────────────

    def acquire_hop_lock(
        self, workflow_id: str, tenant_id: Optional[str] = None, timeout: int = 10
    ) -> bool:
        """
        Acquire a distributed lock before processing a hop.
        Prevents two processes from running the same hop concurrently.
        Returns True if lock acquired, False if already locked.
        """
        tenant   = tenant_id or get_tenant_id()
        lock_key = f"{LOCK_PREFIX}:{tenant}:{workflow_id}"
        try:
            result = self._redis.set(lock_key, "1", nx=True, ex=timeout)
            return result is True
        except redis.exceptions.RedisError:
            return True  # Fail open in degraded mode

    def release_hop_lock(self, workflow_id: str, tenant_id: Optional[str] = None) -> None:
        """Release the distributed lock after hop completes."""
        tenant   = tenant_id or get_tenant_id()
        lock_key = f"{LOCK_PREFIX}:{tenant}:{workflow_id}"
        try:
            self._redis.delete(lock_key)
        except redis.exceptions.RedisError:
            pass

    # ─────────────────────────────────────────
    # Health + Stats
    # ─────────────────────────────────────────

    def health(self) -> dict:
        try:
            self._redis.ping()
            info = self._redis.info("memory")
            return {
                "redis_available": True,
                "used_memory_mb":  round(info["used_memory"] / 1024 / 1024, 1),
                "fallback_count":  len(self._fallback),
            }
        except redis.exceptions.RedisError:
            return {
                "redis_available": False,
                "fallback_count":  len(self._fallback),
            }

    # ─────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────

    def _key(self, workflow_id: str, tenant_id: str) -> str:
        return f"{KEY_PREFIX}:{tenant_id}:{workflow_id}"

    def _get(self, key: str) -> Optional[str]:
        try:
            raw = self._redis.get(key)
            if raw:
                self._redis_available = True
            return raw
        except redis.exceptions.RedisError as e:
            logger.warning("agent_state_redis_read_failed", extra={"error": str(e)})
            self._redis_available = False
            return self._fallback.get(key)


# Singleton
_state_store: Optional[AgentStateStore] = None


def get_state_store() -> AgentStateStore:
    global _state_store
    if _state_store is None:
        _state_store = AgentStateStore()
    return _state_store