# ============================================================
# radiative/cache/semantic_cache.py
# Semantic cache — intercepts repeated queries before LLM
# TTL tiers: HOT=1hr, WARM=24hr, COLD=evict on drift >0.08
# ============================================================

import time
import json
import hashlib
import asyncio
from typing import Optional
from datetime import datetime, timedelta

import redis
import numpy as np

from radiative.embeddings.client import EmbeddingClient
from shared.config.settings import settings
from shared.utils.correlation import get_correlation_id, get_tenant_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class TTLTier:
    HOT  = settings.cache.hot_ttl_seconds
    WARM = settings.cache.warm_ttl_seconds


class SemanticCache:
    """
    GPTCache-style semantic cache for the Radiative Zone.
    """

    CACHE_KEY_PREFIX = "si:cache"
    INDEX_KEY_PREFIX = "si:cache_index"

    def __init__(self):
        self._redis = redis.Redis(
            host=settings.agent.redis_host,
            port=settings.agent.redis_port,
            password=settings.agent.redis_password or None,
            db=1,
            decode_responses=False,
        )
        self._embedder  = EmbeddingClient()
        self._threshold = settings.cache.similarity_threshold
        self._max_size  = settings.cache.max_size
        self._hits      = 0
        self._misses    = 0
        logger.info("semantic_cache_initialized", extra={
            "threshold": self._threshold,
            "max_size":  self._max_size,
        })

    async def get(
        self, query: str, tenant_id: Optional[str] = None
    ) -> Optional[tuple[str, float]]:
        """
        TGI bypass active: returns None until embeddings are ready.
        """
        return None  # TGI bypass - embeddings not ready

    async def put(
        self,
        query: str,
        response: str,
        tenant_id: Optional[str] = None,
        tier: str = "warm",
    ) -> None:
        """Store a query-response pair. Bypassed while TGI is not ready."""
        return None  # TGI bypass

    async def invalidate_on_drift(self, current_centroid: list[float], tenant_id: str) -> int:
        return 0  # TGI bypass

    async def warm(self, queries: list[str], responses: list[str], tenant_id: str) -> None:
        return None  # TGI bypass

    def stats(self) -> dict:
        return {
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  0.0,
            "threshold": self._threshold,
        }

    def _make_key(self, query: str, tenant_id: str) -> str:
        h = hashlib.sha256(f"{tenant_id}:{query}".encode()).hexdigest()[:16]
        return f"{self.CACHE_KEY_PREFIX}:{tenant_id}:{h}"

    def _load_tenant_index(self, tenant_id: str) -> list[tuple[str, bytes]]:
        return []

    def _load_response(self, entry_key: str) -> Optional[str]:
        return None

    @staticmethod
    def _cosine_similarity(a: list[float], b) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))
