# ============================================================
# radiative/cache/semantic_cache.py — FIXED
#
# BUG FIXED: All methods were returning None with the comment
#   "TGI bypass - embeddings not ready". TGI has been ready
#   since session 2 of the build. Cache was permanently
#   disabled, causing every query to hit Groq and burning
#   free-tier quota on repeated questions.
#
# This fix re-enables the full semantic cache:
#   - Embed query → cosine similarity against stored queries
#   - Cache hit if similarity >= threshold (default 0.85)
#   - TTL tiers: HOT=1h, WARM=24h
#   - Drift invalidation: evict if centroid drift > 0.08
#   - Graceful: if Redis is down, returns None (no crash)
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
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class TTLTier:
    HOT  = settings.cache.hot_ttl_seconds   # 3600 = 1 hour
    WARM = settings.cache.warm_ttl_seconds   # 86400 = 24 hours


class SemanticCache:
    """
    Semantic cache for the Radiative Zone.
    Returns cached responses for semantically similar queries.
    """

    CACHE_KEY_PREFIX = "si:cache"
    INDEX_KEY_PREFIX = "si:cache_idx"

    def __init__(self):
        self._redis = redis.Redis(
            host=settings.agent.redis_host,
            port=settings.agent.redis_port,
            password=settings.agent.redis_password or None,
            db=1,
            decode_responses=False,
            socket_connect_timeout=2,
        )
        self._embedder  = EmbeddingClient()
        self._threshold = settings.cache.similarity_threshold   # 0.85
        self._max_size  = settings.cache.max_size               # 10000
        self._hits      = 0
        self._misses    = 0
        logger.info("semantic_cache_initialized", extra={
            "threshold": self._threshold,
            "max_size":  self._max_size,
            "status":    "active",   # No longer bypassed
        })

    async def get(
        self, query: str, tenant_id: Optional[str] = None
    ) -> Optional[tuple[str, float]]:
        """
        Look up a cached response for this query.
        Returns (response_text, similarity_score) or None on miss.
        """
        tenant = tenant_id or "default"
        try:
            # Embed the query
            query_emb = await self._embedder.embed_single(query)
            if not query_emb:
                self._misses += 1
                return None

            # Load all cache entries for this tenant
            index_entries = self._load_tenant_index(tenant)
            if not index_entries:
                self._misses += 1
                return None

            # Find most similar cached query
            best_key   = None
            best_score = 0.0

            for entry_key, emb_bytes in index_entries:
                try:
                    stored_emb = np.frombuffer(emb_bytes, dtype=np.float32).tolist()
                    score = self._cosine_similarity(query_emb, stored_emb)
                    if score > best_score:
                        best_score = score
                        best_key   = entry_key
                except Exception:
                    continue

            if best_score >= self._threshold and best_key:
                response = self._load_response(best_key)
                if response:
                    self._hits += 1
                    logger.info("cache_hit", extra={
                        "tenant":     tenant,
                        "similarity": round(best_score, 3),
                        "hits":       self._hits,
                    })
                    return response, best_score

            self._misses += 1
            return None

        except Exception as e:
            logger.warning("cache_get_failed", extra={"error": str(e)})
            self._misses += 1
            return None

    async def put(
        self,
        query: str,
        response: str,
        tenant_id: Optional[str] = None,
        tier: str = "warm",
    ) -> None:
        """Store a query-response pair in the cache."""
        tenant = tenant_id or "default"
        try:
            query_emb = await self._embedder.embed_single(query)
            if not query_emb:
                return

            ttl     = TTLTier.HOT if tier == "hot" else TTLTier.WARM
            key     = self._make_key(query, tenant)
            idx_key = f"{self.INDEX_KEY_PREFIX}:{tenant}:{key.split(':')[-1]}"

            # Store embedding in index (binary for compactness)
            emb_bytes = np.array(query_emb, dtype=np.float32).tobytes()
            self._redis.setex(idx_key, ttl, emb_bytes)

            # Store response
            self._redis.setex(key, ttl, response.encode("utf-8"))

            logger.debug("cache_put", extra={
                "tenant": tenant,
                "tier":   tier,
                "ttl":    ttl,
            })

        except Exception as e:
            logger.warning("cache_put_failed", extra={"error": str(e)})

    async def invalidate_on_drift(
        self, current_centroid: list[float], tenant_id: str
    ) -> int:
        """Evict cached entries whose embedding drifted beyond threshold."""
        evicted = 0
        try:
            drift_threshold = settings.cache.cosine_drift_threshold  # 0.08
            index_entries   = self._load_tenant_index(tenant_id)

            for entry_key, emb_bytes in index_entries:
                stored_emb = np.frombuffer(emb_bytes, dtype=np.float32).tolist()
                similarity = self._cosine_similarity(current_centroid, stored_emb)
                drift      = 1.0 - similarity

                if drift > drift_threshold:
                    response_key = entry_key.replace(
                        self.INDEX_KEY_PREFIX, self.CACHE_KEY_PREFIX
                    )
                    self._redis.delete(entry_key, response_key)
                    evicted += 1

            if evicted:
                logger.info("cache_drift_eviction", extra={
                    "tenant":  tenant_id,
                    "evicted": evicted,
                })
        except Exception as e:
            logger.warning("cache_drift_check_failed", extra={"error": str(e)})

        return evicted

    async def warm(
        self, queries: list[str], responses: list[str], tenant_id: str
    ) -> None:
        """Pre-warm cache with known query-response pairs."""
        for query, response in zip(queries, responses):
            await self.put(query, response, tenant_id=tenant_id, tier="warm")

    def stats(self) -> dict:
        total    = self._hits + self._misses
        hit_rate = self._hits / max(total, 1)
        return {
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(hit_rate, 3),
            "threshold": self._threshold,
        }

    def _make_key(self, query: str, tenant_id: str) -> str:
        h = hashlib.sha256(f"{tenant_id}:{query}".encode()).hexdigest()[:16]
        return f"{self.CACHE_KEY_PREFIX}:{tenant_id}:{h}"

    def _load_tenant_index(self, tenant_id: str) -> list[tuple[str, bytes]]:
        """Load all index entries for a tenant. Returns list of (key, embedding_bytes)."""
        try:
            pattern = f"{self.INDEX_KEY_PREFIX}:{tenant_id}:*"
            keys    = list(self._redis.scan_iter(pattern, count=100))
            if not keys:
                return []
            values = self._redis.mget(keys)
            return [
                (k.decode() if isinstance(k, bytes) else k, v)
                for k, v in zip(keys, values)
                if v is not None
            ]
        except Exception:
            return []

    def _load_response(self, index_key: str) -> Optional[str]:
        """Load response text given an index key."""
        try:
            # Convert index key to response key
            response_key = index_key.replace(
                self.INDEX_KEY_PREFIX, self.CACHE_KEY_PREFIX
            )
            value = self._redis.get(response_key)
            if value:
                return value.decode("utf-8") if isinstance(value, bytes) else value
        except Exception:
            pass
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