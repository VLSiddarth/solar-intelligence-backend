# ============================================================
# radiative/embeddings/client.py
# BGE-M3 embedding client — calls TGI server
# GoVector feature-flagged: disabled by default, fallback to pgvector
# ============================================================

import time
import asyncio
from typing import Optional
import httpx
from shared.config.settings import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """
    Client for the HuggingFace Text Embeddings Inference server running BGE-M3.
    BGE-M3: 1024-dimensional embeddings, multilingual, best-in-class retrieval.
    """

    def __init__(self):
        self._base_url = settings.embedding.url
        self._timeout  = 30.0
        self._client   = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        logger.info("embedding_client_initialized", extra={
            "url":   self._base_url,
            "model": settings.embedding.model,
        })

    async def embed(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        """
        Embed a batch of texts via TGI server.
        Returns list of 1024-dimensional float vectors.
        """
        if not texts:
            return []

        t_start = time.monotonic()

        # TGI embed endpoint
        response = await self._client.post(
            "/",
            json={
                "inputs":    texts,
                "normalize": normalize,
                "truncate":  True,
            },
        )
        response.raise_for_status()
        embeddings = response.json()

        latency = (time.monotonic() - t_start) * 1000
        logger.debug("embeddings_generated", extra={
            "count":      len(texts),
            "latency_ms": round(latency, 1),
        })

        return embeddings

    async def embed_single(self, text: str, normalize: bool = True) -> list[float]:
        """Embed a single text string. Returns one vector."""
        results = await self.embed([text], normalize=normalize)
        return results[0] if results else []

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> list[list[float]]:
        """
        Embed large batches in chunks to avoid TGI memory limits.
        batch_size=32 is optimal for BGE-M3 on CPU TGI.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            embeddings = await self.embed(chunk, normalize=normalize)
            all_embeddings.extend(embeddings)
            if len(texts) > batch_size:
                await asyncio.sleep(0.01)  # Back-pressure on large batches
        return all_embeddings

    async def health(self) -> bool:
        """Health check — returns True if TGI server is responsive."""
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


# ─────────────────────────────────────────────
# GoVector Feature Flag
# ─────────────────────────────────────────────

class VectorBackendSelector:
    """
    Feature-flags GoVector vs pgvector+HNSW.
    GoVector is a 2026 research paper — disabled by default.
    Auto-switches back to pgvector if GoVector error rate exceeds 5%.
    """

    ERROR_RATE_THRESHOLD = 0.05
    ERROR_WINDOW_SECONDS = 60

    def __init__(self):
        self._use_govector  = settings.vector.backend == "govector"
        self._errors: list[float] = []
        self._calls:  list[float] = []

    def should_use_govector(self) -> bool:
        """Returns True only if GoVector is enabled AND error rate is acceptable."""
        if not self._use_govector:
            return False

        import time
        now    = time.monotonic()
        cutoff = now - self.ERROR_WINDOW_SECONDS
        recent_errors = [t for t in self._errors if t > cutoff]
        recent_calls  = [t for t in self._calls  if t > cutoff]

        if len(recent_calls) < 5:
            return True  # Not enough data — try GoVector

        rate = len(recent_errors) / len(recent_calls)
        if rate > self.ERROR_RATE_THRESHOLD:
            logger.warning("govector_fallback_triggered", extra={
                "error_rate": round(rate, 3),
                "threshold":  self.ERROR_RATE_THRESHOLD,
            })
            self._use_govector = False  # Permanently disable for this session
            return False

        return True

    def record_success(self) -> None:
        import time
        self._calls.append(time.monotonic())

    def record_error(self) -> None:
        import time
        now = time.monotonic()
        self._errors.append(now)
        self._calls.append(now)


# Singleton embedding client
_embedding_client: Optional[EmbeddingClient] = None
_vector_selector  = VectorBackendSelector()


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def get_vector_selector() -> VectorBackendSelector:
    return _vector_selector
