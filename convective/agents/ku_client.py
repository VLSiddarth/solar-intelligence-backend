# ============================================================
# convective/agents/ku_client.py
# Knowledge Universe API client
# Handles discovery, freshness scoring, and usage tracking.
# ============================================================

import time
from typing import Optional
import httpx
from shared.config.settings import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class KUAPIError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"KU API {status}: {detail}")


class KUDocument:
    """A document discovered from the KU API."""
    def __init__(self, data: dict):
        self.doc_id        = data.get("id") or data.get("doc_id", "")
        self.title         = data.get("title", "Untitled")
        self.content       = data.get("content") or data.get("summary", "")
        self.source_url    = data.get("source_url") or data.get("url", "")
        self.decay_score   = float(data.get("decay_score", 1.0))
        self.knowledge_velocity = float(data.get("knowledge_velocity", 0.0))
        self.topic         = data.get("topic", "")
        self.published_at  = data.get("published_at", "")
        self.metadata      = {
            "ku_doc_id":          self.doc_id,
            "decay_score":        self.decay_score,
            "knowledge_velocity": self.knowledge_velocity,
            "source":             "knowledgeuniverse",
            "topic":              self.topic,
        }


class KUAPIClient:
    """
    Client for the Knowledge Universe API.

    Free tier: 500 calls/month.
    The 72-hour agent polls every 30 min = 144 calls over 72h.
    Fits comfortably in the free tier.

    Endpoints used:
        POST /v1/discover  — find new/updated documents for a topic
        GET  /v1/usage     — check remaining quota
    """

    def __init__(self):
        self._base_url   = settings.ku_api_base_url
        self._api_key    = settings.ku_api_key
        self._calls_made = 0
        self._client     = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "X-API-Key":    self._api_key,
                "Content-Type": "application/json",
                "User-Agent":   "SolarIntelligence/1.0 AutonomousAgent",
            },
            timeout=30.0,
        )

    async def discover(
        self,
        topic: str,
        difficulty: int = 3,
        max_results: int = 10,
        since_hours: int = 1,
    ) -> list[KUDocument]:
        """
        Discover new/updated documents for a topic.

        Args:
            topic:       Topic query (e.g. "regulatory NLP clinical data")
            difficulty:  1-5, complexity level of results
            max_results: Max number of documents returned
            since_hours: Only return content newer than N hours
        """
        try:
            resp = await self._client.post("/discover", json={
                "topic":       topic,
                "difficulty":  difficulty,
                "max_results": max_results,
                "since_hours": since_hours,
            })
            self._calls_made += 1

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", data if isinstance(data, list) else [])
                docs = [KUDocument(r) for r in results]
                logger.info("ku_discover_success", extra={
                    "topic":   topic,
                    "found":   len(docs),
                    "calls":   self._calls_made,
                })
                return docs

            elif resp.status_code == 429:
                logger.warning("ku_rate_limited", extra={"topic": topic})
                return []

            else:
                raise KUAPIError(resp.status_code, resp.text[:200])

        except httpx.TimeoutException:
            logger.error("ku_discover_timeout", extra={"topic": topic})
            return []
        except KUAPIError:
            raise
        except Exception as e:
            logger.error("ku_discover_failed", extra={"topic": topic, "error": str(e)})
            return []

    async def check_usage(self) -> dict:
        """Return current API usage stats."""
        try:
            resp = await self._client.get("/usage")
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("ku_usage_check_failed", extra={"error": str(e)})
        return {"calls_used": self._calls_made, "calls_remaining": "unknown"}

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def calls_made(self) -> int:
        return self._calls_made