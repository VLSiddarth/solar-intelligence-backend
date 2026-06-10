import time
import asyncio
from enum import Enum
from typing import Optional

try:
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility, MilvusException
    MILVUS_AVAILABLE = True
except Exception:
    MILVUS_AVAILABLE = False
    Collection = None
    MilvusException = Exception

import psycopg2
try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None

from shared.config.settings import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class IndexColor(str, Enum):
    BLUE  = "v_blue"
    GREEN = "v_green"


class BlueGreenIndexManager:
    def __init__(self):
        self._active = IndexColor.BLUE
        self._shadow = IndexColor.GREEN
        self._backend = "pgvector"
        self._lock = asyncio.Lock()
        self._pg_conn = None
        self._init_pgvector()
        logger.info("vector_backend_initialized", extra={
            "backend": self._backend,
            "active_index": self._active.value,
        })

    def _init_pgvector(self):
        try:
            self._pg_conn = psycopg2.connect(
                host=settings.vector.postgres_host,
                port=settings.vector.postgres_port,
                dbname=settings.vector.postgres_db,
                user=settings.vector.postgres_user,
                password=settings.vector.postgres_password,
            )
            if register_vector:
                register_vector(self._pg_conn)
            for color in IndexColor:
                self._ensure_pgvector_table(color.value)
        except Exception as e:
            logger.warning("pgvector_init_warning", extra={"error": str(e)})

    def _ensure_pgvector_table(self, table_name: str):
        dim = settings.vector.dimension
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        vector_id    TEXT PRIMARY KEY,
                        entity_id    TEXT NOT NULL,
                        tenant_id    TEXT NOT NULL,
                        embedding    vector({dim}),
                        content_hash TEXT,
                        created_at   TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS {table_name}_tenant_idx ON {table_name} (tenant_id);
                """)
                self._pg_conn.commit()
        except Exception as e:
            logger.warning("pgvector_table_warning", extra={"table": table_name, "error": str(e)})
            try:
                self._pg_conn.rollback()
            except Exception:
                pass

    def query(self, embedding, tenant_id: str, top_k: int = 10):
        try:
            return self._query_pgvector(self._active.value, embedding, tenant_id, top_k)
        except Exception as e:
            logger.warning("vector_query_warning", extra={"error": str(e)})
            return []

    def _query_pgvector(self, table: str, embedding, tenant_id: str, top_k: int):
        if self._pg_conn is None:
            return []
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT vector_id, entity_id,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {table}
                    WHERE tenant_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, tenant_id, embedding, top_k),
                )
                rows = cur.fetchall()
            return [{"vector_id": r[0], "entity_id": r[1], "score": float(r[2])} for r in rows]
        except Exception as e:
            logger.warning("pgvector_query_warning", extra={"error": str(e)})
            return []

    def insert(self, records: list):
        if not records or self._pg_conn is None:
            return
        try:
            self._insert_pgvector(self._active.value, records)
        except Exception as e:
            logger.warning("vector_insert_warning", extra={"error": str(e)})

    def _insert_pgvector(self, table: str, records: list):
        with self._pg_conn.cursor() as cur:
            for r in records:
                cur.execute(
                    f"""
                    INSERT INTO {table} (vector_id, entity_id, tenant_id, embedding, content_hash)
                    VALUES (%s, %s, %s, %s::vector, %s)
                    ON CONFLICT (vector_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            content_hash = EXCLUDED.content_hash
                    """,
                    (r["vector_id"], r["entity_id"], r["tenant_id"],
                     r["embedding"], r.get("content_hash", "")),
                )
        self._pg_conn.commit()

    async def promote_shadow(self, test_queries=None):
        async with self._lock:
            self._active, self._shadow = self._shadow, self._active
            logger.info("promotion_complete", extra={
                "new_active": self._active.value,
                "new_shadow": self._shadow.value,
            })
            return True

    @property
    def active_index(self):
        return self._active.value

    @property
    def shadow_index(self):
        return self._shadow.value
