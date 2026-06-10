# ============================================================
# shared/config/settings.py
# Central configuration — all layers read from here
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from functools import lru_cache
from typing import Literal, Optional
import os


class KafkaSettings(BaseSettings):
    bootstrap_servers: str = Field("localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    topic_raw_docs: str    = Field("si.core.raw_documents", alias="KAFKA_TOPIC_RAW_DOCS")
    topic_fused: str       = Field("si.core.fused_entities", alias="KAFKA_TOPIC_FUSED")
    topic_audit: str       = Field("si.core.audit", alias="KAFKA_TOPIC_AUDIT")
    consumer_group: str    = Field("si-core-group", alias="KAFKA_CONSUMER_GROUP")
    transactional_id: str  = Field("si-core-fusion-producer-1", alias="KAFKA_TRANSACTIONAL_ID")


class GraphRAGSettings(BaseSettings):
    falkordb_host: str              = Field("localhost", alias="FALKORDB_HOST")
    falkordb_port: int              = Field(6379, alias="FALKORDB_PORT")
    falkordb_password: str          = Field("", alias="FALKORDB_PASSWORD")
    dedup_cosine_threshold: float   = Field(0.92)
    max_parallel_workers: int       = Field(4)


class OpenAISettings(BaseSettings):
    api_key: str             = Field("", alias="OPENAI_API_KEY")
    model: str               = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    max_tokens_per_job: int  = Field(100000, alias="OPENAI_MAX_TOKENS_PER_JOB")
    hard_cost_ceiling_usd: float = Field(50.0, alias="OPENAI_HARD_COST_CEILING_USD")


class VectorSettings(BaseSettings):
    backend: Literal["pgvector", "milvus", "govector"] = Field("pgvector", alias="VECTOR_BACKEND")
    dimension: int              = Field(1024, alias="VECTOR_DIMENSION")
    milvus_host: str            = Field("localhost", alias="MILVUS_HOST")
    milvus_port: int            = Field(19530, alias="MILVUS_PORT")
    milvus_collection: str      = Field("si_intelligence", alias="MILVUS_COLLECTION")
    postgres_host: str          = Field("localhost", alias="POSTGRES_HOST")
    postgres_port: int          = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str            = Field("si_vectors", alias="POSTGRES_DB")
    postgres_user: str          = Field("si_user", alias="POSTGRES_USER")
    postgres_password: str      = Field("si_password", alias="POSTGRES_PASSWORD")
    govector_index_path: str    = Field("", alias="GOVECTOR_INDEX_PATH")


class CacheSettings(BaseSettings):
    similarity_threshold: float = Field(0.85, alias="GPTCACHE_SIMILARITY_THRESHOLD")
    max_size: int               = Field(10000, alias="GPTCACHE_MAX_SIZE")
    hot_ttl_seconds: int        = Field(3600, alias="CACHE_HOT_TTL_SECONDS")
    warm_ttl_seconds: int       = Field(86400, alias="CACHE_WARM_TTL_SECONDS")
    cosine_drift_threshold: float = Field(0.08, alias="CACHE_COSINE_DRIFT_THRESHOLD")


class EmbeddingSettings(BaseSettings):
    host: str  = Field("localhost", alias="EMBEDDING_HOST")
    port: int  = Field(8080, alias="EMBEDDING_PORT")
    model: str = Field("BAAI/bge-m3", alias="EMBEDDING_MODEL")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RouterSettings(BaseSettings):
    vllm_host: str          = Field("localhost", alias="VLLM_ROUTER_HOST")
    vllm_port: int          = Field(8100, alias="VLLM_ROUTER_PORT")
    cosine_threshold: float = Field(0.78, alias="ROUTER_COSINE_THRESHOLD")
    fallback_error_rate: float     = Field(0.05, alias="ROUTER_FALLBACK_ERROR_RATE")
    fallback_window_seconds: int   = Field(60, alias="ROUTER_FALLBACK_WINDOW_SECONDS")


class AgentSettings(BaseSettings):
    redis_host: str     = Field("localhost", alias="REDIS_HOST")
    redis_port: int     = Field(6379, alias="REDIS_PORT")
    redis_password: str = Field("", alias="REDIS_PASSWORD")
    redis_db: int       = Field(0, alias="REDIS_DB")
    max_workflow_duration: int   = Field(3600, alias="AGENT_MAX_WORKFLOW_DURATION")
    state_ttl_multiplier: float  = Field(1.2, alias="AGENT_STATE_TTL_MULTIPLIER")


class TokenSettings(BaseSettings):
    cot_soft: int      = Field(4096, alias="COT_SOFT_TOKEN_LIMIT")
    cot_hard: int      = Field(8192, alias="COT_HARD_TOKEN_LIMIT")
    standard_soft: int = Field(2048, alias="STANDARD_SOFT_TOKEN_LIMIT")
    standard_hard: int = Field(4096, alias="STANDARD_HARD_TOKEN_LIMIT")
    edge_soft: int     = Field(512, alias="EDGE_SOFT_TOKEN_LIMIT")
    edge_hard: int     = Field(1024, alias="EDGE_HARD_TOKEN_LIMIT")


class APISettings(BaseSettings):
    host: str      = Field("0.0.0.0", alias="API_HOST")
    port: int      = Field(8888, alias="API_PORT")
    workers: int   = Field(4, alias="API_WORKERS")
    secret_key: str = Field("change-me", alias="SI_SECRET_KEY")


class TelemetrySettings(BaseSettings):
    otlp_endpoint: str      = Field("http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name: str       = Field("solar-intelligence", alias="OTEL_SERVICE_NAME")
    trace_sample_rate: float = Field(1.0, alias="OTEL_TRACE_SAMPLE_RATE")


class WebhookSettings(BaseSettings):
    max_retries: int      = Field(3, alias="WEBHOOK_MAX_RETRIES")
    dlq_topic: str        = Field("si.corona.webhook_dlq", alias="WEBHOOK_DLQ_TOPIC")
    pagerduty_key: str    = Field("", alias="PAGERDUTY_INTEGRATION_KEY")
    backoff_seconds: list[int] = Field(default=[1, 4, 16])


class SLASettings(BaseSettings):
    p50_ms: int           = Field(150, alias="SLA_P50_MS")
    p95_ms: int           = Field(500, alias="SLA_P95_MS")
    p99_ms: int           = Field(1500, alias="SLA_P99_MS")
    availability_pct: float = Field(99.5, alias="SLA_AVAILABILITY_PCT")
    rto_minutes: int      = Field(30, alias="SLA_RTO_MINUTES")
    rpo_minutes: int      = Field(5, alias="SLA_RPO_MINUTES")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["development", "staging", "production"] = Field(
        "development", alias="SI_ENV"
    )
    log_level: str = Field("INFO", alias="SI_LOG_LEVEL")

    # ── LLM Provider (NEW) ─────────────────────────────────────────
    # Controls which LLM backend is used across all layers.
    # Values: "groq" | "openai" | "ollama"
    llm_provider: str = Field("groq", alias="LLM_PROVIDER")

    # Groq settings (free tier, fastest inference)
    groq_api_key: str  = Field("", alias="GROQ_API_KEY")
    groq_model: str    = Field("llama-3.1-8b-instant", alias="GROQ_MODEL")

    # Ollama settings (local, fully offline)
    ollama_model: str      = Field("mistral", alias="OLLAMA_MODEL")
    ollama_base_url: str   = Field("http://host.docker.internal:11434/v1", alias="OLLAMA_BASE_URL")

    # KU API settings
    ku_api_key: str        = Field("", alias="KU_API_KEY")
    ku_api_base_url: str   = Field("https://api.knowledgeuniverse.tech/v1", alias="KU_API_BASE_URL")

    # Sub-settings loaded from same .env via model_config inheritance
    kafka:      KafkaSettings     = Field(default_factory=KafkaSettings)
    graphrag:   GraphRAGSettings  = Field(default_factory=GraphRAGSettings)
    openai:     OpenAISettings    = Field(default_factory=OpenAISettings)
    vector:     VectorSettings    = Field(default_factory=VectorSettings)
    cache:      CacheSettings     = Field(default_factory=CacheSettings)
    embedding:  EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    router:     RouterSettings    = Field(default_factory=RouterSettings)
    agent:      AgentSettings     = Field(default_factory=AgentSettings)
    token:      TokenSettings     = Field(default_factory=TokenSettings)
    api:        APISettings       = Field(default_factory=APISettings)
    telemetry:  TelemetrySettings = Field(default_factory=TelemetrySettings)
    webhook:    WebhookSettings   = Field(default_factory=WebhookSettings)
    sla:        SLASettings       = Field(default_factory=SLASettings)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        return self.env == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance — cached after first call."""
    return Settings()


# Convenience shortcut used across all layers
settings = get_settings()