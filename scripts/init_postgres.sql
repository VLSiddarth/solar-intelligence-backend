-- ============================================================
-- scripts/init_postgres.sql  — FIXED (was a broken 21-byte fragment)
--
-- WHAT WAS BROKEN:
--   The file only contained the text "embedding vector(384)"
--   which is not valid SQL at all. This means:
--     1. CREATE EXTENSION vector never ran
--     2. The 'vector' type never existed in postgres
--     3. BlueGreenIndexManager._ensure_pgvector_table() failed
--        silently on every startup with "type vector does not exist"
--     4. pgvector_init_warning appeared in every si-api and
--        si-worker log since the project first started
--     5. Zero embeddings were ever stored
--
-- This file runs automatically when the postgres container first
-- initializes (mounted at /docker-entrypoint-initdb.d/init.sql).
-- To apply it you must wipe the postgres data volume (see below).
-- ============================================================

-- Step 1: Install the pgvector extension
-- The pgvector/pgvector:pg16 image has the .so file pre-installed
-- but you MUST run this to register the 'vector' type.
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create the blue index table (384-dim for bge-small-en-v1.5)
-- VECTOR_DIMENSION=384 matches the TGI model in docker-compose:
--   command: --model-id BAAI/bge-small-en-v1.5
-- If you switch to bge-m3 later, change 384 → 1024 here AND in .env
CREATE TABLE IF NOT EXISTS v_blue (
    vector_id    TEXT PRIMARY KEY,
    entity_id    TEXT NOT NULL,
    tenant_id    TEXT NOT NULL,
    embedding    vector(384),
    content_hash TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS v_blue_tenant_idx
    ON v_blue (tenant_id);

-- Cosine distance index (IVFFlat — fast approximate nearest neighbour)
-- lists=100 is good for up to ~1M vectors
CREATE INDEX IF NOT EXISTS v_blue_embedding_idx
    ON v_blue USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Step 3: Create the green index table (identical schema — blue/green swap)
CREATE TABLE IF NOT EXISTS v_green (
    vector_id    TEXT PRIMARY KEY,
    entity_id    TEXT NOT NULL,
    tenant_id    TEXT NOT NULL,
    embedding    vector(384),
    content_hash TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS v_green_tenant_idx
    ON v_green (tenant_id);

CREATE INDEX IF NOT EXISTS v_green_embedding_idx
    ON v_green USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Step 4: Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO si_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO si_user;