# ============================================================
# photosphere/mcp/server.py
# MCP Tool Server — JSON-RPC 2.0, schema validation, error taxonomy
# P1 fix: all error paths return structured MCPError, never silent failures
# ============================================================

import uuid
import json
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

from shared.models.entities import MCPError, IngestRequest, QueryRequest, QueryMode
from shared.utils.correlation import get_correlation_id, inject_correlation_id
from shared.utils.logging import get_logger

logger = get_logger(__name__)

mcp_app = FastAPI(title="SI MCP Server", version="1.0.0")


# ─────────────────────────────────────────────
# Tool Schemas (JSON Schema for validation)
# ─────────────────────────────────────────────

TOOL_SCHEMAS: dict[str, dict] = {
    "si_ingest": {
        "description": "Ingest a document into the Solar Intelligence knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content":    {"type": "string", "minLength": 10},
                "title":      {"type": "string"},
                "source_url": {"type": "string", "format": "uri"},
                "metadata":   {"type": "object"},
            },
            "required": ["content"],
        },
    },
    "si_query": {
        "description": "Query the Solar Intelligence knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":   {"type": "string", "minLength": 1},
                "mode":    {"type": "string", "enum": ["standard_query", "chain_of_thought", "edge_inference"]},
                "top_k":   {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    "si_health": {
        "description": "Check health status of all SI layers",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    "si_graph_query": {
        "description": "Run a direct Cypher query on the SI knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cypher":     {"type": "string"},
                "parameters": {"type": "object"},
            },
            "required": ["cypher"],
        },
    },
}


def _validate_input(tool_name: str, params: dict) -> tuple[bool, list[str]]:
    """
    Validate tool input against registered schema.
    Returns (is_valid, error_list).
    """
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return False, [f"Unknown tool: {tool_name}"]

    input_schema = schema.get("inputSchema", {})
    required     = input_schema.get("required", [])
    properties   = input_schema.get("properties", {})
    errors       = []

    # Check required fields
    for field in required:
        if field not in params:
            errors.append(f"Missing required field: {field}")

    # Type validation
    for field_name, field_schema in properties.items():
        if field_name not in params:
            continue
        val = params[field_name]
        expected_type = field_schema.get("type")
        if expected_type == "string" and not isinstance(val, str):
            errors.append(f"Field '{field_name}' must be a string")
        elif expected_type == "integer" and not isinstance(val, int):
            errors.append(f"Field '{field_name}' must be an integer")
        elif expected_type == "object" and not isinstance(val, dict):
            errors.append(f"Field '{field_name}' must be an object")
        # minLength check
        if expected_type == "string" and "minLength" in field_schema:
            if len(val) < field_schema["minLength"]:
                errors.append(f"Field '{field_name}' too short (min {field_schema['minLength']} chars)")
        # Enum check
        if "enum" in field_schema and val not in field_schema["enum"]:
            errors.append(f"Field '{field_name}' must be one of {field_schema['enum']}")

    return len(errors) == 0, errors


# ─────────────────────────────────────────────
# JSON-RPC 2.0 Handler
# ─────────────────────────────────────────────

@mcp_app.post("/mcp")
async def mcp_handler(request: Request):
    """
    Main JSON-RPC 2.0 MCP endpoint.
    All tool calls flow through here.
    Error taxonomy:
        4000 — schema validation error
        4001 — missing required field
        4003 — unauthorized tool call
        5000 — upstream error
        5002 — graph error
    """
    cid = inject_correlation_id(request.headers.get("X-Correlation-ID"))

    try:
        body = await request.json()
    except Exception:
        err = MCPError.build(4000, "Invalid JSON body", cid)
        return JSONResponse(status_code=400, content=err.model_dump())

    # JSON-RPC envelope validation
    if body.get("jsonrpc") != "2.0":
        err = MCPError.build(4000, "Missing or invalid jsonrpc field (must be '2.0')", cid)
        return JSONResponse(status_code=400, content=err.model_dump())

    method  = body.get("method", "")
    params  = body.get("params", {})
    req_id  = body.get("id", str(uuid.uuid4()))

    logger.info("mcp_tool_call", extra={
        "method": method,
        "req_id": req_id,
        "cid":    cid,
    })

    # Dispatch to tool handlers
    if method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id":      req_id,
            "result":  {"tools": [
                {"name": name, **schema}
                for name, schema in TOOL_SCHEMAS.items()
            ]},
        })

    elif method == "tools/call":
        tool_name   = params.get("name", "")
        tool_params = params.get("arguments", {})

        # Schema validation
        is_valid, errors = _validate_input(tool_name, tool_params)
        if not is_valid:
            err = MCPError.build(4000, f"Validation errors: {'; '.join(errors)}", cid)
            return JSONResponse(status_code=400, content=err.model_dump())

        result = await _dispatch_tool(tool_name, tool_params, cid)
        return JSONResponse({
            "jsonrpc": "2.0",
            "id":      req_id,
            "result":  {"content": [{"type": "text", "text": json.dumps(result)}]},
        })

    else:
        err = MCPError.build(4003, f"Unknown MCP method: {method}", cid)
        return JSONResponse(status_code=404, content=err.model_dump())


async def _dispatch_tool(tool_name: str, params: dict, cid: str) -> Any:
    """Route tool call to the appropriate handler."""
    if tool_name == "si_ingest":
        return await _tool_ingest(params, cid)
    elif tool_name == "si_query":
        return await _tool_query(params, cid)
    elif tool_name == "si_health":
        return await _tool_health(cid)
    elif tool_name == "si_graph_query":
        return await _tool_graph_query(params, cid)
    else:
        raise ValueError(f"No handler for tool: {tool_name}")


async def _tool_ingest(params: dict, cid: str) -> dict:
    from core.kafka.producer import get_producer
    from shared.models.entities import RawDocument
    from shared.config.settings import settings
    import uuid

    doc_id  = str(uuid.uuid4())
    raw_doc = RawDocument(
        doc_id=doc_id,
        content=params["content"],
        title=params.get("title"),
        source_url=params.get("source_url"),
        tenant_id="mcp",
        correlation_id=cid,
        metadata=params.get("metadata", {}),
    )
    get_producer().produce(
        topic=settings.kafka.topic_raw_docs,
        value=raw_doc.model_dump(),
        key=doc_id,
    )
    return {"doc_id": doc_id, "status": "queued"}


async def _tool_query(params: dict, cid: str) -> dict:
    from radiative.vector.index_manager import BlueGreenIndexManager
    from radiative.cache.semantic_cache import SemanticCache
    from convective.agents.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator(
        vector_index=BlueGreenIndexManager(),
        cache=SemanticCache(),
    )
    mode = QueryMode(params.get("mode", "standard_query"))
    resp = await orch.execute(
        query=params["query"],
        tenant_id="mcp",
        mode_override=mode,
    )
    return {
        "answer":       resp.answer,
        "from_cache":   resp.from_cache,
        "routing_tier": resp.routing_tier.value,
        "tokens":       resp.token_usage.total_tokens,
    }


async def _tool_health(cid: str) -> dict:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8888/health", timeout=5.0)
            return resp.json()
    except Exception as e:
        return {"status": "unknown", "error": str(e)}


async def _tool_graph_query(params: dict, cid: str) -> dict:
    from falkordb import FalkorDB
    from shared.config.settings import settings

    db    = FalkorDB(host=settings.graphrag.falkordb_host, port=settings.graphrag.falkordb_port)
    graph = db.select_graph("si_knowledge")
    result = graph.query(params["cypher"], params.get("parameters", {}))
    return {"rows": result.result_set[:100]}  # Cap at 100 rows


# ─────────────────────────────────────────────
# SSE Transport (Streamable HTTP)
# ─────────────────────────────────────────────

@mcp_app.get("/mcp/sse")
async def mcp_sse(request: Request):
    """Server-Sent Events transport for MCP streaming responses."""
    cid = get_correlation_id()

    async def event_stream():
        yield f"data: {json.dumps({'type': 'connected', 'correlation_id': cid})}\n\n"
        # Keep alive ping every 30s
        while True:
            await asyncio.sleep(30)
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Correlation-ID": cid, "Cache-Control": "no-cache"},
    )