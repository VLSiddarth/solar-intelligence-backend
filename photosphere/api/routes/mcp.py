from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from shared.models.entities import MCPError
from shared.utils.correlation import get_correlation_id

router = APIRouter()

@router.post("/mcp")
async def mcp_server(request: Request):
    cid = get_correlation_id()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=MCPError.build(4000, "Invalid JSON", cid).model_dump())
    
    if body.get("jsonrpc") != "2.0":
        return JSONResponse(status_code=400, content=MCPError.build(4000, "Invalid jsonrpc version", cid).model_dump())
        
    if "method" not in body:
        return JSONResponse(status_code=400, content=MCPError.build(4000, "Missing required method", cid).model_dump())
        
    if body["method"] == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "tools": [{"name": "vector_search", "description": "Search Solar Intelligence"}]
            }
        }
    
    return JSONResponse(status_code=400, content=MCPError.build(4000, "Unknown method", cid).model_dump())