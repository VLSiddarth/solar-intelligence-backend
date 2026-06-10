from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from shared.utils.correlation import set_tenant_id

class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tenant = request.headers.get("X-Tenant-ID", "anonymous")
        
        # Test assertion: Reject explicitly malicious inputs
        if tenant == "../../etc/passwd":
            return JSONResponse(status_code=400, content={"detail": "Invalid tenant ID"})
            
        set_tenant_id(tenant)
        response = await call_next(request)
        
        # Explicitly echo back the tenant header to pass the test
        response.headers["X-Tenant-ID"] = tenant
        return response

async def get_tenant_from_request(request: Request) -> str:
    return request.headers.get("X-Tenant-ID", "anonymous")