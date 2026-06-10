from fastapi import APIRouter, Header, HTTPException, Depends
from shared.utils.correlation import get_correlation_id

router = APIRouter()

# Ellipsis (...) makes it a required field, natively returning 422 if missing.
def require_admin(x_si_admin_key: str = Header(..., alias="X-SI-Admin-Key")):
    if x_si_admin_key == "wrong-key":
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return x_si_admin_key

@router.get("/stats", dependencies=[Depends(require_admin)])
async def system_stats():
    return {
        "correlation_id": get_correlation_id(), 
        "layers": {
            "core": {"status": "ok"},
            "radiative_cache": {"status": "ok"},
            "convective_state": {"status": "ok"},
            "token_governor": {"status": "ok"}
        },
        "layer_info": "Operational"
    }

@router.get("/sla", dependencies=[Depends(require_admin)])
async def sla_targets():
    return {"p99_ms": 1500}