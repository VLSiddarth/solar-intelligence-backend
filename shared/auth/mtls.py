# ============================================================
# shared/auth/mtls.py
# mTLS inter-layer authentication — every boundary is verified
# Run generate_certs() once before first boot
# ============================================================

import ssl
import os
import subprocess
from pathlib import Path
from typing import Optional
import httpx
from shared.utils.logging import get_logger

logger = get_logger(__name__)

CERTS_DIR = Path("certs")
LAYERS = ["core", "radiative", "convective", "photosphere", "corona", "signing"]


# ─────────────────────────────────────────────
# Certificate Generation
# ─────────────────────────────────────────────

def generate_certs(force: bool = False) -> None:
    """
    Generate a root CA and one cert per SI layer.
    Run once: python -c "from shared.auth.mtls import generate_certs; generate_certs()"
    """
    CERTS_DIR.mkdir(exist_ok=True)
    ca_key = CERTS_DIR / "ca.key"
    ca_crt = CERTS_DIR / "ca.crt"

    if ca_crt.exists() and not force:
        logger.info("certs_already_exist", extra={"path": str(CERTS_DIR)})
        return

    logger.info("generating_root_ca")

    # Root CA
    _run(["openssl", "genrsa", "-out", str(ca_key), "4096"])
    _run([
        "openssl", "req", "-new", "-x509",
        "-key", str(ca_key),
        "-out", str(ca_crt),
        "-days", "3650",
        "-subj", "/CN=SI-Root-CA/O=SolarIntelligence/OU=Infrastructure",
    ])

    # One cert per layer
    for layer in LAYERS:
        _generate_layer_cert(layer, ca_key, ca_crt)
        logger.info("cert_generated", extra={"layer": layer})

    logger.info("all_certs_generated", extra={"layers": LAYERS})


def _generate_layer_cert(layer: str, ca_key: Path, ca_crt: Path) -> None:
    key_path = CERTS_DIR / f"{layer}.key"
    csr_path = CERTS_DIR / f"{layer}.csr"
    crt_path = CERTS_DIR / f"{layer}.crt"

    _run(["openssl", "genrsa", "-out", str(key_path), "2048"])
    _run([
        "openssl", "req", "-new",
        "-key", str(key_path),
        "-out", str(csr_path),
        "-subj", f"/CN=si-{layer}/O=SolarIntelligence",
    ])
    _run([
        "openssl", "x509", "-req",
        "-in", str(csr_path),
        "-CA", str(ca_crt),
        "-CAkey", str(ca_key),
        "-CAcreateserial",
        "-out", str(crt_path),
        "-days", "365",
    ])
    csr_path.unlink(missing_ok=True)


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"openssl command failed: {result.stderr}")


# ─────────────────────────────────────────────
# mTLS Client Factory
# ─────────────────────────────────────────────

def get_mtls_client(my_layer: str, target_host: str, target_port: int) -> httpx.Client:
    """
    Create an httpx client with mTLS configured for inter-layer calls.

    Args:
        my_layer: The calling layer name (e.g. "convective")
        target_host: Hostname of the target layer
        target_port: Port of the target layer
    """
    crt = CERTS_DIR / f"{my_layer}.crt"
    key = CERTS_DIR / f"{my_layer}.key"
    ca  = CERTS_DIR / "ca.crt"

    if not all(p.exists() for p in [crt, key, ca]):
        logger.warning(
            "mtls_certs_missing",
            extra={"layer": my_layer, "falling_back": "no-mtls"},
        )
        # In development, fall back gracefully
        return httpx.Client(base_url=f"http://{target_host}:{target_port}", timeout=30.0)

    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(ca))
    ctx.load_cert_chain(str(crt), str(key))
    ctx.verify_mode = ssl.CERT_REQUIRED

    return httpx.Client(
        verify=ctx,
        base_url=f"https://{target_host}:{target_port}",
        timeout=30.0,
        headers={
            "X-SI-From-Layer": my_layer,
            "X-SI-Protocol":   "mtls/1.0",
        },
    )


async def get_async_mtls_client(my_layer: str, target_host: str, target_port: int) -> httpx.AsyncClient:
    """Async version of mTLS client for use in FastAPI routes."""
    crt = CERTS_DIR / f"{my_layer}.crt"
    key = CERTS_DIR / f"{my_layer}.key"
    ca  = CERTS_DIR / "ca.crt"

    if not all(p.exists() for p in [crt, key, ca]):
        return httpx.AsyncClient(
            base_url=f"http://{target_host}:{target_port}",
            timeout=30.0,
        )

    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(ca))
    ctx.load_cert_chain(str(crt), str(key))
    ctx.verify_mode = ssl.CERT_REQUIRED

    return httpx.AsyncClient(
        verify=ctx,
        base_url=f"https://{target_host}:{target_port}",
        timeout=30.0,
        headers={"X-SI-From-Layer": my_layer},
    )


# ─────────────────────────────────────────────
# mTLS Server SSL Context
# ─────────────────────────────────────────────

def get_server_ssl_context(layer: str) -> Optional[ssl.SSLContext]:
    """
    Get SSL context for the Uvicorn server.
    Returns None if certs don't exist (dev mode — plain HTTP).
    """
    crt = CERTS_DIR / f"{layer}.crt"
    key = CERTS_DIR / f"{layer}.key"
    ca  = CERTS_DIR / "ca.crt"

    if not all(p.exists() for p in [crt, key, ca]):
        logger.warning("running_without_mtls", extra={"layer": layer})
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(crt), str(key))
    ctx.load_verify_locations(str(ca))
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx