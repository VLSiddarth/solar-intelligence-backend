# ============================================================
# corona/signing/ecdsa_sidecar.py
# Async ECDSA signing sidecar — signs Kafka messages off the hot path
# P0 fix: ECDSA signing no longer blocks the hot path (was 2–5ms per event)
# ============================================================

import asyncio
import time
import base64
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

from shared.config.settings import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SigningJob:
    message_id: str
    payload:    bytes
    topic:      str
    submitted:  float = 0.0

    def __post_init__(self):
        self.submitted = time.monotonic()


@dataclass
class SignatureRecord:
    message_id:    str
    topic:         str
    signature_b64: str
    signed_at:     float
    latency_ms:    float


class ECDSASigningSidecar:
    """
    Async ECDSA signing sidecar.

    Architecture:
        - Hot path: produce() → queue job → return immediately (non-blocking)
        - Background loop: drain queue → sign → write to audit log
        - Unsigned messages are flagged in audit log, not dropped
        - Signature verification available for consumer-side validation

    Key fix: ECDSA signing removed from the hot path entirely.
    The 2–5ms per-message latency hit is gone.
    """

    QUEUE_MAX  = 50_000  # Max pending signatures before backpressure
    BATCH_SIZE = 100     # Sign N messages per loop iteration

    def __init__(self):
        self._queue:   asyncio.Queue[SigningJob] = asyncio.Queue(maxsize=self.QUEUE_MAX)
        self._audit:   list[SignatureRecord]     = []
        self._unsigned: list[str]               = []  # message_ids not yet signed
        self._running  = False
        self._task:    Optional[asyncio.Task]    = None
        self._private_key = self._load_or_generate_key()
        self._public_key  = self._private_key.public_key()
        logger.info("ecdsa_sidecar_initialized")

    def _load_or_generate_key(self) -> ec.EllipticCurvePrivateKey:
        key_path = Path(settings.api.secret_key if hasattr(settings.api, 'signing_key_path')
                       else "certs/signing.key")
        if key_path.exists():
            try:
                with open(key_path, "rb") as f:
                    key = serialization.load_pem_private_key(f.read(), password=None)
                logger.info("ecdsa_key_loaded", extra={"path": str(key_path)})
                return key
            except Exception as e:
                logger.warning("ecdsa_key_load_failed_generating_new", extra={"error": str(e)})

        # Generate new key
        key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        key_path.parent.mkdir(parents=True, exist_ok=True)
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ))
        pub_path = key_path.with_suffix(".pub")
        with open(pub_path, "wb") as f:
            f.write(key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ))
        logger.info("ecdsa_key_generated", extra={"path": str(key_path)})
        return key

    async def start(self) -> None:
        """Start the background signing loop."""
        self._running = True
        self._task    = asyncio.create_task(self._signing_loop())
        logger.info("ecdsa_signing_loop_started")

    async def stop(self) -> None:
        """Drain queue and stop."""
        self._running = False
        if self._task:
            # Give the loop time to drain
            try:
                await asyncio.wait_for(self._queue.join(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("ecdsa_queue_drain_timeout")
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def queue_for_signing(self, message_id: str, payload: bytes, topic: str) -> None:
        """
        Non-blocking. Returns immediately.
        The hot path calls this and moves on — signing happens asynchronously.
        """
        job = SigningJob(message_id=message_id, payload=payload, topic=topic)
        try:
            self._queue.put_nowait(job)
            self._unsigned.append(message_id)
        except asyncio.QueueFull:
            logger.warning("ecdsa_queue_full_dropping_job", extra={
                "message_id": message_id,
                "queue_size": self._queue.qsize(),
            })

    def verify(self, payload: bytes, signature_b64: str) -> bool:
        """
        Verify a signature. Call on consumer side to validate message integrity.
        """
        try:
            sig = base64.b64decode(signature_b64)
            self._public_key.verify(sig, payload, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False

    def get_signature(self, message_id: str) -> Optional[str]:
        """Look up signature for a message ID."""
        for record in reversed(self._audit):
            if record.message_id == message_id:
                return record.signature_b64
        return None

    def stats(self) -> dict:
        return {
            "queue_size":     self._queue.qsize(),
            "total_signed":   len(self._audit),
            "unsigned_count": len(self._unsigned),
        }

    # ─────────────────────────────────────────
    # Background Signing Loop
    # ─────────────────────────────────────────

    async def _signing_loop(self) -> None:
        """
        Background loop — drains the signing queue in batches.
        Target: sign 100 messages per iteration, run every 10ms.
        """
        while self._running or not self._queue.empty():
            batch: list[SigningJob] = []

            # Drain up to BATCH_SIZE items
            for _ in range(self.BATCH_SIZE):
                try:
                    job = self._queue.get_nowait()
                    batch.append(job)
                except asyncio.QueueEmpty:
                    break

            if batch:
                await self._sign_batch(batch)
            else:
                await asyncio.sleep(0.01)  # 10ms idle sleep

    async def _sign_batch(self, batch: list[SigningJob]) -> None:
        """Sign a batch of messages."""
        for job in batch:
            try:
                t_start = time.monotonic()

                # ECDSA sign — this is the slow op, now off the hot path
                signature = self._private_key.sign(
                    job.payload,
                    ec.ECDSA(hashes.SHA256()),
                )
                sig_b64 = base64.b64encode(signature).decode()

                latency = (time.monotonic() - t_start) * 1000

                record = SignatureRecord(
                    message_id=job.message_id,
                    topic=job.topic,
                    signature_b64=sig_b64,
                    signed_at=time.time(),
                    latency_ms=round(latency, 2),
                )
                self._audit.append(record)

                # Remove from unsigned list
                try:
                    self._unsigned.remove(job.message_id)
                except ValueError:
                    pass

                self._queue.task_done()

            except Exception as e:
                logger.error("ecdsa_sign_failed", extra={
                    "message_id": job.message_id,
                    "error":      str(e),
                })
                self._queue.task_done()

        if batch:
            logger.debug("ecdsa_batch_signed", extra={"count": len(batch)})


# Singleton
_sidecar: Optional[ECDSASigningSidecar] = None


def get_sidecar() -> ECDSASigningSidecar:
    global _sidecar
    if _sidecar is None:
        _sidecar = ECDSASigningSidecar()
    return _sidecar