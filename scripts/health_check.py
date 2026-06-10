#!/usr/bin/env python3
# ============================================================
# scripts/health_check.py
# Verifies all 5 SI layers are up after docker compose up -d
# Run: python scripts/health_check.py
# ============================================================

import sys
import time
import socket
import subprocess
from dataclasses import dataclass
from typing import Callable


# ─── ANSI colors ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


@dataclass
class Check:
    name:    str
    layer:   str
    fn:      Callable[[], bool]
    timeout: int = 60   # seconds to wait before giving up


def tcp_check(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def http_check(url: str, timeout: float = 5.0) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


# ─── All checks ──────────────────────────────────────────────
CHECKS = [
    # Layer I — Core
    Check("Redpanda (Kafka)",   "CORE",       lambda: tcp_check("localhost", 9092),     timeout=90),
    Check("FalkorDB (GraphRAG)","CORE",       lambda: tcp_check("localhost", 6380),     timeout=60),
    Check("Flink JobManager",   "CORE",       lambda: http_check("http://localhost:8081/overview"), timeout=90),

    # Layer II — Radiative
    Check("Milvus",             "RADIATIVE",  lambda: tcp_check("localhost", 19530),    timeout=120),
    Check("PostgreSQL+pgvector","RADIATIVE",  lambda: tcp_check("localhost", 5432),     timeout=60),
    Check("TGI (BGE-M3)",       "RADIATIVE",  lambda: http_check("http://localhost:8080/health"), timeout=180),

    # Layer III — Convective
    Check("Redis Stack",        "CONVECTIVE", lambda: tcp_check("localhost", 6379),     timeout=30),

    # Layer IV — Photosphere
    Check("SI API",             "PHOTOSPHERE",lambda: http_check("http://localhost:8888/health"), timeout=60),
    Check("Kong Gateway",       "PHOTOSPHERE",lambda: http_check("http://localhost:8000"), timeout=30),

    # Layer V — Corona
    Check("OTel Collector",     "CORONA",     lambda: tcp_check("localhost", 4317),     timeout=30),
    Check("Jaeger UI",          "CORONA",     lambda: http_check("http://localhost:16686"), timeout=30),
    Check("Prometheus",         "CORONA",     lambda: http_check("http://localhost:9090/-/ready"), timeout=30),
    Check("Grafana",            "CORONA",     lambda: http_check("http://localhost:3000/api/health"), timeout=30),
]

LAYER_COLORS = {
    "CORE":        "\033[91m",   # red
    "RADIATIVE":   "\033[33m",   # orange/yellow
    "CONVECTIVE":  "\033[93m",   # yellow
    "PHOTOSPHERE": "\033[97m",   # white
    "CORONA":      "\033[94m",   # blue
}


def run_check_with_retry(check: Check) -> tuple[bool, float]:
    start    = time.monotonic()
    deadline = start + check.timeout
    attempts = 0

    while time.monotonic() < deadline:
        attempts += 1
        try:
            if check.fn():
                elapsed = time.monotonic() - start
                return True, elapsed
        except Exception:
            pass
        time.sleep(2)

    elapsed = time.monotonic() - start
    return False, elapsed


def print_banner():
    print(f"\n{BOLD}{CYAN}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║          ☀  SOLAR INTELLIGENCE HEALTH CHECK         ║")
    print("  ║         Verifying all 5 stellar layers...            ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")


def main():
    print_banner()

    results  = []
    passed   = 0
    failed   = 0
    current_layer = ""

    for check in CHECKS:
        if check.layer != current_layer:
            current_layer = check.layer
            color = LAYER_COLORS.get(check.layer, CYAN)
            print(f"  {color}{BOLD}── Layer {check.layer} ─────────────────────────────────────{RESET}")

        print(f"  Checking {check.name:<30}", end="", flush=True)
        ok, elapsed = run_check_with_retry(check)

        if ok:
            print(f"{GREEN}✓ UP{RESET}  ({elapsed:.1f}s)")
            passed += 1
        else:
            print(f"{RED}✗ FAILED{RESET} (timeout after {check.timeout}s)")
            failed += 1

        results.append((check.name, check.layer, ok, elapsed))

    # ── Summary ──────────────────────────────────────────────
    total = len(CHECKS)
    print(f"\n  {'─'*54}")
    print(f"  {BOLD}Results:{RESET} {GREEN}{passed}/{total} passing{RESET}", end="")
    if failed > 0:
        print(f"  {RED}{failed} failed{RESET}")
    else:
        print()

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}☀  All systems nominal. SI is ready for launch.{RESET}")
        print(f"\n  {CYAN}Dashboard URLs:{RESET}")
        print(f"    API docs:       http://localhost:8888/docs")
        print(f"    Kong admin:     http://localhost:8002")
        print(f"    Grafana:        http://localhost:3000  (admin / si_admin)")
        print(f"    Jaeger:         http://localhost:16686")
        print(f"    Flink UI:       http://localhost:8081")
        print(f"    Redpanda UI:    http://localhost:8085")
        print(f"    RedisInsight:   http://localhost:8001")
        print()
    else:
        print(f"\n  {RED}{BOLD}⚠  {failed} service(s) failed. Check docker compose logs.{RESET}")
        failed_checks = [r for r in results if not r[2]]
        for name, layer, _, _ in failed_checks:
            print(f"    {RED}✗{RESET} {name} ({layer})")
        print(f"\n  Run: {YELLOW}docker compose logs --tail=50 <service-name>{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()