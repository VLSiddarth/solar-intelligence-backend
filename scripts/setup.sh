#!/usr/bin/env bash
# ============================================================
# scripts/setup.sh
# One-command SI environment setup
# Usage: chmod +x scripts/setup.sh && ./scripts/setup.sh
# ============================================================

set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[SI]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*"; exit 1; }

echo -e "\n${BOLD}${CYAN}☀  Solar Intelligence — Environment Setup${RESET}\n"

# ─── Prerequisites check ────────────────────────────────────
info "Checking prerequisites..."

command -v docker      >/dev/null 2>&1 || error "Docker not found. Install Docker Desktop."
command -v python3     >/dev/null 2>&1 || error "Python 3.11+ required."
command -v openssl     >/dev/null 2>&1 || error "OpenSSL required."

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" < "3.11" ]]; then
    error "Python 3.11+ required. Found: $PYTHON_VERSION"
fi
success "Prerequisites OK (Python $PYTHON_VERSION)"

# ─── Environment file ───────────────────────────────────────
if [ ! -f .env ]; then
    info "Creating .env from .env.example..."
    cp .env.example .env
    success ".env created. Edit it with your OPENAI_API_KEY before ingesting documents."
else
    info ".env already exists — skipping copy."
fi

# ─── Python virtual environment ─────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv .venv
    success "Virtual environment created at .venv/"
fi

info "Installing Python dependencies..."
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
success "Python dependencies installed."

# ─── mTLS certificates ──────────────────────────────────────
if [ ! -f "certs/ca.crt" ]; then
    info "Generating mTLS certificates for all 5 layers..."
    mkdir -p certs
    python3 -c "from shared.auth.mtls import generate_certs; generate_certs()"
    success "mTLS certificates generated in certs/"
else
    info "mTLS certificates already exist — skipping generation."
fi

# ─── Kafka topics ────────────────────────────────────────────
create_topics() {
    info "Creating Kafka topics..."
    sleep 5   # Wait for Redpanda to be ready

    docker exec si-redpanda rpk topic create si.core.raw_documents \
        --partitions 6 --replicas 1 2>/dev/null && \
        success "Topic: si.core.raw_documents" || warn "Topic may already exist."

    docker exec si-redpanda rpk topic create si.core.fused_entities \
        --partitions 6 --replicas 1 2>/dev/null && \
        success "Topic: si.core.fused_entities" || warn "Topic may already exist."

    docker exec si-redpanda rpk topic create si.core.audit \
        --partitions 3 --replicas 1 2>/dev/null && \
        success "Topic: si.core.audit" || warn "Topic may already exist."

    docker exec si-redpanda rpk topic create si.corona.webhook_dlq \
        --partitions 3 --replicas 1 2>/dev/null && \
        success "Topic: si.corona.webhook_dlq" || warn "Topic may already exist."
}

# ─── Docker Compose ─────────────────────────────────────────
info "Starting Docker Compose stack (all 5 layers)..."
docker compose pull --quiet 2>/dev/null || warn "Some images couldn't be pulled — using cached versions."
docker compose up -d --build

success "Docker Compose started."
info "Waiting for services to become healthy..."
sleep 10

# Create Kafka topics after Redpanda is up
create_topics

# ─── Run health check ────────────────────────────────────────
info "Running full health check..."
python3 scripts/health_check.py

echo -e "\n${GREEN}${BOLD}Setup complete!${RESET}"
echo -e "Next steps:"
echo -e "  1. Add your OPENAI_API_KEY to .env"
echo -e "  2. Run: ${CYAN}python scripts/smoke_test.py${RESET}"
echo -e "  3. Run: ${CYAN}pytest tests/ -v${RESET}"
echo -e "  4. Open: ${CYAN}http://localhost:8888/docs${RESET}\n"cls