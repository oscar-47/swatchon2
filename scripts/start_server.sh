#!/usr/bin/env bash
# start_server.sh — Start FabricFlow HTTPS server on port 8443
# Auto-generates SSL certs if missing. Optionally serves ca.pem over HTTP for iPad install.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CERTS_DIR="$REPO_ROOT/certs"
HTTPS_PORT="${HTTPS_PORT:-8443}"
HTTP_CA_PORT="${HTTP_CA_PORT:-8080}"
HTTP_CA_PID=""

cleanup() {
  echo ""
  echo "[Server] Shutting down..."
  if [[ -n "$HTTP_CA_PID" ]]; then
    kill "$HTTP_CA_PID" 2>/dev/null || true
    echo "[Server] Stopped CA download server (PID $HTTP_CA_PID)"
  fi
  # Clean up temp dir
  if [[ -n "${CA_TMPDIR:-}" && -d "$CA_TMPDIR" ]]; then
    rm -rf "$CA_TMPDIR"
  fi
  exit 0
}
trap cleanup INT TERM

# Detect LAN IP
detect_ip() {
  if [[ "$(uname)" == "Darwin" ]]; then
    ipconfig getifaddr en0 2>/dev/null || echo "127.0.0.1"
  else
    ip -4 addr show scope global | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1 || echo "127.0.0.1"
  fi
}

LAN_IP="$(detect_ip)"

# Auto-generate certs if missing
if [[ ! -f "$CERTS_DIR/server.crt" || ! -f "$CERTS_DIR/server.key" ]]; then
  echo "[Server] SSL certificates not found, generating..."
  bash "$REPO_ROOT/scripts/generate_ssl_cert.sh"
fi

# Verify certs exist after generation
if [[ ! -f "$CERTS_DIR/server.crt" || ! -f "$CERTS_DIR/server.key" ]]; then
  echo "[Server] ERROR: SSL certificates still missing after generation. Aborting."
  exit 1
fi

echo ""
echo "============================================"
echo "  FabricFlow HTTPS Server"
echo "============================================"
echo ""
echo "  Tablet URL:  https://$LAN_IP:$HTTPS_PORT/tablet"
echo "  Desktop URL: https://$LAN_IP:$HTTPS_PORT/"
echo "  API docs:    https://$LAN_IP:$HTTPS_PORT/docs"
echo ""

# Optionally start temporary HTTP server for CA certificate download
if [[ -f "$CERTS_DIR/ca.pem" ]]; then
  CA_TMPDIR="$(mktemp -d)"
  cp "$CERTS_DIR/ca.pem" "$CA_TMPDIR/ca.pem"

  echo "  iPad CA install: http://$LAN_IP:$HTTP_CA_PORT/ca.pem"
  echo "  (HTTP cert server auto-stops after 60s)"
  echo ""

  # Start a simple Python HTTP server in the temp dir, auto-kill after 60s
  (
    cd "$CA_TMPDIR"
    python3 -m http.server "$HTTP_CA_PORT" --bind 0.0.0.0 &>/dev/null &
    PID=$!
    sleep 60
    kill "$PID" 2>/dev/null || true
  ) &
  HTTP_CA_PID=$!
else
  echo "  Note: ca.pem not found, skipping HTTP cert server"
  echo ""
fi

echo "============================================"
echo ""

# Start uvicorn with HTTPS
cd "$REPO_ROOT"
exec uvicorn server.app:app \
  --host 0.0.0.0 \
  --port "$HTTPS_PORT" \
  --ssl-keyfile "$CERTS_DIR/server.key" \
  --ssl-certfile "$CERTS_DIR/server.crt" \
  --reload
