#!/usr/bin/env bash
# start_https.sh — Start FabricFlow server with HTTPS for iPad getUserMedia
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CERTS_DIR="$REPO_ROOT/certs"
VENV="$REPO_ROOT/.venv/bin/python3"

# Check certs exist
if [[ ! -f "$CERTS_DIR/server.crt" ]] || [[ ! -f "$CERTS_DIR/server.key" ]]; then
  echo "[HTTPS] Certificates not found. Generating..."
  bash "$REPO_ROOT/scripts/generate_ssl_cert.sh"
fi

# Detect LAN IP
if [[ "$(uname)" == "Darwin" ]]; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || echo '127.0.0.1')"
else
  LAN_IP="$(ip -4 addr show scope global | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1 || echo '127.0.0.1')"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  FabricFlow HTTPS Server                        ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  Desktop:  https://$LAN_IP:8443/tool       ║"
echo "║  Tablet:   https://$LAN_IP:8443/tablet     ║"
echo "║                                                  ║"
echo "║  CA cert:  https://$LAN_IP:8443/certs/ca.pem║"
echo "║                                                  ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  iPad setup:                                     ║"
echo "║  1. Open CA cert URL in Safari                   ║"
echo "║  2. Settings → Profile Downloaded → Install      ║"
echo "║  3. Settings → General → About →                 ║"
echo "║     Certificate Trust Settings → Enable           ║"
echo "║  4. Open tablet URL in Safari                    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Kill any existing server on 8443
lsof -ti:8443 2>/dev/null | xargs kill -9 2>/dev/null || true

exec "$VENV" -m uvicorn server.app:app \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-keyfile "$CERTS_DIR/server.key" \
  --ssl-certfile "$CERTS_DIR/server.crt"
