#!/usr/bin/env bash
# generate_ssl_cert.sh — Generate SSL certificates for local HTTPS development
# Supports mkcert (preferred) or falls back to openssl self-signed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CERTS_DIR="$REPO_ROOT/certs"

mkdir -p "$CERTS_DIR"

# Detect LAN IP
detect_ip() {
  if [[ "$(uname)" == "Darwin" ]]; then
    ipconfig getifaddr en0 2>/dev/null || echo "127.0.0.1"
  else
    ip -4 addr show scope global | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1 || echo "127.0.0.1"
  fi
}

LAN_IP="$(detect_ip)"
echo "[SSL] Detected LAN IP: $LAN_IP"

if command -v mkcert &>/dev/null; then
  echo "[SSL] Using mkcert (locally-trusted certificates)"
  mkcert -install 2>/dev/null || true
  mkcert \
    -cert-file "$CERTS_DIR/server.crt" \
    -key-file "$CERTS_DIR/server.key" \
    "$LAN_IP" localhost 127.0.0.1

  # Copy the CA root certificate for iPad installation
  CAROOT="$(mkcert -CAROOT)"
  if [[ -f "$CAROOT/rootCA.pem" ]]; then
    cp "$CAROOT/rootCA.pem" "$CERTS_DIR/ca.pem"
    echo "[SSL] CA certificate copied to $CERTS_DIR/ca.pem"
  else
    echo "[SSL] Warning: Could not find mkcert rootCA.pem at $CAROOT"
  fi

  echo "[SSL] mkcert certificates generated successfully"
else
  echo "[SSL] mkcert not found, falling back to openssl self-signed certificate"

  # Generate self-signed certificate with SAN for LAN IP
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$CERTS_DIR/server.key" \
    -out "$CERTS_DIR/server.crt" \
    -days 365 \
    -subj "/CN=FabricFlow Dev" \
    -addext "subjectAltName=IP:$LAN_IP,IP:127.0.0.1,DNS:localhost"

  # For self-signed, the cert itself is the CA
  cp "$CERTS_DIR/server.crt" "$CERTS_DIR/ca.pem"

  echo "[SSL] Self-signed certificate generated (browser will show warning)"
  echo "[SSL] Install $CERTS_DIR/ca.pem on iPad to trust this certificate"
fi

echo ""
echo "[SSL] Output files:"
echo "  Key:  $CERTS_DIR/server.key"
echo "  Cert: $CERTS_DIR/server.crt"
echo "  CA:   $CERTS_DIR/ca.pem"
