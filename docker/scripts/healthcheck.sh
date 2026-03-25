#!/bin/bash
# Container health check - verifies all services are responding
set -euo pipefail

fail() { echo "UNHEALTHY: $*" >&2; exit 1; }

# Check WebRTC gateway
curl -sf http://localhost:8080/health > /dev/null || fail "WebRTC gateway not responding"

# Check Ollama
curl -sf http://localhost:11434/ > /dev/null || fail "Ollama not responding"

echo "HEALTHY"
exit 0
