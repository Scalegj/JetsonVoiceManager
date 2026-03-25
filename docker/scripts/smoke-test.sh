#!/bin/bash
# Smoke test - quick end-to-end sanity check (no GPU required for basic tests)
set -euo pipefail

pass() { echo "✅ PASS: $*"; }
fail() { echo "❌ FAIL: $*"; exit 1; }

# Test Python imports
python3 -c "from jetson_voice.config.models import AppConfig; AppConfig()" && pass "Config imports" || fail "Config import"
python3 -c "from jetson_voice.services import LlamaCppClient" && pass "LlamaCppClient imports" || fail "LlamaCppClient import"
python3 -c "from jetson_voice.services import WhisperClient, PiperClient" && pass "Wyoming clients import" || fail "Wyoming clients import"

# Test WebRTC gateway import
python3 -c "import gateway.webrtc_server" 2>/dev/null && pass "WebRTC gateway imports" || fail "WebRTC gateway import"

echo ""
echo "All smoke tests passed."
