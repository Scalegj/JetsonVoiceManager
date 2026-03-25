#!/bin/bash
# Download AI models on first container startup
set -euo pipefail

# S6 oneshot runs as root without HOME — Ollama CLI panics without it
export HOME="${HOME:-/root}"

WHISPER_MODEL="${WHISPER_MODEL:-base.en}"
PIPER_VOICE="${PIPER_VOICE:-en_US-ryan-medium}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"

WHISPER_DIR="/models/whisper"
PIPER_DIR="/models/piper"

log() { echo "[model-downloader] $*"; }

# ── Whisper ──────────────────────────────────────────────────────────────────
if [ ! -d "${WHISPER_DIR}/${WHISPER_MODEL}" ]; then
    log "Downloading Whisper model: ${WHISPER_MODEL}"
    python3 -c "
from faster_whisper import WhisperModel
WhisperModel('${WHISPER_MODEL}', device='cpu', download_root='${WHISPER_DIR}')
" || log "WARNING: Whisper download failed - wyoming-faster-whisper will retry at startup"
else
    log "Whisper model already present: ${WHISPER_MODEL}"
fi

# ── Piper voice ───────────────────────────────────────────────────────────────
# Voice name format: {locale}-{name}-{quality}  e.g. en_US-ryan-medium
# HuggingFace path:  {lang}/{locale}/{name}/{quality}/{voice}.onnx
PIPER_ONNX="${PIPER_DIR}/${PIPER_VOICE}.onnx"
if [ ! -f "$PIPER_ONNX" ]; then
    log "Downloading Piper voice: ${PIPER_VOICE}"
    mkdir -p "${PIPER_DIR}"

    # Parse voice name into path components
    LOCALE="${PIPER_VOICE%%-*}"                          # en_US
    REMAINDER="${PIPER_VOICE#*-}"                        # ryan-medium
    VOICE_NAME="${REMAINDER%%-*}"                        # ryan
    QUALITY="${REMAINDER##*-}"                           # medium
    LANG="${LOCALE%%_*}"                                 # en
    HF_PATH="${LANG}/${LOCALE}/${VOICE_NAME}/${QUALITY}"
    HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/${HF_PATH}"

    wget -q -O "${PIPER_ONNX}" "${HF_BASE}/${PIPER_VOICE}.onnx" \
        && wget -q -O "${PIPER_ONNX}.json" "${HF_BASE}/${PIPER_VOICE}.onnx.json" \
        || log "WARNING: Piper download failed - wyoming-piper will retry at startup"
else
    log "Piper voice already present: ${PIPER_VOICE}"
fi

# ── Ollama model ──────────────────────────────────────────────────────────────
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
# Strip protocol prefix so curl can use it directly
OLLAMA_ADDR="${OLLAMA_HOST#http://}"
OLLAMA_ADDR="${OLLAMA_ADDR#https://}"

log "Waiting for Ollama at ${OLLAMA_ADDR}..."
until curl -sf "http://${OLLAMA_ADDR}/api/tags" >/dev/null 2>&1; do sleep 2; done

if OLLAMA_HOST="http://${OLLAMA_ADDR}" ollama show "${OLLAMA_MODEL}" >/dev/null 2>&1; then
    log "Ollama model already present: ${OLLAMA_MODEL}"
else
    log "Pulling Ollama model: ${OLLAMA_MODEL}"
    OLLAMA_HOST="http://${OLLAMA_ADDR}" ollama pull "${OLLAMA_MODEL}"
fi

# Fix ownership so the jetson user can write cache files (refs/main, etc.)
chown -R jetson:jetson /models 2>/dev/null || true

log "All models ready."
