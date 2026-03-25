# syntax=docker/dockerfile:1.7
# Multi-arch JetsonCompanion container
# Supports: linux/arm64 (Jetson Orin Nano) and linux/amd64 (RTX 5080 testing)

ARG TARGETPLATFORM
ARG TARGETARCH=amd64

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1a: ARM64 base (JetPack + TensorRT)
# ─────────────────────────────────────────────────────────────────────────────
FROM dustynv/l4t-pytorch:r36.2.0 AS base-arm64

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b: x86_64 base (CUDA 12.6 for RTX 5080)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS base-amd64
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: System dependencies (common to all stages)
# ─────────────────────────────────────────────────────────────────────────────
FROM base-${TARGETARCH} AS system-deps

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl wget \
        python3-dev \
        portaudio19-dev libsndfile1 \
        ca-certificates \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Download Ollama (binary + GPU backends)
# ─────────────────────────────────────────────────────────────────────────────
# Pull Ollama binary + GPU backends directly from the official image.
# The official image ships CUDA 12, CUDA 13 (Blackwell/RTX 5000), and Vulkan
# backends pre-built — no tarball version pinning required.
FROM ollama/ollama:latest AS ollama-builder

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Download Piper binary
# ─────────────────────────────────────────────────────────────────────────────
FROM system-deps AS piper-builder

ARG TARGETARCH=amd64
RUN PIPER_VERSION=2023.11.14-2 \
    && if [ "$TARGETARCH" = "arm64" ]; then ARCH="aarch64"; else ARCH="x86_64"; fi \
    && wget -q "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${ARCH}.tar.gz" \
        -O /tmp/piper.tar.gz \
    && tar -xzf /tmp/piper.tar.gz -C /opt \
    && rm /tmp/piper.tar.gz

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Download S6-Overlay
# ─────────────────────────────────────────────────────────────────────────────
FROM system-deps AS s6-overlay

ARG TARGETARCH=amd64
RUN S6_VERSION=3.1.6.2 \
    && if [ "$TARGETARCH" = "arm64" ]; then S6_ARCH="aarch64"; else S6_ARCH="x86_64"; fi \
    && wget -q "https://github.com/just-containers/s6-overlay/releases/download/v${S6_VERSION}/s6-overlay-noarch.tar.xz" \
        -O /tmp/s6-noarch.tar.xz \
    && wget -q "https://github.com/just-containers/s6-overlay/releases/download/v${S6_VERSION}/s6-overlay-${S6_ARCH}.tar.xz" \
        -O /tmp/s6-arch.tar.xz \
    && mkdir -p /s6-overlay \
    && tar -Jxf /tmp/s6-noarch.tar.xz -C /s6-overlay \
    && tar -Jxf /tmp/s6-arch.tar.xz -C /s6-overlay \
    && rm /tmp/s6-*.tar.xz

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Final image
# ─────────────────────────────────────────────────────────────────────────────
FROM system-deps AS final

# S6-Overlay process supervisor
COPY --from=s6-overlay /s6-overlay /

# Ollama binary + GPU backend libraries.
# Libs must land in /usr/lib/ollama — that is the hardcoded search path in
# Ollama 0.17+ (visible as libdirs=ollama in runner logs).
COPY --from=ollama-builder /usr/bin/ollama /usr/local/bin/ollama
COPY --from=ollama-builder /usr/lib/ollama /usr/lib/ollama

# Piper TTS binary
COPY --from=piper-builder /opt/piper /opt/piper

# Install Python dependencies (cached separately from app code)
# Copy pyproject.toml first so this layer is only invalidated when deps change
COPY pyproject.toml /tmp/pyproject.toml
RUN pip3 install --no-cache-dir \
        wyoming>=1.5.0 \
        pyaudio>=0.2.14 \
        numpy>=1.24.0 \
        aiohttp>=3.9.0 \
        aiohttp-cors>=0.7.0 \
        cryptography>=42.0.0 \
        tiktoken>=0.5.0 \
        jiwer>=3.0.0 \
        wyoming-faster-whisper \
        wyoming-piper

# Create non-root user
RUN groupadd -g 1000 jetson && useradd -u 1000 -g jetson -s /bin/bash -m jetson

# Create runtime directories
RUN mkdir -p /models/whisper /models/piper /models/ollama /config /app \
    && chown -R jetson:jetson /models /config /app

# Copy application code (last - changes most often)
COPY --chown=jetson:jetson src/jetson_voice /app/jetson_voice
COPY --chown=jetson:jetson docker/gateway /app/gateway
COPY --chown=jetson:jetson configs/config.docker.yml /config/config.yml

# S6 service definitions and scripts
# Strip Windows CRLF line endings so s6-rc-compile can read type/run files correctly
COPY docker/s6-rc.d /etc/s6-overlay/s6-rc.d/
RUN find /etc/s6-overlay/s6-rc.d -type f | xargs sed -i 's/\r//'
COPY --chmod=755 docker/scripts /opt/scripts
RUN find /opt/scripts -type f | xargs sed -i 's/\r//'

ENV PYTHONPATH=/app
ENV PATH=/usr/local/bin:/opt/piper:$PATH
ENV OLLAMA_MODELS=/models/ollama
ENV OLLAMA_HOST=127.0.0.1:11434
# Tell the NVIDIA Container Toolkit which GPUs and driver capabilities to expose.
# Required for Ollama's NVML-based GPU detection to work; mirrors Ollama's own image.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Match the official Ollama image's LD_LIBRARY_PATH so NVML is found at runtime.
# /usr/local/nvidia/lib{,64} are bind-mounted by the NVIDIA Container Toolkit.
ENV LD_LIBRARY_PATH=/usr/lib/ollama:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# Allow unlimited time for model downloads and service startup
ENV S6_CMD_WAIT_FOR_SERVICES_MAXTIME=0

VOLUME ["/models", "/config"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /opt/scripts/healthcheck.sh

ENTRYPOINT ["/init"]
