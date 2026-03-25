"""
WebSocket audio gateway for JetsonCompanion.

Replaces WebRTC with WebSocket — TCP-based, works reliably over LAN and
through Docker without ICE/STUN/TURN.  The browser records audio via
MediaRecorder, decodes it to 16-bit 16kHz PCM in JavaScript, and sends
the raw bytes over the WebSocket.  The server passes them through the
Whisper → Ollama → Piper pipeline and sends back WAV bytes to play.
"""

import asyncio
import ipaddress
import json
import logging
import os
import socket
import ssl
import sys
from pathlib import Path

import aiohttp
import aiohttp_cors
from aiohttp import web

sys.path.insert(0, "/app")

from jetson_voice.config.models import AppConfig
from jetson_voice.voice_chat import VoiceChat

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("webrtc-gateway")

STATIC_DIR = Path(__file__).parent / "static"

_voice_chat: VoiceChat | None = None
_voice_chat_lock = asyncio.Lock()


async def get_voice_chat() -> VoiceChat:
    """Return the shared, connected VoiceChat instance."""
    global _voice_chat
    async with _voice_chat_lock:
        if _voice_chat is None or not _voice_chat.is_connected:
            config = AppConfig()
            vc = VoiceChat(config)
            await vc.connect()
            _voice_chat = vc
    return _voice_chat


# ─── HTTP endpoints ───────────────────────────────────────────────────────────

async def handle_index(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC_DIR / "index.html")


async def handle_health(request: web.Request) -> web.Response:
    status = {"gateway": "ok", "services": {}}
    vc = await get_voice_chat()
    status["services"]["connected"] = vc.is_connected
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:11434/",
                timeout=aiohttp.ClientTimeout(total=2),
            ) as r:
                status["services"]["ollama"] = "ok" if r.status == 200 else "error"
    except Exception:
        status["services"]["ollama"] = "unreachable"
    return web.json_response(status)


# ─── WebSocket handler ────────────────────────────────────────────────────────

async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    """
    Each message from the browser is raw PCM audio (Int16, 16kHz, mono).
    The server runs it through the full pipeline and sends back WAV bytes.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    log.info("WebSocket connected: %s", request.remote)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            try:
                vc = await get_voice_chat()
                wav = await vc.process_audio_bytes(msg.data)
                if wav:
                    await ws.send_bytes(wav)
                else:
                    await ws.send_str(json.dumps({"status": "no_speech"}))
            except Exception as e:
                log.error("Pipeline error: %s", e)
                try:
                    await ws.send_str(json.dumps({"error": str(e)}))
                except Exception:
                    pass
        elif msg.type == aiohttp.WSMsgType.ERROR:
            log.warning("WebSocket error: %s", ws.exception())

    log.info("WebSocket disconnected: %s", request.remote)
    return ws


# ─── App wiring ───────────────────────────────────────────────────────────────

async def on_shutdown(app: web.Application):
    global _voice_chat
    if _voice_chat:
        await _voice_chat.disconnect()
        _voice_chat = None


def build_app() -> web.Application:
    app = web.Application()

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=False,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"],
        )
    })

    routes = [
        app.router.add_get("/", handle_index),
        app.router.add_get("/health", handle_health),
        app.router.add_get("/ws", handle_ws),
        app.router.add_static("/static", STATIC_DIR),
    ]
    for route in routes:
        try:
            cors.add(route)
        except Exception:
            pass

    app.on_shutdown.append(on_shutdown)
    return app


# ─── TLS (self-signed) ────────────────────────────────────────────────────────

def _make_ssl_context(cert_dir: Path) -> ssl.SSLContext | None:
    """Generate a self-signed cert if absent, return an SSL context."""
    cert_file = cert_dir / "cert.pem"
    key_file  = cert_dir / "key.pem"

    if not cert_file.exists():
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            import datetime

            log.info("Generating self-signed TLS certificate...")
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

            try:
                lan_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                lan_ip = "127.0.0.1"

            subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "jetson-companion")])
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(subject)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.datetime.utcnow())
                .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
                .add_extension(
                    x509.SubjectAlternativeName([
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                        x509.IPAddress(ipaddress.IPv4Address(lan_ip)),
                    ]),
                    critical=False,
                )
                .sign(key, hashes.SHA256())
            )

            cert_dir.mkdir(parents=True, exist_ok=True)
            cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            key_file.write_bytes(key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))
            log.info("Certificate written to %s", cert_dir)
        except ImportError:
            log.warning("cryptography package not installed — running HTTP only")
            return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_file, key_file)
    return ctx


if __name__ == "__main__":
    cert_dir = Path(os.environ.get("TLS_CERT_DIR", "/config/tls"))
    ssl_ctx = _make_ssl_context(cert_dir)

    if ssl_ctx:
        log.info("Starting HTTPS/WSS on port 8080")
    else:
        log.info("Starting HTTP/WS on port 8080 (localhost only)")

    web.run_app(build_app(), host="0.0.0.0", port=8080, ssl_context=ssl_ctx)
