"""Wyoming Piper text-to-speech client"""
import asyncio
import wave
import io
from typing import Optional
from wyoming.client import AsyncClient
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.tts import Synthesize, SynthesizeVoice
from jetson_voice.config.models import AppConfig


class PiperClient:
    """Client for Wyoming Piper text-to-speech service"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client: Optional[AsyncClient] = None

    async def connect(self):
        self.client = AsyncClient.from_uri(self.config.piper_uri)
        await self.client.connect()

    async def disconnect(self):
        if self.client:
            await self.client.disconnect()

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV audio bytes"""
        if not self.client:
            raise RuntimeError("Not connected to Piper service")

        try:
            kwargs = {"text": text}
            if self.config.piper_voice:
                kwargs["voice"] = SynthesizeVoice(name=self.config.piper_voice)
            await self.client.write_event(Synthesize(**kwargs).event())

            audio_chunks = []
            audio_info = None

            while True:
                try:
                    event = await asyncio.wait_for(
                        self.client.read_event(), timeout=30
                    )
                    if event is None:
                        break
                    if AudioStart.is_type(event.type):
                        s = AudioStart.from_event(event)
                        audio_info = {
                            "rate": s.rate,
                            "width": s.width,
                            "channels": s.channels,
                        }
                    elif AudioChunk.is_type(event.type):
                        audio_chunks.append(AudioChunk.from_event(event).audio)
                    elif AudioStop.is_type(event.type):
                        break
                except asyncio.TimeoutError:
                    break

            if not audio_info or not audio_chunks:
                return b""

            return self._create_wav(b"".join(audio_chunks), audio_info)

        except Exception:
            await self.disconnect()
            await self.connect()
            raise

    def _create_wav(self, audio_data: bytes, audio_info: dict) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(audio_info["channels"])
            wf.setsampwidth(audio_info["width"])
            wf.setframerate(audio_info["rate"])
            wf.writeframes(audio_data)
        return buf.getvalue()
