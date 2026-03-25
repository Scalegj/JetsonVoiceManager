"""Wyoming Whisper speech-to-text client"""
import asyncio
import wave
import io
from typing import Optional
from wyoming.client import AsyncClient
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript
from jetson_voice.config.models import AppConfig


class WhisperClient:
    """Client for Wyoming Whisper speech-to-text service"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client: Optional[AsyncClient] = None

    async def connect(self):
        self.client = AsyncClient.from_uri(self.config.whisper_uri)
        await self.client.connect()

    async def disconnect(self):
        if self.client:
            await self.client.disconnect()

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe raw 16-bit PCM audio to text"""
        # Reconnect before each transcription for clean state
        await self.disconnect()
        await self.connect()

        try:
            await self.client.write_event(Transcribe(language="en").event())
            await self.client.write_event(
                AudioStart(
                    rate=self.config.sample_rate,
                    width=2,
                    channels=self.config.channels,
                ).event()
            )

            chunk_size = self.config.chunk_size * 2
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await self.client.write_event(
                    AudioChunk(
                        rate=self.config.sample_rate,
                        width=2,
                        channels=self.config.channels,
                        audio=chunk,
                    ).event()
                )

            await self.client.write_event(AudioStop().event())

            while True:
                try:
                    event = await asyncio.wait_for(
                        self.client.read_event(), timeout=10
                    )
                    if event is None:
                        break
                    if Transcript.is_type(event.type):
                        return Transcript.from_event(event).text
                except asyncio.TimeoutError:
                    break

            return ""

        except Exception:
            await self.disconnect()
            await self.connect()
            raise
