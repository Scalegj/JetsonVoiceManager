"""Wyoming Protocol clients for Whisper (STT) and Piper (TTS)"""
import asyncio
import wave
import io
from typing import Optional
from wyoming.client import AsyncClient
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.tts import Synthesize, SynthesizeVoice
from config import JetsonConfig


class WhisperClient:
    """Client for Wyoming Whisper speech-to-text service"""
    
    def __init__(self, config: JetsonConfig):
        self.config = config
        self.client: Optional[AsyncClient] = None
    
    async def connect(self):
        """Connect to the Whisper service"""
        self.client = AsyncClient.from_uri(self.config.whisper_uri)
        await self.client.connect()
    
    async def disconnect(self):
        """Disconnect from the Whisper service"""
        if self.client:
            await self.client.disconnect()
    
    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text

        Args:
            audio_data: Raw audio bytes (16-bit PCM)

        Returns:
            Transcribed text
        """
        # Reconnect before each transcription to ensure clean state
        await self.disconnect()
        await self.connect()

        try:
            # Send transcription request
            await self.client.write_event(Transcribe(language="en").event())
            
            # Send audio start event
            await self.client.write_event(
                AudioStart(
                    rate=self.config.sample_rate,
                    width=2,  # 16-bit = 2 bytes
                    channels=self.config.channels
                ).event()
            )
            
            # Send audio data in chunks
            chunk_size = self.config.chunk_size * 2  # 2 bytes per sample
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await self.client.write_event(
                    AudioChunk(
                        rate=self.config.sample_rate,
                        width=2,
                        channels=self.config.channels,
                        audio=chunk
                    ).event()
                )
            
            # Send audio stop event
            await self.client.write_event(AudioStop().event())
            
            # Wait for transcript
            transcript_text = ""
            event_timeout = 10  # seconds

            while True:
                try:
                    event = await asyncio.wait_for(
                        self.client.read_event(),
                        timeout=event_timeout
                    )

                    if event is None:
                        break

                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        transcript_text = transcript.text
                        # Got the transcript, we're done
                        break

                except asyncio.TimeoutError:
                    # Timeout waiting for events
                    break

            return transcript_text
        
        except Exception as e:
            # Reconnect for next attempt
            await self.disconnect()
            await self.connect()
            raise


class PiperClient:
    """Client for Wyoming Piper text-to-speech service"""
    
    def __init__(self, config: JetsonConfig):
        self.config = config
        self.client: Optional[AsyncClient] = None
    
    async def connect(self):
        """Connect to the Piper service"""
        self.client = AsyncClient.from_uri(self.config.piper_uri)
        await self.client.connect()
    
    async def disconnect(self):
        """Disconnect from the Piper service"""
        if self.client:
            await self.client.disconnect()
    
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
        
        Returns:
            Audio data as WAV bytes
        """
        if not self.client:
            raise RuntimeError("Not connected to Piper service")
        
        try:
            # Send synthesis request with optional voice parameter
            synthesize_kwargs = {"text": text}
            if self.config.piper_voice:
                synthesize_kwargs["voice"] = SynthesizeVoice(name=self.config.piper_voice)
            await self.client.write_event(Synthesize(**synthesize_kwargs).event())
            
            # Collect audio chunks
            audio_chunks = []
            audio_info = None
            event_timeout = 30  # Longer timeout for TTS (can take time for long text)
            
            while True:
                try:
                    event = await asyncio.wait_for(
                        self.client.read_event(),
                        timeout=event_timeout
                    )
                    
                    if event is None:
                        break
                    
                    if AudioStart.is_type(event.type):
                        audio_start = AudioStart.from_event(event)
                        audio_info = {
                            'rate': audio_start.rate,
                            'width': audio_start.width,
                            'channels': audio_start.channels
                        }
                    elif AudioChunk.is_type(event.type):
                        chunk = AudioChunk.from_event(event)
                        audio_chunks.append(chunk.audio)
                    elif AudioStop.is_type(event.type):
                        break
                
                except asyncio.TimeoutError:
                    break
            
            if not audio_info or not audio_chunks:
                return b""
            
            # Combine audio chunks and create WAV
            audio_data = b"".join(audio_chunks)
            return self._create_wav(audio_data, audio_info)
        
        except Exception as e:
            # Reconnect for next attempt
            await self.disconnect()
            await self.connect()
            raise
    
    def _create_wav(self, audio_data: bytes, audio_info: dict) -> bytes:
        """Create WAV file from raw audio data"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(audio_info['channels'])
            wav_file.setsampwidth(audio_info['width'])
            wav_file.setframerate(audio_info['rate'])
            wav_file.writeframes(audio_data)
        
        return wav_buffer.getvalue()
