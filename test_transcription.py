"""
Standalone Speech-to-Text Accuracy Test Program

This program continuously listens to your microphone, transcribes speech using
Wyoming Whisper, and logs the results to a file with timestamps.

Usage:
    python test_transcription.py

Press Ctrl+C to stop the program.
All transcriptions are saved to 'transcription_log.txt' with timestamps.
"""

import asyncio
import pyaudio
import numpy as np
from datetime import datetime
from wyoming.client import AsyncClient
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript


# Configuration
WHISPER_HOST = "10.0.0.40"
WHISPER_PORT = 10300
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 2048
SILENCE_THRESHOLD = 500.0  # RMS threshold for silence detection
SILENCE_DURATION = 2.5  # Seconds of silence before stopping recording
LOG_FILE = "transcription_log.txt"


class TranscriptionTester:
    """Simple transcription tester using Wyoming Whisper"""

    def __init__(self):
        self.pyaudio = pyaudio.PyAudio()
        self.whisper_uri = f"tcp://{WHISPER_HOST}:{WHISPER_PORT}"

    def close(self):
        """Cleanup resources"""
        self.pyaudio.terminate()

    def record_audio(self, max_duration: float = 60.0) -> bytes:
        """Record audio with voice activity detection"""
        print("\n🎤 Listening... (speak now)")

        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        frames = []
        silence_chunks = 0
        max_silence_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        max_chunks = int(max_duration * SAMPLE_RATE / CHUNK_SIZE)

        started_speaking = False
        pre_speech_buffer = []  # Buffer to keep audio before speech detected
        pre_speech_buffer_size = 5

        try:
            for _ in range(max_chunks):
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception as e:
                    print(f"⚠️  Audio read error: {e}")
                    break

                # Calculate RMS for voice activity detection
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    mean_square = np.mean(audio_data.astype(np.float64) ** 2)
                    rms = np.sqrt(max(0, mean_square))
                else:
                    rms = 0

                # Check if speaking
                if rms > SILENCE_THRESHOLD:
                    # First time detecting speech - add pre-speech buffer
                    if not started_speaking:
                        frames.extend(pre_speech_buffer)
                        pre_speech_buffer = []

                    frames.append(data)
                    silence_chunks = 0
                    started_speaking = True
                    print("🔊", end="", flush=True)
                else:
                    if started_speaking:
                        frames.append(data)
                        silence_chunks += 1
                        print(".", end="", flush=True)
                    else:
                        # Before speech starts, maintain rolling buffer
                        pre_speech_buffer.append(data)
                        if len(pre_speech_buffer) > pre_speech_buffer_size:
                            pre_speech_buffer.pop(0)

                # Stop if silence detected after speaking
                if started_speaking and silence_chunks >= max_silence_chunks:
                    print()
                    break

        finally:
            stream.stop_stream()
            stream.close()

        if not started_speaking:
            print("No speech detected")
            return b""

        return b"".join(frames)

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio using Wyoming Whisper"""
        client = AsyncClient.from_uri(self.whisper_uri)

        try:
            await client.connect()

            # Send transcription request
            await client.write_event(Transcribe(language="en").event())

            # Send audio start event
            await client.write_event(
                AudioStart(
                    rate=SAMPLE_RATE,
                    width=2,  # 16-bit = 2 bytes
                    channels=CHANNELS
                ).event()
            )

            # Send audio data in chunks
            chunk_size = CHUNK_SIZE * 2  # 2 bytes per sample
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await client.write_event(
                    AudioChunk(
                        rate=SAMPLE_RATE,
                        width=2,
                        channels=CHANNELS,
                        audio=chunk
                    ).event()
                )

            # Send audio stop event
            await client.write_event(AudioStop().event())

            # Wait for transcript
            transcript_text = ""
            event_timeout = 10

            while True:
                try:
                    event = await asyncio.wait_for(
                        client.read_event(),
                        timeout=event_timeout
                    )

                    if event is None:
                        break

                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        transcript_text = transcript.text
                        break

                except asyncio.TimeoutError:
                    break

            return transcript_text

        finally:
            await client.disconnect()

    def log_transcription(self, text: str, timestamp: str):
        """Write transcription to log file"""
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {text}\n")

    async def run(self):
        """Main testing loop"""
        print("=" * 60)
        print("Speech-to-Text Accuracy Test")
        print("=" * 60)
        print(f"Whisper Server: {WHISPER_HOST}:{WHISPER_PORT}")
        print(f"Log File: {LOG_FILE}")
        print("\nPress Ctrl+C to stop\n")

        # Write header to log file
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Test Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")

        try:
            while True:
                # Record audio
                audio_data = self.record_audio()

                if not audio_data:
                    continue

                # Transcribe
                print("📝 Transcribing...")
                try:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    text = await self.transcribe(audio_data)

                    if text and text.strip():
                        print(f"✓ Transcribed: \"{text}\"")
                        self.log_transcription(text, timestamp)
                        print(f"  Logged to {LOG_FILE}")
                    else:
                        print("⚠️  No transcription returned")

                except Exception as e:
                    print(f"❌ Transcription error: {e}")

                print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nStopping test...")

            # Write footer to log file
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Test Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")

            print(f"All transcriptions saved to {LOG_FILE}")

        finally:
            self.close()


def main():
    """Entry point"""
    tester = TranscriptionTester()
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
