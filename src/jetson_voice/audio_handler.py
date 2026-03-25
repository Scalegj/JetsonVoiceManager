"""Audio recording and playback"""
import asyncio
import io
import time
import wave
from typing import Optional

import numpy as np
import pyaudio

from jetson_voice.config.models import AppConfig


class AudioHandler:
    """Handle local audio recording and playback via PyAudio"""

    def __init__(self, config: AppConfig):
        self.config = config
        self._pyaudio: pyaudio.PyAudio | None = None

    @property
    def pyaudio(self) -> pyaudio.PyAudio:
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio

    def close(self):
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def record_audio(self, max_duration: float = 60.0) -> bytes:
        """Record audio with voice activity detection. Returns raw PCM bytes."""
        print("\n🎤 Listening...")

        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
        )

        frames = []
        silence_chunks = 0
        max_silence_chunks = int(
            self.config.silence_duration * self.config.sample_rate / self.config.chunk_size
        )
        max_chunks = int(max_duration * self.config.sample_rate / self.config.chunk_size)

        started_speaking = False
        pre_speech_buffer = []
        pre_speech_buffer_size = 5
        chunk_count = 0

        try:
            for _ in range(max_chunks):
                chunk_count += 1
                try:
                    data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"\n⚠️  Audio read error: {e}")
                    break

                audio_array = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2)) if len(audio_array) > 0 else 0

                if rms > self.config.silence_threshold:
                    if not started_speaking:
                        frames.extend(pre_speech_buffer)
                        pre_speech_buffer = []
                    frames.append(data)
                    silence_chunks = 0
                    started_speaking = True
                    if chunk_count % 3 == 0:
                        print("🔊", end="", flush=True)
                else:
                    if started_speaking:
                        frames.append(data)
                        silence_chunks += 1
                        if chunk_count % 5 == 0:
                            print(".", end="", flush=True)
                    else:
                        pre_speech_buffer.append(data)
                        if len(pre_speech_buffer) > pre_speech_buffer_size:
                            pre_speech_buffer.pop(0)

                if started_speaking and silence_chunks >= max_silence_chunks:
                    print()
                    break
        finally:
            stream.stop_stream()
            stream.close()

        if not started_speaking:
            print("\nNo speech detected")
            return b""

        return b"".join(frames)

    def play_audio(self, audio_data: bytes):
        """Play WAV audio bytes synchronously"""
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                rate = wf.getframerate()
                channels = wf.getnchannels()
                width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

            stream = self.pyaudio.open(
                format=self.pyaudio.get_format_from_width(width),
                channels=channels,
                rate=rate,
                output=True,
            )
            stream.write(frames)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Audio playback error: {e}")

    async def play_audio_stream(self, audio_queue: asyncio.Queue, debug: bool = False):
        """Continuously play WAV chunks from an async queue (None = sentinel to stop)"""
        output_stream = None
        sentence_count = 0

        try:
            while True:
                audio_data = await audio_queue.get()
                if audio_data is None:
                    break

                sentence_count += 1
                if debug:
                    print(f"\n[PLAY] Starting playback of sentence {sentence_count}")
                start = time.time()

                with wave.open(io.BytesIO(audio_data), "rb") as wf:
                    rate = wf.getframerate()
                    channels = wf.getnchannels()
                    width = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())

                if output_stream is None:
                    output_stream = self.pyaudio.open(
                        format=self.pyaudio.get_format_from_width(width),
                        channels=channels,
                        rate=rate,
                        output=True,
                    )

                output_stream.write(frames)

                if debug:
                    print(f"[PLAY] Sentence {sentence_count} played in {time.time() - start:.2f}s")

        except Exception as e:
            print(f"Audio stream playback error: {e}")
        finally:
            if output_stream is not None:
                output_stream.stop_stream()
                output_stream.close()

    def test_microphone(self) -> bool:
        try:
            stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
            )
            stream.read(self.config.chunk_size)
            stream.stop_stream()
            stream.close()
            return True
        except Exception as e:
            print(f"Microphone error: {e}")
            return False

    def list_audio_devices(self):
        print("\nAvailable audio devices:")
        info = self.pyaudio.get_host_api_info_by_index(0)
        for i in range(info.get("deviceCount")):
            dev = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
            print(f"  [{i}] {dev.get('name')}")
            print(f"      In: {dev.get('maxInputChannels')}  Out: {dev.get('maxOutputChannels')}")
            print()
