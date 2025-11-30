"""Audio recording and playback"""
import pyaudio
import numpy as np
import wave
import io
from typing import Optional
from config import JetsonConfig


class AudioHandler:
    """Handle audio recording and playback"""
    
    def __init__(self, config: JetsonConfig):
        self.config = config
        self.pyaudio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
    
    def __del__(self):
        """Cleanup audio resources"""
        self.close()
    
    def close(self):
        """Close audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
    
    def record_audio(self, max_duration: float = 60.0) -> bytes:
        """Record audio with voice activity detection"""
        print("\n🎤 Listening...")

        # Open stream for recording
        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size
        )

        frames = []
        silence_chunks = 0
        max_silence_chunks = int(self.config.silence_duration * self.config.sample_rate / self.config.chunk_size)
        max_chunks = int(max_duration * self.config.sample_rate / self.config.chunk_size)

        started_speaking = False
        pre_speech_buffer = []  # Buffer to keep audio before speech detected
        pre_speech_buffer_size = 5  # Keep last 5 chunks before speech
        chunk_count = 0  # Track chunks for print throttling

        try:
            for _ in range(max_chunks):
                chunk_count += 1
                try:
                    data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"\n⚠️  Audio read error: {e}")
                    break

                # Calculate RMS for voice activity detection
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    mean_square = np.mean(audio_data.astype(np.float64) ** 2)
                    rms = np.sqrt(max(0, mean_square))  # Ensure non-negative before sqrt
                else:
                    rms = 0

                # Check if speaking
                if rms > self.config.silence_threshold:
                    # First time detecting speech - add pre-speech buffer
                    if not started_speaking:
                        frames.extend(pre_speech_buffer)
                        pre_speech_buffer = []

                    frames.append(data)
                    silence_chunks = 0
                    started_speaking = True
                    # Print every 3rd chunk to reduce overhead
                    if chunk_count % 3 == 0:
                        print("🔊", end="", flush=True)
                else:
                    if started_speaking:
                        # After speech started, keep adding frames
                        frames.append(data)
                        silence_chunks += 1
                        # Print every 5th chunk to reduce overhead
                        if chunk_count % 5 == 0:
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
            print("\nNo speech detected")
            return b""
        
        # Convert frames to bytes
        return b"".join(frames)
    
    def play_audio(self, audio_data: bytes):
        """Play audio from WAV bytes"""
        try:
            with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                audio_frames = wav_file.readframes(wav_file.getnframes())

            stream = self.pyaudio.open(
                format=self.pyaudio.get_format_from_width(sample_width),
                channels=channels,
                rate=sample_rate,
                output=True
            )

            stream.write(audio_frames)
            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"Audio playback error: {e}")

    async def play_audio_stream(self, audio_queue, debug=False):
        """Play audio sentences continuously from an async queue without gaps.

        Args:
            audio_queue: asyncio.Queue that yields WAV audio bytes.
                         Queue should receive None as sentinel to stop.
            debug: If True, print timing information for each sentence.
        """
        import time
        stream = None
        sentence_count = 0

        try:
            while True:
                audio_data = await audio_queue.get()
                if audio_data is None:  # Sentinel to stop
                    break

                sentence_count += 1
                if debug:
                    print(f"\n[PLAY] Starting playback of sentence {sentence_count}")
                playback_start = time.time()

                # Extract audio frames from WAV bytes
                with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    audio_frames = wav_file.readframes(wav_file.getnframes())

                # Open stream on first sentence
                if stream is None:
                    stream = self.pyaudio.open(
                        format=self.pyaudio.get_format_from_width(sample_width),
                        channels=channels,
                        rate=sample_rate,
                        output=True
                    )

                # Write frames directly without closing stream
                stream.write(audio_frames)

                if debug:
                    playback_elapsed = time.time() - playback_start
                    print(f"[PLAY] Sentence {sentence_count} played in {playback_elapsed:.2f}s")

        except Exception as e:
            print(f"Audio stream playback error: {e}")

        finally:
            # Close stream when done
            if stream is not None:
                stream.stop_stream()
                stream.close()
    
    def test_microphone(self) -> bool:
        """Test if microphone is accessible"""
        try:
            stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            stream.read(self.config.chunk_size)
            stream.stop_stream()
            stream.close()
            return True
        except Exception as e:
            print(f"Microphone error: {e}")
            return False
    
    def list_audio_devices(self):
        """List all available audio devices"""
        print("\nAvailable audio devices:")
        info = self.pyaudio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
            print(f"  [{i}] {device_info.get('name')}")
            print(f"      Max Input Channels: {device_info.get('maxInputChannels')}")
            print(f"      Max Output Channels: {device_info.get('maxOutputChannels')}")
            print()
