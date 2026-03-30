"""Debug mode data collection and logging"""
import csv
import io
import os
import re
import time
import wave
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import tiktoken
from jiwer import wer

# Ordered list of CSV column names — must match TurnMetrics fields exactly
_CSV_FIELDS = [
    'turn_number',
    'timestamp',
    'audio_input_duration_ms',
    'transcribed_text',
    'transcription_tokens',
    'transcription_latency_ms',
    'reference_text',
    'reference_tokens',
    'wer',
    'llm_latency_ms',
    'llm_time_to_first_token_ms',
    'llm_prompt_tokens',
    'llm_response',
    'llm_response_tokens',
    'llm_response_chars',
    'llm_tokens_per_second',
    'tts_latency_ms',
    'response_audio_duration_ms',
    'time_to_first_audio_ms',
]


@dataclass
class TurnMetrics:
    """
    Metrics collected for a single conversation turn.

    Timing definitions:
      transcription_latency_ms   — wall time for Whisper to return a transcript
      llm_latency_ms             — wall time for Ollama to return the full response
                                   (streaming: time until last token; non-streaming: round-trip)
      llm_time_to_first_token_ms — streaming only: time from request sent to first token received
      tts_latency_ms             — wall time for Piper to finish all audio synthesis for this turn
      time_to_first_audio_ms     — wall time from end-of-recording to when the first audio chunk
                                   is ready for playback (the user-perceived response latency).
                                   In streaming mode this benefits from LLM/TTS concurrency, so it
                                   will typically be less than llm_latency_ms + tts_latency_ms.
    """

    # Session context
    turn_number: int                            # 1-based turn index in this session
    timestamp: str                              # ISO 8601 wall-clock timestamp

    # Input audio
    audio_input_duration_ms: float             # Duration of the recorded user utterance

    # STT — Whisper
    transcribed_text: str
    transcription_tokens: int
    transcription_latency_ms: float

    # WER testing (optional — only populated when debug_wer_test_mode is enabled)
    reference_text: Optional[str]
    reference_tokens: Optional[int]
    wer: Optional[float]                        # Word Error Rate: 0.0 = perfect

    # LLM — Ollama
    # Token counts below come directly from Ollama's API response and are exact for the
    # model in use. They should be preferred over any client-side estimation.
    llm_latency_ms: float
    llm_time_to_first_token_ms: Optional[float]  # None in non-streaming mode
    llm_prompt_tokens: Optional[int]             # Full prompt tokens (system + history + user message)
    llm_response: str
    llm_response_tokens: Optional[int]           # Tokens generated — from Ollama eval_count
    llm_response_chars: int
    llm_tokens_per_second: Optional[float]       # Derived from Ollama's eval_count / eval_duration

    # TTS — Piper
    tts_latency_ms: Optional[float]
    response_audio_duration_ms: Optional[float]  # Duration of synthesized speech

    # End-to-end user-perceived latency
    time_to_first_audio_ms: Optional[float]


class DebugLogger:
    """Handles debug data collection and CSV writing"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                writer.writeheader()

    def log_turn(self, metrics: TurnMetrics):
        """Write turn metrics to CSV immediately (append, sync)."""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writerow(asdict(metrics))

    # ------------------------------------------------------------------
    # Text / token helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace — for WER comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> Optional[float]:
        """
        Word Error Rate between reference and hypothesis (both normalized first).
        Returns a float >= 0.0, or None if either string is empty.
        """
        if not reference or not hypothesis:
            return None
        try:
            return wer(
                DebugLogger.normalize_text(reference),
                DebugLogger.normalize_text(hypothesis),
            )
        except Exception as e:
            print(f"WER calculation error: {e}")
            return None

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Estimate token count using tiktoken's cl100k_base encoding.

        IMPORTANT — when to use this vs Ollama's counts:
          - For LLM input/output tokens, use the values from OllamaClient.last_stats
            (prompt_eval_count / eval_count). Those are exact for whatever model is running.
          - Use this method only for text that does NOT pass through Ollama, such as
            reference text in WER testing or the raw transcription measured independently.

        Why cl100k_base: Llama 3.x uses a tiktoken-based tokenizer with a 128k-token
        vocabulary. tiktoken does not ship Llama's vocab, but cl100k_base (100k vocab,
        also tiktoken) is the closest available encoding. For typical English prose the
        counts differ by roughly 1–5%.
        """
        if not text:
            return 0
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Token counting error: {e}")
            return len(text) // 4  # rough fallback: ~4 chars per token

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pcm_duration_ms(pcm_bytes: bytes, sample_rate: int,
                        channels: int = 1, bit_depth: int = 16) -> float:
        """Duration of raw signed PCM audio in milliseconds."""
        bytes_per_sample = bit_depth // 8
        total_samples = len(pcm_bytes) / (bytes_per_sample * channels)
        return (total_samples / sample_rate) * 1000.0

    @staticmethod
    def wav_duration_ms(wav_bytes: bytes) -> Optional[float]:
        """Duration of a WAV file (bytes) in milliseconds, or None on parse error."""
        if not wav_bytes:
            return None
        try:
            with wave.open(io.BytesIO(wav_bytes)) as wf:
                return (wf.getnframes() / wf.getframerate()) * 1000.0
        except Exception:
            return None


class TimingContext:
    """Context manager for high-resolution timing of a code block."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        return False  # do not suppress exceptions

    def get_elapsed_ms(self) -> Optional[float]:
        return self.elapsed_ms