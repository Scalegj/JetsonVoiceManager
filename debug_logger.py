"""Debug mode data collection and logging"""
import csv
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import tiktoken
from jiwer import wer


@dataclass
class TurnMetrics:
    """Metrics collected for a single conversation turn"""
    timestamp: str
    reference_text: Optional[str]  # Ground truth for WER testing
    reference_tokens: Optional[int]  # Token count for reference text
    transcribed_text: str
    transcription_tokens: int
    wer: Optional[float]  # Word Error Rate (0.0-1.0)
    transcription_latency_ms: float  # Transcription time
    llm_response: str
    llm_response_tokens: int
    end_to_end_latency_ms: Optional[float]  # User stops speaking -> AI starts speaking  # LLM inference time


class DebugLogger:
    """Handles debug data collection and CSV writing"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp',
                    'reference_text',
                    'reference_tokens',
                    'transcribed_text',
                    'transcription_tokens',
                    'wer',
                    'transcription_latency_ms',
                    'llm_response',
                    'llm_response_tokens',
                    'end_to_end_latency_ms'
                ])
                writer.writeheader()

    def log_turn(self, metrics: TurnMetrics):
        """
        Write turn metrics to CSV immediately (sync write).

        Args:
            metrics: TurnMetrics dataclass instance
        """
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp',
                'reference_text',
                'reference_tokens',
                'transcribed_text',
                'transcription_tokens',
                'wer',
                'transcription_latency_ms',
                'llm_response',
                'llm_response_tokens',
                'end_to_end_latency_ms'
            ])
            writer.writerow(asdict(metrics))

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for WER calculation by removing punctuation and extra whitespace.

        Args:
            text: Text to normalize

        Returns:
            Normalized text (lowercase, no punctuation, single spaces)
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation (commas, periods, question marks, exclamation points, etc.)
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace (multiple spaces to single space)
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> Optional[float]:
        """
        Calculate Word Error Rate between reference and hypothesis text.
        Text is normalized (punctuation removed, lowercase) before calculation.

        Args:
            reference: Ground truth text
            hypothesis: Transcribed/predicted text

        Returns:
            WER as float between 0.0 (perfect) and 1.0+ (very poor)
        """
        if not reference or not hypothesis:
            return None

        try:
            # Normalize both texts to ignore punctuation and capitalization
            normalized_reference = DebugLogger.normalize_text(reference)
            normalized_hypothesis = DebugLogger.normalize_text(hypothesis)

            return wer(normalized_reference, normalized_hypothesis)
        except Exception as e:
            print(f"WER calculation error: {e}")
            return None

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to tokenize
            model: Model name for tokenizer (default: gpt-4)

        Returns:
            Token count
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to cl100k_base encoding if model not recognized
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as e2:
                print(f"Token counting error: {e2}")
                # Rough approximation: 1 token ≈ 4 characters
                return len(text) // 4


class TimingContext:
    """Context manager for timing operations"""

    def __init__(self):
        self.start_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        return False

    def get_elapsed_ms(self) -> Optional[float]:
        """Get elapsed time in milliseconds"""
        return self.elapsed_ms if self.elapsed_ms is not None else None
