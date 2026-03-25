"""Debug data collection and CSV logging"""
import csv
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

import tiktoken
from jiwer import wer


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn"""
    timestamp: str
    reference_text: Optional[str]
    reference_tokens: Optional[int]
    transcribed_text: str
    transcription_tokens: int
    wer: Optional[float]
    transcription_latency_ms: float
    llm_response: str
    llm_response_tokens: int
    end_to_end_latency_ms: Optional[float]


_CSV_FIELDS = list(TurnMetrics.__dataclass_fields__.keys())


class DebugLogger:
    """Collect per-turn metrics and write to CSV"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    def log_turn(self, metrics: TurnMetrics):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(asdict(metrics))

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> Optional[float]:
        if not reference or not hypothesis:
            return None
        try:
            return wer(DebugLogger.normalize_text(reference), DebugLogger.normalize_text(hypothesis))
        except Exception as e:
            print(f"WER calculation error: {e}")
            return None

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        try:
            return len(tiktoken.encoding_for_model(model).encode(text))
        except Exception:
            try:
                return len(tiktoken.get_encoding("cl100k_base").encode(text))
            except Exception:
                return len(text) // 4


class TimingContext:
    """Context manager for timing code blocks"""

    def __init__(self):
        self.start_time = None
        self.elapsed_ms: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        return False

    def get_elapsed_ms(self) -> Optional[float]:
        return self.elapsed_ms
