from .stt import WhisperClient
from .tts import PiperClient
from .llm import OllamaClient

__all__ = ["WhisperClient", "PiperClient", "OllamaClient"]


def create_llm_client(config):
    """Factory: returns an OllamaClient."""
    return OllamaClient(config)
