"""Application configuration"""
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """Configuration for JetsonCompanion services"""

    # Connection - set to 'localhost' when running inside Docker
    jetson_ip: str = "localhost"

    # Service ports
    whisper_port: int = 10300
    piper_port: int = 10200
    ollama_port: int = 11434

    # LLM settings (Ollama)
    ollama_model: str = "llama3.2:3b"

    # Piper TTS settings
    piper_voice: str = None

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 2048

    # Voice activity detection
    silence_threshold: float = 500.0
    silence_duration: float = 2.5

    # Performance
    num_tts_clients: int = 3
    debug_pipeline: bool = False

    # Debug mode
    debug_mode: bool = False
    debug_csv_path: str = "debug_output.csv"
    debug_wer_test_mode: bool = False

    # System prompt
    system_prompt: str = (
        "You are a friendly AI companion for an elderly individual who may be lonely. "
        "You will be used in a voice conversation with them. "
        "Keep responses very short and natural - like talking to a friend. "
        "Use 1-3 sentences maximum. Speak conversationally, not formally. "
        "Avoid lists, bullet points, symbols, or anything that doesn't sound natural when spoken aloud. "
        "Don't say things like 'here are some options' or use numbered lists. "
        "Just speak naturally as if you're having a casual chat and trying to alleviate their loneliness. "
        "Do not explicitly mention their loneliness. "
        "Simply be there for them as a friend and companion. "
    )

    def __post_init__(self):
        if not self.jetson_ip:
            raise ValueError("jetson_ip must be provided")

    @property
    def whisper_uri(self) -> str:
        return f"tcp://{self.jetson_ip}:{self.whisper_port}"

    @property
    def piper_uri(self) -> str:
        return f"tcp://{self.jetson_ip}:{self.piper_port}"

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.jetson_ip}:{self.ollama_port}"


def create_config(jetson_ip: str = "localhost", **kwargs) -> AppConfig:
    """Create configuration with optional overrides"""
    return AppConfig(jetson_ip=jetson_ip, **kwargs)
