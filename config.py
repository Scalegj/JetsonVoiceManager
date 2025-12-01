"""Configuration for Jetson Voice Chat"""
from dataclasses import dataclass


@dataclass
class JetsonConfig:
    """Configuration for connecting to Jetson services"""
    
    # Jetson IP address
    jetson_ip: str
    
    # Service ports
    whisper_port: int = 10300
    piper_port: int = 10200
    ollama_port: int = 11434
    
    # Ollama settings
    ollama_model: str = "llama3.2"

    # Piper TTS settings
    piper_voice: str = None  # Optional: specific voice model for Piper TTS

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 2048  # Larger chunks to prevent buffer overflow
    
    # Voice activity detection
    silence_threshold: float = 500.0  # RMS threshold for silence detection
    silence_duration: float = 2.5  # Seconds of silence before stopping recording

    # Interrupt settings
    enable_interrupts: bool = True  # Allow user to interrupt AI speech
    interrupt_threshold: float = 600.0  # RMS threshold for interrupt detection (slightly higher than silence)
    interrupt_confirmation_chunks: int = 2  # Number of consecutive chunks above threshold to confirm interrupt

    # Performance settings
    num_tts_clients: int = 3  # Number of concurrent TTS connections for parallel synthesis
    debug_pipeline: bool = False  # Enable detailed pipeline timing diagnostics


    # System prompt for the chatbot
    system_prompt: str = (
        "You are a friendly AI assistant in a voice conversation. "
        "Keep responses very short and natural - like talking to a friend. "
        "Use 1-3 sentences maximum. Speak conversationally, not formally. "
        "Avoid lists, bullet points, symbols, or anything that doesn't sound natural when spoken aloud. "
        "Don't say things like 'here are some options' or use numbered lists. "
        "Just speak naturally as if you're having a casual chat."
    )
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.jetson_ip:
            raise ValueError("Jetson IP address must be provided")
    
    @property
    def whisper_uri(self) -> str:
        """Get Whisper service URI"""
        return f"tcp://{self.jetson_ip}:{self.whisper_port}"
    
    @property
    def piper_uri(self) -> str:
        """Get Piper service URI"""
        return f"tcp://{self.jetson_ip}:{self.piper_port}"
    
    @property
    def ollama_base_url(self) -> str:
        """Get Ollama API base URL"""
        return f"http://{self.jetson_ip}:{self.ollama_port}"


def create_config(jetson_ip: str, **kwargs) -> JetsonConfig:
    """Create configuration with Jetson IP and optional overrides"""
    return JetsonConfig(jetson_ip=jetson_ip, **kwargs)
