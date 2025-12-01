# Jetson AI Voice Chat

A voice-enabled AI chatbot that runs locally on a Jetson Orin Nano. Designed as a companion for elderly individuals with full local processing for privacy.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Windows users**: If PyAudio fails, download the wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

### 2. Verify Jetson Services

Ensure Docker containers are running on your Jetson:

```bash
docker compose ps
```

Required services:
- Ollama (port 11434)
- Whisper-TRT (port 10300)
- Piper TTS (port 10200)

### 3. Run

```bash
python main.py <JETSON_IP>
```

Example:
```bash
python main.py 192.168.1.100
```

## Usage

```bash
# Basic usage
python main.py 192.168.1.100

# Use faster model
python main.py 192.168.1.100 --model llama3.2:1b

# Enable streaming responses
python main.py 192.168.1.100 --streaming

# Adjust for noisy environment
python main.py 192.168.1.100 --silence-threshold 800

# Disable interrupts to save resources
python main.py 192.168.1.100 --no-interrupts

# List audio devices
python main.py 0.0.0.0 --list-devices
```

## Project Structure

```
├── main.py              # Entry point and CLI
├── voice_chat.py        # Conversation orchestrator
├── audio_handler.py     # Audio recording/playback
├── wyoming_client.py    # Whisper & Piper clients
├── ollama_client.py     # Ollama LLM client
├── config.py            # Configuration
└── requirements.txt     # Dependencies
```

## How It Works

1. Records audio from microphone with voice detection
2. Sends audio to Whisper (STT) on Jetson
3. Processes text with Ollama (LLM)
4. Synthesizes speech with Piper (TTS)
5. Plays response through speakers
6. Monitors for user interruptions during AI speech (optional)

## Interrupts

The system supports natural conversation interrupts - you can speak while the AI is responding to cut it off and provide new input. This feature:

- Uses minimal resources (simple RMS-based voice detection)
- Can be disabled with `--no-interrupts` to save even more resources
- Automatically stops AI speech and TTS synthesis when you start speaking
- Processes your interruption as a new conversation turn

**Configuration:**
- `enable_interrupts`: Enable/disable interrupt detection (default: True)
- `interrupt_threshold`: RMS threshold for interrupt detection (default: 600.0)
- `interrupt_confirmation_chunks`: Consecutive chunks needed to confirm interrupt (default: 2)

## Troubleshooting

### Microphone Issues
- List devices: `python main.py 0.0.0.0 --list-devices`
- Check Windows microphone permissions
- Adjust threshold: `--silence-threshold 800`

### Connection Issues
- Verify Jetson IP: `ping <JETSON_IP>`
- Check Docker containers: `docker ps`
- Test Ollama: `curl http://<JETSON_IP>:11434/api/tags`

### Audio Detection
- Cuts off early: `--silence-duration 2.5`
- Too sensitive: `--silence-threshold 800`

## Configuration

Default settings:
- Sample Rate: 16000 Hz
- Channels: Mono
- Silence Threshold: 500 RMS
- Silence Duration: 2.5 seconds
- Model: llama3.2:3b
- Interrupts: Enabled
- Interrupt Threshold: 600 RMS

Customize settings:
```python
from config import create_config

config = create_config(
    jetson_ip="192.168.1.100",
    system_prompt="Custom prompt here",
    enable_interrupts=True,
    interrupt_threshold=600.0
)
```

## ISEF Project

This project demonstrates local AI processing for elderly companions, emphasizing privacy and accessibility through edge computing.

## License

Created for educational and research purposes.
