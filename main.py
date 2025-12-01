"""Jetson Voice Chat Client - Main Entry Point"""
import asyncio
import argparse
import configparser
import os
from config import create_config
from voice_chat import VoiceChat
from audio_handler import AudioHandler


def load_config_file():
    """Load configuration from config.ini file"""
    config_file = os.path.join(os.path.dirname(__file__), "config.ini")
    config = configparser.ConfigParser()

    if os.path.exists(config_file):
        config.read(config_file)
        return config['DEFAULT']
    return None


def parse_arguments():
    """Parse command line arguments"""
    # Load defaults from config file if it exists
    file_config = load_config_file()

    parser = argparse.ArgumentParser(description="Jetson AI Voice Chat")

    # Make jetson_ip optional if config file exists
    if file_config and 'jetson_ip' in file_config:
        parser.add_argument("jetson_ip", nargs='?',
                          default=file_config['jetson_ip'],
                          help="Jetson IP address (default: from config.ini)")
    else:
        parser.add_argument("jetson_ip", help="Jetson IP address")

    # Other arguments with config file defaults
    model_default = file_config.get('ollama_model', 'llama3.2:3b') if file_config else 'llama3.2:3b'
    parser.add_argument("--model", default=model_default, help="Ollama model")

    streaming_default = file_config.getboolean('streaming', False) if file_config else False
    parser.add_argument("--streaming", action="store_true", default=streaming_default,
                       help="Enable streaming")

    interrupts_default = file_config.getboolean('enable_interrupts', True) if file_config else True
    parser.add_argument("--no-interrupts", action="store_false", dest="enable_interrupts",
                       default=interrupts_default,
                       help="Disable interrupt detection (saves resources)")

    parser.add_argument("--list-devices", action="store_true", help="List audio devices")

    silence_threshold_default = file_config.getfloat('silence_threshold', 500.0) if file_config else 500.0
    parser.add_argument("--silence-threshold", type=float, default=silence_threshold_default)

    silence_duration_default = file_config.getfloat('silence_duration', 2.5) if file_config else 2.5
    parser.add_argument("--silence-duration", type=float, default=silence_duration_default)

    system_prompt_default = file_config.get('system_prompt') if file_config else None
    parser.add_argument("--system-prompt", type=str, default=system_prompt_default,
                       help="Custom system prompt")

    piper_voice_default = file_config.get('piper_voice') if file_config else None
    parser.add_argument("--piper-voice", type=str, default=piper_voice_default,
                       help="Piper TTS voice model")

    return parser.parse_args()


async def main():
    """Main application entry point"""
    args = parse_arguments()

    # List audio devices if requested
    if args.list_devices:
        config = create_config(jetson_ip="0.0.0.0")
        audio_handler = AudioHandler(config)
        audio_handler.list_audio_devices()
        audio_handler.close()
        return

    # Create configuration
    config_kwargs = {
        "ollama_model": args.model,
        "silence_threshold": args.silence_threshold,
        "silence_duration": args.silence_duration,
        "enable_interrupts": args.enable_interrupts,
    }
    if args.system_prompt:
        config_kwargs["system_prompt"] = args.system_prompt
    if args.piper_voice:
        config_kwargs["piper_voice"] = args.piper_voice

    config = create_config(jetson_ip=args.jetson_ip, **config_kwargs)

    # Display configuration
    print(f"\nJetson Voice Chat")
    print(f"Connecting to {config.jetson_ip}")
    print(f"Model: {config.ollama_model}")
    print(f"Interrupts: {'Enabled' if config.enable_interrupts else 'Disabled'}\n")

    # Run voice chat
    voice_chat = VoiceChat(config)
    try:
        await voice_chat.run(streaming=args.streaming)
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")
