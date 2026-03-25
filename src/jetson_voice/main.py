"""JetsonCompanion - Main entry point"""
import argparse
import asyncio
import configparser
import os

from jetson_voice.config.models import create_config
from jetson_voice.voice_chat import VoiceChat
from jetson_voice.audio_handler import AudioHandler


def _load_config_file() -> configparser.SectionProxy | None:
    path = os.path.join(os.path.dirname(__file__), "..", "..", "config.ini")
    config = configparser.ConfigParser()
    if os.path.exists(path):
        config.read(path)
        return config["DEFAULT"]
    return None


def parse_arguments():
    fc = _load_config_file()

    parser = argparse.ArgumentParser(description="Jetson AI Voice Chat")

    default_ip = fc.get("jetson_ip", "localhost") if fc else "localhost"
    parser.add_argument("jetson_ip", nargs="?", default=default_ip,
                        help="Jetson/service host IP (default: localhost when in Docker)")

    parser.add_argument("--model", default=fc.get("ollama_model", "llama3.2:3b") if fc else "llama3.2:3b",
                        help="Ollama model name")
    parser.add_argument("--streaming", action="store_true",
                        default=fc.getboolean("streaming", False) if fc else False)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--silence-threshold", type=float,
                        default=fc.getfloat("silence_threshold", 500.0) if fc else 500.0)
    parser.add_argument("--silence-duration", type=float,
                        default=fc.getfloat("silence_duration", 2.5) if fc else 2.5)
    parser.add_argument("--system-prompt", type=str,
                        default=fc.get("system_prompt") if fc else None)
    parser.add_argument("--piper-voice", type=str,
                        default=fc.get("piper_voice") if fc else None)
    parser.add_argument("--debug-mode", action="store_true",
                        default=fc.getboolean("debug_mode", False) if fc else False)
    parser.add_argument("--debug-csv", type=str,
                        default=fc.get("debug_csv_path", "debug_output.csv") if fc else "debug_output.csv")
    parser.add_argument("--debug-wer-test", action="store_true",
                        default=fc.getboolean("debug_wer_test_mode", False) if fc else False)

    return parser.parse_args()


async def main():
    args = parse_arguments()

    if args.list_devices:
        config = create_config(jetson_ip="0.0.0.0")
        handler = AudioHandler(config)
        handler.list_audio_devices()
        handler.close()
        return

    kwargs = {
        "ollama_model": args.model,
        "silence_threshold": args.silence_threshold,
        "silence_duration": args.silence_duration,
        "debug_mode": args.debug_mode,
        "debug_csv_path": args.debug_csv,
        "debug_wer_test_mode": args.debug_wer_test,
    }
    if args.system_prompt:
        kwargs["system_prompt"] = args.system_prompt
    if args.piper_voice:
        kwargs["piper_voice"] = args.piper_voice

    config = create_config(jetson_ip=args.jetson_ip, **kwargs)

    print(f"\nJetson Voice Chat")
    print(f"Connecting to {config.jetson_ip}")
    print(f"LLM: {config.ollama_model} (Ollama)")
    if config.debug_mode:
        print(f"Debug Mode: Enabled (CSV: {config.debug_csv_path})")
    print()

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
