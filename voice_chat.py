"""
Main voice chat orchestrator
"""
import asyncio
from typing import Optional
from config import JetsonConfig
from audio_handler import AudioHandler
from wyoming_client import WhisperClient, PiperClient
from ollama_client import OllamaClient


class VoiceChat:
    """Orchestrates the voice chat conversation"""
    
    def __init__(self, config: JetsonConfig):
        self.config = config
        self.audio_handler = AudioHandler(config)
        self.whisper_client = WhisperClient(config)
        self.piper_client = PiperClient(config)
        self.ollama_client = OllamaClient(config)
        self.is_connected = False
    
    async def connect(self):
        """Connect to all services"""
        print("Connecting...")
        try:
            await self.whisper_client.connect()
            await self.piper_client.connect()
            await self.ollama_client.connect()
            self.is_connected = True
            print("Connected\n")
        except Exception as e:
            print(f"Connection failed: {e}")
            await self.disconnect()
            raise

    async def disconnect(self):
        """Disconnect from all services"""
        await self.whisper_client.disconnect()
        await self.piper_client.disconnect()
        await self.ollama_client.disconnect()
        self.audio_handler.close()
        self.is_connected = False
    
    async def process_turn(self) -> bool:
        """Process one conversation turn. Returns False if user wants to exit."""
        if not self.is_connected:
            raise RuntimeError("Not connected to services")

        # Record audio
        audio_data = self.audio_handler.record_audio()
        if not audio_data:
            print("No audio recorded.")
            return True

        # Transcribe
        print("📝 Transcribing...")
        try:
            user_text = await self.whisper_client.transcribe(audio_data)
            if not user_text or not user_text.strip():
                print("Could not transcribe audio.")
                return True

            print(f'You: "{user_text}"')

            # Check for exit
            if user_text.lower().strip() in ["exit", "quit", "goodbye", "bye", "stop"]:
                print("Goodbye!")
                return False

        except Exception as e:
            print(f"Transcription error: {e}")
            return True

        # Get LLM response
        print("🤔 Thinking...")
        try:
            assistant_text = await self.ollama_client.chat(user_text)
            print(f'Assistant: "{assistant_text}"')
        except Exception as e:
            print(f"LLM error: {e}")
            return True

        # Synthesize and play
        print("🗣️  Speaking...")
        try:
            audio_response = await self.piper_client.synthesize(assistant_text)
            if audio_response:
                self.audio_handler.play_audio(audio_response)
        except Exception as e:
            print(f"Speech error: {e}")

        return True
    
    async def run(self, streaming: bool = False):
        """Run the main conversation loop"""
        try:
            # Test microphone
            if not self.audio_handler.test_microphone():
                print("Please check your microphone and try again.")
                return

            # Connect to services
            await self.connect()

            # Welcome message
            mode = "Streaming" if streaming else "Standard"
            print(f"Voice Chat Active ({mode})")
            print("Say 'exit', 'quit', or 'goodbye' to end\n")

            # Main conversation loop
            while True:
                try:
                    if streaming:
                        if not await self._streaming_turn():
                            break
                    else:
                        if not await self.process_turn():
                            break
                    print("-" * 60 + "\n")

                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    print("Continuing...\n")

        finally:
            await self.disconnect()

    async def _streaming_turn(self) -> bool:
        """Process one turn with streaming LLM response"""
        # Record and transcribe
        audio_data = self.audio_handler.record_audio()
        if not audio_data:
            return True

        print("📝 Transcribing...")
        user_text = await self.whisper_client.transcribe(audio_data)
        if not user_text or not user_text.strip():
            print("Could not transcribe audio.")
            return True

        print(f'You: "{user_text}"')

        # Check for exit
        if user_text.lower().strip() in ["exit", "quit", "goodbye", "bye", "stop"]:
            print("Goodbye!")
            return False

        # Stream LLM response
        print("Assistant: ", end="", flush=True)
        full_response = ""
        async for chunk in self.ollama_client.chat_stream(user_text):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()

        # Synthesize and play
        if full_response:
            print("🗣️  Speaking...")
            audio_response = await self.piper_client.synthesize(full_response)
            if audio_response:
                self.audio_handler.play_audio(audio_response)

        return True
