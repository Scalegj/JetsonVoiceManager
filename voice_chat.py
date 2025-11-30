"""
Main voice chat orchestrator
"""
import asyncio
import re
import time
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
        # Create multiple Piper clients for concurrent synthesis
        self.piper_clients = [PiperClient(config) for _ in range(config.num_tts_clients)]
        self.ollama_client = OllamaClient(config)
        self.is_connected = False
    
    async def connect(self):
        """Connect to all services"""
        print("Connecting...")
        try:
            await self.whisper_client.connect()
            # Connect all Piper clients
            for i, piper_client in enumerate(self.piper_clients):
                await piper_client.connect()
                if self.config.debug_pipeline:
                    print(f"  TTS client {i+1}/{len(self.piper_clients)} connected")
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
        # Disconnect all Piper clients
        for piper_client in self.piper_clients:
            await piper_client.disconnect()
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

        # Synthesize and play (use first Piper client for non-streaming mode)
        print("🗣️  Speaking...")
        try:
            audio_response = await self.piper_clients[0].synthesize(assistant_text)
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
        """Process one turn with streaming LLM response and concurrent TTS"""
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

        # Stream LLM response with concurrent TTS processing
        print("Assistant: ", end="", flush=True)

        # Queue for audio chunks ready to play
        audio_queue = asyncio.Queue()

        # Task to handle TTS synthesis with concurrent client pool
        async def tts_worker():
            """Process sentences for TTS using multiple concurrent clients"""
            sentence_queue = asyncio.Queue()
            sentence_count = [0]  # Use list for mutable counter

            async def synthesize_one(text, order_id, client_id):
                """Synthesize one sentence using a specific Piper client"""
                try:
                    if self.config.debug_pipeline:
                        print(f"\n[TTS-{client_id}] Starting synthesis of sentence {order_id}...")
                    start_time = time.time()

                    # Use assigned Piper client
                    audio_data = await self.piper_clients[client_id].synthesize(text)

                    elapsed = time.time() - start_time
                    if self.config.debug_pipeline:
                        print(f"[TTS-{client_id}] Sentence {order_id} synthesized in {elapsed:.2f}s")

                    return (order_id, audio_data)
                except Exception as e:
                    print(f"\n⚠️  TTS-{client_id} error: {e}")
                    return (order_id, None)

            async def synthesis_coordinator():
                """Coordinate concurrent synthesis across multiple clients"""
                active_tasks = {}  # {task: (order_id, client_id)}
                available_clients = list(range(len(self.piper_clients)))
                next_order_id = 0
                next_to_play = 0
                pending_audio = {}  # Buffer for out-of-order completions

                while True:
                    # Get new sentences when we have available clients
                    while available_clients:
                        try:
                            sentence = await asyncio.wait_for(
                                sentence_queue.get(),
                                timeout=0.01
                            )

                            if sentence is None:  # Sentinel to stop
                                # Wait for all active tasks to complete
                                if active_tasks:
                                    done_tasks = await asyncio.gather(*active_tasks.keys())
                                    for order_id, audio_data in done_tasks:
                                        pending_audio[order_id] = audio_data

                                # Drain pending audio in order
                                while next_to_play in pending_audio:
                                    audio_data = pending_audio.pop(next_to_play)
                                    if audio_data:
                                        await audio_queue.put(audio_data)
                                    next_to_play += 1

                                await audio_queue.put(None)  # Signal playback done
                                return

                            # Assign to available client
                            client_id = available_clients.pop(0)
                            task = asyncio.create_task(
                                synthesize_one(sentence, next_order_id, client_id)
                            )
                            active_tasks[task] = (next_order_id, client_id)
                            next_order_id += 1

                        except asyncio.TimeoutError:
                            break  # No new sentences yet

                    # Check for completed tasks
                    if active_tasks:
                        done, pending = await asyncio.wait(
                            active_tasks.keys(),
                            timeout=0.01,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        for task in done:
                            order_id, audio_data = await task
                            _, client_id = active_tasks.pop(task)
                            available_clients.append(client_id)  # Return client to pool

                            pending_audio[order_id] = audio_data

                            # Send audio in order
                            while next_to_play in pending_audio:
                                audio_data = pending_audio.pop(next_to_play)
                                if audio_data:
                                    await audio_queue.put(audio_data)
                                next_to_play += 1

                    # Small yield to prevent busy waiting
                    await asyncio.sleep(0.001)

            return sentence_queue, asyncio.create_task(synthesis_coordinator())


        # Start workers
        sentence_queue, tts_task = await tts_worker()

        # Start continuous audio player (no gaps between sentences)
        speaking_started = False
        playback_count = [0]  # Track playback progress

        async def monitor_audio_start():
            """Print speaking message when first audio is ready"""
            nonlocal speaking_started
            # Wait for first audio chunk
            while audio_queue.empty():
                await asyncio.sleep(0.01)
            if not speaking_started:
                print("\n🗣️  Speaking...")
                speaking_started = True

        async def audio_player_with_timing():
            """Play audio with timing information"""
            await self.audio_handler.play_audio_stream(audio_queue, debug=self.config.debug_pipeline)

        monitor_task = asyncio.create_task(monitor_audio_start())
        player_task = asyncio.create_task(audio_player_with_timing())

        # Stream LLM and extract sentences
        buffer = ""
        # Break only on sentence endings (.!?) for natural prosody
        sentence_pattern = re.compile(r'([^.!?]*[.!?]+)')
        detected_count = [0]

        try:
            async for chunk in self.ollama_client.chat_stream(user_text):
                print(chunk, end="", flush=True)
                buffer += chunk

                # Check for complete sentences
                while True:
                    match = sentence_pattern.search(buffer)
                    if not match:
                        break

                    sentence = match.group(1).strip()
                    if sentence:
                        detected_count[0] += 1
                        if self.config.debug_pipeline:
                            print(f"\n[LLM] Sentence {detected_count[0]} complete, sending to TTS")
                        await sentence_queue.put(sentence)
                        if self.config.debug_pipeline:
                            print("Assistant: ", end="", flush=True)  # Resume printing

                    # Remove processed sentence from buffer (even if empty)
                    buffer = buffer[match.end():].lstrip()

            print()

            # Process any remaining text in buffer
            if buffer.strip():
                detected_count[0] += 1
                if self.config.debug_pipeline:
                    print(f"\n[LLM] Final fragment (sentence {detected_count[0]}), sending to TTS")
                await sentence_queue.put(buffer.strip())

        finally:
            # Signal completion to workers
            await sentence_queue.put(None)

            # Wait for all synthesis and playback to complete
            await tts_task
            await player_task

            # Cancel monitor task if still running
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

        return True
