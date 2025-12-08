"""
Main voice chat orchestrator
"""
import asyncio
import re
import time
from typing import Optional
from datetime import datetime
from config import JetsonConfig
from audio_handler import AudioHandler
from wyoming_client import WhisperClient, PiperClient
from ollama_client import OllamaClient
from debug_logger import DebugLogger, TurnMetrics, TimingContext


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

        # Initialize debug logger if debug mode enabled
        self.debug_logger = DebugLogger(config.debug_csv_path) if config.debug_mode else None
    
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

        # WER test mode: get reference text first
        reference_text = self._get_reference_text() if self.debug_logger else None

        # Record audio
        audio_data = self.audio_handler.record_audio()
        if not audio_data:
            print("No audio recorded.")
            return True

        # Start end-to-end timer (from sending to Whisper until AI starts speaking)
        e2e_timer = TimingContext() if self.debug_logger else None
        if e2e_timer:
            e2e_timer.__enter__()

        # Transcribe with timing
        print("📝 Transcribing...")
        transcription_timer = TimingContext() if self.debug_logger else None
        try:
            if transcription_timer:
                transcription_timer.__enter__()

            user_text = await self.whisper_client.transcribe(audio_data)

            if transcription_timer:
                transcription_timer.__exit__(None, None, None)

            if not user_text or not user_text.strip():
                print("Could not transcribe audio.")
                if e2e_timer:
                    e2e_timer.__exit__(None, None, None)
                return True

            print(f'You: "{user_text}"')

            # Check for exit
            if user_text.lower().strip() in ["exit", "quit", "goodbye", "bye", "stop"]:
                print("Goodbye!")
                if e2e_timer:
                    e2e_timer.__exit__(None, None, None)
                return False

        except Exception as e:
            print(f"Transcription error: {e}")
            if transcription_timer:
                transcription_timer.__exit__(None, None, None)
            if e2e_timer:
                e2e_timer.__exit__(None, None, None)
            return True

        # Get LLM response with timing
        print("🤔 Thinking...")
        llm_timer = TimingContext() if self.debug_logger else None
        try:
            if llm_timer:
                llm_timer.__enter__()

            assistant_text = await self.ollama_client.chat(user_text)

            if llm_timer:
                llm_timer.__exit__(None, None, None)

            print(f'Assistant: "{assistant_text}"')
        except Exception as e:
            print(f"LLM error: {e}")
            if llm_timer:
                llm_timer.__exit__(None, None, None)
            if e2e_timer:
                e2e_timer.__exit__(None, None, None)
            return True

        # Synthesize and play (use first Piper client for non-streaming mode)
        print("🗣️  Speaking...")
        try:
            audio_response = await self.piper_clients[0].synthesize(assistant_text)
            if audio_response:
                self.audio_handler.play_audio(audio_response)

        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            # End-to-end timer ends when AI starts speaking (or tries to)
            if e2e_timer:
                e2e_timer.__exit__(None, None, None)

        # Log debug metrics if enabled
        if self.debug_logger:
            try:
                metrics = TurnMetrics(
                    timestamp=datetime.now().isoformat(),
                    reference_text=reference_text,
                    reference_tokens=self.debug_logger.count_tokens(reference_text) if reference_text else None,
                    transcribed_text=user_text,
                    transcription_tokens=self.debug_logger.count_tokens(user_text),
                    wer=self.debug_logger.calculate_wer(reference_text, user_text) if reference_text else None,
                    transcription_latency_ms=transcription_timer.get_elapsed_ms(),
                    llm_response=assistant_text,
                    llm_response_tokens=self.debug_logger.count_tokens(assistant_text),
                    end_to_end_latency_ms=e2e_timer.get_elapsed_ms()
                )
                self.debug_logger.log_turn(metrics)
                print(f"\n[DEBUG] Metrics logged to {self.config.debug_csv_path}")
            except Exception as e:
                print(f"\n[DEBUG] Failed to log metrics: {e}")

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

    def _get_reference_text(self) -> Optional[str]:
        """
        Get reference text for WER testing in interactive mode.

        Returns:
            Reference text or None if user wants to skip
        """
        if not self.config.debug_wer_test_mode:
            return None

        print("\n" + "="*60)
        print("WER TEST MODE")
        print("="*60)
        print("Enter the reference text you will read into the microphone.")
        print("This text will stay visible as a teleprompter.")
        print("Press ENTER twice to skip WER testing for this turn.\n")

        reference = input("Reference text: ").strip()

        if not reference:
            print("Skipping WER test for this turn.\n")
            return None

        print("\n" + "-"*60)
        print("TELEPROMPTER - Read this text:")
        print("-"*60)
        print(f"\n{reference}\n")
        print("-"*60)
        print("Press ENTER when ready to start recording...")
        input()

        return reference

    async def _streaming_turn(self) -> bool:
        """Process one turn with streaming LLM response and concurrent TTS"""

        # WER test mode: get reference text first
        reference_text = self._get_reference_text() if self.debug_logger else None

        # Record and transcribe
        audio_data = self.audio_handler.record_audio()
        if not audio_data:
            return True

        # Start end-to-end timer (from sending to Whisper until AI starts speaking)
        e2e_timer = TimingContext() if self.debug_logger else None
        if e2e_timer:
            e2e_timer.__enter__()

        print("📝 Transcribing...")
        transcription_timer = TimingContext() if self.debug_logger else None
        if transcription_timer:
            transcription_timer.__enter__()

        user_text = await self.whisper_client.transcribe(audio_data)

        if transcription_timer:
            transcription_timer.__exit__(None, None, None)

        if not user_text or not user_text.strip():
            print("Could not transcribe audio.")
            if e2e_timer:
                e2e_timer.__exit__(None, None, None)
            return True

        print(f'You: "{user_text}"')

        # Check for exit
        if user_text.lower().strip() in ["exit", "quit", "goodbye", "bye", "stop"]:
            print("Goodbye!")
            if e2e_timer:
                e2e_timer.__exit__(None, None, None)
            return False

        # Stream LLM response with concurrent TTS processing
        print("Assistant: ", end="", flush=True)

        # Start LLM timer
        llm_timer = TimingContext() if self.debug_logger else None
        if llm_timer:
            llm_timer.__enter__()

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
            """Print speaking message when first audio is ready and end e2e timer"""
            nonlocal speaking_started
            # Wait for first audio chunk
            while audio_queue.empty():
                await asyncio.sleep(0.01)
            if not speaking_started:
                # End-to-end timer ends when first audio is ready to play
                if e2e_timer:
                    e2e_timer.__exit__(None, None, None)
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
        full_response = ""  # Collect full LLM response for debug logging

        try:
            async for chunk in self.ollama_client.chat_stream(user_text):
                print(chunk, end="", flush=True)
                buffer += chunk
                full_response += chunk  # Accumulate for debug logging

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

            # End LLM timer when streaming completes
            if llm_timer:
                llm_timer.__exit__(None, None, None)

            print()

            # Process any remaining text in buffer
            if buffer.strip():
                detected_count[0] += 1
                if self.config.debug_pipeline:
                    print(f"\n[LLM] Final fragment (sentence {detected_count[0]}), sending to TTS")
                await sentence_queue.put(buffer.strip())
                full_response += buffer  # Add final fragment

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

            # Ensure e2e timer is exited (in case monitor task was cancelled before exiting it)
            if e2e_timer and e2e_timer.elapsed_ms is None:
                e2e_timer.__exit__(None, None, None)

            # Log debug metrics if enabled
            if self.debug_logger:
                try:
                    metrics = TurnMetrics(
                        timestamp=datetime.now().isoformat(),
                        reference_text=reference_text,
                        reference_tokens=self.debug_logger.count_tokens(reference_text) if reference_text else None,
                        transcribed_text=user_text,
                        transcription_tokens=self.debug_logger.count_tokens(user_text),
                        wer=self.debug_logger.calculate_wer(reference_text, user_text) if reference_text else None,
                        transcription_latency_ms=transcription_timer.get_elapsed_ms() if transcription_timer else 0,
                        llm_response=full_response,
                        llm_response_tokens=self.debug_logger.count_tokens(full_response),
                        end_to_end_latency_ms=e2e_timer.get_elapsed_ms() if e2e_timer else None
                    )
                    self.debug_logger.log_turn(metrics)
                    print(f"\n[DEBUG] Metrics logged to {self.config.debug_csv_path}")
                except Exception as e:
                    print(f"\n[DEBUG] Failed to log metrics: {e}")

        return True
