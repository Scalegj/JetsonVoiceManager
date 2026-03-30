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
        self.turn_number = 0  # 1-based counter incremented at the start of each turn

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
        """Process one conversation turn (non-streaming). Returns False if user wants to exit."""
        if not self.is_connected:
            raise RuntimeError("Not connected to services")

        self.turn_number += 1

        # WER test mode: get reference text first
        reference_text = self._get_reference_text() if self.debug_logger else None

        # Record audio
        audio_data = self.audio_handler.record_audio()
        if not audio_data:
            print("No audio recorded.")
            return True

        # Compute input audio duration from raw PCM bytes
        audio_input_duration_ms = DebugLogger.pcm_duration_ms(
            audio_data, self.config.sample_rate, self.config.channels
        ) if self.debug_logger else 0.0

        # TTFA timer: from end-of-recording until the first audio chunk is ready for playback.
        # This is the latency the user perceives as "waiting for a response".
        # Covers: transcription + LLM inference + TTS synthesis (all sequential in non-streaming mode).
        ttfa_start = time.perf_counter() if self.debug_logger else None

        # --- Transcription ---
        print("📝 Transcribing...")
        transcription_timer = TimingContext()
        try:
            with transcription_timer:
                user_text = await self.whisper_client.transcribe(audio_data)

            if not user_text or not user_text.strip():
                print("Could not transcribe audio.")
                return True

            print(f'You: "{user_text}"')

            if user_text.lower().strip() in ["exit", "quit", "goodbye", "bye", "stop"]:
                print("Goodbye!")
                return False

        except Exception as e:
            print(f"Transcription error: {e}")
            return True

        # --- LLM inference ---
        print("🤔 Thinking...")
        llm_timer = TimingContext()
        try:
            with llm_timer:
                assistant_text = await self.ollama_client.chat(user_text)
            print(f'Assistant: "{assistant_text}"')
        except Exception as e:
            print(f"LLM error: {e}")
            return True

        # --- TTS synthesis ---
        print("🗣️  Speaking...")
        tts_timer = TimingContext()
        audio_response = None
        time_to_first_audio_ms = None
        try:
            with tts_timer:
                audio_response = await self.piper_clients[0].synthesize(assistant_text)

            # TTFA ends here: synthesis is done, audio is ready.
            # We stop the clock BEFORE play_audio() so we measure latency, not playback duration.
            if ttfa_start is not None:
                time_to_first_audio_ms = (time.perf_counter() - ttfa_start) * 1000.0

            if audio_response:
                self.audio_handler.play_audio(audio_response)

        except Exception as e:
            print(f"Speech error: {e}")
            if ttfa_start is not None and time_to_first_audio_ms is None:
                time_to_first_audio_ms = (time.perf_counter() - ttfa_start) * 1000.0

        # --- Log debug metrics ---
        if self.debug_logger:
            try:
                llm_latency_ms = llm_timer.get_elapsed_ms() or 0.0
                # Use Ollama's server-reported token counts — exact for the model in use
                ollama_stats = self.ollama_client.last_stats
                llm_response_tokens = ollama_stats.get('eval_count')
                llm_prompt_tokens = ollama_stats.get('prompt_eval_count')
                # tokens/sec from server-side timing (eval_duration is nanoseconds)
                eval_ns = ollama_stats.get('eval_duration_ns')
                llm_tokens_per_second = (
                    llm_response_tokens / (eval_ns / 1e9)
                    if llm_response_tokens and eval_ns and eval_ns > 0
                    else None
                )

                metrics = TurnMetrics(
                    turn_number=self.turn_number,
                    timestamp=datetime.now().isoformat(),
                    audio_input_duration_ms=audio_input_duration_ms,
                    transcribed_text=user_text,
                    transcription_tokens=self.debug_logger.count_tokens(user_text),
                    transcription_latency_ms=transcription_timer.get_elapsed_ms() or 0.0,
                    reference_text=reference_text,
                    reference_tokens=self.debug_logger.count_tokens(reference_text) if reference_text else None,
                    wer=self.debug_logger.calculate_wer(reference_text, user_text) if reference_text else None,
                    llm_latency_ms=llm_latency_ms,
                    llm_time_to_first_token_ms=None,  # N/A in non-streaming mode
                    llm_prompt_tokens=llm_prompt_tokens,
                    llm_response=assistant_text,
                    llm_response_tokens=llm_response_tokens,
                    llm_response_chars=len(assistant_text),
                    llm_tokens_per_second=llm_tokens_per_second,
                    tts_latency_ms=tts_timer.get_elapsed_ms(),
                    response_audio_duration_ms=DebugLogger.wav_duration_ms(audio_response) if audio_response else None,
                    time_to_first_audio_ms=time_to_first_audio_ms,
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
        """Process one turn with streaming LLM response and concurrent TTS."""

        self.turn_number += 1

        # WER test mode: get reference text first
        reference_text = self._get_reference_text() if self.debug_logger else None

        # Record and transcribe
        audio_data = self.audio_handler.record_audio()
        if not audio_data:
            return True

        # Compute input audio duration
        audio_input_duration_ms = DebugLogger.pcm_duration_ms(
            audio_data, self.config.sample_rate, self.config.channels
        ) if self.debug_logger else 0.0

        # TTFA timer: from end-of-recording to first audio chunk ready.
        # In streaming mode, LLM and TTS overlap, so TTFA < llm_latency + tts_latency.
        e2e_start = time.perf_counter() if self.debug_logger else None
        first_audio_time_ref = [None]  # set inside synthesis_coordinator on first audio put

        # --- Transcription ---
        print("📝 Transcribing...")
        transcription_timer = TimingContext() if self.debug_logger else None
        if transcription_timer:
            transcription_timer.__enter__()

        user_text = await self.whisper_client.transcribe(audio_data)

        if transcription_timer:
            transcription_timer.__exit__(None, None, None)

        if not user_text or not user_text.strip():
            print("Could not transcribe audio.")
            return True

        print(f'You: "{user_text}"')

        if user_text.lower().strip() in ["exit", "quit", "goodbye", "bye", "stop"]:
            print("Goodbye!")
            return False

        # --- Streaming LLM + concurrent TTS ---
        print("Assistant: ", end="", flush=True)

        # Shared mutable state for timing across closures
        tts_start_ref = [None]          # perf_counter when first sentence is sent to TTS
        audio_duration_ref = [0.0]      # accumulates total synthesized audio duration (ms)
        llm_ttft_ref = [None]           # ms from LLM request to first token received

        # LLM timer starts here (covers full streaming duration)
        llm_timer = TimingContext() if self.debug_logger else None
        if llm_timer:
            llm_timer.__enter__()
        llm_request_start = time.perf_counter()

        # Queue for ordered audio chunks ready to play
        audio_queue = asyncio.Queue()

        # --- TTS worker with concurrent client pool ---
        async def tts_worker():
            """Process sentences for TTS using multiple concurrent clients"""
            sentence_queue = asyncio.Queue()
            sentence_count = [0]

            async def synthesize_one(text, order_id, client_id):
                """Synthesize one sentence using a specific Piper client"""
                try:
                    if self.config.debug_pipeline:
                        print(f"\n[TTS-{client_id}] Starting synthesis of sentence {order_id}...")
                    start_time = time.perf_counter()

                    audio = await self.piper_clients[client_id].synthesize(text)

                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    if self.config.debug_pipeline:
                        print(f"[TTS-{client_id}] Sentence {order_id} synthesized in {elapsed_ms:.0f}ms")

                    # Accumulate total response audio duration
                    if audio and self.debug_logger:
                        dur = DebugLogger.wav_duration_ms(audio)
                        if dur:
                            audio_duration_ref[0] += dur

                    return (order_id, audio)
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
                                    for order_id, audio in done_tasks:
                                        pending_audio[order_id] = audio

                                # Drain pending audio in order
                                while next_to_play in pending_audio:
                                    audio = pending_audio.pop(next_to_play)
                                    if audio:
                                        if first_audio_time_ref[0] is None and e2e_start is not None:
                                            first_audio_time_ref[0] = time.perf_counter()
                                        await audio_queue.put(audio)
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
                            order_id, audio = await task
                            _, client_id = active_tasks.pop(task)
                            available_clients.append(client_id)  # Return client to pool

                            pending_audio[order_id] = audio

                            # Send audio in order
                            while next_to_play in pending_audio:
                                audio = pending_audio.pop(next_to_play)
                                if audio:
                                    if first_audio_time_ref[0] is None and e2e_start is not None:
                                        first_audio_time_ref[0] = time.perf_counter()
                                    await audio_queue.put(audio)
                                next_to_play += 1

                    await asyncio.sleep(0.001)  # Yield to prevent busy-wait

            return sentence_queue, asyncio.create_task(synthesis_coordinator())

        # Start TTS workers
        sentence_queue, tts_task = await tts_worker()

        async def audio_player():
            await self.audio_handler.play_audio_stream(audio_queue, debug=self.config.debug_pipeline)

        player_task = asyncio.create_task(audio_player())

        # --- Stream LLM response and dispatch sentences to TTS ---
        buffer = ""
        sentence_pattern = re.compile(r'([^.!?]*[.!?]+)')
        detected_count = [0]
        full_response = ""

        try:
            async for chunk in self.ollama_client.chat_stream(user_text):
                # Capture time-to-first-token on first chunk
                if self.debug_logger and llm_ttft_ref[0] is None:
                    llm_ttft_ref[0] = (time.perf_counter() - llm_request_start) * 1000.0

                print(chunk, end="", flush=True)
                buffer += chunk
                full_response += chunk

                # Dispatch complete sentences to TTS
                while True:
                    match = sentence_pattern.search(buffer)
                    if not match:
                        break

                    sentence = match.group(1).strip()
                    if sentence:
                        detected_count[0] += 1
                        # Record when TTS starts receiving its first sentence
                        if self.debug_logger and tts_start_ref[0] is None:
                            tts_start_ref[0] = time.perf_counter()
                        if self.config.debug_pipeline:
                            print(f"\n[LLM] Sentence {detected_count[0]} complete, sending to TTS")
                        await sentence_queue.put(sentence)
                        if self.config.debug_pipeline:
                            print("Assistant: ", end="", flush=True)

                    buffer = buffer[match.end():].lstrip()

            # LLM streaming complete
            if llm_timer:
                llm_timer.__exit__(None, None, None)

            print()

            # Dispatch any remaining text fragment
            if buffer.strip():
                detected_count[0] += 1
                if self.debug_logger and tts_start_ref[0] is None:
                    tts_start_ref[0] = time.perf_counter()
                if self.config.debug_pipeline:
                    print(f"\n[LLM] Final fragment (sentence {detected_count[0]}), sending to TTS")
                await sentence_queue.put(buffer.strip())
                full_response += buffer

        finally:
            # Signal TTS workers to finish up
            await sentence_queue.put(None)

            # Wait for all synthesis to complete, then record TTS end time
            await tts_task
            tts_end_time = time.perf_counter()

            # Wait for playback to finish
            await player_task

            # Ensure llm timer is closed (in case of exception before the loop ended)
            if llm_timer and llm_timer.elapsed_ms is None:
                llm_timer.__exit__(None, None, None)

            # --- Log debug metrics ---
            if self.debug_logger:
                try:
                    llm_latency_ms = llm_timer.get_elapsed_ms() or 0.0
                    # Use Ollama's server-reported token counts — exact for the model in use
                    ollama_stats = self.ollama_client.last_stats
                    llm_response_tokens = ollama_stats.get('eval_count')
                    llm_prompt_tokens = ollama_stats.get('prompt_eval_count')
                    eval_ns = ollama_stats.get('eval_duration_ns')
                    llm_tokens_per_second = (
                        llm_response_tokens / (eval_ns / 1e9)
                        if llm_response_tokens and eval_ns and eval_ns > 0
                        else None
                    )

                    tts_latency_ms = (
                        (tts_end_time - tts_start_ref[0]) * 1000.0
                        if tts_start_ref[0] is not None else None
                    )

                    metrics = TurnMetrics(
                        turn_number=self.turn_number,
                        timestamp=datetime.now().isoformat(),
                        audio_input_duration_ms=audio_input_duration_ms,
                        transcribed_text=user_text,
                        transcription_tokens=self.debug_logger.count_tokens(user_text),
                        transcription_latency_ms=transcription_timer.get_elapsed_ms() if transcription_timer else 0.0,
                        reference_text=reference_text,
                        reference_tokens=self.debug_logger.count_tokens(reference_text) if reference_text else None,
                        wer=self.debug_logger.calculate_wer(reference_text, user_text) if reference_text else None,
                        llm_latency_ms=llm_latency_ms,
                        llm_time_to_first_token_ms=llm_ttft_ref[0],
                        llm_prompt_tokens=llm_prompt_tokens,
                        llm_response=full_response,
                        llm_response_tokens=llm_response_tokens,
                        llm_response_chars=len(full_response),
                        llm_tokens_per_second=llm_tokens_per_second,
                        tts_latency_ms=tts_latency_ms,
                        response_audio_duration_ms=audio_duration_ref[0] if audio_duration_ref[0] > 0 else None,
                        time_to_first_audio_ms=(
                            (first_audio_time_ref[0] - e2e_start) * 1000.0
                            if first_audio_time_ref[0] is not None and e2e_start is not None else None
                        ),
                    )
                    self.debug_logger.log_turn(metrics)
                    print(f"\n[DEBUG] Metrics logged to {self.config.debug_csv_path}")
                except Exception as e:
                    print(f"\n[DEBUG] Failed to log metrics: {e}")

        return True