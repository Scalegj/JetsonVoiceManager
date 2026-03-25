"""Voice chat orchestrator"""
import asyncio
import re
import time
from datetime import datetime
from typing import Optional

from jetson_voice.audio_handler import AudioHandler
from jetson_voice.config.models import AppConfig
from jetson_voice.debug_logger import DebugLogger, TimingContext, TurnMetrics
from jetson_voice.services import WhisperClient, PiperClient, create_llm_client


class VoiceChat:
    """Orchestrates the voice chat conversation pipeline"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.audio_handler = AudioHandler(config)
        self.whisper_client = WhisperClient(config)
        self.piper_clients = [PiperClient(config) for _ in range(config.num_tts_clients)]
        self.llm_client = create_llm_client(config)
        self.is_connected = False
        self.debug_logger = DebugLogger(config.debug_csv_path) if config.debug_mode else None

    async def connect(self):
        print("Connecting...")
        try:
            await self.whisper_client.connect()
            for piper in self.piper_clients:
                await piper.connect()
            await self.llm_client.connect()
            self.is_connected = True
            print("Connected\n")
        except Exception as e:
            print(f"Connection failed: {e}")
            await self.disconnect()
            raise

    async def disconnect(self):
        await self.whisper_client.disconnect()
        for piper in self.piper_clients:
            await piper.disconnect()
        await self.llm_client.disconnect()
        self.audio_handler.close()
        self.is_connected = False

    async def process_turn(self) -> bool:
        """Standard (non-streaming) conversation turn. Returns False to exit."""
        if not self.is_connected:
            raise RuntimeError("Not connected to services")

        reference_text = self._get_reference_text() if self.debug_logger else None

        audio_data = self.audio_handler.record_audio()
        if not audio_data:
            print("No audio recorded.")
            return True

        e2e_timer = TimingContext() if self.debug_logger else None
        if e2e_timer:
            e2e_timer.__enter__()

        # --- Transcribe ---
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

            if user_text.lower().strip() in {"exit", "quit", "goodbye", "bye", "stop"}:
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

        # --- LLM ---
        print("🤔 Thinking...")
        llm_timer = TimingContext() if self.debug_logger else None
        try:
            if llm_timer:
                llm_timer.__enter__()
            assistant_text = await self.llm_client.chat(user_text)
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

        # --- TTS + Playback ---
        print("🗣️  Speaking...")
        try:
            audio_response = await self.piper_clients[0].synthesize(assistant_text)
            if audio_response:
                self.audio_handler.play_audio(audio_response)
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            if e2e_timer:
                e2e_timer.__exit__(None, None, None)

        self._log_turn(
            reference_text=reference_text,
            user_text=user_text,
            assistant_text=assistant_text,
            transcription_timer=transcription_timer,
            e2e_timer=e2e_timer,
        )
        return True

    async def run(self, streaming: bool = False):
        """Main conversation loop"""
        try:
            if not self.audio_handler.test_microphone():
                print("Please check your microphone and try again.")
                return

            await self.connect()
            mode = "Streaming" if streaming else "Standard"
            print(f"Voice Chat Active ({mode})")
            print("Say 'exit', 'quit', or 'goodbye' to end\n")

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

    async def process_audio_bytes(self, audio_bytes: bytes) -> bytes:
        """
        Process raw PCM audio and return synthesized WAV response.
        Used by the WebRTC gateway.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to services")

        user_text = await self.whisper_client.transcribe(audio_bytes)
        if not user_text or not user_text.strip():
            return b""

        assistant_text = await self.llm_client.chat(user_text)
        if not assistant_text:
            return b""

        return await self.piper_clients[0].synthesize(assistant_text)

    def _get_reference_text(self) -> Optional[str]:
        if not self.config.debug_wer_test_mode:
            return None
        print("\n" + "=" * 60)
        print("WER TEST MODE")
        print("=" * 60)
        print("Enter reference text (press ENTER twice to skip):\n")
        reference = input("Reference text: ").strip()
        if not reference:
            print("Skipping WER test.\n")
            return None
        print(f"\n--- TELEPROMPTER ---\n{reference}\n--------------------")
        print("Press ENTER when ready...")
        input()
        return reference

    def _log_turn(self, reference_text, user_text, assistant_text, transcription_timer, e2e_timer):
        if not self.debug_logger:
            return
        try:
            dl = self.debug_logger
            metrics = TurnMetrics(
                timestamp=datetime.now().isoformat(),
                reference_text=reference_text,
                reference_tokens=dl.count_tokens(reference_text) if reference_text else None,
                transcribed_text=user_text,
                transcription_tokens=dl.count_tokens(user_text),
                wer=dl.calculate_wer(reference_text, user_text) if reference_text else None,
                transcription_latency_ms=transcription_timer.get_elapsed_ms() if transcription_timer else 0,
                llm_response=assistant_text,
                llm_response_tokens=dl.count_tokens(assistant_text),
                end_to_end_latency_ms=e2e_timer.get_elapsed_ms() if e2e_timer else None,
            )
            dl.log_turn(metrics)
            print(f"\n[DEBUG] Metrics logged to {self.config.debug_csv_path}")
        except Exception as e:
            print(f"\n[DEBUG] Failed to log metrics: {e}")

    async def _streaming_turn(self) -> bool:
        """Streaming turn: concurrent LLM streaming + TTS pipeline"""
        reference_text = self._get_reference_text() if self.debug_logger else None

        audio_data = self.audio_handler.record_audio()
        if not audio_data:
            return True

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

        if user_text.lower().strip() in {"exit", "quit", "goodbye", "bye", "stop"}:
            print("Goodbye!")
            if e2e_timer:
                e2e_timer.__exit__(None, None, None)
            return False

        print("Assistant: ", end="", flush=True)

        llm_timer = TimingContext() if self.debug_logger else None
        if llm_timer:
            llm_timer.__enter__()

        audio_queue = asyncio.Queue()

        async def tts_worker():
            """Concurrent TTS synthesis across the Piper client pool"""
            sentence_queue = asyncio.Queue()

            async def synthesize_one(text, order_id, client_id):
                try:
                    if self.config.debug_pipeline:
                        print(f"\n[TTS-{client_id}] sentence {order_id}...")
                    t = time.time()
                    data = await self.piper_clients[client_id].synthesize(text)
                    if self.config.debug_pipeline:
                        print(f"[TTS-{client_id}] sentence {order_id} done in {time.time()-t:.2f}s")
                    return (order_id, data)
                except Exception as e:
                    print(f"\n⚠️  TTS-{client_id} error: {e}")
                    return (order_id, None)

            async def coordinator():
                active_tasks = {}
                available_clients = list(range(len(self.piper_clients)))
                next_order_id = 0
                next_to_play = 0
                pending_audio = {}

                while True:
                    while available_clients:
                        try:
                            sentence = await asyncio.wait_for(sentence_queue.get(), timeout=0.01)
                            if sentence is None:
                                if active_tasks:
                                    done_results = await asyncio.gather(*active_tasks.keys())
                                    for oid, adata in done_results:
                                        pending_audio[oid] = adata
                                while next_to_play in pending_audio:
                                    data = pending_audio.pop(next_to_play)
                                    if data:
                                        await audio_queue.put(data)
                                    next_to_play += 1
                                await audio_queue.put(None)
                                return
                            client_id = available_clients.pop(0)
                            task = asyncio.create_task(synthesize_one(sentence, next_order_id, client_id))
                            active_tasks[task] = (next_order_id, client_id)
                            next_order_id += 1
                        except asyncio.TimeoutError:
                            break

                    if active_tasks:
                        done, _ = await asyncio.wait(active_tasks.keys(), timeout=0.01, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            oid, adata = await task
                            _, client_id = active_tasks.pop(task)
                            available_clients.append(client_id)
                            pending_audio[oid] = adata
                            while next_to_play in pending_audio:
                                data = pending_audio.pop(next_to_play)
                                if data:
                                    await audio_queue.put(data)
                                next_to_play += 1

                    await asyncio.sleep(0.001)

            return sentence_queue, asyncio.create_task(coordinator())

        sentence_queue, tts_task = await tts_worker()

        speaking_started = False

        async def monitor_start():
            nonlocal speaking_started
            while audio_queue.empty():
                await asyncio.sleep(0.01)
            if not speaking_started:
                if e2e_timer:
                    e2e_timer.__exit__(None, None, None)
                print("\n🗣️  Speaking...")
                speaking_started = True

        monitor_task = asyncio.create_task(monitor_start())
        player_task = asyncio.create_task(
            self.audio_handler.play_audio_stream(audio_queue, debug=self.config.debug_pipeline)
        )

        buffer = ""
        sentence_pattern = re.compile(r"([^.!?]*[.!?]+)")
        full_response = ""

        try:
            async for chunk in self.llm_client.chat_stream(user_text):
                print(chunk, end="", flush=True)
                buffer += chunk
                full_response += chunk

                while True:
                    match = sentence_pattern.search(buffer)
                    if not match:
                        break
                    sentence = match.group(1).strip()
                    if sentence:
                        await sentence_queue.put(sentence)
                    buffer = buffer[match.end():].lstrip()

            if llm_timer:
                llm_timer.__exit__(None, None, None)

            print()

            if buffer.strip():
                await sentence_queue.put(buffer.strip())
                full_response += buffer

        finally:
            await sentence_queue.put(None)
            await tts_task
            await player_task

            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            if e2e_timer and e2e_timer.elapsed_ms is None:
                e2e_timer.__exit__(None, None, None)

            self._log_turn(
                reference_text=reference_text,
                user_text=user_text,
                assistant_text=full_response,
                transcription_timer=transcription_timer,
                e2e_timer=e2e_timer,
            )

        return True
