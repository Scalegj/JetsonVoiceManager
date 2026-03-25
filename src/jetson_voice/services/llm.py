"""Ollama LLM client"""
import json
import aiohttp
from typing import List, Dict, AsyncIterator
from jetson_voice.config.models import AppConfig


class OllamaClient:
    """Client for Ollama LLM service (/api/chat endpoint)"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.session: aiohttp.ClientSession = None
        self.conversation_history: List[Dict[str, str]] = []

    async def connect(self):
        self.session = aiohttp.ClientSession()

    async def disconnect(self):
        if self.session:
            await self.session.close()

    def reset_conversation(self):
        self.conversation_history = []

    def _build_messages(self) -> list:
        return [{"role": "system", "content": self.config.system_prompt}] + self.conversation_history

    async def chat(self, user_message: str) -> str:
        if not self.session:
            raise RuntimeError("Not connected to Ollama service")

        self.conversation_history.append({"role": "user", "content": user_message})

        try:
            async with self.session.post(
                f"{self.config.ollama_base_url}/api/chat",
                json={
                    "model": self.config.ollama_model,
                    "messages": self._build_messages(),
                    "stream": False,
                },
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama error {resp.status}: {await resp.text()}")
                result = await resp.json()
                reply = result["message"]["content"]
                self.conversation_history.append({"role": "assistant", "content": reply})
                return reply

        except Exception:
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            raise

    async def chat_stream(self, user_message: str) -> AsyncIterator[str]:
        if not self.session:
            raise RuntimeError("Not connected to Ollama service")

        self.conversation_history.append({"role": "user", "content": user_message})
        full_response = ""

        try:
            async with self.session.post(
                f"{self.config.ollama_base_url}/api/chat",
                json={
                    "model": self.config.ollama_model,
                    "messages": self._build_messages(),
                    "stream": True,
                },
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama error {resp.status}: {await resp.text()}")

                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                chunk = data["message"]["content"]
                                full_response += chunk
                                yield chunk
                        except json.JSONDecodeError:
                            continue

            self.conversation_history.append({"role": "assistant", "content": full_response})

        except Exception:
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            raise
