"""Ollama LLM client"""
import aiohttp
from typing import List, Dict, AsyncIterator
from config import JetsonConfig


class OllamaClient:
    """Client for Ollama LLM service"""
    
    def __init__(self, config: JetsonConfig):
        self.config = config
        self.session: aiohttp.ClientSession = None
        self.conversation_history: List[Dict[str, str]] = []
        # Populated after every chat() or chat_stream() call with Ollama's server-reported stats.
        # Keys: eval_count, prompt_eval_count, eval_duration_ns, prompt_eval_duration_ns, total_duration_ns
        self.last_stats: dict = {}
    
    async def connect(self):
        """Initialize the HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def disconnect(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
    
    def reset_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    async def chat(self, user_message: str) -> str:
        """
        Send a chat message and get a response
        
        Args:
            user_message: The user's message
        
        Returns:
            The assistant's response
        """
        if not self.session:
            raise RuntimeError("Not connected to Ollama service")
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare the request
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ] + self.conversation_history
        
        payload = {
            "model": self.config.ollama_model,
            "messages": messages,
            "stream": False
        }
        
        try:
            async with self.session.post(
                f"{self.config.ollama_base_url}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                
                result = await response.json()
                assistant_message = result["message"]["content"]

                # Capture Ollama's server-side token counts and durations (exact for this model)
                self.last_stats = {
                    'eval_count': result.get('eval_count'),               # tokens generated
                    'prompt_eval_count': result.get('prompt_eval_count'), # tokens in full prompt
                    'eval_duration_ns': result.get('eval_duration'),      # generation time (ns)
                    'prompt_eval_duration_ns': result.get('prompt_eval_duration'),
                    'total_duration_ns': result.get('total_duration'),
                }

                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                return assistant_message
        
        except Exception as e:
            # Remove user message from history on error
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            raise
    
    async def chat_stream(self, user_message: str) -> AsyncIterator[str]:
        """
        Send a chat message and stream the response
        
        Args:
            user_message: The user's message
        
        Yields:
            Chunks of the assistant's response
        """
        if not self.session:
            raise RuntimeError("Not connected to Ollama service")
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare the request
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ] + self.conversation_history
        
        payload = {
            "model": self.config.ollama_model,
            "messages": messages,
            "stream": True
        }
        
        full_response = ""
        
        try:
            async with self.session.post(
                f"{self.config.ollama_base_url}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                
                import json
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            # The final chunk (done=True) carries stats but no content
                            if data.get('done'):
                                self.last_stats = {
                                    'eval_count': data.get('eval_count'),
                                    'prompt_eval_count': data.get('prompt_eval_count'),
                                    'eval_duration_ns': data.get('eval_duration'),
                                    'prompt_eval_duration_ns': data.get('prompt_eval_duration'),
                                    'total_duration_ns': data.get('total_duration'),
                                }
                            if "message" in data and "content" in data["message"]:
                                chunk = data["message"]["content"]
                                full_response += chunk
                                yield chunk
                        except json.JSONDecodeError:
                            continue
            
            # Add complete response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
        
        except Exception as e:
            # Remove user message from history on error
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            raise
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history"""
        if not self.conversation_history:
            return "No conversation yet."
        
        summary = f"Conversation history ({len(self.conversation_history)} messages):\n"
        for i, message in enumerate(self.conversation_history[-6:], 1):  # Last 6 messages
            role = message["role"].capitalize()
            content = message["content"][:50] + "..." if len(message["content"]) > 50 else message["content"]
            summary += f"{i}. {role}: {content}\n"
        
        return summary
