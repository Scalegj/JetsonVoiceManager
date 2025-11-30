"""
Example of how to use the voice chat library programmatically
with custom configurations
"""
import asyncio
from config import create_config
from voice_chat import VoiceChat


async def example_basic_conversation():
    """
    Basic example: Simple conversation with default settings
    """
    print("\n" + "="*60)
    print("Example 1: Basic Conversation")
    print("="*60)
    
    # Create configuration
    config = create_config(
        jetson_ip="192.168.1.100",  # Replace with your Jetson IP
        ollama_model="llama3.2:3b"
    )
    
    # Create voice chat instance
    chat = VoiceChat(config)
    
    # Run the conversation
    await chat.run()


async def example_custom_personality():
    """
    Example: Chatbot with custom personality
    """
    print("\n" + "="*60)
    print("Example 2: Custom Personality")
    print("="*60)
    
    # Create configuration with custom system prompt
    config = create_config(
        jetson_ip="192.168.1.100",  # Replace with your Jetson IP
        system_prompt=(
            "You are a friendly tech support assistant who loves to help people "
            "with their computer problems. You explain things in simple terms and "
            "always ask if the user needs clarification. Keep responses brief."
        ),
        ollama_model="llama3.2:3b"
    )
    
    chat = VoiceChat(config)
    await chat.run()


async def example_sensitive_detection():
    """
    Example: Adjusted for sensitive microphone (picks up background noise)
    """
    print("\n" + "="*60)
    print("Example 3: Sensitive Microphone Settings")
    print("="*60)
    
    # Increase thresholds for noisy environment
    config = create_config(
        jetson_ip="192.168.1.100",  # Replace with your Jetson IP
        silence_threshold=800.0,    # Higher threshold
        silence_duration=2.0,        # Longer wait time
        ollama_model="llama3.2:3b"
    )
    
    chat = VoiceChat(config)
    await chat.run()


async def example_streaming_mode():
    """
    Example: Streaming mode to see LLM thinking in real-time
    """
    print("\n" + "="*60)
    print("Example 4: Streaming Mode")
    print("="*60)
    
    config = create_config(
        jetson_ip="192.168.1.100",  # Replace with your Jetson IP
        ollama_model="llama3.2:3b"
    )
    
    chat = VoiceChat(config)
    await chat.run_with_streaming()


async def example_manual_control():
    """
    Example: Manual control over conversation flow
    (e.g., for integration with other systems)
    """
    print("\n" + "="*60)
    print("Example 5: Manual Control")
    print("="*60)
    
    config = create_config(
        jetson_ip="192.168.1.100",  # Replace with your Jetson IP
        ollama_model="llama3.2:3b"
    )
    
    chat = VoiceChat(config)
    
    try:
        # Connect to services
        await chat.connect()
        
        # Have 3 conversation turns
        for turn in range(3):
            print(f"\n--- Turn {turn + 1} ---")
            should_continue = await chat.process_turn()
            if not should_continue:
                break
        
        print("\n--- Conversation complete ---")
        
    finally:
        # Always disconnect
        await chat.disconnect()


async def example_different_model():
    """
    Example: Using a different (smaller/faster) model
    """
    print("\n" + "="*60)
    print("Example 6: Using Smaller Model for Speed")
    print("="*60)
    
    # Use smaller 1B model for faster responses
    config = create_config(
        jetson_ip="192.168.1.100",  # Replace with your Jetson IP
        ollama_model="llama3.2:1b"  # Smaller, faster model
    )
    
    chat = VoiceChat(config)
    await chat.run()


def main():
    """
    Main function - uncomment the example you want to run
    """
    print("\n🎯 Voice Chat Library - Usage Examples\n")
    print("Edit example_custom_usage.py to run different examples")
    print("Make sure to replace '192.168.1.100' with your Jetson IP!\n")
    
    # Uncomment ONE of the following lines to run that example:
    
    # asyncio.run(example_basic_conversation())
    # asyncio.run(example_custom_personality())
    # asyncio.run(example_sensitive_detection())
    # asyncio.run(example_streaming_mode())
    # asyncio.run(example_manual_control())
    # asyncio.run(example_different_model())
    
    print("\n💡 Tip: Uncomment one of the example functions in main() to run it")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
