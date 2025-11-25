import json

from litellm import get_model_info

from litellm_utils import Conversation
from litellm_utils.handler import request_ai, stream_ai, list_models, supports_pdf_input


def basic_example():
    response = request_ai(
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant",
        user_text="What is the capital of France?"
    )
    print(response)


def stream_example():
    print("Streaming response: ", end="", flush=True)
    for chunk in stream_ai(
        provider="ollama",
        model="gpt-oss",
        system_prompt="You are a helpful assistant",
        user_text="Write a short poem about coding"
    ):
        print(chunk, end="", flush=True)
    print("\n")


def json_example():
    print(request_ai(
        provider="cerebras",
        model="llama-3.3-70b",
        system_prompt="You're an intelligent json converter tool",
        user_text="Convert to json",
        file="test/2024-10-31_aliexpress_02.pdf",
        json_output=True,
    ))


def list_models_example():
    print(json.dumps(list(list_models("cerebras")), indent=4))


def conversation_example():
    conv = Conversation(
        provider="ollama",
        model="gpt-oss",
        system_prompt="You are a helpful assistant",
        temperature=0.7
    )

    print("Turn 1:")
    for chunk in conv.stream("What is Python?"):
        print(chunk, end="", flush=True)
    print("\n")

    print("Turn 2:")
    for chunk in conv.stream("What are its main features?"):
        print(chunk, end="", flush=True)
    print("\n")

    print("Turn 3:")
    for chunk in conv.stream("Give me a simple code example"):
        print(chunk, end="", flush=True)
    print("\n")

    print(f"\nConversation history length: {len(conv.get_history())} messages")


if __name__ == "__main__":
    print(json_example())