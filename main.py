from multi_ai_handler.handler import request_ai


def main():
    pass

if __name__ == "__main__":
    for chunk in request_ai(
        provider="ollama",
        model="gpt-oss",
        system_prompt="You're a helpful assistant",
        user_text="What is in this file?",
        file="test/2024-10-31_aliexpress_02.pdf",
        stream=True
    ):
        print(chunk, end="", flush=True)