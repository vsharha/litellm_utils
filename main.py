from multi_ai_handler.handler import request_ai

def json_example():
    print(request_ai(
        provider="cerebras",
        model="gpt-oss-120b",
        system_prompt="You're an intelligent json converter tool",
        user_text="Convert to json",
        file="test/2024-10-31_aliexpress_02.pdf",
        json_output=True,
    ))

if __name__ == "__main__":
    json_example()