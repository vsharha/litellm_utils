from multi_ai_handler import request_ai, OpenAIProvider, AnthropicProvider, OpenrouterProvider, OllamaProvider, \
    GoogleProvider


def main():
    client = OllamaProvider()
    for m in client.list_models():
        print(m)

    client = GoogleProvider()

    for m in client.list_models():
        print(m)
    # print(request_ai(system_prompt="You're a helpful assistant", user_text="Hello", file="test/2024-10-31_aliexpress_02.pdf", provider="google"))

if __name__ == "__main__":
    main()
