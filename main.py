from multi_ai_handler import request_ai

def main():
    print(request_ai(provider="cerebras", model="gpt-oss-120b", user_text="What's in the file", file="test/2024-10-31_aliexpress_02.pdf"))

if __name__ == "__main__":
    main()
