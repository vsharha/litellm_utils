import asyncio
from multi_ai_handler import request_ai, stream_ai, get_model_info, list_models, arequest_ai, astream_ai

def main():
    print(request_ai(provider="google", model="gemini-2.5-pro", user_text="What's in the file", file="test/2024-10-31_aliexpress_02.pdf"))

def stream_example():
    for chunk in stream_ai(provider="cerebras", model="gpt-oss-120b", user_text="What's in the file", file="test/2024-10-31_aliexpress_02.pdf"):
        print(chunk, end="", flush=True)
    print()

def model_info_example():
    all_models = list_models()

    provider = "ollama"
    models = all_models.get(provider, [])
    print(f"Available {provider} models: {len(models)}")

    if models:
        model = models[0]
        print(f"\nGetting info for: {model}")
        info = get_model_info(provider, model)
        print(info)

async def async_example():
    response = await arequest_ai(provider="google", model="gemini-2.0-flash", user_text="Write a haiku about async programming")
    print(response)

async def async_stream_example():
    async for chunk in astream_ai(provider="google", model="gemini-2.0-flash", user_text="Write a haiku about async programming"):
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(async_stream_example())
