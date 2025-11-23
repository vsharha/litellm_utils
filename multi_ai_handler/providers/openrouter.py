from multi_ai_handler.providers.openai import OpenAIProvider
import os

class OpenrouterProvider(OpenAIProvider):
    def __init__(self):
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )