from openai import OpenAI
from openai.pagination import SyncPage

from multi_ai_handler.ai_provider import AIProvider
import os
from pathlib import Path

from multi_ai_handler.generate_payload import generate_openai_payload

class OpenAIProvider(AIProvider):
    def __init__(self, base_url: str | None=None, api_key: str | None=None, local: bool=False) -> None:
        super().__init__()
        self.local = local
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, local: bool=False) -> str:
        if self.local:
            local = True

        messages: list = generate_openai_payload(user_text, system_prompt, file, local=local)

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return completion.choices[0].message.content

    def list_models(self) -> SyncPage:
        return self.client.models.list()