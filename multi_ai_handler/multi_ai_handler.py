from pathlib import Path
from typing import Iterator, AsyncIterator

from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.providers.openrouter import OpenrouterProvider
from multi_ai_handler.providers.anthropic import AnthropicProvider
from multi_ai_handler.providers.cerebras import CerebrasProvider
from multi_ai_handler.providers.google import GoogleProvider
from multi_ai_handler.providers.ollama import OllamaProvider
from multi_ai_handler.providers.openai import OpenAIProvider


class AIProviderManager:
    def __init__(self):
        self.providers: dict[str, type[AIProvider]] = {
            "google": GoogleProvider,
            "anthropic": AnthropicProvider,
            "openai": OpenAIProvider,
            "openrouter": OpenrouterProvider,
            "ollama": OllamaProvider,
            "cerebras": CerebrasProvider,
        }

    def register_provider(self, name: str, provider: type[AIProvider]) -> None:
        self.providers[name] = provider

    def generate(self, provider: str, model: str, system_prompt: str | None=None, user_text: str=None, file: str | Path | dict | None=None, temperature: float=0.2, local: bool=False, json_output: bool=False) -> dict | str:
        Provider = self.providers[provider]
        client = Provider()

        return client.generate(system_prompt, user_text, file, model, temperature, local=local, json_output=json_output)

    def stream(self, provider: str, model: str, system_prompt: str | None=None, user_text: str=None, file: str | Path | dict | None=None, temperature: float=0.2, local: bool=False) -> Iterator[str]:
        Provider = self.providers[provider]
        client = Provider()

        yield from client.stream(system_prompt, user_text, file, model, temperature, local=local)

    def list_models(self) -> dict[str, list[str]]:
        models = {}

        for name, Provider in self.providers.items():
            try:
                client = Provider()
                models[name] = client.list_models()
            except Exception:
                models[name] = []

        return models

    def get_model_info(self, provider: str, model: str) -> dict:
        Provider = self.providers[provider]
        client = Provider()

        return client.get_model_info(model)

    async def agenerate(self, provider: str, model: str, system_prompt: str | None=None, user_text: str=None, file: str | Path | dict | None=None, temperature: float=0.2, local: bool=False, json_output: bool=False) -> dict | str:
        Provider = self.providers[provider]
        client = Provider()

        return await client.agenerate(system_prompt, user_text, file, model, temperature, local=local, json_output=json_output)

    async def astream(self, provider: str, model: str, system_prompt: str | None=None, user_text: str=None, file: str | Path | dict | None=None, temperature: float=0.2, local: bool=False) -> AsyncIterator[str]:
        Provider = self.providers[provider]
        client = Provider()

        async for chunk in client.astream(system_prompt, user_text, file, model, temperature, local=local):
            yield chunk
