import json
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

    def generate(self, provider: str, model:str, system_prompt: str | None=None, user_text: str=None, file: str | Path | dict | None=None, temperature: float=0.2, local:bool=False, json_output: bool = False) -> dict | str:
        Provider = self.providers[provider]
        client = Provider()

        response_text: str = client.generate(system_prompt, user_text, file, model, temperature, local=local)

        if json_output:
            return parse_ai_response(response_text)

        else:
            return response_text

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

    async def agenerate(self, provider: str, model: str, system_prompt: str | None=None, user_text: str=None, file: str | Path | dict | None=None, temperature: float=0.2, local: bool=False, json_output: bool = False) -> dict | str:
        Provider = self.providers[provider]
        client = Provider()

        response_text: str = await client.agenerate(system_prompt, user_text, file, model, temperature, local=local)

        if json_output:
            return parse_ai_response(response_text)

        return response_text

    async def astream(self, provider: str, model: str, system_prompt: str | None=None, user_text: str=None, file: str | Path | dict | None=None, temperature: float=0.2, local: bool=False) -> AsyncIterator[str]:
        Provider = self.providers[provider]
        client = Provider()

        async for chunk in client.astream(system_prompt, user_text, file, model, temperature, local=local):
            yield chunk

_handler = AIProviderManager()

def request_ai(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    temperature: float = 0.2,
    json_output: bool = False,
    local: bool = False,
) -> dict | str:
    return _handler.generate(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        temperature=temperature,
        json_output=json_output,
        local=local,
    )

def stream_ai(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    temperature: float = 0.2,
    local: bool = False,
) -> Iterator[str]:
    yield from _handler.stream(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        temperature=temperature,
        local=local,
    )

def get_model_info(provider: str, model: str) -> dict:
    return _handler.get_model_info(provider=provider, model=model)

def list_models() -> dict[str, list[str]]:
    return _handler.list_models()

async def arequest_ai(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    temperature: float = 0.2,
    json_output: bool = False,
    local: bool = False,
) -> dict | str:
    return await _handler.agenerate(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        temperature=temperature,
        json_output=json_output,
        local=local,
    )

async def astream_ai(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    temperature: float = 0.2,
    local: bool = False,
) -> AsyncIterator[str]:
    async for chunk in _handler.astream(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        temperature=temperature,
        local=local,
    ):
        yield chunk

def parse_ai_response(response_text: str) -> dict:
    response_text = response_text.strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start: int = 0
        if "```json" in response_text:
            start = response_text.find("```json") + 7
        elif "```" in response_text:
            start = response_text.find("```") + 3

        if start != 0:
            end: int = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end]

    try:
        return json.loads(response_text)
    except json.decoder.JSONDecodeError as e:
        raise Exception(e)