from anthropic import Anthropic, AsyncAnthropic

from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.utils import parse_ai_response
from pathlib import Path
from typing import Iterator, AsyncIterator

from multi_ai_handler.generate_payload import generate_claude_payload


class AnthropicProvider(AIProvider):
    def __init__(self):
        super().__init__()
        self.client = Anthropic()
        self.async_client = AsyncAnthropic()

    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> str | dict:
        messages: list = generate_claude_payload(user_text, file, local=local)

        response: str = ""

        with self.client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                response += text

        if json_output:
            return parse_ai_response(response)
        return response

    def stream(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        messages: list = generate_claude_payload(user_text, file, local=local)

        with self.client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return [model.id for model in response.data]

    def get_model_info(self, model: str) -> dict:
        response = self.client.models.retrieve(model)
        return {
            "id": response.id,
            "created_at": response.created_at,
            "display_name": response.display_name,
        }

    async def agenerate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> str | dict:
        messages: list = generate_claude_payload(user_text, file, local=local)

        response: str = ""

        async with self.async_client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        ) as stream:
            async for text in stream.text_stream:
                response += text

        if json_output:
            return parse_ai_response(response)
        return response

    async def astream(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> AsyncIterator[str]:
        messages: list = generate_claude_payload(user_text, file, local=local)

        async with self.async_client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        ) as stream:
            async for text in stream.text_stream:
                yield text