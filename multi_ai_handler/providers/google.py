from google import genai
from google.genai import types
from pathlib import Path
from typing import Iterator, AsyncIterator

from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.utils import parse_ai_response
from multi_ai_handler.generate_payload import generate_google_payload

class GoogleProvider(AIProvider):
    def __init__(self):
        super().__init__()
        self.client = genai.Client()
        self.async_client = self.client.aio

    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> str | dict:
        payload: list = generate_google_payload(user_text, file, local=local)

        response = self.client.models.generate_content(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        if json_output:
            return parse_ai_response(response.text)
        return response.text

    def stream(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        payload: list = generate_google_payload(user_text, file, local=local)

        response = self.client.models.generate_content_stream(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return [model.name for model in response]

    def get_model_info(self, model: str) -> dict:
        response = self.client.models.get(model=model)
        return {
            "name": response.name,
            "display_name": response.display_name,
            "input_token_limit": response.input_token_limit,
            "output_token_limit": response.output_token_limit,
        }

    async def agenerate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> str | dict:
        payload: list = generate_google_payload(user_text, file, local=local)

        response = await self.async_client.models.generate_content(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        if json_output:
            return parse_ai_response(response.text)
        return response.text

    async def astream(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> AsyncIterator[str]:
        payload: list = generate_google_payload(user_text, file, local=local)

        response = await self.async_client.models.generate_content_stream(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text