from anthropic import Anthropic
from multi_ai_handler.ai_provider import AIProvider
import os
from pathlib import Path

from multi_ai_handler.generate_payload import generate_claude_payload


class AnthropicProvider(AIProvider):
    def __init__(self):
        super().__init__()
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0) -> str:
        messages: list = generate_claude_payload(user_text, file)

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

        return response
