from typing import Iterator
from pathlib import Path

from litellm_utils.handler import request_ai, stream_ai


class Conversation:
    def __init__(
        self,
        provider: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ):
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.messages: list[dict] = []

    def send(
        self,
        user_text: str | None = None,
        file: str | Path | dict | None = None,
        json_output: bool = False,
        preprocess_file_content: bool = False,
    ) -> str | dict:
        response = request_ai(
            provider=self.provider,
            model=self.model,
            system_prompt=self.system_prompt if not self.messages else None,
            user_text=user_text,
            messages=self.messages,
            file=file,
            temperature=self.temperature,
            preprocess_file_content=preprocess_file_content,
            json_output=json_output,
        )

        if isinstance(response, str):
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": response})
        else:
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": str(response)})

        return response

    def stream(
        self,
        user_text: str | None = None,
        file: str | Path | dict | None = None,
        preprocess_file_content: bool = False,
    ) -> Iterator[str]:
        full_response = ""

        for chunk in stream_ai(
            provider=self.provider,
            model=self.model,
            system_prompt=self.system_prompt if not self.messages else None,
            user_text=user_text,
            messages=self.messages,
            file=file,
            temperature=self.temperature,
            preprocess_file_content=preprocess_file_content,
        ):
            full_response += chunk
            yield chunk

        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": full_response})

    def get_history(self) -> list[dict]:
        return self.messages.copy()

    def clear_history(self):
        self.messages = []

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def __repr__(self) -> str:
        return f"Conversation(provider='{self.provider}', model='{self.model}', messages={len(self.messages)})"
