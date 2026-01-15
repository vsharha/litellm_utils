from typing import Iterator, Optional

from litellm_utils.types import FileType

from litellm_utils.handler import request_ai, stream_ai
from litellm_utils.generate_payload import build_openai_user_content


class Conversation:
    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.messages: list[dict] = []

    def send(
        self,
        user_text: Optional[str] = None,
        file: Optional[FileType | list[FileType]] = None,
        json_output: bool = False,
        preprocess_file_content: Optional[bool] = None,
    ) -> str | dict:
        response = request_ai(
            model=self.model,
            system_prompt=self.system_prompt if not self.messages else None,
            user_text=user_text,
            messages=self.messages,
            file=file,
            temperature=self.temperature,
            preprocess_file_content=preprocess_file_content,
            json_output=json_output,
        )

        # Build proper user content (includes files if provided)
        user_content = build_openai_user_content(user_text, file, preprocess_file_content or False)

        self.messages.append({"role": "user", "content": user_content})
        self.messages.append({"role": "assistant", "content": str(response)})

        return response

    def stream(
        self,
        user_text: Optional[str] = None,
        file: Optional[FileType | list[FileType]] = None,
        preprocess_file_content: Optional[bool] = None,
    ) -> Iterator[str]:
        full_response = ""

        for chunk in stream_ai(
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

        # Build proper user content (includes files if provided)
        user_content = build_openai_user_content(user_text, file, preprocess_file_content or False)

        self.messages.append({"role": "user", "content": user_content})
        self.messages.append({"role": "assistant", "content": full_response})

    def get_history(self) -> list[dict]:
        return self.messages.copy()

    def clear_history(self):
        self.messages = []

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def __repr__(self) -> str:
        return f"Conversation(model='{self.model}', messages={len(self.messages)})"
