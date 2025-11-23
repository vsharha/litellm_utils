from multi_ai_handler.ai_provider import AIProvider
from pathlib import Path
import os

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False

from multi_ai_handler.generate_payload import generate_openai_payload_local


class CerebrasProvider(AIProvider):
    def __init__(self) -> None:
        super().__init__()
        self.client = Cerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY")
        )

    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0) -> str:
        if not CEREBRAS_AVAILABLE:
            raise ImportError(
                "Cerebras is not installed. Install it with: pip install multi-ai-handler[extra]"
            )

        messages: list = generate_openai_payload_local(user_text, system_prompt, file)

        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=False,
            max_completion_tokens=20000,
            temperature=temperature
        )

        return response.choices[0].message.content