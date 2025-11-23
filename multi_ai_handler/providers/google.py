from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path

from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.generate_payload import generate_google_payload

load_dotenv()

class GoogleProvider(AIProvider):
    def __init__(self):
        super().__init__()
        self.client = genai.Client()

    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0) -> str:
        payload: list = generate_google_payload(user_text, file)

        response = self.client.models.generate_content(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        return response.text