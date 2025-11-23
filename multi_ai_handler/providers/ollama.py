from multi_ai_handler.ai_provider import AIProvider
from pathlib import Path

from multi_ai_handler.generate_payload import generate_local_payload

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class OllamaProvider(AIProvider):
    def generate(self, system_prompt: str, user_text: str = None, file: str | Path | dict | None = None, model: str = None, temperature: float = 0.0) -> str:
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama is not installed. Install it with: pip install multi-ai-handler[ollama]"
            )

        messages: list = generate_local_payload(user_text, system_prompt, file)

        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )

        return response['message']['content']