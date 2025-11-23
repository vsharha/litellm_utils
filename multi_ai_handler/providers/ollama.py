from multi_ai_handler.ai_provider import AIProvider
from pathlib import Path
import requests

from multi_ai_handler.generate_payload import generate_local_payload

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class OllamaServerError(RuntimeError):
    pass


class OllamaProvider(AIProvider):
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__()
        self.base_url = base_url.rstrip("/")

    def _check_server(self):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama is not installed. Install it with: pip install multi-ai-handler[ollama]"
            )

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
        except requests.exceptions.ConnectionError:
            raise OllamaServerError(
                f"Ollama server is not running at {self.base_url}. "
                f"Start it with: `ollama serve`"
            )
        except requests.exceptions.RequestException as e:
            raise OllamaServerError(
                f"Could not communicate with Ollama server at {self.base_url}: {e}"
            )

        if resp.status_code >= 500:
            raise OllamaServerError(
                f"Ollama server responded with {resp.status_code} (server error)"
            )

    def generate(self, system_prompt: str, user_text: str = None, file: str | Path | dict | None = None, model: str = None, temperature: float = 0.0) -> str:
        self._check_server()

        messages: list = generate_local_payload(user_text, system_prompt, file)

        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )

        return response['message']['content']

    def list_models(self) -> list:
        self._check_server()

        resp = requests.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()

        return data.get("models")
