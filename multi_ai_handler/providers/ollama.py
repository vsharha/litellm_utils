from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.utils import parse_ai_response
from pathlib import Path
from typing import Iterator, AsyncIterator
import requests

from multi_ai_handler.generate_payload import generate_ollama_payload

try:
    import ollama
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    AsyncClient = None

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

    def generate(self, system_prompt: str, user_text: str = None, file: str | Path | dict | None = None, model: str = None, temperature: float = 0.0, local: bool=False, json_output: bool=False) -> str | dict:
        self._check_server()

        messages: list = generate_ollama_payload(user_text, system_prompt, file)

        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )

        response_text = response['message']['content']
        if json_output:
            return parse_ai_response(response_text)
        return response_text

    def stream(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        self._check_server()

        messages: list = generate_ollama_payload(user_text, system_prompt, file)

        stream = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=True
        )

        for chunk in stream:
            if chunk['message']['content']:
                yield chunk['message']['content']

    def list_models(self) -> list[str]:
        self._check_server()

        resp = requests.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()

        return [model["name"] for model in data.get("models", [])]

    def get_model_info(self, model: str) -> dict:
        self._check_server()

        resp = requests.post(f"{self.base_url}/api/show", json={"name": model})
        resp.raise_for_status()
        data = resp.json()

        return {
            "name": model,
            "modified_at": data.get("modified_at"),
            "size": data.get("size"),
            "parameters": data.get("details", {}).get("parameter_size"),
        }

    async def agenerate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> str | dict:
        self._check_server()

        messages: list = generate_ollama_payload(user_text, system_prompt, file)

        async_client = AsyncClient(host=self.base_url)
        response = await async_client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )

        response_text = response['message']['content']
        if json_output:
            return parse_ai_response(response_text)
        return response_text

    async def astream(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> AsyncIterator[str]:
        self._check_server()

        messages: list = generate_ollama_payload(user_text, system_prompt, file)

        async_client = AsyncClient(host=self.base_url)
        stream = await async_client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=True
        )

        async for chunk in stream:
            if chunk['message']['content']:
                yield chunk['message']['content']
