import litellm
from pathlib import Path
import json

from multi_ai_handler.generate_payload import generate_openai_payload

NO_FILE_UPLOAD_PROVIDERS = ["ollama", "cerebras"]

def request_ai(provider: str, model: str, system_prompt: str | None=None, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, temperature: float=0.2, extract_file_content: bool=False, json_output: bool=False) -> str | dict:
    model_name = f"{provider}/{model}"

    if provider in NO_FILE_UPLOAD_PROVIDERS:
        extract_file_content = True

    payload: list = generate_openai_payload(user_text, system_prompt, file, extract_file_content, messages)

    response = litellm.completion(
        model=model_name,
        messages=payload,
        temperature=temperature,
    )
    response_str = response.choices[0].message.content

    if json_output:
        return parse_ai_response(response_str)
    else:
        return response_str

def list_models(provider: str) -> list[dict]:
    models = litellm.models_by_provider.get(provider)
    if models is None:
        return []
    return models

def parse_ai_response(response_text: str) -> dict:
    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start: int = 0
        if "```json" in response_text:
            start = response_text.find("```json") + 7
        elif "```" in response_text:
            start = response_text.find("```") + 3

        if start != 0:
            end: int = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
            else:
                response_text = response_text[start:].strip()

    try:
        return json.loads(response_text)
    except json.decoder.JSONDecodeError as e:
        preview = response_text[:500] if len(response_text) > 500 else response_text
        raise ValueError(
            f"Failed to parse JSON from AI response.\n"
            f"Error: {str(e)}\n"
            f"Response content:\n{preview}"
        ) from e