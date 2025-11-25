from typing import Iterator

import litellm
from pathlib import Path
import json

from litellm import get_model_info

from litellm_utils.generate_payload import generate_openai_payload
from litellm_utils.utils import parse_ai_response


def supports_pdf_input(provider: str, model: str) -> bool:
    model_name = f"{provider}/{model}"

    model_info = get_model_info(model_name)

    # print(json.dumps(model_info, indent=2))
    return "supports_pdf_input" in model_info and model_info["supports_pdf_input"]

def request_ai(provider: str, model: str, system_prompt: str | None=None, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, temperature: float=0.2, preprocess_file_content: bool=False, json_output: bool=False) -> str | dict:
    model_name = f"{provider}/{model}"

    if not supports_pdf_input(provider, model):
        preprocess_file_content = True

    payload: list = generate_openai_payload(user_text, system_prompt, file, preprocess_file_content, messages)

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

def stream_ai(provider: str, model: str, system_prompt: str | None=None, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, temperature: float=0.2, preprocess_file_content: bool=False) -> Iterator[str]:
    model_name = f"{provider}/{model}"

    if not supports_pdf_input(provider, model):
        preprocess_file_content = True

    payload: list = generate_openai_payload(user_text, system_prompt, file, preprocess_file_content, messages)

    response = litellm.completion(
        model=model_name,
        messages=payload,
        temperature=temperature,
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def list_models(provider: str) -> list[dict]:
    models = litellm.models_by_provider.get(provider)
    if models is None:
        return []
    return models