import logging
from typing import Iterator

import litellm
from pathlib import Path

from litellm import get_model_info

from litellm_utils.generate_payload import generate_openai_payload
from litellm_utils.utils import parse_ai_response

logger = logging.getLogger("litellm_utils")


def requires_preprocessing(model: str) -> bool:
    try:
        info = get_model_info(model)
    except Exception as e:
        logger.debug(f"Could not get model info for {model}: {e}")
        return False

    return not info.get("supports_pdf_input", False)


def _validate_preprocessing_config(
    model: str,
    preprocess_file_content: bool | None,
) -> bool:
    needs_preprocessing = requires_preprocessing(model)

    if preprocess_file_content is False and needs_preprocessing:
        raise ValueError(
            f"Model {model} requires preprocessing for file inputs, but preprocess_file_content was explicitly set to False"
        )

    return needs_preprocessing



def request_ai(
        model: str,
        system_prompt: str | None=None,
        user_text: str | None=None,
        messages: list[dict]=None,
        file: str | Path | dict | list[str | Path | dict] | None=None,
        temperature: float=0.2,
        preprocess_file_content: bool | None=None,
        json_output: bool=False
    ) -> str | dict:
    if file is not None and preprocess_file_content is None:
        needs_preprocessing = _validate_preprocessing_config(model, preprocess_file_content)

        preprocess_file_content = needs_preprocessing

    payload: list = generate_openai_payload(user_text, system_prompt, file, preprocess_file_content, messages)

    response = litellm.completion(
        model=model,
        messages=payload,
        temperature=temperature,
    )
    response_str = response.choices[0].message.content

    if json_output:
        return parse_ai_response(response_str)
    else:
        return response_str

def stream_ai(
        model: str,
        system_prompt: str | None=None,
        user_text: str | None=None,
        messages: list[dict]=None,
        file: str | Path | dict | list[str | Path | dict] | None=None,
        temperature: float=0.2,
        preprocess_file_content: bool | None=None
    ) -> Iterator[str]:
    if file is not None and preprocess_file_content is None:
        needs_preprocessing = _validate_preprocessing_config(model, preprocess_file_content)

        preprocess_file_content = needs_preprocessing

    payload: list = generate_openai_payload(user_text, system_prompt, file, preprocess_file_content, messages)

    response = litellm.completion(
        model=model,
        messages=payload,
        temperature=temperature,
        stream=True,
    )

    for chunk in response:
        if content:=chunk.choices[0].delta.content:
            yield content

def list_models(provider: str) -> list[dict]:
    models = litellm.models_by_provider.get(provider)
    if models is None:
        return []
    return models