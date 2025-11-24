import litellm
from pathlib import Path
from typing import Iterator, Any, Generator

from multi_ai_handler.generate_payload import generate_openai_payload

LOCAL_PROVIDERS = ["ollama", "cerebras"]

def request_ai(provider: str, model: str, system_prompt: str | None=None, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, temperature: float=0.2, local: bool=False, json_output: bool=False, stream: bool=False) -> \
Generator[Any, Any, str | None]:
    if provider in LOCAL_PROVIDERS:
        local = True

    payload: list = generate_openai_payload(user_text, system_prompt, file, local, messages)

    response = litellm.completion(
        model=f"{provider}/{model}",
        messages=payload,
        temperature=temperature,
        stream=stream,
        #JSON
    )

    if stream:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.get("content"):
                yield chunk.choices[0].delta["content"]
    else:
        return response.choices[0].message.content