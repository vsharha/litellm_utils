import litellm
from pathlib import Path

from multi_ai_handler.generate_payload import generate_openai_payload

LOCAL_PROVIDERS = ["ollama", "cerebras"]

def request_ai(provider: str, model: str, system_prompt: str | None=None, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, temperature: float=0.2, local: bool=False, json_output: bool=False):
    if provider in LOCAL_PROVIDERS:
        local = True

    payload: list = generate_openai_payload(user_text, system_prompt, file, local, messages)

    litellm.completion(
        model=f"{provider}/{model}",
        messages=payload,
        temperature=temperature,
        #JSON
    )