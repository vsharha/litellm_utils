import os
from dotenv import load_dotenv
import logging

import google.generativeai as genai
from openai import OpenAI
import anthropic

from multi_ai_handler.generate_payload import generate_openai_payload, generate_gemini_payload, generate_claude_payload

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google.generativeai").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

def request_gemini(system_prompt: str, user_text: str=None, filename: str=None, encoded_data: str=None, model:str=None, temperature: float=0.0) -> str:
    genai.configure(api_key=GEMINI_API_KEY)

    generative_model = genai.GenerativeModel(model, system_instruction=system_prompt)

    payload: dict = generate_gemini_payload(filename, encoded_data, user_text)

    response = generative_model.generate_content(
        contents=[
            payload,
        ],
        generation_config=genai.GenerationConfig(
            temperature=temperature,
        ),
    )

    usage = getattr(response, "usage_metadata", None)

    if usage:
        print(f"Prompt tokens: {usage.prompt_token_count}")
        print(f"Output tokens: {usage.candidates_token_count}")
        print(f"Total tokens: {usage.total_token_count}")
    else:
        print("No usage metadata returned.")

    return response.text

def request_anthropic(system_prompt: str, user_text: str=None, filename: str=None, encoded_data: str=None, model:str=None, temperature: float=0.0):
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    payload: dict = generate_claude_payload(filename, encoded_data, user_text)

    response: str = ""

    with client.messages.stream(
        model=model,
        max_tokens=20000,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {
                'role': 'user',
                'content': [payload]
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            response += text

    return response

def request_openrouter(system_prompt: str, user_text: str=None, filename: str=None, encoded_data: str=None, model:str=None, temperature: float=0.0) -> str:
    link: str="https://openrouter.ai/api/v1"
    return request_openai(system_prompt, user_text, filename, encoded_data, model, temperature)

def request_openai(system_prompt: str, user_text: str=None, filename: str=None, encoded_data: str=None, model:str=None, temperature: float=0.0, link:str | None=None) -> str:
    client = OpenAI(
        base_url=link,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    payload: dict = generate_openai_payload(filename, encoded_data, user_text)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            payload
        ],
        temperature=temperature
    )

    return completion.choices[0].message.content