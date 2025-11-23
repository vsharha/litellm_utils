import json


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
                response_text = response_text[start:end]

    try:
        return json.loads(response_text)
    except json.decoder.JSONDecodeError as e:
        raise Exception(e)
