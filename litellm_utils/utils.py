import json

def parse_ai_response(response_text: str) -> dict:
    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = None
        if "```json" in response_text:
            start = response_text.find("```json") + 7
        elif "```" in response_text:
            start = response_text.find("```") + 3

        if start is not None:
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