import json
from pathlib import Path
import base64

def process_file(file: str | Path | dict | None) -> tuple[str | None, str | None]:
    if file is None:
        return None, None

    if isinstance(file, dict):
        return file.get("filename"), file.get("encoded_data")

    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    with open(file_path, "rb") as f:
        file_data = f.read()

    encoded = base64.b64encode(file_data).decode()
    return file_path.name, encoded

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