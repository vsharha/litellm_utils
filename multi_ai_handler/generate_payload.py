import mimetypes
from typing import Any
import base64
from pathlib import Path

from multi_ai_handler.extract_md import extract_structured_md

def _process_file(file: str | Path | dict | None) -> tuple[str | None, str | None]:
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


def generate_openai_payload(user_text: str | None, system_prompt: str, file: str | Path | dict | None=None) -> list[dict[str, Any]]:
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    content = []

    if user_text:
        content.append({
            "type": "text",
            "text": user_text
        })

    if file:
        filename, encoded_data = _process_file(file)
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            raise ValueError("Could not detect MIME type from filename.")

        data_url = f"data:{mime_type};base64,{encoded_data}"

        if mime_type.startswith("image/"):
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        elif mime_type == "application/pdf":
            content.append({
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": data_url
                }
            })

    messages.append({
        "role": "user",
        "content": content
    })

    return messages

def generate_google_payload(user_text: str | None, file: str | Path | dict | None=None) -> list[dict[str, Any]]:
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    parts = []

    if user_text:
        parts.append({"text": user_text})

    if file:
        filename, encoded_data = _process_file(file)
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            raise ValueError("Could not detect MIME type from filename.")

        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": encoded_data
            }
        })

    return parts

def generate_claude_payload(user_text: str | None, file: str | Path | dict | None=None) -> list[dict[str, Any]]:
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    content = []

    if user_text:
        content.append({
            "type": "text",
            "text": user_text
        })

    if file:
        filename, encoded_data = _process_file(file)
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            raise ValueError("Could not detect MIME type from filename.")

        if mime_type.startswith("image/"):
            content_type = "image"
        else:
            content_type = "document"

        content.append({
            "type": content_type,
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": encoded_data
            }
        })

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages

def generate_local_payload(user_text: str | None, system_prompt: str, file: str | Path | dict | None=None) -> list[dict[str, Any]]:
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    content = []

    if user_text:
        content.append(user_text)
    if file:
        filename, encoded_data = _process_file(file)
        file_text = extract_structured_md(filename, encoded_data)
        if file_text:
            content.append(f"""
<<<FILE CONTENT ({filename})>>>
{file_text}
<<<END FILE CONTENT>>>
""")

    user_message = {
        "role": "user",
        "content": "\n".join(content) if content else ""
    }

    messages.append(user_message)

    return messages
