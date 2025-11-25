import mimetypes
from typing import Any
import base64
from pathlib import Path
import logging

from litellm_utils.extract_md import extract_structured_md

logger = logging.getLogger("litellm_utils")

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


def process_local_file(filename: str, encoded_data: str) -> str:
    file_text = extract_structured_md(filename, encoded_data)
    return (f"""
<<<FILE CONTENT ({filename})>>>
{file_text}
<<<END FILE CONTENT>>>
""")


def build_openai_user_content(user_text: str | None, file: str | Path | dict | None=None, preprocess_file_content: bool=False) -> list[dict[str, Any]]:
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    content = []

    if user_text and not file:
        content.append({
            "type": "text",
            "text": user_text
        })

    if file:
        filename, encoded_data = _process_file(file)

        if preprocess_file_content:
            logger.info(f"Preprocessing file content locally for: {filename}")
            content.append({
                "type": "text",
                "text": (user_text + "\n" if user_text else "") + process_local_file(filename, encoded_data)
            })
        else:
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                logger.warning(f"Could not detect MIME type for {filename}, using application/octet-stream")
                mime_type = "application/octet-stream"

            if user_text:
                content.append({
                    "type": "text",
                    "text": user_text
                })

            data_url = f"data:{mime_type};base64,{encoded_data}"

            if mime_type.startswith("image/"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            else:
                # Handle all non-image files (PDFs, documents, text files, etc.)
                content.append({
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": data_url
                    }
                })

    return content


def generate_openai_payload(user_text: str | None, system_prompt: str, file: str | Path | dict | None=None, preprocess_file_content: bool=False, messages: list[dict] | None=None) -> list[dict[str, Any]]:
    result = []

    if system_prompt:
        result.append({
            "role": "system",
            "content": system_prompt
        })

    # Add previous conversation history
    if messages:
        result.extend(messages)

    # Add new user message
    content = build_openai_user_content(user_text, file, preprocess_file_content)
    result.append({
        "role": "user",
        "content": content
    })

    return result