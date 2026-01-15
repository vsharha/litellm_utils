import mimetypes
from typing import Any, Optional
import logging

from litellm_utils.extract_md import extract_structured_md
from litellm_utils.types import FileType
from litellm_utils.utils import process_file

logger = logging.getLogger("litellm_utils")

def process_local_file(file: FileType) -> str:
    filename, _ = process_file(file)

    file_text = extract_structured_md(file)
    return (f"""
<<<FILE CONTENT ({filename})>>>
{file_text}
<<<END FILE CONTENT>>>
""")


def build_openai_user_content(user_text: Optional[str]=None, file: Optional[FileType | list[FileType]]=None, preprocess_file_content: bool=False) -> list[dict[str, Any]]:
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    content = []

    files = []
    if file is not None:
        if isinstance(file, list):
            files = file
        else:
            files = [file]

    if user_text and not files:
        content.append({
            "type": "text",
            "text": user_text
        })

    if files:
        if preprocess_file_content:
            file_texts = []
            for f in files:
                logger.info(f"Preprocessing file content locally")
                file_texts.append(process_local_file(file))

            combined_text = (user_text + "\n" if user_text else "") + "\n".join(file_texts)
            content.append({
                "type": "text",
                "text": combined_text
            })
        else:
            # Add user text first if present
            if user_text:
                content.append({
                    "type": "text",
                    "text": user_text
                })

            # Add each file to content
            for f in files:
                filename, encoded_data = process_file(f)
                mime_type, _ = mimetypes.guess_type(filename)
                if not mime_type:
                    logger.warning(f"Could not detect MIME type for {filename}, using application/octet-stream")
                    mime_type = "application/octet-stream"

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

def generate_openai_payload(user_text: Optional[str]=None, system_prompt: Optional[str]=None, file: Optional[FileType | list[FileType]]=None, preprocess_file_content: bool=False, messages: Optional[list[dict]]=None) -> list[dict[str, Any]]:
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