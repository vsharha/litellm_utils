import logging

from litellm_utils.conversation import Conversation
from litellm_utils.handler import request_ai, stream_ai, list_models
from litellm_utils.generate_payload import build_openai_user_content, generate_openai_payload

# Configure logger
logger = logging.getLogger("litellm_utils")
logger.setLevel(logging.INFO)
logger.propagate = False

# Add console handler if no handlers exist
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

__all__ = [
    "Conversation",
    "request_ai",
    "stream_ai",
    "list_models",
    "logger",
    "build_openai_user_content",
    "generate_openai_payload"
]