from dotenv import load_dotenv

load_dotenv()

from multi_ai_handler.multi_ai_handler import (
    request_ai,
    AIProviderManager,
    parse_ai_response,
)

from multi_ai_handler.providers.anthropic import AnthropicProvider
from multi_ai_handler.providers.cerebras import CerebrasProvider
from multi_ai_handler.providers.google import GoogleProvider
from multi_ai_handler.providers.ollama import OllamaProvider
from multi_ai_handler.providers.openai import OpenAIProvider
from multi_ai_handler.providers.openrouter import OpenrouterProvider

from multi_ai_handler.ai_provider import AIProvider

__all__ = [
    # Main unified interface
    "request_ai",
    "AIProviderManager",
    "parse_ai_response",
    # Provider-specific classes
    "AnthropicProvider",
    "CerebrasProvider",
    "GoogleProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenrouterProvider",
    "AIProvider"
]