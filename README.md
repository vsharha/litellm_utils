# Multi AI Handler

A unified Python library for interacting with multiple AI providers through a consistent interface. Supports text and file inputs across OpenAI, Anthropic Claude, Google Gemini, OpenRouter, Cerebras and Ollama (local LLMs).

## Features

- Unified interface for multiple AI providers
- **Conversation history** for multi-turn interactions
- **Streaming support** for real-time token output
- **Async support** for concurrent workloads
- Support for images and documents (PDF)
- Local LLM support with Ollama
- Advanced document processing with Docling (OCR, table extraction)
- Model information retrieval

## Installation

```bash
pip install multi-ai-handler
```

**Optional dependencies:**
```bash
pip install multi-ai-handler[ollama]   # Local LLM support
pip install multi-ai-handler[docling]  # Document processing (OCR, tables)
pip install multi-ai-handler[all]      # All optional dependencies
```

## Setup

Create a `.env` file with your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Usage

### Basic Request

```python
from multi_ai_handler import request_ai

response = request_ai(
    provider="google",  # or "anthropic", "openai", "openrouter", "cerebras", "ollama"
    model="gemini-2.5-flash",
    system_prompt="You are a helpful assistant.",
    user_text="What is the capital of France?"
)
```

### JSON Output

```python
data = request_ai(
    provider="google",
    model="gemini-2.5-flash",
    system_prompt="Return valid JSON only.",
    user_text="Convert to JSON: Name: Alice, Age: 25",
    json_output=True
)
# Returns: {'name': 'Alice', 'age': 25}
```

### File Processing

```python
response = request_ai(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    system_prompt="Summarize this document.",
    file="document.pdf"
)
```

### Streaming

```python
from multi_ai_handler import stream_ai

for chunk in stream_ai(provider="google", model="gemini-2.0-flash", user_text="Write a poem"):
    print(chunk, end="", flush=True)
```

### Async Support

```python
import asyncio
from multi_ai_handler import arequest_ai, astream_ai

async def main():
    # Concurrent requests
    responses = await asyncio.gather(
        arequest_ai(provider="google", model="gemini-2.0-flash", user_text="Hello"),
        arequest_ai(provider="anthropic", model="claude-sonnet-4-20250514", user_text="Hello"),
    )

    # Async streaming
    async for chunk in astream_ai(provider="google", model="gemini-2.0-flash", user_text="Hi"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Conversation History

Use the `Conversation` class for multi-turn interactions:

```python
from multi_ai_handler import AIProviderManager

manager = AIProviderManager()
conv = manager.conversation(
    provider="google",
    model="gemini-2.0-flash",
    system_prompt="You are a helpful assistant.",
)

response = conv.send("My name is Alice.")
print(response.content)

response = conv.send("What's my name?")  # Remembers context
print(response.content)

conv.clear()  # Reset conversation
```

With file processing:

```python
conv = manager.conversation(provider="google", model="gemini-2.0-flash")

response = conv.send("Summarize this document", file="report.pdf")
print(response.content)

response = conv.send("What are the key findings?")  # Follow-up without re-sending file
print(response.content)
```

### Model Information

```python
from multi_ai_handler import list_models, get_model_info

all_models = list_models()  # {'google': [...], 'anthropic': [...], ...}
info = get_model_info(provider="google", model="models/gemini-2.0-flash")
```

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `request_ai(provider, model, ...)` | Generate a response |
| `stream_ai(provider, model, ...)` | Stream response tokens |
| `arequest_ai(provider, model, ...)` | Async generation |
| `astream_ai(provider, model, ...)` | Async streaming |
| `list_models()` | List all available models |
| `get_model_info(provider, model)` | Get model metadata |

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | str | `"google"`, `"anthropic"`, `"openai"`, `"openrouter"`, `"cerebras"`, `"ollama"` |
| `model` | str | Model name (e.g., `"gemini-2.5-flash"`, `"claude-sonnet-4-5-20250929"`) |
| `system_prompt` | str | System instruction |
| `user_text` | str | User input text |
| `messages` | list[dict] | Conversation history from previous `response.history` |
| `file` | str/Path | File path for images or documents |
| `temperature` | float | Randomness (0.0-1.0), default: 0.2 |
| `json_output` | bool | Parse response as JSON, default: False |
| `local` | bool | Use local text extraction (Docling), default: False |

### Classes

- `AIProviderManager` - Manage providers, register custom providers
- `Conversation` - Multi-turn conversation with automatic history management
- `AIProvider` - Abstract base class for implementing custom providers
- Provider classes: `AnthropicProvider`, `GoogleProvider`, `OpenAIProvider`, `OpenrouterProvider`, `OllamaProvider`, `CerebrasProvider`

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions, please open an issue on the GitHub repository.

