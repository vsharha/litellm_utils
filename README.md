# litellm_utils

A lightweight Python wrapper around [LiteLLM](https://github.com/BerriAI/litellm) that provides a simplified interface for interacting with multiple AI providers. Supports text and file inputs across 100+ LLMs including OpenAI, Anthropic Claude, Google Gemini, and more.

## Features

- Simple unified interface built on LiteLLM
- **Conversation history** for multi-turn interactions
- **Streaming support** for real-time token output
- Support for images and documents (PDF, single or multiple files)
- Advanced document processing with Docling (OCR, table extraction)
- Automatic PDF handling based on model capabilities
- Access to 100+ LLMs through LiteLLM

## Installation

```bash
pip install litellm_utils
```

**Optional dependencies:**
```bash
pip install litellm_utils[docling]  # Document processing (OCR, tables)
pip install litellm_utils[all]      # All optional dependencies
```

## Setup

Configure your API keys as environment variables. LiteLLM supports many providers:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
# ... and many more providers supported by LiteLLM
```

See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for all supported providers.

## Usage

### Basic Request

```python
from litellm_utils import request_ai

response = request_ai(
    provider="gemini",  # or "anthropic", "openai", "openrouter", "cerebras", "ollama"
    model="gemini-2.5-flash",
    system_prompt="You are a helpful assistant.",
    user_text="What is the capital of France?"
)
```

### JSON Output

```python
data = request_ai(
    provider="openai",
    model="gpt-4o-mini",
    system_prompt="Return valid JSON only.",
    user_text="Convert to JSON: Name: Alice, Age: 25",
    json_output=True
)
# Returns: {'name': 'Alice', 'age': 25}
```

### File Processing

Single file:
```python
response = request_ai(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    system_prompt="Summarize this document.",
    file="document.pdf"
)
```

Multiple files:
```python
response = request_ai(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    system_prompt="Compare these documents.",
    file=["document1.pdf", "document2.pdf", "image.jpg"]
)
```

### Streaming

```python
from litellm_utils import stream_ai

for chunk in stream_ai(
    provider="openai",
    model="gpt-4o-mini",
    user_text="Write a poem about Python"
):
    print(chunk, end="", flush=True)
```

### Conversation History

Use the `Conversation` class for multi-turn interactions:

```python
from litellm_utils import Conversation

conv = Conversation(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    system_prompt="You are a helpful assistant."
)

response = conv.send("My name is Alice.")
print(response)

response = conv.send("What's my name?")  # Remembers context
print(response)

conv.clear_history()  # Reset conversation
```

With streaming:

```python
conv = Conversation(provider="openai", model="gpt-4o-mini")

for chunk in conv.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

With file processing:

```python
conv = Conversation(provider="gemini", model="gemini-2.0-flash")

# Single file
response = conv.send("Summarize this document", file="report.pdf")
print(response)

# Multiple files
response = conv.send("Compare these documents", file=["doc1.pdf", "doc2.pdf"])
print(response)

response = conv.send("What are the key findings?")  # Follows up on context
print(response)
```

### Model Information

```python
from litellm_utils import list_models

# List models for a specific provider
openai_models = list_models("openai")
anthropic_models = list_models("anthropic")
```

## API Reference

### Functions

#### `request_ai(provider, model, **kwargs)`
Generate a response from an AI model.

**Parameters:**
- `provider` (str): Provider name (e.g., `"openai"`, `"anthropic"`, `"gemini"`)
- `model` (str): Model name (e.g., `"gpt-4o-mini"`, `"claude-sonnet-4-20250514"`)
- `system_prompt` (str, optional): System instruction
- `user_text` (str, optional): User input text
- `messages` (list[dict], optional): Conversation history in OpenAI format
- `file` (str/Path/list, optional): Path to image or document file, or list of file paths for multiple files
- `temperature` (float, optional): Sampling temperature (0.0-1.0), default: 0.2
- `json_output` (bool, optional): Parse response as JSON, default: False
- `preprocess_file_content` (bool, optional): Use Docling for document processing, default: False (auto-detected)

**Returns:** `str` or `dict` (if `json_output=True`)

#### `stream_ai(provider, model, **kwargs)`
Stream response tokens from an AI model.

**Parameters:** Same as `request_ai()` except `json_output`

**Returns:** `Iterator[str]`

#### `list_models(provider)`
List available models for a provider.

**Parameters:**
- `provider` (str): Provider name

**Returns:** `list[dict]`

### Classes

#### `Conversation`
Multi-turn conversation with automatic history management.

**Methods:**
- `__init__(provider, model, system_prompt=None, temperature=0.2)` - Initialize conversation
- `send(user_text, file=None, json_output=False, preprocess_file_content=False)` - Send a message
- `stream(user_text, file=None, preprocess_file_content=False)` - Stream a response
- `get_history()` - Get conversation history
- `clear_history()` - Clear conversation history
- `set_system_prompt(system_prompt)` - Update system prompt

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions, please open an issue on the GitHub repository.

