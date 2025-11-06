# Multi AI Handler

A unified Python library for interacting with multiple AI providers through a consistent interface. Supports text and file inputs across OpenAI, Anthropic Claude, Google Gemini, and OpenRouter APIs.

## Features

- Unified interface for multiple AI providers
- Support for text-only, file-only, or combined text and file inputs
- Automatic payload formatting for each provider's API requirements
- Support for images and documents (PDF)
- Built-in token usage tracking for Gemini
- Streaming support for Anthropic Claude
- Environment-based API key management

## Supported Providers

- Anthropic Claude
- Google Gemini
- OpenAI
- OpenRouter

## Installation

### Prerequisites

- Python 3.14 or higher
- uv package manager

### Setup

```bash
git clone https://github.com/vsharha/multi-ai-handler.git
cd multi-ai-handler
uv sync
```

## Setup

### 1. Create a `.env` file

Create a `.env` file in your project root with your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Import the library

```python
from multi_ai_handler.ai_handlers import (
    request_anthropic,
    request_gemini,
    request_openai,
    request_openrouter
)
```

## Usage

### Text-only requests

```python
from multi_ai_handler.ai_handlers import request_anthropic

response = request_anthropic(
    system_prompt="You are a helpful assistant.",
    user_text="What is the capital of France?",
    model="claude-3-5-sonnet-20241022",
    temperature=0.7
)
print(response)
```

### Image analysis

```python
import base64
from multi_ai_handler.ai_handlers import request_gemini

# Read and encode image
with open("image.jpg", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode()

response = request_gemini(
    system_prompt="You are an image analysis expert.",
    user_text="Describe what you see in this image.",
    filename="image.jpg",
    encoded_data=encoded_image,
    model="gemini-1.5-flash",
    temperature=0.0
)
print(response)
```

### Document processing

```python
import base64
from multi_ai_handler.ai_handlers import request_anthropic

# Read and encode PDF
with open("document.pdf", "rb") as f:
    encoded_pdf = base64.b64encode(f.read()).decode()

response = request_anthropic(
    system_prompt="You are a document analysis assistant.",
    user_text="Summarize the key points from this document.",
    filename="document.pdf",
    encoded_data=encoded_pdf,
    model="claude-3-5-sonnet-20241022",
    temperature=0.0
)
print(response)
```

### File-only requests (no text)

```python
import base64
from multi_ai_handler.ai_handlers import request_gemini

with open("chart.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode()

# User text can be None when only analyzing a file
response = request_gemini(
    system_prompt="Extract all text and data from images.",
    filename="chart.png",
    encoded_data=encoded_image,
    model="gemini-1.5-pro"
)
print(response)
```

## API Reference

### Common Parameters

All request functions share the following parameters:

- `system_prompt` (str, required): The system instruction for the AI model
- `user_text` (str, optional): The user's text input
- `filename` (str, optional): Name of the file being uploaded (used for MIME type detection)
- `encoded_data` (str, optional): Base64-encoded file data
- `model` (str, required): The specific model to use
- `temperature` (float, optional): Controls randomness (0.0 = deterministic, 1.0 = creative). Default: 0.0

**Note**: Either `user_text` or `filename`/`encoded_data` must be provided.

### `request_anthropic()`

Makes a request to Anthropic's Claude API with streaming support.

```python
def request_anthropic(
    system_prompt: str,
    user_text: str = None,
    filename: str = None,
    encoded_data: str = None,
    model: str = None,
    temperature: float = 0.0
) -> str
```

**Supported file types**: Images (PNG, JPEG, GIF, WebP), Documents (PDF, DOCX, TXT, etc.)

### `request_gemini()`

Makes a request to Google's Gemini API with token usage reporting.

```python
def request_gemini(
    system_prompt: str,
    user_text: str = None,
    filename: str = None,
    encoded_data: str = None,
    model: str = None,
    temperature: float = 0.0
) -> str
```

**Supported file types**: Images, videos, audio, documents

**Note**: Prints token usage (prompt, output, and total tokens) to console.

### `request_openai()`

Makes a request to OpenAI's API.

```python
def request_openai(
    system_prompt: str,
    user_text: str = None,
    filename: str = None,
    encoded_data: str = None,
    model: str = None,
    temperature: float = 0.0,
    link: str = None
) -> str
```

**Supported file types**: Images (PNG, JPEG, GIF, WebP), PDFs (via file API)

**Additional parameter**:
- `link` (str, optional): Custom base URL for API endpoint

### `request_openrouter()`

Makes a request through OpenRouter's unified API.

```python
def request_openrouter(
    system_prompt: str,
    user_text: str = None,
    filename: str = None,
    encoded_data: str = None,
    model: str = None,
    temperature: float = 0.0
) -> str
```

Uses OpenAI-compatible format. Requires `OPENROUTER_API_KEY` in environment.

## Payload Generation

The library automatically formats payloads for each provider using specialized functions:

- `generate_openai_payload()`: Creates OpenAI-compatible content blocks
- `generate_gemini_payload()`: Creates Gemini-compatible parts
- `generate_claude_payload()`: Creates Claude-compatible content blocks

These functions handle:
- MIME type detection from filenames
- Base64 data URL formatting
- Provider-specific content structure
- Validation of required inputs

## Error Handling

The library raises `ValueError` in the following cases:

- Neither `user_text` nor `filename` is provided
- MIME type cannot be detected from filename
- Invalid API credentials

Example:

```python
try:
    response = request_anthropic(
        system_prompt="You are helpful.",
        model="claude-3-5-sonnet-20241022"
    )
except ValueError as e:
    print(f"Error: {e}")
```

## Best Practices

1. **Use specific model names**: Always specify the exact model version (e.g., `claude-3-5-sonnet-20241022`)
2. **Handle errors**: Wrap API calls in try-except blocks
3. **Manage API keys securely**: Never commit `.env` files to version control
4. **Optimize temperature**: Use lower values (0.0-0.3) for factual tasks, higher (0.7-1.0) for creative tasks
5. **Monitor token usage**: Check console output for Gemini token counts

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions, please open an issue on the GitHub repository.

