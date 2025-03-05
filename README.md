# Document RAG System

A Retrieval Augmented Generation (RAG) system for processing PDF documents and answering queries based on their content.

## Project Structure

The project has been organized into the following modules:

- `document_rag.py`: Main RAG system implementation
- `llm_client.py`: Client for interacting with LLM APIs
- `text_processing.py`: Utilities for text processing and chunking
- `pdf_utils.py`: Utilities for PDF handling
- `main.py`: Command-line interface and entry point
- `.env`: Environment variables configuration
- `.gitignore`: Git ignore configuration

## Requirements

- Python 3.7+
- Dependencies (install via `pip install -r requirements.txt`):
  - sentence-transformers
  - faiss-cpu (or faiss-gpu)
  - PyPDF2
  - tiktoken
  - openai
  - requests
  - numpy
  - python-dotenv

## Requesty.ai Integration

This project uses [Requesty.ai](https://requesty.ai) as a router to access various LLM providers through a unified API. Requesty.ai allows you to use models from OpenAI, Anthropic, and other providers with a single API key. (Since Requesty.ai uses OpenAI's API, you can use it with your OpenAI API key. You just have to switch the base URL.)

### Getting a Requesty.ai API Key

1. Create an account at [https://app.requesty.ai/sign-up](https://app.requesty.ai/sign-up)
2. Set up an API key at [https://app.requesty.ai/router](https://app.requesty.ai/router)
3. Copy your API key and add it to your `.env` file

## Environment Variables

The system can be configured using environment variables in a `.env` file:

```
# API Keys
REQUESTY_API_KEY=your_api_key_here

# Model Configuration
DEFAULT_MODEL=deepseek-v3  # Options: claude-3-sonnet, gpt-4, deepseek-v3

# System Configuration
MAX_CONTEXT_TOKENS=3500
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Model IDs

The following model IDs are used for each option:

- `claude-3-sonnet`: "claude-3-sonnet-20240229" (Anthropic)
- `gpt-4`: "gpt-4-turbo-2024-04-09" (OpenAI)
- `deepseek-v3`: "deepinfra/deepseek-ai/DeepSeek-V3" (DeepSeek)

## Usage

### Basic Usage

```bash
python main.py --docs_dir /path/to/documents --query "Your question about the documents?"
```

### Command Line Arguments

- `--docs_dir`: Directory containing PDF documents
- `--pdf`: One or more paths to PDF documents
- `--query`: Search query
- `--api_key`: Requesty API key (can also be set in .env file)
- `--model`: LLM model to use (can also be set in .env file)
  - `claude-3-sonnet`: Anthropic's Claude 3 Sonnet model
  - `gpt-4`: OpenAI's GPT-4 Turbo model
  - `deepseek-v3`: DeepSeek's DeepSeek-V3 model (deepinfra/deepseek-ai/DeepSeek-V3)

### Example

```bash
python main.py --pdf document1.pdf document2.pdf --query "What are the main topics covered?" --model gpt-4
```

## Using as a Library

You can also use the Document RAG system as a library in your own Python code:

```python
from document_rag import DocumentRAGSystem
from llm_client import RequestyLLMClient

# Initialize the RAG system
# Configuration can be provided directly or via environment variables
rag_system = DocumentRAGSystem(
    docs_directory="./documents",
    pdf_paths=["important_doc.pdf"]
    # embedding_model and max_context_tokens can be set in .env file
)

# Initialize the LLM client with your Requesty.ai API key
# API key and default model can be provided directly or via environment variables
llm_client = RequestyLLMClient(api_key="your-requesty-api-key", default_model="gpt-4")

# Generate a response
response = rag_system.generate_response(
    query="What are the key findings?",
    api_call_function=lambda prompt: llm_client.generate_response(prompt)
)

print(response)
```

Note: The LLM client now uses the OpenAI SDK with Requesty.ai's router URL.

## Extending the System

### Adding New LLM Providers

To add support for new LLM providers, update the `AVAILABLE_MODELS` dictionary in the `RequestyLLMClient` class. Requesty.ai supports various providers, and you can specify the provider in the model configuration.

### How the Requesty.ai Integration Works

The system uses the OpenAI SDK with a custom base URL (`https://router.requesty.ai/v1`) to route requests to different LLM providers. The provider is specified in the `extra_body` parameter of the API request.

### Custom Text Processing

You can customize the text chunking algorithm by modifying the `split_text` function in `text_processing.py`.