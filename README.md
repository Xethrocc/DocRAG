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
  - tqdm
  - fastapi
  - uvicorn

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
- `--data_dir`: Directory for storing processed documents (default: "rag_data")
- `--add_document`: Add a single document to the existing system
- `--force_reprocess`: Force reprocessing of documents even if already processed

### Example

```bash
python main.py --pdf document1.pdf document2.pdf --query "What are the main topics covered?" --model gpt-4
```

### Document Persistence

The system now supports persistent storage of processed documents, allowing you to:

1. Save processed documents to disk
2. Load previously processed documents without reprocessing
3. Check if a document has already been processed
4. Add new documents to an existing system

#### Basic Persistence Usage

```bash
# Process documents and save to default location (rag_data/)
python main.py --pdf document1.pdf document2.pdf

# Later, query the same documents without reprocessing
python main.py --query "What are the key findings?"

# Add a new document to the existing system
python main.py --add_document document3.pdf

# Specify a custom location for document storage
python main.py --pdf document1.pdf --data_dir my_documents

# Force reprocessing of documents even if already processed
python main.py --pdf document1.pdf --force_reprocess
```

#### Example Script

An example script demonstrating the persistence features is included:

```bash
# Run the example persistence script
python example_persistence.py
```

This script will:
1. Check if a saved system exists and load it
2. If no system exists, create a new one with example documents
3. Allow you to add a new document to the system
4. Let you query the documents

## Using as a Library

You can also use the Document RAG system as a library in your own Python code:

```python
from document_rag import DocumentRAGSystem
from llm_client import RequestyLLMClient

# Option 1: Initialize a new RAG system
rag_system = DocumentRAGSystem(
    docs_directory="./documents",
    pdf_paths=["important_doc.pdf"]
    # embedding_model and max_context_tokens can be set in .env file
)

# Save the processed documents for future use
rag_system.save_system_state("my_rag_data")

# Option 2: Load a previously saved RAG system
rag_system = DocumentRAGSystem(load_from="my_rag_data")

# Add a new document to the system
rag_system.add_document("new_document.pdf", save_directory="my_rag_data")

# Check if a document is already processed
is_processed = rag_system.is_document_processed("document.pdf", "my_rag_data")

# Initialize the LLM client with your Requesty.ai API key
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

### Document Persistence Implementation

The system uses a hybrid approach for document persistence:

1. **Metadata (JSON)**: Document paths and basic information are stored in a JSON file (`metadata.json`).
2. **FAISS Index**: The vector index is stored as a binary file (`document_index.faiss`).
3. **Text Chunks**: Original text chunks are stored using pickle (`index_texts.pkl`).
4. **Document Data**: Each document's chunks and embeddings are stored in separate pickle files.

This approach provides:
- Fast loading and saving of the system state
- Ability to check if documents are already processed
- Incremental updates by adding new documents
- Separation of concerns for easier debugging and maintenance

You can customize the persistence behavior by modifying the following methods in `DocumentRAGSystem`:
- `save_system_state`: Controls how data is saved to disk
- `load_system_state`: Controls how data is loaded from disk
- `is_document_processed`: Checks if a document is already in the system

### Error Handling

The system has been improved with more specific exception types and logging for better error handling. This ensures that errors are properly logged and can be debugged more easily.

### Performance Optimization

The `split_text` function in `text_processing.py` has been optimized for better performance. This ensures that text chunking is done efficiently, even for large documents.

### Handling Large PDFs

To handle large PDFs efficiently, the system processes documents in batches and uses a progress bar to track the processing progress. This helps in optimizing memory usage and provides a visual indication of the progress.

### Using Progress Bar

The `tqdm` library is used to add a progress bar in the `process_documents` method. This helps in tracking the progress of processing large PDFs.

### Unit Tests

Unit tests have been added for all major functions to ensure code reliability. This includes tests for the following functions:

- `document_checker.py`: `is_document_processed`, `get_processed_documents`, `system_exists`
- `document_rag.py`: `process_documents`, `add_document`, `create_faiss_index`, `retrieve_relevant_context`, `prepare_prompt_with_context`, `truncate_to_max_tokens`, `generate_response`, `save_system_state`, `load_system_state`, `is_document_processed`
- `example_persistence.py`: Example script
- `llm_client.py`: `RequestyLLMClient`, `example_api_call`
- `main.py`: Main entry point
- `pdf_utils.py`: `extract_text_from_pdf`, `collect_pdf_paths`
- `text_processing.py`: `split_text`

These tests ensure that the system functions correctly and can handle various edge cases.

### New Behavior: Skipping Transformer Model Initialization

The `DocumentRAGSystem` class now checks if documents are already processed before initializing the transformer model. This optimization helps in reducing unnecessary loading of the transformer model, improving the system's efficiency.

#### Updated Initialization

The `__init__` method of `DocumentRAGSystem` now uses the `is_document_processed` function from `document_checker.py` to determine if the transformer model should be loaded. If all documents are already processed, the transformer model is not loaded during initialization.

#### Ensuring Model Loading in `add_document`

The `add_document` method in `document_rag.py` still checks if a document is already processed before adding it, but now it also ensures the transformer model is loaded if needed.

### API Usage with FastAPI

The system now includes a REST API using FastAPI to handle inputs and outputs. This allows other programs to interact with the Document RAG system through standardized API endpoints.

#### Installation

To use the FastAPI server, you need to install the following additional dependencies:

```bash
pip install fastapi uvicorn
```

#### Running the FastAPI Server

You can run the FastAPI server using the following command:

```bash
python main.py --run_server
```

The server will start and listen for requests on `http://0.0.0.0:8000`.

#### API Endpoints

The FastAPI server provides the following endpoints:

- `POST /process`: Process a new document and add it to the system
- `POST /query`: Query the system with a search query
- `POST /switch_folder`: Switch the RAG folder

#### Example Usage

To process a new document, send a POST request to `/process` with the path to the PDF document:

```bash
curl -X POST "http://0.0.0.0:8000/process" -H "Content-Type: application/json" -d '{"pdf_path": "path/to/document.pdf"}'
```

To query the system, send a POST request to `/query` with the search query:

```bash
curl -X POST "http://0.0.0.0:8000/query" -H "Content-Type: application/json" -d '{"query": "Your question about the documents?"}'
```

To switch the RAG folder, send a POST request to `/switch_folder` with the path to the new folder:

```bash
curl -X POST "http://0.0.0.0:8000/switch_folder" -H "Content-Type: application/json" -d '{"new_folder": "path/to/new/folder"}'
```

### Advanced Metadata Extraction Techniques

The system now includes advanced metadata extraction techniques to enhance document processing:

- **Named Entity Recognition (NER)**: Extracts entities such as names, dates, locations, and organizations from the text.
- **Topic Modeling**: Identifies the main topics within each document using techniques like Latent Dirichlet Allocation (LDA).
- **Sentiment Analysis**: Determines the overall sentiment of each document or specific sections.
- **Keyword Extraction**: Identifies important keywords and phrases within the document.

### Enhanced Metadata Storage and Retrieval

The system has been enhanced to support advanced metadata storage and retrieval:

- **Database Integration**: Stores metadata in a structured database (e.g., SQLite, PostgreSQL) to enable efficient querying and retrieval.
- **Metadata Versioning**: Implements versioning for metadata to track changes and updates over time.
- **Metadata Enrichment**: Integrates external data sources (e.g., Wikipedia, DBpedia) to enrich the metadata with additional information.
