# DocRAG API Documentation

This document provides detailed information about the DocRAG API endpoints, including how to call them and what they do.

## Base URL

When running locally:
```
http://localhost:8000
```

## Authentication

The API doesn't require authentication for its endpoints, but you need to have a Requesty API key configured in your environment (through .env file or environment variables) to use the LLM services.

## Document Management

### List Documents

Retrieve a list of all documents in the system, with optional filtering.

**Endpoint:** `GET /documents`

**Query Parameters:**
- `category` (optional): Filter by category (academic, business, technical, legal, general)
- `processed` (optional): Filter by processed status (true/false)

**Response:**
```json
[
  {
    "id": "document1",
    "name": "document1.pdf",
    "path": "documents/technical/document1.pdf",
    "category": "technical",
    "size": 1234567,
    "processed": true,
    "date_added": "2025-03-08T10:30:45.123456"
  },
  {
    "id": "document2",
    "name": "document2.pdf",
    "path": "documents/academic/document2.pdf",
    "category": "academic",
    "size": 987654,
    "processed": false,
    "date_added": "2025-03-08T11:15:30.654321"
  }
]
```

**Example cURL:**
```bash
curl -X GET "http://localhost:8000/documents?category=technical&processed=true"
```

### Upload Document

Upload a new document to the system.

**Endpoint:** `POST /documents/upload`

**Form Parameters:**
- `file`: The PDF file to upload (multipart/form-data)
- `category` (optional): The category to place the document in (default: "general")

**Response:**
```json
{
  "id": "document3",
  "name": "document3.pdf",
  "path": "documents/technical/document3.pdf",
  "category": "technical",
  "size": 2345678,
  "processed": false,
  "date_added": "2025-03-08T14:25:10.987654"
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf" \
  -F "category=technical"
```

### Delete Document

Delete a document from the system.

**Endpoint:** `DELETE /documents/{document_id}`

**Path Parameters:**
- `document_id`: The ID of the document to delete

**Response:**
```json
{
  "message": "Document with ID document3 deleted successfully"
}
```

**Example cURL:**
```bash
curl -X DELETE "http://localhost:8000/documents/document3"
```

### Process Document

Start processing a document that has been uploaded but not yet processed for RAG.

**Endpoint:** `POST /documents/{document_id}/process`

**Path Parameters:**
- `document_id`: The ID of the document to process

**Query Parameters:**
- `use_graph` (optional): Whether to build and use semantic graph (default: false)

**Response:**
```json
{
  "message": "Processing started for document documents/technical/document2.pdf",
  "task_id": "process_document2_20250308142822"
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/documents/document2/process?use_graph=false"
```

## RAG System Management

### List RAG Systems

Get a list of all available RAG systems.

**Endpoint:** `GET /rag/systems`

**Response:**
```json
[
  {
    "name": "default",
    "document_count": 10,
    "size_mb": 25.34,
    "description": "Default RAG system",
    "date_created": "2025-03-01T09:15:30.123456"
  },
  {
    "name": "project_x",
    "document_count": 5,
    "size_mb": 12.67,
    "description": "RAG system for Project X",
    "date_created": "2025-03-05T14:30:22.654321"
  }
]
```

**Example cURL:**
```bash
curl -X GET "http://localhost:8000/rag/systems"
```

### Create RAG System

Create a new RAG system.

**Endpoint:** `POST /rag/systems`

**Request Body:**
```json
{
  "name": "project_y",
  "description": "RAG system for Project Y documents"
}
```

**Response:**
```json
{
  "name": "project_y",
  "document_count": 0,
  "size_mb": 0.0,
  "description": "RAG system for Project Y documents",
  "date_created": "2025-03-08T14:35:12.654321"
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/rag/systems" \
  -H "Content-Type: application/json" \
  -d '{"name": "project_y", "description": "RAG system for Project Y documents"}'
```

### Delete RAG System

Delete a RAG system.

**Endpoint:** `DELETE /rag/systems/{system_name}`

**Path Parameters:**
- `system_name`: Name of the RAG system to delete

**Response:**
```json
{
  "message": "System 'project_y' deleted successfully"
}
```

**Example cURL:**
```bash
curl -X DELETE "http://localhost:8000/rag/systems/project_y"
```

## Query Management

### Query Documents

Query documents using a RAG system.

**Endpoint:** `POST /query`

**Request Body:**
```json
{
  "query": "What is the main topic of document X?",
  "system_name": "default",
  "model": "o3-mini",
  "use_graph": false
}
```

**Parameters:**
- `query`: The question to ask about your documents
- `system_name` (optional): RAG system to use (default="default")
- `model` (optional): LLM model to use (default="o3-mini", options: "o3-mini", "deepseek-r1", "deepseek-v3", "claude-3-sonnet", "gpt-4")
- `use_graph` (optional): Whether to use graph-based retrieval (slower but more accurate)

**Response:**
```json
{
  "response": "Based on the document excerpts, the main topic of document X appears to be machine learning algorithms and their applications in natural language processing..."
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of document X?", "system_name": "default", "model": "o3-mini", "use_graph": false}'
```

## Background Tasks

### Get Task Status

Get the status of a background task (like document processing).

**Endpoint:** `GET /tasks/{task_id}`

**Path Parameters:**
- `task_id`: ID of the task to check

**Response:**
```json
{
  "task_id": "process_document2_20250308142822",
  "status": "completed",
  "progress": 100,
  "message": "Document documents/technical/document2.pdf processed successfully"
}
```

**Example cURL:**
```bash
curl -X GET "http://localhost:8000/tasks/process_document2_20250308142822"
```

## Legacy Endpoints

These endpoints are maintained for backward compatibility:

### Process Document (Legacy)

**Endpoint:** `POST /process`

**Request Body:**
```json
{
  "pdf_path": "path/to/document.pdf"
}
```

### Query Document (Legacy)

**Endpoint:** `POST /query`

**Request Body:**
```json
{
  "query": "What is the main topic of the document?",
  "model": "o3-mini",
  "use_graph": false
}
```

### Switch Folder (Legacy)

**Endpoint:** `POST /switch_folder`

**Request Body:**
```json
{
  "new_folder": "path/to/new/folder"
}
```

## Running the API Server

You can start the API server with the following command:

```bash
python main_improved.py --host 0.0.0.0 --port 8000 --model o3-mini
```

Additional command-line parameters:

- `--host`: Host to bind the server to (default: "0.0.0.0")
- `--port`: Port to bind the server to (default: 8000)
- `--model`: Default LLM model to use (default: "o3-mini")

## Example Usage Scenarios

### Scenario 1: Upload and Query a Technical Document

1. **Upload a document:**
   ```bash
   curl -X POST "http://localhost:8000/documents/upload" \
     -F "file=@technical_spec.pdf" \
     -F "category=technical"
   ```

2. **Process the document:**
   ```bash
   curl -X POST "http://localhost:8000/documents/technical_spec/process"
   ```

3. **Check processing status:**
   ```bash
   curl -X GET "http://localhost:8000/tasks/process_technical_spec_20250308150000"
   ```

4. **Query the document:**
   ```bash
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the key technical specifications?", "model": "o3-mini"}'
   ```

### Scenario 2: Create a New RAG System for a Project

1. **Create a new RAG system:**
   ```bash
   curl -X POST "http://localhost:8000/rag/systems" \
     -H "Content-Type: application/json" \
     -d '{"name": "project_z", "description": "Project Z documentation"}'
   ```

2. **Upload documents to the system:**
   ```bash
   curl -X POST "http://localhost:8000/documents/upload" \
     -F "file=@project_z_doc1.pdf" \
     -F "category=business"
   ```

3. **Process each document:**
   ```bash
   curl -X POST "http://localhost:8000/documents/project_z_doc1/process"
   ```

4. **Query the custom RAG system:**
   ```bash
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Project Z's timeline?", "system_name": "project_z", "model": "deepseek-r1"}'
   ```

## Performance Considerations

- Use the `o3-mini` model for fastest responses
- Set `use_graph=false` for faster document processing and querying
- For higher quality responses (at the cost of speed), set `use_graph=true` and use models like `deepseek-v3`