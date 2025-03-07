# DocRAG - Improved Document Retrieval Augmented Generation System

## Overview

DocRAG is a system for processing PDF documents and answering questions about them using Retrieval Augmented Generation (RAG). This improved version includes a comprehensive FastAPI implementation with enhanced document management, flexible RAG system organization, and better API structure.

## New Features

- **Better Document Organization**
  - Documents are now organized in category folders (academic, business, technical, legal, general)
  - Direct upload functionality through API endpoints
  - Document management with CRUD operations

- **Enhanced RAG System Management**
  - Support for multiple independent RAG systems
  - System-specific configurations and queries
  - Background processing for document ingestion

- **Improved API Structure**
  - RESTful API design with proper resource organization
  - Detailed API documentation with Swagger UI
  - Asynchronous processing with task tracking

## Folder Structure

```
DocRAG/
│
├── documents/               # Document storage organized by category
│   ├── academic/
│   ├── business/
│   ├── technical/
│   ├── legal/
│   └── general/
│
├── uploads/                 # Temporary storage for uploaded documents
│
├── rag_data/                # RAG system data
│   ├── default/             # Default RAG system
│   └── [custom_systems]/    # User-created RAG systems
│
├── main_improved.py         # New FastAPI implementation
├── main.py                  # Original FastAPI implementation (for compatibility)
├── document_rag.py          # RAG system implementation
├── pdf_utils.py             # PDF processing utilities
├── text_processing.py       # Text processing utilities
├── document_checker.py      # Document status checker
└── llm_client.py            # LLM API client
```

## API Documentation

### Document Management

#### List Documents
```
GET /documents
```
List all documents or filter by category and processed status.

Query parameters:
- `category` (optional): Filter by category (academic, business, technical, legal, general)
- `processed` (optional): Filter by processed status (true/false)

#### Upload Document
```
POST /documents/upload
```
Upload a document to the specified category.

Form parameters:
- `file`: PDF file to upload
- `category` (optional, default='general'): Document category

#### Delete Document
```
DELETE /documents/{document_id}
```
Delete a document by ID.

#### Process Document
```
POST /documents/{document_id}/process
```
Process a document by ID. This starts a background task to process the document.

Response includes a `task_id` that can be used to track progress.

### RAG System Management

#### List RAG Systems
```
GET /rag/systems
```
List all available RAG systems.

#### Create RAG System
```
POST /rag/systems
```
Create a new RAG system.

Request body:
```json
{
  "name": "my_system",
  "description": "Description of my system"
}
```

#### Delete RAG System
```
DELETE /rag/systems/{system_name}
```
Delete a RAG system.

### Query Endpoints

#### Query Document
```
POST /query
```
Query documents using the RAG system.

Request body:
```json
{
  "query": "What is the main topic of the document?",
  "system_name": "default"
}
```

### Background Tasks

#### Get Task Status
```
GET /tasks/{task_id}
```
Get the status of a background task.

## Running the Application

### Prerequisites

- Python 3.8+
- Dependencies from requirements.txt

### Installation

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys (see `.env.example.txt`)

### Starting the API Server

```
python main_improved.py
```

Access the Swagger UI documentation at: http://localhost:8000/docs

## Migration Guide

If you've been using the original DocRAG system, here's how to migrate:

1. Your existing documents in the `rag_data` directory will automatically be recognized as the "default" RAG system
2. Legacy endpoints (/process, /query, /switch_folder) remain available for backward compatibility
3. For best results, move your existing PDF documents to the appropriate category folders under `documents/`

## Example Usage

### Adding a Document and Querying

1. Upload a document:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/documents/upload' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@document.pdf' \
     -F 'category=technical'
   ```

2. Process the document (replace with actual document ID):
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/documents/document_id/process' \
     -H 'accept: application/json'
   ```

3. Check task status (replace with actual task ID):
   ```bash
   curl -X 'GET' \
     'http://localhost:8000/tasks/task_id' \
     -H 'accept: application/json'
   ```

4. Query the document:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/query' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
       "query": "What is the main topic of the document?",
       "system_name": "default"
     }'
   ```

### Creating a New RAG System

1. Create a new system:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/rag/systems' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
       "name": "project_x",
       "description": "RAG system for Project X documents"
     }'
   ```

2. When processing documents or querying, specify the system name:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/query' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
       "query": "What is the main topic of the document?",
       "system_name": "project_x"
     }'