import os
import argparse
import logging
import shutil
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Import document_checker for lightweight operations
from document_checker import is_document_processed, get_processed_documents, system_exists
from llm_client import RequestyLLMClient, example_api_call
from pdf_utils import extract_text_from_pdf, collect_pdf_paths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default directories
DEFAULT_DOCUMENTS_DIR = "documents"
DEFAULT_RAG_DATA_DIR = "rag_data"
DEFAULT_UPLOADS_DIR = "uploads"

# Ensure directories exist
os.makedirs(DEFAULT_DOCUMENTS_DIR, exist_ok=True)
os.makedirs(DEFAULT_RAG_DATA_DIR, exist_ok=True)
os.makedirs(DEFAULT_UPLOADS_DIR, exist_ok=True)

# Define document categories
DOCUMENT_CATEGORIES = ["academic", "business", "technical", "legal", "general"]
for category in DOCUMENT_CATEGORIES:
    os.makedirs(os.path.join(DEFAULT_DOCUMENTS_DIR, category), exist_ok=True)

# Background tasks list to track progress
active_tasks = {}

# Pydantic models for request/response
class DocumentRequest(BaseModel):
    pdf_path: str
    
class UploadRequest(BaseModel):
    category: str = Field("general", description="Document category")
    
class DocumentResponse(BaseModel):
    id: str
    name: str
    path: str
    category: str
    size: int
    processed: bool
    date_added: str
    
class QueryRequest(BaseModel):
    query: str
    system_name: Optional[str] = None
    
class SystemRequest(BaseModel):
    name: str
    description: Optional[str] = None
    
class RAGSystemResponse(BaseModel):
    name: str
    document_count: int
    size_mb: float
    description: Optional[str] = None
    date_created: str
    
class TaskResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str

# Utility functions
def get_file_info(file_path):
    """Get file information as a dictionary"""
    path = Path(file_path)
    return {
        "id": str(path.stem),
        "name": path.name,
        "path": str(path),
        "category": path.parent.name if path.parent.name in DOCUMENT_CATEGORIES else "general",
        "size": path.stat().st_size,
        "processed": is_document_processed(str(path), DEFAULT_RAG_DATA_DIR),
        "date_added": datetime.fromtimestamp(path.stat().st_ctime).isoformat()
    }

def process_document_task(task_id, pdf_path, rag_dir=DEFAULT_RAG_DATA_DIR):
    """Background task to process a document"""
    try:
        active_tasks[task_id] = {"status": "processing", "progress": 0, "message": "Starting document processing"}
        
        # Update progress
        active_tasks[task_id]["progress"] = 10
        active_tasks[task_id]["message"] = "Loading document"
        
        # Import document_rag only when needed
        from document_rag import DocumentRAGSystem
        
        # Try to load existing system
        try:
            active_tasks[task_id]["progress"] = 20
            active_tasks[task_id]["message"] = "Loading RAG system"
            rag_system = DocumentRAGSystem(load_from=rag_dir)
            logging.info(f"Loaded existing system from {rag_dir}")
        except FileNotFoundError:
            active_tasks[task_id]["message"] = "Creating new RAG system"
            rag_system = DocumentRAGSystem()
            logging.info(f"Created new RAG system")
        
        active_tasks[task_id]["progress"] = 40
        active_tasks[task_id]["message"] = f"Processing document {pdf_path}"
        
        # Add document to the system
        rag_system.add_document(pdf_path, save_directory=rag_dir)
        
        active_tasks[task_id]["progress"] = 90
        active_tasks[task_id]["message"] = "Saving system state"
        
        active_tasks[task_id] = {"status": "completed", "progress": 100, "message": f"Document {pdf_path} processed successfully"}
        logging.info(f"Document {pdf_path} processed successfully")
    except Exception as e:
        active_tasks[task_id] = {"status": "failed", "progress": 0, "message": f"Error: {str(e)}"}
        logging.error(f"Error processing document {pdf_path}: {str(e)}")

# Create FastAPI app
app = FastAPI(
    title="Document RAG API",
    description="API for Document Retrieval Augmented Generation System",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create routers
documents_router = APIRouter(prefix="/documents", tags=["Documents"])
rag_router = APIRouter(prefix="/rag", tags=["RAG Systems"])
query_router = APIRouter(prefix="/query", tags=["Queries"])
tasks_router = APIRouter(prefix="/tasks", tags=["Background Tasks"])

# Document management endpoints
@documents_router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    category: Optional[str] = Query(None, description="Filter by category"),
    processed: Optional[bool] = Query(None, description="Filter by processed status")
):
    """List all documents or filter by category and processed status"""
    try:
        documents = []
        search_dir = os.path.join(DEFAULT_DOCUMENTS_DIR, category) if category else DEFAULT_DOCUMENTS_DIR
        
        # Collect all pdf files in the directory
        pdf_files = []
        if os.path.exists(search_dir):
            if category:
                # If a specific category is requested
                pdf_files.extend([os.path.join(search_dir, f) for f in os.listdir(search_dir) 
                                if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(search_dir, f))])
            else:
                # If all categories are requested
                for cat in DOCUMENT_CATEGORIES:
                    cat_dir = os.path.join(search_dir, cat)
                    if os.path.exists(cat_dir):
                        pdf_files.extend([os.path.join(cat_dir, f) for f in os.listdir(cat_dir) 
                                        if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(cat_dir, f))])
        
        # Apply processed filter if specified
        for pdf_file in pdf_files:
            file_info = get_file_info(pdf_file)
            if processed is None or file_info["processed"] == processed:
                documents.append(file_info)
                
        return documents
    except Exception as e:
        logging.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@documents_router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("general")
):
    """Upload a document to the specified category"""
    try:
        # Validate category
        if category not in DOCUMENT_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {', '.join(DOCUMENT_CATEGORIES)}")
            
        # Save uploaded file to the uploads directory first
        temp_path = os.path.join(DEFAULT_UPLOADS_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Check if it's a valid PDF
        try:
            extract_text_from_pdf(temp_path)
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")
            
        # Move to the appropriate category folder
        dest_path = os.path.join(DEFAULT_DOCUMENTS_DIR, category, file.filename)
        shutil.move(temp_path, dest_path)
        
        # Return file info
        return get_file_info(dest_path)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@documents_router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document by ID"""
    try:
        # Find the document across all categories
        found = False
        for category in DOCUMENT_CATEGORIES:
            category_dir = os.path.join(DEFAULT_DOCUMENTS_DIR, category)
            for file in os.listdir(category_dir):
                if file.lower().endswith('.pdf') and Path(file).stem == document_id:
                    file_path = os.path.join(category_dir, file)
                    os.remove(file_path)
                    found = True
                    break
            if found:
                break
                
        if not found:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
            
        return {"message": f"Document with ID {document_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@documents_router.post("/{document_id}/process")
async def process_document(document_id: str, background_tasks: BackgroundTasks):
    """Process a document by ID"""
    try:
        # Find the document across all categories
        document_path = None
        for category in DOCUMENT_CATEGORIES:
            category_dir = os.path.join(DEFAULT_DOCUMENTS_DIR, category)
            if not os.path.exists(category_dir):
                continue
                
            for file in os.listdir(category_dir):
                if file.lower().endswith('.pdf') and Path(file).stem == document_id:
                    document_path = os.path.join(category_dir, file)
                    break
            if document_path:
                break
                
        if not document_path:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
            
        # Check if document is already processed
        if is_document_processed(document_path, DEFAULT_RAG_DATA_DIR):
            return {"message": f"Document {document_path} is already processed."}
            
        # Create task ID
        task_id = f"process_{document_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Start background processing
        background_tasks.add_task(process_document_task, task_id, document_path, DEFAULT_RAG_DATA_DIR)
        
        # Initialize task status
        active_tasks[task_id] = {"status": "queued", "progress": 0, "message": "Task queued"}
        
        return {"message": f"Processing started for document {document_path}", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# RAG System management endpoints
@rag_router.get("/systems", response_model=List[RAGSystemResponse])
async def list_rag_systems():
    """List all available RAG systems"""
    try:
        systems = []
        rag_root = Path(DEFAULT_RAG_DATA_DIR)
        
        # Check the base RAG directory
        if system_exists(str(rag_root)):
            # Get document count and size
            document_count = len(get_processed_documents(str(rag_root)))
            size_mb = sum(f.stat().st_size for f in rag_root.glob('**/*') if f.is_file()) / (1024 * 1024)
            
            systems.append({
                "name": "default",
                "document_count": document_count,
                "size_mb": round(size_mb, 2),
                "description": "Default RAG system",
                "date_created": datetime.fromtimestamp(rag_root.stat().st_ctime).isoformat()
            })
        
        # Check subdirectories that might be RAG systems
        for subdir in rag_root.iterdir():
            if subdir.is_dir() and system_exists(str(subdir)):
                document_count = len(get_processed_documents(str(subdir)))
                size_mb = sum(f.stat().st_size for f in subdir.glob('**/*') if f.is_file()) / (1024 * 1024)
                
                systems.append({
                    "name": subdir.name,
                    "document_count": document_count,
                    "size_mb": round(size_mb, 2),
                    "description": None,
                    "date_created": datetime.fromtimestamp(subdir.stat().st_ctime).isoformat()
                })
                
        return systems
    except Exception as e:
        logging.error(f"Error listing RAG systems: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing RAG systems: {str(e)}")

@rag_router.post("/systems", response_model=RAGSystemResponse)
async def create_rag_system(system: SystemRequest):
    """Create a new RAG system"""
    try:
        if system.name == "default":
            raise HTTPException(status_code=400, detail="Cannot create a system named 'default'")
            
        # Create the directory for the new system
        system_dir = os.path.join(DEFAULT_RAG_DATA_DIR, system.name)
        
        if os.path.exists(system_dir):
            raise HTTPException(status_code=400, detail=f"System with name '{system.name}' already exists")
            
        os.makedirs(system_dir, exist_ok=True)
        
        # Initialize an empty RAG system
        from document_rag import DocumentRAGSystem
        rag_system = DocumentRAGSystem()
        rag_system.save_system_state(system_dir)
        
        return {
            "name": system.name,
            "document_count": 0,
            "size_mb": 0.0,
            "description": system.description,
            "date_created": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating RAG system: {str(e)}")

@rag_router.delete("/systems/{system_name}")
async def delete_rag_system(system_name: str):
    """Delete a RAG system"""
    try:
        if system_name == "default":
            raise HTTPException(status_code=400, detail="Cannot delete the default system")
            
        system_dir = os.path.join(DEFAULT_RAG_DATA_DIR, system_name)
        
        if not os.path.exists(system_dir):
            raise HTTPException(status_code=404, detail=f"System with name '{system_name}' not found")
            
        # Delete the system directory
        shutil.rmtree(system_dir)
        
        return {"message": f"System '{system_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting RAG system: {str(e)}")

# Query endpoints
@query_router.post("/")
async def query_document(request: QueryRequest):
    """Query documents using the RAG system"""
    try:
        # Determine which RAG system to use
        rag_dir = DEFAULT_RAG_DATA_DIR
        if request.system_name and request.system_name != "default":
            rag_dir = os.path.join(DEFAULT_RAG_DATA_DIR, request.system_name)
            
        if not system_exists(rag_dir):
            raise HTTPException(status_code=404, detail=f"RAG system not found or not initialized in {rag_dir}")
            
        # Import and load RAG system
        from document_rag import DocumentRAGSystem
        rag_system = DocumentRAGSystem(load_from=rag_dir)
        
        # Get API key from environment
        api_key = os.getenv('REQUESTY_API_KEY')
        if not api_key:
            raise HTTPException(status_code=400, detail="API key not found in environment variables.")
            
        # Initialize LLM client
        llm_client = RequestyLLMClient(api_key=api_key)
        
        # Generate response
        response = rag_system.generate_response(
            query=request.query,
            api_call_function=lambda prompt: llm_client.generate_response(prompt)
        )
        
        return {"response": response}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error querying document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

# Task status endpoints
@tasks_router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
        
    task_info = active_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task_info.get("status", "unknown"),
        "progress": task_info.get("progress", 0),
        "message": task_info.get("message", "")
    }

# Register routers
app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(query_router)
app.include_router(tasks_router)

# Legacy endpoints for backward compatibility
@app.post("/process")
async def process_document_legacy(request: DocumentRequest):
    """Legacy endpoint for processing documents"""
    try:
        pdf_path = request.pdf_path
        if is_document_processed(pdf_path, DEFAULT_RAG_DATA_DIR):
            return {"message": f"Document {pdf_path} is already processed."}
        else:
            from document_rag import DocumentRAGSystem
            rag_system = DocumentRAGSystem(load_from=DEFAULT_RAG_DATA_DIR)
            rag_system.add_document(pdf_path, save_directory=DEFAULT_RAG_DATA_DIR)
            return {"message": f"Document {pdf_path} added to system."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document_legacy(request: QueryRequest):
    """Legacy endpoint for querying documents"""
    try:
        query = request.query
        from document_rag import DocumentRAGSystem
        rag_system = DocumentRAGSystem(load_from=DEFAULT_RAG_DATA_DIR)
        api_key = os.getenv('REQUESTY_API_KEY')
        if not api_key:
            raise HTTPException(status_code=400, detail="API key not found in environment variables.")
        llm_client = RequestyLLMClient(api_key=api_key)
        response = rag_system.generate_response(query=query, api_call_function=lambda prompt: llm_client.generate_response(prompt))
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch_folder")
async def switch_folder_legacy(request: BaseModel):
    """Legacy endpoint for switching folders"""
    try:
        new_folder = request.new_folder
        from document_rag import DocumentRAGSystem
        rag_system = DocumentRAGSystem(load_from=DEFAULT_RAG_DATA_DIR)
        rag_system.switch_rag_folder(new_folder)
        return {"message": f"Switched to new RAG folder: {new_folder}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Document RAG System API')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    args = parser.parse_args()
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()