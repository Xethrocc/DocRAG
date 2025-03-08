import os
import argparse
import logging
import time
from dotenv import load_dotenv
from llm_client import RequestyLLMClient, example_api_call
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Import document_checker for lightweight operations
from document_checker import is_document_processed, get_processed_documents, system_exists

# Default directory for storing processed documents
DEFAULT_RAG_DATA_DIR = "rag_data"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_document_status(pdf_path, data_dir=DEFAULT_RAG_DATA_DIR):
    """
    Check if a document is already processed without loading TensorFlow
    
    Parameters:
    pdf_path (str): Path to the PDF document
    data_dir (str): Directory containing the system state
    
    Returns:
    bool: True if the document is already processed, False otherwise
    """
    return is_document_processed(pdf_path, data_dir)

class DocumentRequest(BaseModel):
    pdf_path: str

class QueryRequest(BaseModel):
    query: str

class SwitchFolderRequest(BaseModel):
    new_folder: str

def create_app():
    app = FastAPI()

    @app.post("/process")
    async def process_document(request: DocumentRequest):
        try:
            pdf_path = request.pdf_path
            if is_document_processed(pdf_path, DEFAULT_RAG_DATA_DIR):
                return {"message": f"Document {pdf_path} is already processed."}
            else:
                start_time = time.time()
                from document_rag import DocumentRAGSystem
                logging.info(f"Loading RAG system from {DEFAULT_RAG_DATA_DIR}...")
                rag_system = DocumentRAGSystem(load_from=DEFAULT_RAG_DATA_DIR)
                logging.info(f"Adding document {pdf_path} to system...")
                rag_system.add_document(pdf_path, save_directory=DEFAULT_RAG_DATA_DIR)
                logging.info(f"Document processed in {time.time() - start_time:.2f} seconds")
                return {"message": f"Document {pdf_path} added to system."}
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query")
    async def query_document(request: QueryRequest):
        try:
            query = request.query
            start_time = time.time()
            from document_rag import DocumentRAGSystem
            logging.info(f"Loading RAG system for query: '{query[:50]}...'")
            rag_system = DocumentRAGSystem(load_from=DEFAULT_RAG_DATA_DIR)
            logging.info(f"RAG system loaded in {time.time() - start_time:.2f} seconds")
            
            api_key = os.getenv('REQUESTY_API_KEY')
            if not api_key:
                raise HTTPException(status_code=400, detail="API key not found in environment variables.")
            
            logging.info("Initializing LLM client...")
            llm_client = RequestyLLMClient(api_key=api_key)
            
            logging.info("Generating response...")
            query_start = time.time()
            response = rag_system.generate_response(
                query=query, 
                api_call_function=lambda prompt: llm_client.generate_response(prompt)
            )
            logging.info(f"Response generated in {time.time() - query_start:.2f} seconds")
            
            return {"response": response}
        except Exception as e:
            logging.error(f"Error querying document: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/switch_folder")
    async def switch_folder(request: SwitchFolderRequest):
        try:
            new_folder = request.new_folder
            from document_rag import DocumentRAGSystem
            logging.info(f"Switching to RAG folder: {new_folder}")
            start_time = time.time()
            rag_system = DocumentRAGSystem(load_from=DEFAULT_RAG_DATA_DIR)
            rag_system.switch_rag_folder(new_folder)
            logging.info(f"Switched folders in {time.time() - start_time:.2f} seconds")
            return {"message": f"Switched to new RAG folder: {new_folder}"}
        except Exception as e:
            logging.error(f"Error switching folders: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return app

def main():
    """
    Main entry point for the Document RAG System
    """
    try:
        # Command line arguments
        parser = argparse.ArgumentParser(description='Document RAG System')
        parser.add_argument('--docs_dir', type=str, help='Directory containing PDF documents')
        parser.add_argument('--pdf', type=str, nargs='+', help='Paths to PDF documents')
        parser.add_argument('--query', type=str, help='Search query')
        parser.add_argument('--api_key', type=str,
                            default=os.getenv('REQUESTY_API_KEY'),
                            help='Requesty API key (can also be set in .env file)')
        parser.add_argument('--model', type=str,
                            default=os.getenv('DEFAULT_MODEL', 'o3-mini'),
                            choices=['claude-3-sonnet', 'gpt-4', 'deepseek-v3', 'deepseek-r1', 'o3-mini'],
                            help='LLM model (claude-3-sonnet, gpt-4, deepseek-v3, deepseek-r1, o3-mini)')
        parser.add_argument('--data_dir', type=str,
                            default=DEFAULT_RAG_DATA_DIR,
                            help=f'Directory for storing processed documents (default: {DEFAULT_RAG_DATA_DIR})')
        parser.add_argument('--add_document', type=str,
                            help='Add a single document to the existing system')
        parser.add_argument('--check_document', type=str,
                            help='Check if a document is already processed (without loading TensorFlow)')
        parser.add_argument('--force_reprocess', action='store_true',
                            help='Force reprocessing of documents even if already processed')
        parser.add_argument('--no_graph', action='store_true',
                            help='Disable chunk graph building for faster loading (reduces context quality)')
        parser.add_argument('--run_server', action='store_true', 
                            help='Run the FastAPI server')
        
        args = parser.parse_args()
        
        # Print the arguments for debugging
        logging.info("\nCommand line arguments:")
        logging.info(f"  docs_dir: {args.docs_dir}")
        logging.info(f"  pdf: {args.pdf}")
        logging.info(f"  query: {args.query}")
        logging.info(f"  data_dir: {args.data_dir}")
        logging.info(f"  add_document: {args.add_document}")
        logging.info(f"  check_document: {args.check_document}")
        logging.info(f"  force_reprocess: {args.force_reprocess}")
        logging.info(f"  no_graph: {args.no_graph}")
        logging.info(f"  run_server: {args.run_server}")
        logging.info(f"  model: {args.model}")
        
        if args.run_server:
            app = create_app()
            uvicorn.run(app, host="0.0.0.0", port=8000)
            return
        
        # Handle checking if a document is already processed
        if args.check_document:
            is_processed = is_document_processed(args.check_document, args.data_dir)
            if is_processed:
                logging.info(f"Document '{args.check_document}' is already processed in the system.")
            else:
                logging.info(f"Document '{args.check_document}' is NOT processed in the system.")
            
            # If this is the only operation requested, return
            if not any([args.docs_dir, args.pdf, args.query, args.add_document]):
                return
        
        # Handle adding a single document to existing system
        if args.add_document:
            # First check if document is already processed without loading TensorFlow
            if is_document_processed(args.add_document, args.data_dir):
                logging.info(f"Document {args.add_document} is already processed. Skipping.")
                if not args.query:
                    return
            else:
                try:
                    # Only import and load the full system if we need to process a new document
                    from document_rag import DocumentRAGSystem
                    
                    # Try to load existing system
                    start_time = time.time()
                    logging.info(f"Loading system from {args.data_dir}...")
                    rag_system = DocumentRAGSystem(load_from=args.data_dir, use_graph=not args.no_graph)
                    logging.info(f"System loaded in {time.time() - start_time:.2f} seconds")
                    
                    # Add the document
                    start_time = time.time()
                    logging.info(f"Adding document {args.add_document}...")
                    rag_system.add_document(args.add_document, save_directory=args.data_dir)
                    logging.info(f"Document added in {time.time() - start_time:.2f} seconds")
                    
                    # Exit if no query was provided
                    if not args.query:
                        return
                except FileNotFoundError:
                    logging.error(f"No existing system found in {args.data_dir}. Creating new system...")
                    # Continue with normal initialization
        
        # If no arguments were provided, run example
        if not any([args.docs_dir, args.pdf, args.query, args.add_document, args.check_document]):
            logging.info("No arguments provided. Running example...")
            
            # Import the DocumentRAGSystem only when needed
            from document_rag import DocumentRAGSystem
            
            # Example PDF paths
            pdf_paths = [
                'course_syllabus.pdf',
                'lecture_notes.pdf'
            ]
            
            # Initialize RAG system
            start_time = time.time()
            logging.info("Initializing new RAG system...")
            rag_system = DocumentRAGSystem(pdf_paths=pdf_paths, use_graph=not args.no_graph)
            logging.info(f"RAG system initialized in {time.time() - start_time:.2f} seconds")
            
            # Save the system state
            logging.info(f"Saving system state to {args.data_dir}...")
            start_time = time.time()
            rag_system.save_system_state(args.data_dir)
            logging.info(f"System state saved in {time.time() - start_time:.2f} seconds")
            
            # Example query
            query = "What are the main topics of the course?"
            
            # Since no API key was provided, use an example function
            logging.info(f"Generating example response for query: '{query}'")
            start_time = time.time()
            response = rag_system.generate_response(query, example_api_call)
            logging.info(f"Response generated in {time.time() - start_time:.2f} seconds")
            
        else:
            # Try to load existing system if we haven't already
            if 'rag_system' not in locals():
                # Import the DocumentRAGSystem only when needed
                from document_rag import DocumentRAGSystem
                
                # Collect all PDF paths that might need processing
                all_pdf_paths = []
                if args.docs_dir:
                    from pdf_utils import collect_pdf_paths
                    dir_pdfs = collect_pdf_paths(docs_directory=args.docs_dir)
                    all_pdf_paths.extend(dir_pdfs)
                
                if args.pdf:
                    all_pdf_paths.extend(args.pdf)
                
                # First check if system exists without loading TensorFlow
                if not args.force_reprocess and system_exists(args.data_dir):
                    logging.info(f"Checking for existing system in {args.data_dir}...")
                    
                    # Check if any documents need processing without loading TensorFlow
                    processed_docs = get_processed_documents(args.data_dir)
                    new_docs = [path for path in all_pdf_paths if path not in processed_docs]
                    
                    if not new_docs and processed_docs:
                        # If all documents are already processed and we just need to query
                        if args.query:
                            start_time = time.time()
                            logging.info(f"Loading existing system for query...")
                            rag_system = DocumentRAGSystem(load_from=args.data_dir, use_graph=not args.no_graph)
                            logging.info(f"Existing system loaded in {time.time() - start_time:.2f} seconds")
                        else:
                            logging.info("All documents are already processed. No query provided.")
                            return
                    elif all_pdf_paths:
                        # Load the system and add any new documents
                        start_time = time.time()
                        logging.info("Loading existing system to add new documents...")
                        rag_system = DocumentRAGSystem(load_from=args.data_dir, use_graph=not args.no_graph)
                        logging.info(f"Existing system loaded in {time.time() - start_time:.2f} seconds")
                        
                        # Add any new documents
                        if new_docs:
                            logging.info("Adding new documents to existing system...")
                            for pdf_path in new_docs:
                                doc_start = time.time()
                                logging.info(f"Adding document: {pdf_path}")
                                rag_system.add_document(pdf_path, save_directory=args.data_dir)
                                logging.info(f"Document added in {time.time() - doc_start:.2f} seconds")
                    else:
                        # Just load the system for querying
                        start_time = time.time()
                        logging.info("Loading existing system for querying...")
                        rag_system = DocumentRAGSystem(load_from=args.data_dir, use_graph=not args.no_graph)
                        logging.info(f"System loaded in {time.time() - start_time:.2f} seconds")
                elif all_pdf_paths:
                    # Initialize new RAG system with specified documents
                    start_time = time.time()
                    logging.info("Initializing new RAG system with specified documents...")
                    rag_system = DocumentRAGSystem(
                        docs_directory=args.docs_dir,
                        pdf_paths=args.pdf,
                        use_graph=not args.no_graph
                    )
                    logging.info(f"RAG system initialized in {time.time() - start_time:.2f} seconds")
                    
                    # Save the system state
                    save_start = time.time()
                    logging.info(f"Saving system state to {args.data_dir}...")
                    rag_system.save_system_state(args.data_dir)
                    logging.info(f"System state saved in {time.time() - save_start:.2f} seconds")
                else:
                    logging.error("ERROR: No documents specified. Please provide either --docs_dir or --pdf.")
                    return
            
            # If no API key was provided, ask for one
            api_key = args.api_key
            if not api_key:
                api_key = input("Please enter your Requesty API key: ")
            
            # Initialize LLM client
            logging.info(f"Initializing LLM client with model: {args.model}")
            llm_client = RequestyLLMClient(api_key=api_key, default_model=args.model)
            
            # If no query was provided, ask for one
            query = args.query
            if not query:
                query = input("Please enter your query: ")
            
            # Generate response
            logging.info(f"Generating response for query: '{query[:50]}...'")
            start_time = time.time()
            response = rag_system.generate_response(
                query=query,
                api_call_function=lambda prompt: llm_client.generate_response(prompt)
            )
            logging.info(f"Response generated in {time.time() - start_time:.2f} seconds")
        
        logging.info("\nResponse:")
        logging.info(response)
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
