import os
import argparse
from dotenv import load_dotenv
from document_rag import DocumentRAGSystem
from llm_client import RequestyLLMClient, example_api_call

# Load environment variables from .env file
load_dotenv()

# Default directory for storing processed documents
DEFAULT_RAG_DATA_DIR = "rag_data"

def main():
    """
    Main entry point for the Document RAG System
    """
    # Command line arguments
    parser = argparse.ArgumentParser(description='Document RAG System')
    parser.add_argument('--docs_dir', type=str, help='Directory containing PDF documents')
    parser.add_argument('--pdf', type=str, nargs='+', help='Paths to PDF documents')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--api_key', type=str,
                        default=os.getenv('REQUESTY_API_KEY'),
                        help='Requesty API key (can also be set in .env file)')
    parser.add_argument('--model', type=str,
                        default=os.getenv('DEFAULT_MODEL', 'deepseek-v3'),
                        choices=['claude-3-sonnet', 'gpt-4', 'deepseek-v3'],
                        help='LLM model (claude-3-sonnet, gpt-4, deepseek-v3)')
    parser.add_argument('--data_dir', type=str,
                        default=DEFAULT_RAG_DATA_DIR,
                        help=f'Directory for storing processed documents (default: {DEFAULT_RAG_DATA_DIR})')
    parser.add_argument('--add_document', type=str,
                        help='Add a single document to the existing system')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of documents even if already processed')
    
    args = parser.parse_args()
    
    # Print the arguments for debugging
    print("\nCommand line arguments:")
    print(f"  docs_dir: {args.docs_dir}")
    print(f"  pdf: {args.pdf}")
    print(f"  query: {args.query}")
    print(f"  data_dir: {args.data_dir}")
    print(f"  add_document: {args.add_document}")
    print(f"  force_reprocess: {args.force_reprocess}")
    
    # Handle adding a single document to existing system
    if args.add_document:
        try:
            # Try to load existing system
            rag_system = DocumentRAGSystem(load_from=args.data_dir)
            
            # Add the document
            rag_system.add_document(args.add_document, save_directory=args.data_dir)
            print(f"Document {args.add_document} added to system.")
            
            # Exit if no query was provided
            if not args.query:
                return
        except FileNotFoundError:
            print(f"No existing system found in {args.data_dir}. Creating new system...")
            # Continue with normal initialization
    
    # If no arguments were provided, run example
    if not any([args.docs_dir, args.pdf, args.query, args.add_document]):
        print("No arguments provided. Running example...")
        
        # Example PDF paths
        pdf_paths = [
            'course_syllabus.pdf',
            'lecture_notes.pdf'
        ]
        
        # Initialize RAG system
        rag_system = DocumentRAGSystem(pdf_paths=pdf_paths)
        
        # Save the system state
        rag_system.save_system_state(args.data_dir)
        
        # Example query
        query = "What are the main topics of the course?"
        
        # Since no API key was provided, use an example function
        response = rag_system.generate_response(query, example_api_call)
        
    else:
        # Try to load existing system if we haven't already
        if not 'rag_system' in locals():
            try:
                # Try to load existing system if we have documents to process
                if not args.force_reprocess and (args.docs_dir or args.pdf):
                    print(f"Checking for existing system in {args.data_dir}...")
                    rag_system = DocumentRAGSystem(load_from=args.data_dir)
                    print("Loaded existing system.")
                    
                    # Add any new documents if specified
                    if args.docs_dir or args.pdf:
                        print("Adding any new documents to existing system...")
                        all_pdf_paths = []
                        
                        if args.docs_dir:
                            from pdf_utils import collect_pdf_paths
                            dir_pdfs = collect_pdf_paths(docs_directory=args.docs_dir)
                            all_pdf_paths.extend(dir_pdfs)
                            
                        if args.pdf:
                            all_pdf_paths.extend(args.pdf)
                            
                        for pdf_path in all_pdf_paths:
                            if not rag_system.is_document_processed(pdf_path, args.data_dir):
                                rag_system.add_document(pdf_path, save_directory=args.data_dir)
                
                else:
                    # Initialize new RAG system with specified documents
                    rag_system = DocumentRAGSystem(
                        docs_directory=args.docs_dir,
                        pdf_paths=args.pdf
                    )
                    
                    # Save the system state
                    rag_system.save_system_state(args.data_dir)
                    
            except FileNotFoundError as e:
                print(f"Warning: Directory {args.data_dir} not found. Initializing new system.")
                
                # Make sure we have valid parameters
                if not args.docs_dir and not args.pdf:
                    print("ERROR: No documents specified. Please provide either --docs_dir or --pdf.")
                    return
                
                # Initialize new RAG system with specified documents
                # Pass the actual document directory or PDFs here
                rag_system = DocumentRAGSystem(
                    docs_directory=args.docs_dir,
                    pdf_paths=args.pdf
                )
                
                # Save the system state
                rag_system.save_system_state(args.data_dir)
        
        # If no API key was provided, ask for one
        api_key = args.api_key
        if not api_key:
            api_key = input("Please enter your Requesty API key: ")
        
        # Initialize LLM client
        llm_client = RequestyLLMClient(api_key=api_key, default_model=args.model)
        
        # If no query was provided, ask for one
        query = args.query
        if not query:
            query = input("Please enter your query: ")
        
        # Generate response
        response = rag_system.generate_response(
            query=query,
            api_call_function=lambda prompt: llm_client.generate_response(prompt)
        )
    
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()