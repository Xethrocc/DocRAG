import os
import argparse
from dotenv import load_dotenv
from document_rag import DocumentRAGSystem
from llm_client import RequestyLLMClient, example_api_call

# Load environment variables from .env file
load_dotenv()

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
    
    args = parser.parse_args()
    
    # If no arguments were provided, run example
    if not any(vars(args).values()):
        print("No arguments provided. Running example...")
        
        # Example PDF paths
        pdf_paths = [
            'course_syllabus.pdf',
            'lecture_notes.pdf'
        ]
        
        # Initialize RAG system
        rag_system = DocumentRAGSystem(pdf_paths=pdf_paths)
        
        # Example query
        query = "What are the main topics of the course?"
        
        # Since no API key was provided, use an example function
        response = rag_system.generate_response(query, example_api_call)
        
    else:
        # Initialize RAG system with specified documents
        rag_system = DocumentRAGSystem(
            docs_directory=args.docs_dir,
            pdf_paths=args.pdf
        )
        
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