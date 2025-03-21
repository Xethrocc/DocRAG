"""
Example script demonstrating the document persistence features of the RAG system.
"""
import os
import logging
from document_rag import DocumentRAGSystem
from llm_client import example_api_call

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory for storing processed documents
RAG_DATA_DIR = "rag_data"

def main():
    try:
        # Check if we already have a saved system
        if os.path.exists(RAG_DATA_DIR) and os.path.isfile(os.path.join(RAG_DATA_DIR, "metadata.json")):
            logging.info(f"Loading existing RAG system from {RAG_DATA_DIR}...")
            
            # Load the existing system
            rag_system = DocumentRAGSystem(load_from=RAG_DATA_DIR)
            logging.info(f"System loaded with {len(rag_system.pdf_paths)} documents")
            
            # List the documents in the system
            logging.info("\nDocuments in the system:")
            for i, path in enumerate(rag_system.pdf_paths, 1):
                logging.info(f"{i}. {path}")
        else:
            logging.info(f"No existing system found. Creating a new RAG system...")
            
            # For this example, we'll use some example PDF paths
            # Replace these with actual PDF paths on your system
            pdf_paths = [
                "example_doc1.pdf",
                "example_doc2.pdf"
            ]
            
            # Check if the example files exist
            existing_pdfs = [p for p in pdf_paths if os.path.exists(p)]
            
            if not existing_pdfs:
                logging.warning("No example PDFs found. Please update the pdf_paths list with actual PDF paths.")
                return
                
            logging.info(f"Processing {len(existing_pdfs)} documents...")
            
            # Initialize a new RAG system
            rag_system = DocumentRAGSystem(pdf_paths=existing_pdfs)
            
            # Save the system state
            rag_system.save_system_state(RAG_DATA_DIR)
            logging.info(f"System saved to {RAG_DATA_DIR}")
        
        # Ask for a document to add
        new_doc = input("\nEnter path to a new document to add (or press Enter to skip): ")
        if new_doc and os.path.exists(new_doc):
            # Check if the document is already processed
            if rag_system.is_document_processed(new_doc, RAG_DATA_DIR):
                logging.info(f"Document {new_doc} is already in the system.")
            else:
                # Add the document to the system
                rag_system.add_document(new_doc, save_directory=RAG_DATA_DIR)
                logging.info(f"Document {new_doc} added to the system.")
        
        # Ask for a query
        query = input("\nEnter a query about the documents (or press Enter to exit): ")
        if query:
            # Generate a response using the example API call function
            logging.info("\nGenerating response...")
            response = rag_system.generate_response(query, example_api_call)
            logging.info("\nResponse:")
            logging.info(response)
            
            # In a real application, you would use a real LLM API:
            # from llm_client import RequestyLLMClient
            # llm_client = RequestyLLMClient(api_key="your-api-key")
            # response = rag_system.generate_response(
            #     query=query,
            #     api_call_function=lambda prompt: llm_client.generate_response(prompt)
            # )
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
