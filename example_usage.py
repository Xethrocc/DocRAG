import os
import time
import requests
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
SAMPLE_PDF_PATH = "testdoc/Skript_DBI.pdf"  # Adjust to point to an existing PDF
CATEGORY = "academic"

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def pretty_print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def main():
    """Example usage of the improved DocRAG API"""
    
    print_section("1. Check available documents")
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        documents = response.json()
        print(f"Found {len(documents)} documents")
        if documents:
            print("First document:")
            pretty_print_json(documents[0])
    except Exception as e:
        print(f"Error listing documents: {str(e)}")
    
    print_section("2. Upload a document")
    try:
        # Check if sample document exists
        if not os.path.exists(SAMPLE_PDF_PATH):
            print(f"Sample PDF not found at {SAMPLE_PDF_PATH}. Skipping upload.")
        else:
            with open(SAMPLE_PDF_PATH, "rb") as file:
                response = requests.post(
                    f"{API_BASE_URL}/documents/upload",
                    files={"file": (Path(SAMPLE_PDF_PATH).name, file, "application/pdf")},
                    data={"category": CATEGORY}
                )
                document = response.json()
                print("Uploaded document:")
                pretty_print_json(document)
                
                # Store document ID for later use
                document_id = document["id"]
                
                print_section("3. Process the document")
                response = requests.post(f"{API_BASE_URL}/documents/{document_id}/process")
                process_result = response.json()
                task_id = process_result.get("task_id")
                print("Process started:")
                pretty_print_json(process_result)
                
                if task_id:
                    print("\nChecking task status:")
                    # Check task status a few times
                    for _ in range(10):
                        time.sleep(2)  # Wait 2 seconds between checks
                        response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")
                        task_status = response.json()
                        print(f"Status: {task_status['status']}, Progress: {task_status['progress']}%, Message: {task_status['message']}")
                        
                        if task_status["status"] in ["completed", "failed"]:
                            break
                
                print_section("4. Query the document")
                query = "What is the main topic of this document?"
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"query": query, "system_name": "default"}
                )
                query_result = response.json()
                print(f"Query: {query}")
                print("\nResponse:")
                print(query_result.get("response", "No response generated"))
    except Exception as e:
        print(f"Error during example: {str(e)}")
    
    print_section("5. List RAG systems")
    try:
        response = requests.get(f"{API_BASE_URL}/rag/systems")
        systems = response.json()
        print(f"Found {len(systems)} RAG systems")
        for system in systems:
            print(f"\n- {system['name']}: {system['document_count']} documents, {system['size_mb']} MB")
    except Exception as e:
        print(f"Error listing RAG systems: {str(e)}")

if __name__ == "__main__":
    main()