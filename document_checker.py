import os
import json
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_document_processed(pdf_path, directory="rag_data"):
    """
    Check if a document is already processed without loading TensorFlow
    
    Parameters:
    pdf_path (str): Path to the PDF document
    directory (str): Directory containing the system state
    
    Returns:
    bool: True if the document is already processed, False otherwise
    """
    try:
        metadata_path = os.path.join(directory, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Normalize paths for comparison (handle both forward and backslashes)
            normalized_path = pdf_path.replace('/', '\\')
            if normalized_path in metadata["pdf_paths"]:
                return True
                
            # Also check with forward slashes
            normalized_path = pdf_path.replace('\\', '/')
            if normalized_path in metadata["pdf_paths"]:
                return True
                
            # Check if the basename matches (ignore directory structure)
            basename = os.path.basename(pdf_path)
            for stored_path in metadata["pdf_paths"]:
                if os.path.basename(stored_path) == basename:
                    return True
    except FileNotFoundError as e:
        logging.error(f"Metadata file not found: {str(e)}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        
    return False

def get_processed_documents(directory="rag_data"):
    """
    Get a list of all processed documents without loading TensorFlow
    
    Parameters:
    directory (str): Directory containing the system state
    
    Returns:
    list: List of processed document paths, or empty list if none found
    """
    try:
        metadata_path = os.path.join(directory, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Return both normalized versions of paths (with forward and backslashes)
            paths = []
            for path in metadata["pdf_paths"]:
                paths.append(path)
                paths.append(path.replace('\\', '/'))
                paths.append(path.replace('/', '\\'))
            
            return list(set(paths))  # Remove duplicates
    except FileNotFoundError as e:
        logging.error(f"Metadata file not found: {str(e)}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        
    return []

def system_exists(directory="rag_data"):
    """
    Check if a RAG system exists in the specified directory without loading TensorFlow
    
    Parameters:
    directory (str): Directory to check for system files
    
    Returns:
    bool: True if the system exists, False otherwise
    """
    try:
        if not os.path.exists(directory):
            return False
            
        # Check for essential files
        required_files = [
            os.path.join(directory, "metadata.json"),
            os.path.join(directory, "document_index.faiss"),
            os.path.join(directory, "index_texts.pkl")
        ]
        
        return all(os.path.exists(file) for file in required_files)
    except Exception as e:
        logging.error(f"Error checking system existence: {str(e)}")
        return False

def enrich_metadata_with_external_sources(metadata):
    """
    Enrich metadata with external data sources
    
    Parameters:
    metadata (dict): Metadata to be enriched
    
    Returns:
    dict: Enriched metadata
    """
    try:
        # Example: Enrich with Wikipedia summaries
        for entity in metadata.get('entities', []):
            if entity['label'] in ['PERSON', 'ORG', 'GPE']:
                summary = get_wikipedia_summary(entity['text'])
                if summary:
                    entity['summary'] = summary
    except Exception as e:
        logging.error(f"Error enriching metadata: {str(e)}")
    
    return metadata

def get_wikipedia_summary(entity_name):
    """
    Get Wikipedia summary for a given entity name
    
    Parameters:
    entity_name (str): Name of the entity
    
    Returns:
    str: Wikipedia summary or None if not found
    """
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity_name}")
        if response.status_code == 200:
            data = response.json()
            return data.get('extract')
    except Exception as e:
        logging.error(f"Error fetching Wikipedia summary: {str(e)}")
    
    return None
