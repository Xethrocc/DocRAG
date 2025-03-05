import os
import json

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
    except Exception as e:
        print(f"Error checking if document is processed: {str(e)}")
        
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
    except Exception as e:
        print(f"Error getting processed documents: {str(e)}")
        
    return []

def system_exists(directory="rag_data"):
    """
    Check if a RAG system exists in the specified directory without loading TensorFlow
    
    Parameters:
    directory (str): Directory to check for system files
    
    Returns:
    bool: True if the system exists, False otherwise
    """
    if not os.path.exists(directory):
        return False
        
    # Check for essential files
    required_files = [
        os.path.join(directory, "metadata.json"),
        os.path.join(directory, "document_index.faiss"),
        os.path.join(directory, "index_texts.pkl")
    ]
    
    return all(os.path.exists(file) for file in required_files)