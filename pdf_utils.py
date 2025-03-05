import os
import PyPDF2
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a PDF file.
    
    Parameters:
    pdf_path (str): Path to the PDF file
    
    Returns:
    str: Extracted text content
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} was not found.")
            
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
        
        return full_text
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        raise
    except PyPDF2.errors.PdfReadError as e:
        logging.error(f"Error reading PDF: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise

def collect_pdf_paths(docs_directory: str = None, pdf_paths: List[str] = None) -> List[str]:
    """
    Collects PDF file paths from a directory and/or a list of paths.
    
    Parameters:
    docs_directory (str, optional): Directory containing PDF documents
    pdf_paths (List[str], optional): List of paths to PDF documents
    
    Returns:
    List[str]: Combined list of PDF paths
    """
    import glob
    
    try:
        all_pdf_paths = []
        
        if docs_directory:
            # Find all PDFs in the specified directory
            directory_pdfs = glob.glob(os.path.join(docs_directory, "*.pdf"))
            all_pdf_paths.extend(directory_pdfs)
        
        if pdf_paths:
            all_pdf_paths.extend(pdf_paths)
            
        if not all_pdf_paths:
            raise ValueError("No PDF documents found. Please provide either docs_directory or pdf_paths.")
            
        return all_pdf_paths
    except ValueError as e:
        logging.error(f"Value error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise
