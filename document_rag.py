import os
import numpy as np
import faiss
import tiktoken
import json
import pickle
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our utility modules
from text_processing import split_text
from pdf_utils import extract_text_from_pdf, collect_pdf_paths

class DocumentRAGSystem:
    def __init__(self,
                 docs_directory: str = None,
                 pdf_paths: List[str] = None,
                 embedding_model: str = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
                 max_context_tokens: int = int(os.getenv('MAX_CONTEXT_TOKENS', 3500)),
                 tokenizer_name: str = "cl100k_base",
                 load_from: str = None):
        """
        Initializes a Retrieval Augmented Generation System
        
        Parameters:
        docs_directory (str, optional): Directory containing PDF documents
        pdf_paths (List[str], optional): Paths to PDF documents
        embedding_model (str): Embedding model
        max_context_tokens (int): Maximum token count for context
        tokenizer_name (str): Name of the tokenizer
        load_from (str, optional): Directory to load saved system state from
        """
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Tokenizer for controlling token length
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # Maximum context length
        self.max_context_tokens = max_context_tokens
        
        # Initialize empty state
        self.documents = {}
        self.pdf_paths = []
        self.index_texts = []
        
        # Try to load from saved state if specified
        if load_from:
            if not os.path.exists(load_from):
                raise FileNotFoundError(f"Directory {load_from} does not exist")
            
            try:
                self.load_system_state(load_from)
                print(f"System state loaded from {load_from}")
                return  # Skip the rest of initialization if loaded successfully
            except Exception as e:
                print(f"Error loading system state: {e}")
                # Continue with initialization using the provided parameters
        
        # If we get here, either no load_from was specified or loading failed
        # Collect PDF paths
        print(f"Collecting PDF paths with docs_directory={docs_directory}, pdf_paths={pdf_paths}")
        all_pdf_paths = collect_pdf_paths(docs_directory, pdf_paths)
            
        # Process documents
        self.documents = self.process_documents(all_pdf_paths)
        
        # Create FAISS index
        self.index = self.create_faiss_index()
        
        # Store original paths for later reference
        self.pdf_paths = all_pdf_paths
    
    def process_documents(self, pdf_paths: List[str]) -> Dict:
        """
        Processes PDFs into structured documents
        
        Returns:
        dict: Processed documents with embeddings
        """
        processed_docs = {}
        
        for pdf_path in pdf_paths:
            # Extract text from PDF
            full_text = extract_text_from_pdf(pdf_path)
            
            # Split text into chunks
            chunks = split_text(full_text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks)
            
            processed_docs[pdf_path] = {
                'chunks': chunks,
                'embeddings': embeddings
            }
        
        return processed_docs
    
    def add_document(self, pdf_path: str, save_directory: str = "rag_data") -> None:
        """
        Adds a new document to the existing RAG system
        
        Parameters:
        pdf_path (str): Path to the PDF document
        save_directory (str): Directory to save the updated system state
        
        Returns:
        None
        """
        # Check if the file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} was not found.")
            
        # Check if document is already processed
        if self.is_document_processed(pdf_path, save_directory):
            print(f"Document {pdf_path} is already processed. Skipping.")
            return
            
        # Process document
        new_doc = self.process_documents([pdf_path])
        
        # Add to existing documents dictionary
        self.documents.update(new_doc)
        
        # Add path to the list of PDF paths
        if pdf_path not in self.pdf_paths:
            self.pdf_paths.append(pdf_path)
        
        # Add new embeddings to FAISS index
        for doc_path, doc in new_doc.items():
            embeddings = doc['embeddings'].astype('float32')
            self.index.add(embeddings)
            self.index_texts.extend(doc['chunks'])
            
        print(f"Document {pdf_path} successfully added.")
        
        # Save the updated system state
        self.save_system_state(save_directory)
        print(f"System state updated in {save_directory}")
    
    def create_faiss_index(self) -> faiss.Index:
        """
        Creates FAISS index for fast similarity search
        
        Returns:
        faiss.Index: FAISS index
        """
        # Collect all embeddings
        all_embeddings = []
        all_texts = []
        
        for doc in self.documents.values():
            all_embeddings.extend(doc['embeddings'])
            all_texts.extend(doc['chunks'])
        
        # NumPy array of embeddings
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Additional storage of original texts
        self.index_texts = all_texts
        
        return index
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Finds relevant document sections for the search query
        
        Returns:
        List[str]: Relevant text sections
        """
        # Query embedding
        query_embedding = self.embedding_model.encode([query])[0].astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            top_k
        )
        
        # Extract relevant texts
        relevant_contexts = [self.index_texts[i] for i in indices[0]]
        
        return relevant_contexts
    
    def prepare_prompt_with_context(self, query: str, context: List[str]) -> str:
        """
        Prepares prompt with context
        
        Returns:
        str: Prepared prompt
        """
        # Combine context
        context_text = "\n\n".join(context)
        
        # Construct prompt
        prompt = f"""
        Document Context:
        {context_text}
        
        User Query: {query}
        
        Please answer the question based on the given context. 
        If the answer is not contained in the context, honestly state 
        that you cannot answer the question.
        """
        
        return prompt
    
    def truncate_to_max_tokens(self, prompt: str) -> str:
        """
        Truncates prompt to maximum token length
        
        Returns:
        str: Truncated prompt
        """
        tokens = self.tokenizer.encode(prompt)
        
        if len(tokens) > self.max_context_tokens:
            # Truncate
            tokens = tokens[:self.max_context_tokens]
            return self.tokenizer.decode(tokens)
        
        return prompt
    
    def generate_response(self, 
                          query: str, 
                          api_call_function,  # Function for API call
                          top_k: int = 5) -> str:
        """
        Generates response with Retrieval Augmented Generation
        
        Parameters:
        query (str): User query
        api_call_function (callable): Function for LLM API call
        top_k (int): Number of context documents
        
        Returns:
        str: Generated response
        """
        # Retrieve relevant contexts
        contexts = self.retrieve_relevant_context(query, top_k)
        
        # Prepare prompt with context
        prompt = self.prepare_prompt_with_context(query, contexts)
        
        # Truncate prompt to maximum token length
        truncated_prompt = self.truncate_to_max_tokens(prompt)
        
        # API call with prompt
        response = api_call_function(truncated_prompt)
        
        return response
    
    def save_system_state(self, directory="rag_data"):
        """
        Save the system state using a hybrid approach
        
        Parameters:
        directory (str): Directory to save the system state
        
        Returns:
        None
        """
        os.makedirs(directory, exist_ok=True)
        
        # 1. Save document list and metadata as JSON
        metadata = {
            "pdf_paths": self.pdf_paths,
            "document_info": {path: {"chunk_count": len(data["chunks"])}
                             for path, data in self.documents.items()}
        }
        
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # 2. Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "document_index.faiss"))
        
        # 3. Save text chunks
        with open(os.path.join(directory, "index_texts.pkl"), "wb") as f:
            pickle.dump(self.index_texts, f)
        
        # 4. Save document chunks and embeddings
        for path, data in self.documents.items():
            # Create safe filename
            safe_name = path.replace("/", "_").replace("\\", "_")
            doc_path = os.path.join(directory, f"doc_{safe_name}.pkl")
            with open(doc_path, "wb") as f:
                pickle.dump(data, f)
        
        print(f"System state saved to {directory}")
    
    def load_system_state(self, directory="rag_data"):
        """
        Load the system state using a hybrid approach
        
        Parameters:
        directory (str): Directory to load the system state from
        
        Returns:
        None
        
        Raises:
        FileNotFoundError: If the directory or required files don't exist
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} not found")
        
        # 1. Load metadata
        metadata_path = os.path.join(directory, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.pdf_paths = metadata["pdf_paths"]
        
        # 2. Load FAISS index
        index_path = os.path.join(directory, "document_index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
            
        self.index = faiss.read_index(index_path)
        
        # 3. Load text chunks
        texts_path = os.path.join(directory, "index_texts.pkl")
        if not os.path.exists(texts_path):
            raise FileNotFoundError(f"Index texts not found at {texts_path}")
            
        with open(texts_path, "rb") as f:
            self.index_texts = pickle.load(f)
        
        # 4. Load document chunks and embeddings
        self.documents = {}
        for path in self.pdf_paths:
            safe_name = path.replace("/", "_").replace("\\", "_")
            doc_path = os.path.join(directory, f"doc_{safe_name}.pkl")
            
            if not os.path.exists(doc_path):
                print(f"Warning: Document data not found for {path}")
                continue
                
            with open(doc_path, "rb") as f:
                self.documents[path] = pickle.load(f)
        
        print(f"System state loaded from {directory}")
    
    def is_document_processed(self, pdf_path, directory="rag_data"):
        """
        Check if a document is already processed
        
        Parameters:
        pdf_path (str): Path to the PDF document
        directory (str): Directory containing the system state
        
        Returns:
        bool: True if the document is already processed, False otherwise
        """
        # First check in-memory state
        if hasattr(self, 'pdf_paths') and pdf_path in self.pdf_paths:
            return True
            
        # Then check saved state if directory exists
        try:
            metadata_path = os.path.join(directory, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                return pdf_path in metadata["pdf_paths"]
        except Exception as e:
            print(f"Error checking if document is processed: {str(e)}")
            
        return False