import os
import numpy as np
import faiss
import tiktoken
import json
import pickle
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from rake_nltk import Rake

# Load environment variables
load_dotenv()

# Import our utility modules
from text_processing import split_text, split_text_efficiently
from pdf_utils import extract_text_from_pdf, collect_pdf_paths
from document_checker import is_document_processed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRAGSystem:
    def __init__(self,
                 docs_directory: str = None,
                 pdf_paths: List[str] = None,
                 embedding_model: str = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
                 max_context_tokens: int = int(os.getenv('MAX_CONTEXT_TOKENS', 3500)),
                 tokenizer_name: str = "cl100k_base",
                 load_from: str = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 100,
                 use_graph: bool = True):
        """
        Initializes a Retrieval Augmented Generation System
        
        Parameters:
        docs_directory (str, optional): Directory containing PDF documents
        pdf_paths (List[str], optional): Paths to PDF documents
        embedding_model (str): Embedding model
        max_context_tokens (int): Maximum token count for context
        tokenizer_name (str): Name of the tokenizer
        load_from (str, optional): Directory to load saved system state from
        chunk_size (int): Size of text chunks in words
        chunk_overlap (int): Overlap between chunks in words
        use_graph (bool): Whether to use graph-based retrieval
        """
        # Initialize empty state
        self.documents = {}
        self.pdf_paths = []
        self.index_texts = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_graph = use_graph
        
        # Initialize embedding model and tokenizer regardless of loading path
        # to ensure they're always available
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_context_tokens = max_context_tokens
        
        # Initialize NLP models for advanced metadata extraction
        self.nlp = spacy.load("en_core_web_sm")
        self.lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        self.vectorizer = CountVectorizer(stop_words='english')
        self.rake = Rake()
        
        # Try to load from saved state if specified
        if load_from:
            if not os.path.exists(load_from):
                raise FileNotFoundError(f"Directory {load_from} does not exist")
            
            try:
                self.load_system_state(load_from)
                print(f"System state loaded from {load_from}")
                return  # Skip the rest of initialization if loaded successfully
            except Exception as e:
                logging.error(f"Error loading system state: {str(e)}")
                # Continue with initialization using the provided parameters
        
        # If we get here, either no load_from was specified or loading failed
        # Collect PDF paths
        print(f"Collecting PDF paths with docs_directory={docs_directory}, pdf_paths={pdf_paths}")
        all_pdf_paths = collect_pdf_paths(docs_directory, pdf_paths)
        
        # Check if all documents are already processed
        all_processed = all(is_document_processed(pdf_path) for pdf_path in all_pdf_paths)
        
        if all_processed:
            print("All documents are already processed. Skipping document processing.")
            return
            
        # Process documents
        self.documents = self.process_documents(all_pdf_paths)
        
        # Create FAISS index
        self.index = self.create_faiss_index()
        
        # Store original paths for later reference
        self.pdf_paths = all_pdf_paths
        
        # Build chunk graph for enhanced retrieval if enabled
        if self.use_graph:
            self.build_chunk_graph()
    
    def process_documents(self, pdf_paths: List[str]) -> Dict:
        """
        Processes PDFs into structured documents with metadata
        
        Returns:
        dict: Processed documents with embeddings and metadata
        """
        processed_docs = {}
        
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            try:
                # Extract text from PDF
                full_text = extract_text_from_pdf(pdf_path)
                
                # Split text into chunks with metadata
                chunks, metadata = self.split_text_with_metadata(full_text)
                
                # Process chunks in batches to optimize memory usage
                batch_size = 100
                embeddings = []
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_embeddings = self.embedding_model.encode(batch_chunks)
                    embeddings.extend(batch_embeddings)
                
                # Perform advanced metadata extraction
                entities = self.extract_entities(full_text)
                topics = self.extract_topics(full_text)
                sentiment = self.extract_sentiment(full_text)
                keywords = self.extract_keywords(full_text)
                
                # Store metadata
                metadata = {
                    'entities': entities,
                    'topics': topics,
                    'sentiment': sentiment,
                    'keywords': keywords
                }
                
                processed_docs[pdf_path] = {
                    'chunks': chunks,
                    'embeddings': np.array(embeddings),
                    'metadata': metadata
                }
            except FileNotFoundError as e:
                logging.error(f"File not found: {str(e)}")
            except Exception as e:
                logging.error(f"Error processing document {pdf_path}: {str(e)}")
        
        return processed_docs
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts named entities from text using NER
        
        Parameters:
        text (str): Text to extract entities from
        
        Returns:
        List[Dict[str, str]]: Extracted entities
        """
        doc = self.nlp(text)
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        return entities
    
    def extract_topics(self, text: str) -> List[str]:
        """
        Extracts main topics from text using LDA
        
        Parameters:
        text (str): Text to extract topics from
        
        Returns:
        List[str]: Extracted topics
        """
        text_data = [text]
        text_vectorized = self.vectorizer.fit_transform(text_data)
        lda_output = self.lda_model.fit_transform(text_vectorized)
        topics = [self.vectorizer.get_feature_names_out()[i] for i in lda_output[0].argsort()[-5:]]
        return topics
    
    def extract_sentiment(self, text: str) -> str:
        """
        Extracts sentiment from text using sentiment analysis
        
        Parameters:
        text (str): Text to extract sentiment from
        
        Returns:
        str: Extracted sentiment
        """
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return "positive"
        elif sentiment < 0:
            return "negative"
        else:
            return "neutral"
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extracts keywords from text using keyword extraction
        
        Parameters:
        text (str): Text to extract keywords from
        
        Returns:
        List[str]: Extracted keywords
        """
        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases()
        return keywords
    
    def add_document(self, pdf_path: str, save_directory: str = "rag_data") -> None:
        """
        Adds a new document to the existing RAG system
        
        Parameters:
        pdf_path (str): Path to the PDF document
        save_directory (str): Directory to save the updated system state
        
        Returns:
        None
        """
        try:
            # Check if the file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"The file {pdf_path} was not found.")
                
            # Check if document is already processed
            if self.is_document_processed(pdf_path, save_directory):
                print(f"Document {pdf_path} is already processed. Skipping.")
                return
            
            # Ensure the transformer model and tokenizer are loaded
            # This is now redundant since we initialize in __init__, but keeping for safety
            if not hasattr(self, 'embedding_model') or self.embedding_model is None:
                self.embedding_model = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
            
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                
            if not hasattr(self, 'max_context_tokens'):
                self.max_context_tokens = int(os.getenv('MAX_CONTEXT_TOKENS', 3500))
                
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
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
        except Exception as e:
            logging.error(f"Error adding document {pdf_path}: {str(e)}")
    
    def create_faiss_index(self) -> faiss.Index:
        """
        Creates FAISS index for fast similarity search
        
        Returns:
        faiss.Index: FAISS index
        """
        try:
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
        except Exception as e:
            logging.error(f"Error creating FAISS index: {str(e)}")
            raise
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5,
                                 use_graph: bool = None,
                                 max_tokens: Optional[int] = None) -> List[str]:
        """
        Finds relevant document sections for the search query using enhanced retrieval
        
        Parameters:
        query (str): User query
        top_k (int): Initial number of chunks to retrieve
        use_graph (bool): Whether to use graph-based expansion (defaults to self.use_graph)
        max_tokens (int, optional): Maximum tokens to include in context
        
        Returns:
        List[str]: Relevant text sections
        """
        try:
            # Use instance default if not specified
            if use_graph is None:
                use_graph = self.use_graph
                
            # Query embedding
            query_embedding = self.embedding_model.encode([query])[0].astype('float32')
            
            # Search in FAISS index
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1),
                top_k
            )
            
            # Extract initial relevant texts
            initial_chunks = [self.index_texts[i] for i in indices[0]]
            
            if not use_graph or not hasattr(self, 'chunk_graph'):
                return initial_chunks
            
            # Use graph to expand context
            expanded_chunks = self.expand_context_with_graph(indices[0], query_embedding, max_tokens)
            
            return expanded_chunks
        except Exception as e:
            logging.error(f"Error retrieving relevant context for query '{query}': {str(e)}")
            return []
    
    def prepare_prompt_with_context(self, query: str, context: List[str],
                                   include_metadata: bool = True) -> str:
        """
        Prepares an enhanced prompt with context and metadata
        
        Parameters:
        query (str): User query
        context (List[str]): Context chunks
        include_metadata (bool): Whether to include metadata
        
        Returns:
        str: Prepared prompt
        """
        try:
            # Add metadata to each context chunk if available
            enhanced_context = []
            
            for i, chunk in enumerate(context):
                # Try to find metadata for this chunk
                metadata = {}
                for doc_path, doc_data in self.documents.items():
                    if chunk in doc_data['chunks']:
                        chunk_idx = doc_data['chunks'].index(chunk)
                        if 'metadata' in doc_data and chunk_idx < len(doc_data['metadata']):
                            metadata = doc_data['metadata'][chunk_idx]
                            break
                
                # Format chunk with metadata
                if include_metadata and metadata:
                    heading = f"Section: {metadata.get('potential_heading', 'Unknown')}" if metadata.get('potential_heading') else ""
                    position = f"Position: {metadata.get('position', 0):.2f}" if 'position' in metadata else ""
                    
                    enhanced_chunk = f"--- DOCUMENT EXCERPT {i+1} ---\n"
                    if heading:
                        enhanced_chunk += f"{heading}\n"
                    if position:
                        enhanced_chunk += f"{position}\n"
                    enhanced_chunk += f"{chunk}\n"
                else:
                    enhanced_chunk = f"--- DOCUMENT EXCERPT {i+1} ---\n{chunk}\n"
                
                enhanced_context.append(enhanced_chunk)
            
            # Combine context
            context_text = "\n\n".join(enhanced_context)
            
            # Construct enhanced prompt
            prompt = f"""
            You are a document question-answering assistant. Your task is to answer the following user question based ONLY on the information provided in the document excerpts below.
            
            USER QUESTION: "{query}"
            
            DOCUMENT EXCERPTS:
            {context_text}
            
            Instructions:
            1. Answer ONLY the user question above: "{query}"
            2. Base your answer ONLY on information in the document excerpts
            3. If the answer cannot be determined from the excerpts, state that clearly and summarize what topics ARE covered in the excerpts
            4. Include specific details from the document excerpts when relevant
            5. If different excerpts contain contradictory information, acknowledge this and explain the discrepancy
            
            IMPORTANT: Many document excerpts contain questions as part of their content.
            - IGNORE ALL QUESTIONS that appear in the document excerpts themselves
            - ONLY answer the user question: "{query}"
            
            If you cannot find information directly related to the user question, explain what topics ARE covered in the provided excerpts so the user understands what information is available.
            """
            
            return prompt
        except Exception as e:
            logging.error(f"Error preparing prompt with context for query '{query}': {str(e)}")
            return ""
    
    def truncate_to_max_tokens(self, prompt: str) -> str:
        """
        Truncates prompt to maximum token length
        
        Returns:
        str: Truncated prompt
        """
        try:
            tokens = self.tokenizer.encode(prompt)
            
            if len(tokens) > self.max_context_tokens:
                # Truncate
                tokens = tokens[:self.max_context_tokens]
                return self.tokenizer.decode(tokens)
            
            return prompt
        except Exception as e:
            logging.error(f"Error truncating prompt to max tokens: {str(e)}")
            return prompt
    
    def generate_response(self,
                           query: str,
                           api_call_function,  # Function for API call
                           top_k: int = 5,
                           adaptive: bool = True,
                           include_metadata: bool = True) -> str:
        """
        Generates response with Enhanced Retrieval Augmented Generation
        
        Parameters:
        query (str): User query
        api_call_function (callable): Function for LLM API call
        top_k (int): Number of context documents (if not adaptive)
        adaptive (bool): Whether to use adaptive context selection
        include_metadata (bool): Whether to include metadata in prompt
        
        Returns:
        str: Generated response
        """
        try:
            # Determine context selection method
            if adaptive and self.use_graph:
                # Use 90% of token budget for context
                context_token_budget = int(self.max_context_tokens * 0.9)
                contexts = self.adaptive_context_selection(query, context_token_budget)
            else:
                # Use fixed top-k retrieval
                contexts = self.retrieve_relevant_context(query, top_k=top_k)
            
            # Prepare prompt with context
            prompt = self.prepare_prompt_with_context(query, contexts, include_metadata)
            
            # Truncate prompt to maximum token length
            truncated_prompt = self.truncate_to_max_tokens(prompt)
            
            # API call with prompt
            response = api_call_function(truncated_prompt)
            
            return response
        except Exception as e:
            logging.error(f"Error generating response for query '{query}': {str(e)}")
            return "Error generating response."
    
    def save_system_state(self, directory="rag_data"):
        """
        Save the system state using a hybrid approach
        
        Parameters:
        directory (str): Directory to save the system state
        
        Returns:
        None
        """
        try:
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
        except Exception as e:
            logging.error(f"Error saving system state: {str(e)}")
    
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
        try:
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
            
            # 5. Initialize embedding model and tokenizer
            self.embedding_model = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.max_context_tokens = int(os.getenv('MAX_CONTEXT_TOKENS', 3500))
            
            # 6. Build chunk graph for enhanced retrieval if enabled
            if hasattr(self, 'use_graph') and self.use_graph:
                self.build_chunk_graph()
            
            print(f"System state loaded from {directory}")
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error loading system state: {str(e)}")
            raise
    
    def split_text_with_metadata(self, text: str) -> Tuple[List[str], List[Dict]]:
        """
        Splits text into chunks and generates metadata for each chunk
        
        Parameters:
        text (str): Text to be split
        
        Returns:
        Tuple[List[str], List[Dict]]: Chunks and their metadata
        """
        # Split text into chunks using the efficient function
        chunks = split_text_efficiently(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        
        # Generate metadata for each chunk
        metadata = []
        for i, chunk in enumerate(chunks):
            # Extract potential headings (simple heuristic)
            lines = chunk.split('\n')
            potential_heading = lines[0] if lines and len(lines[0]) < 100 else ""
            
            # Create metadata
            chunk_metadata = {
                'index': i,
                'position': i / len(chunks),  # Normalized position in document
                'potential_heading': potential_heading,
                'length': len(chunk.split()),  # Word count
                'neighbors': [max(0, i-1), min(len(chunks)-1, i+1)]  # Adjacent chunks
            }
            metadata.append(chunk_metadata)
        
        return chunks, metadata
    
    def build_chunk_graph(self):
        """
        Builds a graph representation of chunks to capture relationships
        """
        if not self.use_graph:
            return
            
        self.chunk_graph = nx.Graph()
        
        # Add nodes for all chunks
        all_embeddings = []
        chunk_to_doc = {}  # Maps global chunk index to document
        global_idx = 0
        
        for doc_path, doc_data in self.documents.items():
            for i, embedding in enumerate(doc_data['embeddings']):
                self.chunk_graph.add_node(global_idx,
                                         doc_path=doc_path,
                                         local_idx=i,
                                         text=doc_data['chunks'][i],
                                         metadata=doc_data['metadata'][i] if 'metadata' in doc_data else {})
                all_embeddings.append(embedding)
                chunk_to_doc[global_idx] = (doc_path, i)
                global_idx += 1
        
        # Convert to numpy array for efficient similarity computation
        all_embeddings = np.array(all_embeddings)
        
        # Add edges based on similarity and document structure
        for i in range(len(all_embeddings)):
            doc_path, local_idx = chunk_to_doc[i]
            
            # Add edges to adjacent chunks in the same document
            if local_idx > 0:
                prev_global_idx = next(idx for idx, (path, lidx) in chunk_to_doc.items()
                                      if path == doc_path and lidx == local_idx - 1)
                self.chunk_graph.add_edge(i, prev_global_idx, weight=1.0, type='adjacent')
            
            if local_idx < len(self.documents[doc_path]['chunks']) - 1:
                next_global_idx = next(idx for idx, (path, lidx) in chunk_to_doc.items()
                                      if path == doc_path and lidx == local_idx + 1)
                self.chunk_graph.add_edge(i, next_global_idx, weight=1.0, type='adjacent')
            
            # Add edges to semantically similar chunks (across all documents)
            # This is computationally expensive, so we limit to top-k similar chunks
            similarities = cosine_similarity([all_embeddings[i]], all_embeddings)[0]
            top_k_indices = np.argsort(similarities)[-6:-1]  # Top 5 excluding self
            
            for j in top_k_indices:
                if i != j:  # Avoid self-loops
                    self.chunk_graph.add_edge(i, j, weight=float(similarities[j]), type='semantic')
    
    def expand_context_with_graph(self, seed_indices: List[int],
                                 query_embedding: np.ndarray,
                                 max_tokens: Optional[int] = None) -> List[str]:
        """
        Expands the context using the chunk graph
        
        Parameters:
        seed_indices (List[int]): Initial chunk indices
        query_embedding (np.ndarray): Query embedding
        max_tokens (int, optional): Maximum tokens to include
        
        Returns:
        List[str]: Expanded context chunks
        """
        if not self.use_graph or not hasattr(self, 'chunk_graph'):
            return [self.index_texts[i] for i in seed_indices]
            
        # Convert global indices to graph nodes
        seed_nodes = []
        for idx in seed_indices:
            # Find the corresponding node in the graph
            for node in self.chunk_graph.nodes():
                if self.chunk_graph.nodes[node].get('text') == self.index_texts[idx]:
                    seed_nodes.append(node)
                    break
        
        # Use personalized PageRank to find important related chunks
        personalization = {node: 0 for node in self.chunk_graph.nodes()}
        for node in seed_nodes:
            personalization[node] = 1.0
            
        # Normalize personalization dict
        if sum(personalization.values()) > 0:
            norm_factor = sum(personalization.values())
            personalization = {k: v/norm_factor for k, v in personalization.items()}
            
        # Run personalized PageRank
        pagerank = nx.pagerank(self.chunk_graph, alpha=0.85, personalization=personalization)
        
        # Sort nodes by PageRank score
        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        
        # Get expanded chunks
        expanded_chunks = []
        total_tokens = 0
        
        for node, score in sorted_nodes[:20]:  # Limit to top 20 chunks
            chunk_text = self.chunk_graph.nodes[node]['text']
            
            # Check token count if max_tokens is specified
            if max_tokens is not None:
                chunk_tokens = len(self.tokenizer.encode(chunk_text))
                if total_tokens + chunk_tokens > max_tokens:
                    continue
                total_tokens += chunk_tokens
            
            expanded_chunks.append(chunk_text)
            
            # Stop if we've reached the token limit
            if max_tokens is not None and total_tokens >= max_tokens:
                break
        
        return expanded_chunks
    
    def adaptive_context_selection(self, query: str, max_tokens: int) -> List[str]:
        """
        Adaptively selects context based on query complexity and token budget
        
        Parameters:
        query (str): User query
        max_tokens (int): Maximum tokens for context
        
        Returns:
        List[str]: Selected context chunks
        """
        # Analyze query complexity
        query_words = query.split()
        query_complexity = min(1.0, len(query_words) / 20)  # Normalize, max at 20 words
        
        # Determine initial retrieval parameters based on complexity
        base_k = 3
        complex_k = 8
        initial_k = int(base_k + (complex_k - base_k) * query_complexity)
        
        # Get initial chunks
        initial_chunks = self.retrieve_relevant_context(query, top_k=initial_k, use_graph=False)
        
        # Count tokens in initial chunks
        initial_tokens = sum(len(self.tokenizer.encode(chunk)) for chunk in initial_chunks)
        
        # If we have budget for more context, use graph expansion
        if initial_tokens < max_tokens * 0.8 and self.use_graph:  # Leave 20% buffer
            remaining_tokens = max_tokens - initial_tokens
            query_embedding = self.embedding_model.encode([query])[0].astype('float32')
            
            # Find indices of initial chunks
            initial_indices = []
            for chunk in initial_chunks:
                try:
                    idx = self.index_texts.index(chunk)
                    initial_indices.append(idx)
                except ValueError:
                    continue
            
            # Expand context with remaining token budget
            expanded_chunks = self.expand_context_with_graph(
                initial_indices,
                query_embedding,
                max_tokens=remaining_tokens
            )
            
            # Combine initial and expanded chunks, removing duplicates
            all_chunks = initial_chunks.copy()
            for chunk in expanded_chunks:
                if chunk not in all_chunks:
                    all_chunks.append(chunk)
            
            return all_chunks
        
        return initial_chunks
    
    def is_document_processed(self, pdf_path, directory="rag_data"):
        """
        Check if a document is already processed
        
        Parameters:
        pdf_path (str): Path to the PDF document
        directory (str): Directory containing the system state
        
        Returns:
        bool: True if the document is already processed, False otherwise
        """
        try:
            # First check in-memory state
            if hasattr(self, 'pdf_paths') and pdf_path in self.pdf_paths:
                return True
                
            # Then use the lightweight document checker
            from document_checker import is_document_processed
            return is_document_processed(pdf_path, directory)
        except Exception as e:
            logging.error(f"Error checking if document is processed: {str(e)}")
            return False

    def switch_rag_folder(self, new_folder: str) -> None:
        """
        Switches the RAG folder and reloads the system state
        
        Parameters:
        new_folder (str): Path to the new RAG folder
        
        Returns:
        None
        """
        try:
            # Check if the new folder exists
            if not os.path.exists(new_folder):
                raise FileNotFoundError(f"The folder {new_folder} does not exist.")
            
            # Load the system state from the new folder
            self.load_system_state(new_folder)
            
            print(f"Switched to new RAG folder: {new_folder}")
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
        except Exception as e:
            logging.error(f"Error switching to new RAG folder {new_folder}: {str(e)}")
