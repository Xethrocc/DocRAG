import re
from typing import List

def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks with consideration for semantic boundaries.
    
    Parameters:
    text (str): Text to be split
    chunk_size (int): Maximum number of words per chunk
    overlap (int): Overlap between chunks in words
    
    Returns:
    List[str]: Text chunks with semantic boundaries
    """
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            # If the sentence alone is too large, split it further
            if sentence_length > chunk_size:
                # Add the current chunk if it's not empty
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep overlap for the next chunk
                    overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_words
                    current_size = len(overlap_words)
                
                # Split the long sentence into smaller pieces
                for i in range(0, sentence_length, chunk_size - overlap):
                    sentence_chunk = sentence_words[i:i + chunk_size]
                    if i + chunk_size >= sentence_length:  # Last piece of the sentence
                        current_chunk.extend(sentence_chunk)
                        current_size += len(sentence_chunk)
                    else:  # Complete chunk from the sentence
                        new_chunk = sentence_chunk
                        chunks.append(' '.join(new_chunk))
                        # Keep overlap for the next chunk
                        current_chunk = new_chunk[-overlap:] if len(new_chunk) > overlap else new_chunk
                        current_size = len(current_chunk)
            
            # Normal case: Add sentence to current chunk if it fits
            elif current_size + sentence_length <= chunk_size:
                current_chunk.extend(sentence_words)
                current_size += sentence_length
            
            # Sentence doesn't fit in the current chunk
            else:
                # Save the current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start a new chunk with overlap
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + sentence_words
                current_size = len(current_chunk)
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
