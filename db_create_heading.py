import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import torch
import chromadb
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import string
from typing import List, Dict, Tuple, Optional

# Load the spaCy model
nlp = spacy.load('en_core_web_md')

# -------------------------
# Heading Detection Functions
# -------------------------

def is_heading(line: str, next_line: Optional[str] = None) -> bool:
    """
    Determine if a line is a true section heading based on multiple criteria.
    
    Args:
        line: The line to check
        next_line: The following line (for context)
    
    Returns:
        bool: True if the line is likely a heading
    """
    line = line.strip()
    
    # Must start with one or more '#' symbols
    if not line.startswith('#'):
        return False
    
    # Extract the text after '#' symbols
    heading_text = re.sub(r'^#+\s*', '', line).strip()
    
    # Skip if empty after removing '#'
    if not heading_text:
        return False
    
    # Check length - headings should be short (less than 10 words)
    word_count = len(heading_text.split())
    if word_count >= 10:
        return False
    
    # Check if it ends with punctuation (headings typically don't)
    if heading_text.endswith(('.', ':', '?', ';', '!')):
        return False
    
    # Check if it's mostly uppercase or title case
    if not (heading_text.isupper() or heading_text.istitle() or is_mostly_capitalized(heading_text)):
        return False
    
    # Check if it's followed by normal text (if next_line is provided)
    if next_line:
        next_line = next_line.strip()
        if next_line and not next_line.startswith('#'):
            # Good sign - followed by normal text
            return True
    
    return True

def is_mostly_capitalized(text: str) -> bool:
    """
    Check if text is mostly capitalized (good indicator of headings).
    
    Args:
        text: Text to check
    
    Returns:
        bool: True if mostly capitalized
    """
    # Remove punctuation and numbers for analysis
    letters_only = ''.join(c for c in text if c.isalpha())
    if not letters_only:
        return False
    
    # Consider it mostly capitalized if >70% of letters are uppercase
    uppercase_count = sum(1 for c in letters_only if c.isupper())
    return uppercase_count / len(letters_only) > 0.7

def extract_heading_text(line: str) -> str:
    """
    Extract the actual heading text from a line with '#' markers.
    
    Args:
        line: Line starting with '#' symbols
    
    Returns:
        str: Clean heading text
    """
    return re.sub(r'^#+\s*', '', line.strip())

# -------------------------
# Semantic Chunking within Sections
# -------------------------

def semantic_chunking_within_section(text: str, embedding_model, max_chunk_size: int = 300, 
                                   similarity_threshold: float = 0.75) -> List[str]:
    """
    Split text into semantically coherent chunks using sentence embeddings.
    This is applied within each heading section.
    
    Args:
        text: Input text to chunk
        embedding_model: SentenceTransformer model for embeddings
        max_chunk_size: Maximum number of words per chunk
        similarity_threshold: Cosine similarity threshold for grouping sentences
    
    Returns:
        List[str]: List of text chunks
    """
    # Extract sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        return []
    
    # If text is short enough, return as single chunk
    total_words = len(text.split())
    if total_words <= max_chunk_size:
        return [text]
    
    # Generate embeddings for all sentences
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    # Convert to numpy for sklearn compatibility
    embeddings_np = sentence_embeddings.cpu().numpy()
    
    chunks = []
    current_chunk = []
    current_indices = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_word_count = len(sentence.split())
        
        # If adding this sentence would exceed max_chunk_size, finalize current chunk
        if current_chunk and current_word_count + sentence_word_count > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_indices = []
            current_word_count = 0
            continue
        
        # Add current sentence to chunk
        current_chunk.append(sentence)
        current_indices.append(i)
        current_word_count += sentence_word_count
        
        # Look ahead to group semantically similar sentences
        j = i + 1
        while j < len(sentences) and current_word_count < max_chunk_size:
            next_sentence = sentences[j]
            next_word_count = len(next_sentence.split())
            
            # Check if adding next sentence would exceed limit
            if current_word_count + next_word_count > max_chunk_size:
                break
            
            # Calculate semantic similarity between current chunk and next sentence
            if len(current_chunk) > 0:
                # Use average embedding of current chunk
                current_chunk_embedding = np.mean(embeddings_np[current_indices], axis=0).reshape(1, -1)
                next_sentence_embedding = embeddings_np[j].reshape(1, -1)
                
                similarity = cosine_similarity(current_chunk_embedding, next_sentence_embedding)[0][0]
                
                # If similarity is above threshold, add to current chunk
                if similarity >= similarity_threshold:
                    current_chunk.append(next_sentence)
                    current_indices.append(j)
                    current_word_count += next_word_count
                    j += 1
                else:
                    # Similarity too low, break the sequence
                    break
            else:
                # First sentence in chunk, just add it
                current_chunk.append(next_sentence)
                current_indices.append(j)
                current_word_count += next_word_count
                j += 1
        
        i = j if j > i + 1 else i + 1
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# -------------------------
# Main Heading-Based Chunking Function
# -------------------------

def heading_based_chunking(lines: List[str], embedding_model, max_chunk_size: int = 300, 
                         similarity_threshold: float = 0.75) -> List[Dict]:
    """
    Chunk text using headings as natural boundaries, with semantic chunking within sections.
    
    Args:
        lines: List of text lines from the document
        embedding_model: SentenceTransformer model for embeddings
        max_chunk_size: Maximum words per semantic chunk within sections
        similarity_threshold: Similarity threshold for semantic chunking
    
    Returns:
        List[Dict]: List of chunk dictionaries with heading, text, and chunk_id
    """
    chunks = []
    current_heading = None
    current_section_lines = []
    chunk_counter = 0
    
    # Process each line
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check if this line is a heading
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if is_heading(line, next_line):
            # Process the previous section if it exists
            if current_section_lines:
                section_text = '\n'.join(current_section_lines)
                section_chunks = process_section(section_text, current_heading, embedding_model, 
                                               max_chunk_size, similarity_threshold, chunk_counter)
                chunks.extend(section_chunks)
                chunk_counter += len(section_chunks)
            
            # Start new section
            current_heading = extract_heading_text(line)
            current_section_lines = []
        else:
            # Add line to current section
            current_section_lines.append(line)
    
    # Process the last section
    if current_section_lines:
        section_text = '\n'.join(current_section_lines)
        section_chunks = process_section(section_text, current_heading, embedding_model, 
                                       max_chunk_size, similarity_threshold, chunk_counter)
        chunks.extend(section_chunks)
    
    return chunks

def process_section(section_text: str, heading: Optional[str], embedding_model, 
                   max_chunk_size: int, similarity_threshold: float, 
                   chunk_counter: int) -> List[Dict]:
    """
    Process a section of text under a heading, applying semantic chunking.
    
    Args:
        section_text: The text content of the section
        heading: The section heading (can be None for text before first heading)
        embedding_model: SentenceTransformer model
        max_chunk_size: Maximum words per chunk
        similarity_threshold: Similarity threshold for semantic chunking
        chunk_counter: Starting counter for chunk IDs
    
    Returns:
        List[Dict]: List of chunk dictionaries
    """
    if not section_text.strip():
        return []
    
    # Apply semantic chunking within this section
    text_chunks = semantic_chunking_within_section(
        section_text, embedding_model, max_chunk_size, similarity_threshold
    )
    
    # Create chunk dictionaries
    section_chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_dict = {
            'chunk_id': f"chunk_{chunk_counter + i:04d}",
            'heading': heading if heading else "Introduction",  # Default heading for text before first heading
            'text': chunk_text.strip(),
            'word_count': len(chunk_text.split()),
            'chunk_method': 'heading_semantic'
        }
        section_chunks.append(chunk_dict)
    
    return section_chunks

# -------------------------
# Updated Main Processing Function
# -------------------------

def process_document_with_heading_chunks(text_folder: str, collection_name: str, 
                                       max_chunk_size: int = 300, 
                                       similarity_threshold: float = 0.75):
    """
    Process documents using heading-based chunking with semantic chunking within sections.
    
    Args:
        text_folder: Path to folder containing text files
        collection_name: Name for the ChromaDB collection
        max_chunk_size: Maximum words per semantic chunk
        similarity_threshold: Similarity threshold for semantic chunking
    """
    # Load embedding model
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="chromadb")
    collection = client.get_or_create_collection(f"{collection_name}_heading_semantic")
    
    # Get text files
    txt_files = sorted(
        [f for f in os.listdir(text_folder) if f.startswith("page_") and f.endswith(".txt")],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    
    total_chunks_processed = 0
    
    with tqdm(total=len(txt_files), desc="Processing Pages") as page_bar:
        for txt_file in txt_files:
            page_path = os.path.join(text_folder, txt_file)
            page_number = int(re.search(r'\d+', txt_file).group())
            
            # Read the file
            with open(page_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split into lines
            lines = content.split('\n')
            
            # Apply heading-based chunking
            chunks = heading_based_chunking(lines, embedding_model, max_chunk_size, similarity_threshold)
            
            if not chunks:
                page_bar.update(1)
                continue
            
            # Generate embeddings and store in ChromaDB
            for chunk in chunks:
                # Generate embedding for the chunk text
                embedding = embedding_model.encode(chunk['text'], convert_to_tensor=True)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                
                # Create unique ID
                chunk_id = f"{collection_name}_p{page_number}_{chunk['chunk_id']}"
                
                # Metadata
                metadata = {
                    "chunk": chunk['text'],
                    "heading": chunk['heading'],
                    "page": page_number,
                    "document": collection_name,
                    "chunk_method": chunk['chunk_method'],
                    "chunk_size": chunk['word_count']
                }
                
                # Add to ChromaDB
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata]
                )
                
                total_chunks_processed += 1
            
            page_bar.update(1)
    
    print(f"Heading-based semantic chunking completed!")
    print(f"Total pages processed: {len(txt_files)}")
    print(f"Total chunks created: {total_chunks_processed}")
    print(f"Collection: {collection_name}_heading_semantic")

# -------------------------
# Execute the Processing
# -------------------------

if __name__ == "__main__":
    # Configuration
    TEXT_FOLDER = "ril_pdf_pages"
    COLLECTION_NAME = Path(TEXT_FOLDER).name
    
    # Process the documents using the main function
    process_document_with_heading_chunks(
        text_folder=TEXT_FOLDER,
        collection_name=COLLECTION_NAME,
        max_chunk_size=400,  # Match your original max_chunk_size
        similarity_threshold=0.8  # Match your original similarity_threshold
    )