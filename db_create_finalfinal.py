'''
# -------------------------
# Updated Configuration for Dual-Purpose Chunking
# -------------------------
CONFIG = {
    'TEXT_FILE': "combined_novartis_report/combined_novartis.txt",
    'EMBEDDING_MODEL': 'BAAI/bge-base-en-v1.5',
    'TARGET_CHUNK_SIZE': 250,      # Optimal for both Q&A and search
    'MAX_CHUNK_SIZE': 450,         # Hard limit to prevent huge chunks
    'MIN_CHUNK_SIZE': 50,          # Prevent tiny fragments
    'SIMILARITY_THRESHOLD': 0.68,   # Balanced threshold
    'OVERLAP_SENTENCES': 2,         # Context overlap between chunks
    'SPACY_MODEL': 'en_core_web_md',
    'CHROMADB_PATH': "chromadb"
}

# -------------------------
# Helper Functions
# -------------------------

def get_last_sentences(text: str, num_sentences: int) -> str:
    """Extract the last N sentences from a text."""
    if not text or num_sentences <= 0:
        return ""
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if len(sentences) <= num_sentences:
        return text
    
    return " ".join(sentences[-num_sentences:])


def get_first_sentences(text: str, num_sentences: int) -> str:
    """Extract the first N sentences from a text."""
    if not text or num_sentences <= 0:
        return ""
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if len(sentences) <= num_sentences:
        return text
    
    return " ".join(sentences[:num_sentences])


def split_oversized_sentence(sentence: str, max_words: int) -> List[str]:
    """Intelligently split a sentence that's too long."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    
    # Try to split at natural breakpoints first
    parts = []
    
    # Split by punctuation and conjunctions
    segments = re.split(r'([.!?;]|\s+(?:and|but|or|however|therefore|meanwhile|furthermore|moreover|additionally)\s+)', 
                       sentence, flags=re.IGNORECASE)
    
    current_segment = ""
    for segment in segments:
        if not segment.strip():
            continue
            
        test_segment = current_segment + " " + segment if current_segment else segment
        if len(test_segment.split()) <= max_words:
            current_segment = test_segment
        else:
            if current_segment:
                parts.append(current_segment.strip())
            current_segment = segment
    
    if current_segment:
        parts.append(current_segment.strip())
    
    # If still too long, force split by words
    final_parts = []
    for part in parts:
        part_words = part.split()
        if len(part_words) <= max_words:
            final_parts.append(part)
        else:
            # Force split
            for i in range(0, len(part_words), max_words):
                sub_part = " ".join(part_words[i:i + max_words])
                if sub_part.strip():
                    final_parts.append(sub_part)
    
    return [part for part in final_parts if part.strip()]


# -------------------------
# Enhanced Semantic Chunking Function
# -------------------------

def semantic_chunking_within_section(text: str, embedding_model, 
                                   target_chunk_size: int = None,
                                   max_chunk_size: int = None,
                                   min_chunk_size: int = None,
                                   similarity_threshold: float = None,
                                   overlap_sentences: int = None) -> List[str]:
    """
    Split text into semantically coherent chunks optimized for both Q&A and search.
    
    Args:
        text: Input text to chunk
        embedding_model: SentenceTransformer model for embeddings
        target_chunk_size: Target number of words per chunk (soft limit)
        max_chunk_size: Maximum number of words per chunk (hard limit)
        min_chunk_size: Minimum number of words per chunk
        similarity_threshold: Cosine similarity threshold for grouping sentences
        overlap_sentences: Number of sentences to overlap between chunks
    
    Returns:
        List[str]: List of text chunks with optimal sizing and overlap
    """
    # Use config values if not provided
    if target_chunk_size is None:
        target_chunk_size = CONFIG['TARGET_CHUNK_SIZE']
    if max_chunk_size is None:
        max_chunk_size = CONFIG['MAX_CHUNK_SIZE']
    if min_chunk_size is None:
        min_chunk_size = CONFIG['MIN_CHUNK_SIZE']
    if similarity_threshold is None:
        similarity_threshold = CONFIG['SIMILARITY_THRESHOLD']
    if overlap_sentences is None:
        overlap_sentences = CONFIG['OVERLAP_SENTENCES']
    
    if not text or not text.strip():
        return []
    
    # Extract sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        return []
    
    # Handle oversized sentences intelligently
    processed_sentences = []
    max_sentence_words = max_chunk_size // 2  # Sentences shouldn't be more than half chunk size
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if sentence_words <= max_sentence_words:
            processed_sentences.append(sentence)
        else:
            # Split oversized sentences intelligently
            split_parts = split_oversized_sentence(sentence, max_sentence_words)
            processed_sentences.extend(split_parts)
    
    sentences = processed_sentences
    
    if not sentences:
        return []
    
    try:
        # Generate embeddings for all sentences
        sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # Convert to numpy for sklearn compatibility
        embeddings_np = sentence_embeddings.cpu().numpy()
    except Exception as e:
        logger.warning(f"Error generating embeddings for semantic chunking: {e}")
        # Fallback to simple word-based chunking
        return enhanced_simple_word_chunking(text, target_chunk_size, max_chunk_size, 
                                           min_chunk_size, overlap_sentences)
    
    chunks = []
    current_chunk = []
    current_indices = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_word_count = len(sentence.split())
        
        # If adding this sentence would exceed MAX size, finalize current chunk
        if current_chunk and current_word_count + sentence_word_count > max_chunk_size:
            if current_word_count >= min_chunk_size:  # Only add if meets minimum size
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_indices = [i]
            current_word_count = sentence_word_count
            i += 1
            continue
        
        # Add current sentence to chunk
        current_chunk.append(sentence)
        current_indices.append(i)
        current_word_count += sentence_word_count
        
        # Look ahead to group semantically similar sentences
        j = i + 1
        while j < len(sentences):
            next_sentence = sentences[j]
            next_word_count = len(next_sentence.split())
            
            # Hard limit check - cannot exceed max_chunk_size
            if current_word_count + next_word_count > max_chunk_size:
                break
            
            # If we're already at target size, be more selective about adding more
            similarity_threshold_adjusted = similarity_threshold
            if current_word_count >= target_chunk_size:
                similarity_threshold_adjusted += 0.05  # Require higher similarity for larger chunks
            
            try:
                # Calculate semantic similarity
                current_chunk_embedding = np.mean(embeddings_np[current_indices], axis=0).reshape(1, -1)
                next_sentence_embedding = embeddings_np[j].reshape(1, -1)
                
                similarity = cosine_similarity(current_chunk_embedding, next_sentence_embedding)[0][0]
                
                # Decision logic: semantic similarity vs chunk size
                if similarity >= similarity_threshold_adjusted:
                    # High similarity - add regardless of approaching target size
                    current_chunk.append(next_sentence)
                    current_indices.append(j)
                    current_word_count += next_word_count
                    j += 1
                elif current_word_count < target_chunk_size:
                    # Haven't reached target size - be more lenient with similarity
                    if similarity >= (similarity_threshold - 0.05):
                        current_chunk.append(next_sentence)
                        current_indices.append(j)
                        current_word_count += next_word_count
                        j += 1
                    else:
                        break
                else:
                    # Reached target size and similarity not high enough
                    break
                    
            except Exception as e:
                logger.warning(f"Error calculating similarity: {e}")
                # If similarity calculation fails, use size-based decision
                if current_word_count < target_chunk_size:
                    current_chunk.append(next_sentence)
                    current_indices.append(j)
                    current_word_count += next_word_count
                    j += 1
                else:
                    break
        
        i = j if j > i + 1 else i + 1
    
    # Add the last chunk if it exists and meets minimum size
    if current_chunk and current_word_count >= min_chunk_size:
        chunks.append(" ".join(current_chunk))
    
    # Apply context overlap between chunks
    if overlap_sentences > 0 and len(chunks) > 1:
        chunks = add_context_overlap(chunks, overlap_sentences)
    
    # Final size enforcement and filtering
    final_chunks = []
    for chunk in chunks:
        chunk_words = chunk.split()
        chunk_word_count = len(chunk_words)
        
        # Filter out chunks that are too small
        if chunk_word_count < min_chunk_size:
            continue
            
        # Handle chunks that somehow exceeded max size (should be rare)
        if chunk_word_count <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            logger.warning(f"Chunk exceeded max size ({chunk_word_count} words), splitting...")
            # Smart split at sentence boundaries if possible
            chunk_doc = nlp(chunk)
            chunk_sentences = [sent.text.strip() for sent in chunk_doc.sents if sent.text.strip()]
            
            sub_chunk = []
            sub_word_count = 0
            
            for sent in chunk_sentences:
                sent_words = len(sent.split())
                if sub_word_count + sent_words <= max_chunk_size:
                    sub_chunk.append(sent)
                    sub_word_count += sent_words
                else:
                    if sub_chunk and sub_word_count >= min_chunk_size:
                        final_chunks.append(" ".join(sub_chunk))
                    sub_chunk = [sent]
                    sub_word_count = sent_words
            
            if sub_chunk and sub_word_count >= min_chunk_size:
                final_chunks.append(" ".join(sub_chunk))
    
    return final_chunks


def add_context_overlap(chunks: List[str], overlap_sentences: int) -> List[str]:
    """Add overlapping sentences between chunks for better context continuity."""
    if len(chunks) <= 1 or overlap_sentences <= 0:
        return chunks
    
    enhanced_chunks = []
    
    for i, chunk in enumerate(chunks):
        enhanced_chunk = chunk
        
        # Add overlap from previous chunk (except for first chunk)
        if i > 0:
            prev_overlap = get_last_sentences(chunks[i-1], overlap_sentences)
            if prev_overlap and prev_overlap.strip():
                enhanced_chunk = prev_overlap + " " + chunk
        
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks


def enhanced_simple_word_chunking(text: str, target_size: int, max_size: int, 
                                min_size: int, overlap_sentences: int) -> List[str]:
    """
    Enhanced fallback chunking method with overlap support.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for word in words:
        # If adding word would exceed max size, finalize chunk
        if current_word_count + 1 > max_size and current_chunk:
            if current_word_count >= min_size:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_word_count = 1
        # If at target size, finalize chunk (soft boundary)
        elif current_word_count >= target_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_word_count = 1
        else:
            current_chunk.append(word)
            current_word_count += 1
    
    # Add final chunk if it meets minimum size
    if current_chunk and current_word_count >= min_size:
        chunks.append(" ".join(current_chunk))
    
    # Add overlap if requested
    if overlap_sentences > 0 and len(chunks) > 1:
        chunks = add_context_overlap(chunks, overlap_sentences)
    
    return chunks



'''
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
import logging

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    'TEXT_FILE': "combined_novartis.txt",
    'EMBEDDING_MODEL': 'BAAI/bge-base-en-v1.5',
    'MAX_CHUNK_SIZE': 200,
    'SIMILARITY_THRESHOLD': 0.7,
    'SPACY_MODEL': 'en_core_web_md',
    'CHROMADB_PATH': "chromadb"
}

# Load the spaCy model with error handling
def load_spacy_model():
    """Load spaCy model with error handling."""
    try:
        return spacy.load(CONFIG['SPACY_MODEL'])
    except OSError:
        logger.error(f"SpaCy model '{CONFIG['SPACY_MODEL']}' not found. Please install it with:")
        logger.error(f"python -m spacy download {CONFIG['SPACY_MODEL']}")
        raise

nlp = load_spacy_model()

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
    if not line or not line.strip():
        return False
        
    line = line.strip()

    # Check if line is page number
    if is_pageno(line):
        return False

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
    first_word = heading_text.split()[0] if heading_text.split() else ""
    if not (heading_text.isupper() or heading_text.istitle() or 
            is_mostly_capitalized(heading_text) or 
            (first_word and first_word[0].isupper())):
        return False
    
    # Check if it's followed by normal text (if next_line is provided)
    if next_line:
        next_line = next_line.strip()
        if next_line and next_line.startswith('#'):
            # If next line is also a heading, current line might not be a content heading
            return False
    
    return True


def is_pageno(line: str) -> bool:
    """
    Check if a line is a page number.
    
    Args:
        line: The line to check
    
    Returns:
        bool: True if the line is a page number
    """
    if not line:
        return False
    # Match patterns like 'Page 2', 'p. 2', '2', or surrounded by hashes
    return re.match(r'^(#*\s*)?(Page|p\.)?\s*\d+\s*(#*\s*)?$', line.strip(), re.IGNORECASE) is not None


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


#_-------------------------
# Markdown Table Extraction
# -------------------------
def extract_markdown_tables(text: str) -> Tuple[List[str], str]:
    """
    Extract markdown-style tables and return both the tables and cleaned text.
    
    Returns:
        tables: List of table strings (each table as a block)
        cleaned_text: Text with tables removed
    """
    table_pattern = re.compile(r'(?:\|.*\|\n)+', re.MULTILINE)
    tables = table_pattern.findall(text)
    
    # Remove tables from the original text
    cleaned_text = table_pattern.sub('', text)
    
    return tables, cleaned_text

#--------------------------
# Chunk taking care of tables
#--------------------------
def chunk_with_tables(text: str, embedding_model, max_chunk_size=None, similarity_threshold=None) -> List[str]:
    """
    Handles table extraction before performing semantic chunking.
    
    Returns:
        List of chunks (including each table as a standalone chunk)
    """
    # Step 1: Extract tables and clean text
    tables, cleaned_text = extract_markdown_tables(text)
    
    # Step 2: Chunk the cleaned (non-table) text
    semantic_chunks = semantic_chunking_within_section(
        cleaned_text, 
        embedding_model, 
        max_chunk_size=max_chunk_size, 
        similarity_threshold=similarity_threshold
    )
    
    # Step 3: Combine table chunks and semantic chunks
    all_chunks = semantic_chunks + tables
    return all_chunks



# -------------------------
# Semantic Chunking within Sections
# -------------------------

def semantic_chunking_within_section(text: str, embedding_model, max_chunk_size: int = None, 
                                   similarity_threshold: float = None) -> List[str]:
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
    # Use config values if not provided
    if max_chunk_size is None:
        max_chunk_size = CONFIG['MAX_CHUNK_SIZE']
    if similarity_threshold is None:
        similarity_threshold = CONFIG['SIMILARITY_THRESHOLD']
    
    if not text or not text.strip():
        return []
    
    # Extract sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # Before the main loop, add:
    # Split any sentences that are too long
    max_sentence_words = max_chunk_size // 2 # Allow sentences up to half chunk size
    validated_sentences = []
    for sent in sentences:
        sent_words = sent.split()
        if len(sent_words) <= max_sentence_words:
            validated_sentences.append(sent)
        else:
            # Force split long sentences
            for i in range(0, len(sent_words), max_sentence_words):
                sub_sent = " ".join(sent_words[i:i + max_sentence_words])
                if sub_sent.strip():
                    validated_sentences.append(sub_sent)

    sentences = validated_sentences  # Use the validated sentences


    if not sentences:
        return []
    '''
    # If text is short enough, return as single chunk
    total_words = len(text.split())
    if total_words <= max_chunk_size:
        return [text]
    '''
    try:
        # Generate embeddings for all sentences
        sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        # Convert to numpy for sklearn compatibility
        embeddings_np = sentence_embeddings.cpu().numpy()
    except Exception as e:
        logger.warning(f"Error generating embeddings for semantic chunking: {e}")
        # Fallback to simple word-based chunking
        return simple_word_chunking(text, max_chunk_size)
    
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
            current_chunk = [sentence]  # ← Start new chunk with current sentence
            current_indices = [i]
            current_word_count = sentence_word_count
            i += 1
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
                try:
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
                except Exception as e:
                    logger.warning(f"Error calculating similarity: {e}")
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


def simple_word_chunking(text: str, max_chunk_size: int) -> List[str]:
    """
    Fallback chunking method that splits text by words when semantic chunking fails.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum number of words per chunk
    
    Returns:
        List[str]: List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for word in words:
        if current_word_count + 1 > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_word_count = 1
        else:
            current_chunk.append(word)
            current_word_count += 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# -------------------------
# Page Number Mapping
# -------------------------

def create_line_to_page_mapping(lines: List[str]) -> Dict[int, int]:
    """
    Create a mapping from line indices to page numbers by scanning the document once.
    
    Args:
        lines: List of all lines from the document
    
    Returns:
        Dict[int, int]: Mapping from line index to page number
    """
    line_to_page = {}
    current_page = 1
    
    for i, line in enumerate(lines):
        if not line:
            line_to_page[i] = current_page
            continue
            
        line = line.strip()
        if is_pageno(line):
            # Extract the page number from the line
            match = re.search(r'\d+', line)
            if match:
                current_page = int(match.group())
        
        line_to_page[i] = current_page
    
    return line_to_page

# -------------------------
# Section Processing
# -------------------------

def process_section(section_text: str, heading: Optional[str], embedding_model, 
                   max_chunk_size: int, similarity_threshold: float, 
                   chunk_counter: int, line_indices: List[int], 
                   line_to_page: Dict[int, int]) -> List[Dict]:
    """
    Process a section of text under a heading, applying semantic chunking.
    
    Args:
        section_text: The text content of the section
        heading: The section heading (can be None for text before first heading)
        embedding_model: SentenceTransformer model
        max_chunk_size: Maximum words per chunk
        similarity_threshold: Similarity threshold for semantic chunking
        chunk_counter: Starting counter for chunk IDs
        line_indices: List of line indices for this section
        line_to_page: Mapping from line index to page number
    
    Returns:
        List[Dict]: List of chunk dictionaries
    """
    if not section_text or not section_text.strip():
        return []
    
    # Get page number for this section (from first line of section)
    page_number = line_to_page.get(line_indices[0], 1) if line_indices else 1
    
    # Apply semantic chunking within this section
    text_chunks = chunk_with_tables(section_text, embedding_model)
    
    # Create chunk dictionaries
    section_chunks = []
    for i, chunk_text in enumerate(text_chunks):
        if chunk_text.strip():  # Only add non-empty chunks
            chunk_dict = {
                'chunk_id': f"chunk_{chunk_counter + i:04d}",
                'heading': heading if heading else "No Heading",
                'text': chunk_text.strip(),
                'word_count': len(chunk_text.split()),
                'chunk_method': 'heading_semantic',
                'page_number': page_number
            }
            section_chunks.append(chunk_dict)
    
    return section_chunks

# -------------------------
# Main Heading-Based Chunking Function
# -------------------------

def heading_based_chunking(lines: List[str], embedding_model, max_chunk_size: int = None, 
                         similarity_threshold: float = None) -> List[Dict]:
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
    # Use config values if not provided
    if max_chunk_size is None:
        max_chunk_size = CONFIG['MAX_CHUNK_SIZE']
    if similarity_threshold is None:
        similarity_threshold = CONFIG['SIMILARITY_THRESHOLD']
    
    # Create line-to-page mapping once at the beginning
    logger.info("Creating line-to-page mapping...")
    line_to_page = create_line_to_page_mapping(lines)
    
    chunks = []
    current_heading = None
    current_section_lines = []
    current_section_line_indices = []
    chunk_counter = 0
    
    # Process each line with progress tracking
    logger.info("Processing lines for headings...")
    with tqdm(total=len(lines), desc="Analyzing document structure") as pbar:
        for i, line in enumerate(lines):
            line = line.strip() if line else ""
            
            # Skip empty lines
            if not line:
                pbar.update(1)
                continue
            
            # Check if this line is a heading
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if is_heading(line, next_line):
                # Process the previous section if it exists
                if current_section_lines:
                    section_text = '\n'.join(current_section_lines)
                    section_chunks = process_section(section_text, current_heading, embedding_model, 
                                                   max_chunk_size, similarity_threshold, chunk_counter,
                                                   current_section_line_indices, line_to_page)
                    chunks.extend(section_chunks)
                    chunk_counter += len(section_chunks)
                
                # Start new section
                current_heading = extract_heading_text(line)
                current_section_lines = []
                current_section_line_indices = []
            else:
                # Add line to current section
                current_section_lines.append(line)
                current_section_line_indices.append(i)
            
            pbar.update(1)
    
    # Process the last section
    if current_section_lines:
        logger.info("Processing final section...")
        section_text = '\n'.join(current_section_lines)
        section_chunks = process_section(section_text, current_heading, embedding_model, 
                                       max_chunk_size, similarity_threshold, chunk_counter,
                                       current_section_line_indices, line_to_page)
        chunks.extend(section_chunks)
    
    return chunks

# -------------------------
# Main Processing Function
# -------------------------

def process_document_with_heading_chunks(text_file: str = None, collection_name: str = None, 
                                       max_chunk_size: int = None, 
                                       similarity_threshold: float = None):
    """
    Process a single document file using heading-based chunking with semantic chunking within sections.
    
    Args:
        text_file: Path to the text file (uses CONFIG if None)
        collection_name: Name for the ChromaDB collection (derived from file if None)
        max_chunk_size: Maximum words per semantic chunk (uses CONFIG if None)
        similarity_threshold: Similarity threshold for semantic chunking (uses CONFIG if None)
    """
    # Use config values if not provided
    if text_file is None:
        text_file = CONFIG['TEXT_FILE']
    if max_chunk_size is None:
        max_chunk_size = CONFIG['MAX_CHUNK_SIZE']
    if similarity_threshold is None:
        similarity_threshold = CONFIG['SIMILARITY_THRESHOLD']
    if collection_name is None:
        collection_name = Path(text_file).stem
    
    # Validate file exists
    if not Path(text_file).exists():
        logger.error(f"File not found: {text_file}")
        return
    
    logger.info(f"Starting document processing: {text_file}")
    logger.info(f"Configuration: max_chunk_size={max_chunk_size}, similarity_threshold={similarity_threshold}")
    
    # Load embedding model
    logger.info(f"Loading embedding model: {CONFIG['EMBEDDING_MODEL']}")
    try:
        embedding_model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return
    
    # Initialize ChromaDB
    logger.info("Initializing ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CONFIG['CHROMADB_PATH'])
        collection = client.get_or_create_collection(f"{collection_name}_heading_semantic")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return
    
    # Read the file
    logger.info("Reading document...")
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return
    
    if not content.strip():
        logger.error("Document is empty")
        return
    
    # Split into lines
    lines = content.split('\n')
    logger.info(f"Document contains {len(lines)} lines")
    
    # Apply heading-based chunking
    chunks = heading_based_chunking(lines, embedding_model, max_chunk_size, similarity_threshold)
    
    if not chunks:
        logger.error("No chunks created from the document")
        return
    
    logger.info(f"Created {len(chunks)} chunks")
    
    total_chunks_processed = 0
    failed_chunks = 0
    
    # Generate embeddings and store in ChromaDB
    with tqdm(total=len(chunks), desc="Processing and storing chunks") as chunk_bar:
        for chunk in chunks:
            try:
                # Generate embedding for the chunk text
                embedding = embedding_model.encode(chunk['text'], convert_to_tensor=True)
                embedding = embedding.squeeze() # Remove batch dimension
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                
                # Create unique ID
                chunk_id = f"{collection_name}_{chunk['chunk_id']}"
                
                # Metadata
                metadata = {
                    "chunk": chunk['text'][:1000],  # Limit chunk text in metadata to avoid size issues
                    "heading": chunk['heading'],
                    "page": chunk['page_number'],
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
                
            except Exception as e:
                logger.warning(f"Failed to process chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                failed_chunks += 1
            
            chunk_bar.update(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DOCUMENT PROCESSING COMPLETED!")
    print(f"{'='*60}")
    print(f"File processed: {text_file}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunks successfully stored: {total_chunks_processed}")
    print(f"Failed chunks: {failed_chunks}")
    print(f"Collection: {collection_name}_heading_semantic")
    print(f"\nConfiguration used:")
    print(f"  - Max chunk size: {max_chunk_size} words")
    print(f"  - Similarity threshold: {similarity_threshold}")
    print(f"  - Embedding model: {CONFIG['EMBEDDING_MODEL']}")
    print(f"  - ChromaDB path: {CONFIG['CHROMADB_PATH']}")
    
    # Additional statistics
    if chunks:
        word_counts = [chunk['word_count'] for chunk in chunks]
        print(f"\nChunk statistics:")
        print(f"  - Average chunk size: {np.mean(word_counts):.1f} words")
        print(f"  - Min chunk size: {min(word_counts)} words")
        print(f"  - Max chunk size: {max(word_counts)} words")
        
        headings = set(chunk['heading'] for chunk in chunks)
        print(f"  - Unique headings found: {len(headings)}")
    
    print(f"{'='*60}")

# -------------------------
# Execute the Processing
# -------------------------

if __name__ == "__main__":
    # Process the documents using the main function with centralized configuration
    process_document_with_heading_chunks()