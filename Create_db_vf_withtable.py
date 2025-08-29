import os
import re
import torch
import chromadb
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Dict, Tuple, Union
import logging
import json
from datetime import datetime
import hashlib

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    'TEXT_FILES': [
    'combined_output_Zurich_withpage.txt'
        # Add more files here as needed
    ],
    'TEXT_DIRECTORY': None,  # Optional: specify directory to process all .txt files
    'FILE_PATTERN': "*.txt",  # Pattern to match files in directory
    'EMBEDDING_MODEL': 'BAAI/bge-base-en-v1.5',
    'TARGET_CHUNK_SIZE': 120,
    'MAX_CHUNK_SIZE': 250,
    'MIN_CHUNK_SIZE': 10,
    'SIMILARITY_THRESHOLD': 0.6,
    'OVERLAP_SENTENCES': 1,
    'SPACY_MODEL': 'en_core_web_md',
    'CHROMADB_PATH': "chromadb",
    'COLLECTION_NAME': "zurich_collection",  # Single collection for all files
    'SEPARATE_COLLECTIONS': False,  # Set to True to create separate collections per file
}

# -------------------------
# Setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

nlp = spacy.load(CONFIG['SPACY_MODEL'])

# -------------------------
# File Management
# -------------------------
def get_files_to_process() -> List[Path]:
    """Get list of files to process based on configuration."""
    files = []
    
    # Add explicitly specified files
    if CONFIG['TEXT_FILES']:
        for file_path in CONFIG['TEXT_FILES']:
            path = Path(file_path)
            if path.exists():
                files.append(path)
            else:
                logger.warning(f"File not found: {file_path}")
    
    # Add files from directory if specified
    if CONFIG['TEXT_DIRECTORY']:
        directory = Path(CONFIG['TEXT_DIRECTORY'])
        if directory.exists() and directory.is_dir():
            pattern_files = list(directory.glob(CONFIG['FILE_PATTERN']))
            files.extend(pattern_files)
            logger.info(f"Found {len(pattern_files)} files in directory: {directory}")
        else:
            logger.warning(f"Directory not found: {CONFIG['TEXT_DIRECTORY']}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file in files:
        file_resolved = file.resolve()
        if file_resolved not in seen:
            seen.add(file_resolved)
            unique_files.append(file_resolved)
    
    logger.info(f"Total files to process: {len(unique_files)}")
    return unique_files

def get_file_hash(file_path: Path) -> str:
    """Generate a hash for the file to track changes."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_file_metadata(file_path: Path) -> Dict:
    """Extract metadata from file."""
    stat = file_path.stat()
    return {
        'filename': file_path.name,
        'filepath': str(file_path),
        'file_size_bytes': stat.st_size,
        'file_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'file_hash': get_file_hash(file_path),
        'processing_timestamp': datetime.now().isoformat()
    }

# -------------------------
# Helper: Split by Page Marker
# -------------------------
def split_document_by_pages(text: str) -> List[Tuple[int, str]]:
    """Split document into (page_number, page_content) tuples."""
    pattern = re.compile(r'###\s*Page\s+(\d+)\s*###', re.IGNORECASE)
    splits = pattern.split(text)
    pages = []

    if splits[0].strip():  # Handle content before first page marker
        pages.append((0, splits[0].strip()))

    for i in range(1, len(splits), 2):
        try:
            page_num = int(splits[i])
            content = splits[i + 1].strip()
            pages.append((page_num, content))
        except (ValueError, IndexError):
            continue

    return pages

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
    semantic_chunks = semantic_chunking(
        cleaned_text, 
        embedding_model, 
        config=CONFIG
    )
    
    # Step 3: Combine table chunks and semantic chunks
    all_chunks = semantic_chunks + tables
    return all_chunks







# -------------------------
# Sentence Utilities
# -------------------------
def split_oversized_sentence(sentence: str, max_words: int) -> List[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    final_parts = []
    for i in range(0, len(words), max_words):
        part = " ".join(words[i:i + max_words])
        final_parts.append(part)
    return final_parts

def get_last_sentences(text: str, num_sentences: int) -> str:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return " ".join(sentences[-num_sentences:]) if len(sentences) > num_sentences else text

def add_context_overlap(chunks: List[str], overlap_sentences: int) -> List[str]:
    if overlap_sentences <= 0 or len(chunks) < 2:
        return chunks
    enhanced = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            overlap = get_last_sentences(chunks[i - 1], overlap_sentences)
            chunk = f"{overlap} {chunk}"
        enhanced.append(chunk)
    return enhanced

# -------------------------
# Semantic Chunking
# -------------------------
def semantic_chunking(text: str, embedding_model, config=CONFIG) -> List[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    processed = []

    # Handle oversized sentences by splitting them, but keep track of original boundaries
    sentence_boundaries = []  # Track which processed items are sentence starts
    for sent in sentences:
        words = sent.split()
        if len(words) > config['MAX_CHUNK_SIZE'] // 2:
            split_parts = split_oversized_sentence(sent, config['MAX_CHUNK_SIZE'] // 2)
            for idx, part in enumerate(split_parts):
                processed.append(part)
                sentence_boundaries.append(idx == 0)  # Only first part is sentence start
        else:
            processed.append(sent)
            sentence_boundaries.append(True)  # This is a complete sentence

    if not processed:
        return []

    try:
        embs = embedding_model.encode(processed, convert_to_tensor=True)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1).cpu().numpy()
    except Exception as e:
        logger.warning(f"Embedding error: {e}")
        #return [" ".join(processed[i:i + config['TARGET_CHUNK_SIZE']]) for i in range(0, len(processed), config['TARGET_CHUNK_SIZE'])]
        fallback_chunks, bag, bag_len = [], [], 0
        for sent in processed:
            w = len(sent.split())
            if bag and bag_len + w > config['MAX_CHUNK_SIZE']:
                fallback_chunks.append(" ".join(bag))
                bag, bag_len = [], 0
            bag.append(sent)
            bag_len += w
        if bag:
            fallback_chunks.append(" ".join(bag))
        return fallback_chunks
        
    chunks, temp_chunk, idxs, word_count = [], [], [], 0
    i = 0
    
    while i < len(processed):
        sent = processed[i]
        words = len(sent.split())

        # If adding this sentence would exceed MAX_CHUNK_SIZE and we already have a valid chunk
        if word_count + words > config['MAX_CHUNK_SIZE'] and word_count >= config['MIN_CHUNK_SIZE']:
            # Save current chunk and start new one
            chunks.append(" ".join(temp_chunk))
            temp_chunk, idxs, word_count = [sent], [i], words
            i += 1
            continue

        # If this single sentence is larger than MAX_CHUNK_SIZE
        if words > config['MAX_CHUNK_SIZE']:
            # Save current chunk if it exists
            if temp_chunk and word_count >= config['MIN_CHUNK_SIZE']:
                chunks.append(" ".join(temp_chunk))
            # Add the oversized sentence as its own chunk
            chunks.append(sent)
            temp_chunk, idxs, word_count = [], [], 0
            i += 1
            continue

        # Add current sentence to temp chunk
        temp_chunk.append(sent)
        idxs.append(i)
        word_count += words
        j = i + 1

        # Look for similar sentences to add
        exceeded_max = False
        while j < len(processed):
            next_sent = processed[j]
            next_words = len(next_sent.split())
            
            # Calculate similarity first
            curr_emb = np.mean(embs[idxs], axis=0).reshape(1, -1)
            next_emb = embs[j].reshape(1, -1)
            sim = cosine_similarity(curr_emb, next_emb)[0][0]
            
            # Adjust threshold based on current chunk size and whether we've exceeded max
            threshold = config['SIMILARITY_THRESHOLD']
            if word_count >= config['TARGET_CHUNK_SIZE']:
                threshold += 0.015
            if exceeded_max:
                # Once we've exceeded max size, require higher similarity to continue
                threshold += 0.05
                
            if sim >= threshold:
                # Add the sentence even if it exceeds MAX_CHUNK_SIZE (complete sentences priority)
                temp_chunk.append(next_sent)
                idxs.append(j)
                word_count += next_words
                j += 1
                
                # Track if we've exceeded max size
                if word_count > config['MAX_CHUNK_SIZE']:
                    exceeded_max = True

                if word_count > config['MAX_CHUNK_SIZE'] and exceeded_max:
                    break

            else:
                # Similarity too low - stop here
                break
                
        # Move to next unprocessed sentence
        i = j if j > i else i + 1

    # Add final chunk if it meets minimum size
    if temp_chunk and word_count >= config['MIN_CHUNK_SIZE']:
        chunks.append(" ".join(temp_chunk))

    return add_context_overlap(chunks, config['OVERLAP_SENTENCES'])

# -------------------------
# Collection Management
# -------------------------
def get_or_create_collection(client, file_path: Path, config=CONFIG):
    """Get or create ChromaDB collection based on configuration."""
    if config['SEPARATE_COLLECTIONS']:
        collection_name = f"{file_path.stem}_semantic"
    else:
        collection_name = config['COLLECTION_NAME']
    
    return client.get_or_create_collection(collection_name)

def check_file_already_processed(collection, file_path: Path) -> bool:
    """Check if file has already been processed by comparing hash."""
    try:
        current_hash = get_file_hash(file_path)
        # Query for existing chunks from this file
        results = collection.get(
            where={"filepath": str(file_path)},
            limit=1
        )
        
        if results['ids']:
            # Get the file hash from existing metadata
            existing_hash = results['metadatas'][0].get('file_hash')
            if existing_hash == current_hash:
                logger.info(f"File {file_path.name} already processed with same content hash")
                return True
            else:
                logger.info(f"File {file_path.name} has changed, reprocessing...")
                # Delete old chunks from this file
                delete_file_chunks(collection, file_path)
                return False
    except Exception as e:
        logger.warning(f"Error checking if file processed: {e}")
    
    return False

def delete_file_chunks(collection, file_path: Path):
    """Delete all chunks from a specific file."""
    try:
        results = collection.get(
            where={"filepath": str(file_path)}
        )
        if results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} existing chunks from {file_path.name}")
    except Exception as e:
        logger.warning(f"Error deleting existing chunks: {e}")

# -------------------------
# Main Processing Function
# -------------------------
def process_single_file(file_path: Path, embedding_model, collection, file_metadata: Dict) -> int:
    """Process a single file and return number of chunks created."""
    logger.info(f"Processing file: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    pages = split_document_by_pages(full_text)
    if not pages:
        # If no page markers, treat entire file as one page
        pages = [(1, full_text)]
    
    logger.info(f"Detected {len(pages)} pages in {file_path.name}")

    total_chunks = 0
    for page_num, content in tqdm(pages, desc=f"Processing {file_path.name}", leave=False):

        #chunks = semantic_chunking(content, embedding_model)
        chunks= chunk_with_tables(content,embedding_model)
        for i, chunk in enumerate(chunks):
            try:
                emb = embedding_model.encode(chunk, convert_to_tensor=True)
                emb = torch.nn.functional.normalize(emb.squeeze(), p=2, dim=0)
                
                # Create unique chunk ID across all files
                chunk_id = f"{file_path.stem}_page{page_num}_chunk{i:03d}"
                
                # Enhanced metadata combining file info and chunk info
                metadata = {
                    **file_metadata,  # File-level metadata
                    "page": page_num,
                    "chunk_index": i,
                    "chunk_text": chunk[:1000],  # Truncated for storage
                    "chunk_method": "semantic_page_split",
                    "chunk_size_words": len(chunk.split()),
                    "chunk_size_chars": len(chunk),
                    "has_overlap": CONFIG['OVERLAP_SENTENCES'] > 0 and i > 0,
                    "chunk_id": chunk_id
                }
                
                collection.add(
                    ids=[chunk_id],
                    embeddings=[emb.tolist()],
                    metadatas=[metadata]
                )
                total_chunks += 1
                
            except Exception as e:
                logger.warning(f"Chunk embedding failed ({file_path.name}, page {page_num}, chunk {i}): {e}")
                logger.warning(f"Chunk length: {len(chunk)} chars, {len(chunk.split())} words")
                logger.warning(f"Chunk preview: '{chunk[:100]}...'")

    return total_chunks

def process_multiple_files():
    """Main function to process multiple files."""
    files_to_process = get_files_to_process()
    
    if not files_to_process:
        logger.error("No files found to process")
        return

    # Initialize embedding model
    logger.info(f"Loading embedding model: {CONFIG['EMBEDDING_MODEL']}")
    embedding_model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CONFIG['CHROMADB_PATH'])
    
    # Processing statistics
    total_files_processed = 0
    total_chunks_created = 0
    skipped_files = []
    
    for file_path in tqdm(files_to_process, desc="Processing files"):
        try:
            # Get collection for this file
            collection = get_or_create_collection(client, file_path)
            
            # Check if file already processed
            if check_file_already_processed(collection, file_path):
                skipped_files.append(file_path.name)
                continue
            
            # Get file metadata
            file_metadata = get_file_metadata(file_path)
            
            # Process the file
            chunks_created = process_single_file(file_path, embedding_model, collection, file_metadata)
            
            total_files_processed += 1
            total_chunks_created += chunks_created
            
            logger.info(f"✅ {file_path.name}: {chunks_created} chunks created")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path.name}: {e}")
            continue

    # Final summary
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files processed: {total_files_processed}")
    logger.info(f"Files skipped (unchanged): {len(skipped_files)}")
    logger.info(f"Total chunks created: {total_chunks_created}")
    
    if skipped_files:
        logger.info(f"Skipped files: {', '.join(skipped_files)}")
    
    # Collection info
    if not CONFIG['SEPARATE_COLLECTIONS']:
        collection = client.get_collection(CONFIG['COLLECTION_NAME'])
        total_items = collection.count()
        logger.info(f"Total items in collection '{CONFIG['COLLECTION_NAME']}': {total_items}")
    
    logger.info("=" * 60)

# -------------------------
# Utility Functions
# -------------------------
def list_collections():
    """List all collections in the database."""
    client = chromadb.PersistentClient(path=CONFIG['CHROMADB_PATH'])
    collections = client.list_collections()
    logger.info("Available collections:")
    for collection in collections:
        count = collection.count()
        logger.info(f"  - {collection.name}: {count} items")

def query_collection(query_text: str, collection_name: str = None, n_results: int = 5):
    """Query a collection for similar chunks."""
    client = chromadb.PersistentClient(path=CONFIG['CHROMADB_PATH'])
    
    if collection_name is None:
        collection_name = CONFIG['COLLECTION_NAME']
    
    try:
        collection = client.get_collection(collection_name)
        embedding_model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])
        
        query_embedding = embedding_model.encode([query_text])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        logger.info(f"Query results for: '{query_text}'")
        for i, (chunk_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
            logger.info(f"{i+1}. {chunk_id}")
            logger.info(f"   File: {metadata['filename']} (Page: {metadata['page']})")
            logger.info(f"   Text: {metadata['chunk_text'][:200]}...")
            logger.info("")
            
    except Exception as e:
        logger.error(f"Query failed: {e}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-file semantic chunking processor")
    parser.add_argument("--list-collections", action="store_true", help="List all collections")
    parser.add_argument("--query", type=str, help="Query text to search for")
    parser.add_argument("--collection", type=str, help="Collection name to query")
    parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.list_collections:
        list_collections()
    elif args.query:
        query_collection(args.query, args.collection, args.results)
    else:
        process_multiple_files()