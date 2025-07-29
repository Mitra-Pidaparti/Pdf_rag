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
from typing import List, Dict, Tuple
import logging

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    'TEXT_FILE': "combined_new_novartis.txt",
    'EMBEDDING_MODEL': 'BAAI/bge-base-en-v1.5',
    'TARGET_CHUNK_SIZE': 250,
    'MAX_CHUNK_SIZE': 450,
    'MIN_CHUNK_SIZE': 50,
    'SIMILARITY_THRESHOLD': 0.68,
    'OVERLAP_SENTENCES': 2,
    'SPACY_MODEL': 'en_core_web_md',
    'CHROMADB_PATH': "chromadb"
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
# Helper: Split by Page Marker
# -------------------------
def split_document_by_pages(text: str) -> List[Tuple[int, str]]:
    """Split document into (page_number, page_content) tuples."""
    pattern = re.compile(r'### Page (\d+) ###')
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

    for sent in sentences:
        words = sent.split()
        if len(words) > config['MAX_CHUNK_SIZE'] // 2:
            processed.extend(split_oversized_sentence(sent, config['MAX_CHUNK_SIZE'] // 2))
        else:
            processed.append(sent)

    if not processed:
        return []

    try:
        embs = embedding_model.encode(processed, convert_to_tensor=True)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1).cpu().numpy()
    except Exception as e:
        logger.warning(f"Embedding error: {e}")
        return [" ".join(processed[i:i + config['TARGET_CHUNK_SIZE']]) for i in range(0, len(processed), config['TARGET_CHUNK_SIZE'])]

    chunks, temp_chunk, idxs, word_count = [], [], [], 0
    i = 0
    while i < len(processed):
        sent = processed[i]
        words = len(sent.split())

        if word_count + words > config['MAX_CHUNK_SIZE']:
            if word_count >= config['MIN_CHUNK_SIZE']:
                chunks.append(" ".join(temp_chunk))
            temp_chunk, idxs, word_count = [sent], [i], words
            i += 1
            continue

        temp_chunk.append(sent)
        idxs.append(i)
        word_count += words
        j = i + 1

        while j < len(processed):
            next_sent = processed[j]
            next_words = len(next_sent.split())
            if word_count + next_words > config['MAX_CHUNK_SIZE']:
                break
            curr_emb = np.mean(embs[idxs], axis=0).reshape(1, -1)
            next_emb = embs[j].reshape(1, -1)
            sim = cosine_similarity(curr_emb, next_emb)[0][0]
            threshold = config['SIMILARITY_THRESHOLD']
            if word_count >= config['TARGET_CHUNK_SIZE']:
                threshold += 0.05
            if sim >= threshold:
                temp_chunk.append(next_sent)
                idxs.append(j)
                word_count += next_words
                j += 1
            else:
                break
        i = j if j > i else i + 1

    if temp_chunk and word_count >= config['MIN_CHUNK_SIZE']:
        chunks.append(" ".join(temp_chunk))

    return add_context_overlap(chunks, config['OVERLAP_SENTENCES'])

# -------------------------
# Main Processing Function
# -------------------------
def process_by_page(text_file: str = None):
    if text_file is None:
        text_file = CONFIG['TEXT_FILE']
    if not Path(text_file).exists():
        logger.error(f"File not found: {text_file}")
        return

    with open(text_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    pages = split_document_by_pages(full_text)
    logger.info(f"Detected {len(pages)} pages in the document")

    embedding_model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CONFIG['CHROMADB_PATH'])
    collection_name = Path(text_file).stem + "_page_semantic"
    collection = client.get_or_create_collection(collection_name)

    total_chunks = 0
    for page_num, content in tqdm(pages, desc="Processing pages"):
        chunks = semantic_chunking(content, embedding_model)
        for i, chunk in enumerate(chunks):
            try:
                emb = embedding_model.encode(chunk, convert_to_tensor=True)
                emb = torch.nn.functional.normalize(emb.squeeze(), p=2, dim=0)
                chunk_id = f"page{page_num}_chunk{i:03d}"
                metadata = {
                    "page": page_num,
                    "chunk": chunk[:1000],
                    "chunk_method": "semantic_page_split",
                    "chunk_size": len(chunk.split()),
                }
                collection.add(
                    ids=[chunk_id],
                    embeddings=[emb.tolist()],
                    metadatas=[metadata]
                )
                total_chunks += 1
            except Exception as e:
                logger.warning(f"Chunk embedding failed (page {page_num}, chunk {i}): {e}")

    logger.info(f"✅ Total chunks stored: {total_chunks}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    process_by_page()
