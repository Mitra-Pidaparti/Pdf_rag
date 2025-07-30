# save as rag_chunker_with_tables.py
import os
import re
import torch
import chromadb
import numpy as np
import spacy
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    'TEXT_FILES': ["novartis-annual-report-2024_docx.txt", "novartis-integrated-report-2024_docx.txt"],
    'EMBEDDING_MODEL': 'BAAI/bge-base-en-v1.5',
    'TARGET_CHUNK_SIZE': 150,
    'MAX_CHUNK_SIZE': 250,
    'MIN_CHUNK_SIZE': 15,
    'SIMILARITY_THRESHOLD': 0.65,
    'OVERLAP_SENTENCES': 1,
    'SPACY_MODEL': 'en_core_web_md',
    'CHROMADB_PATH': "chromadb",
    'CHROMA_COLLECTION': "novartis_combined_chunks_docx",
    'SPACY_BLOCK_CHAR_LIMIT': 1000000  # Safe limit per block
}
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# -------------------------
# Setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load(CONFIG['SPACY_MODEL'])
#nlp.max_length = 2_000_000

# -------------------------
# Table Handling
# -------------------------
def extract_tables(text: str) -> List[str]:
    pattern = re.compile(r"((?:\|.*\n)+\|\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\n(?:\|.*\n?)+)", re.MULTILINE)
    return pattern.findall(text)

def remove_tables(text: str, tables: List[str]) -> str:
    for table in tables:
        text = text.replace(table, "")
    return text

# -------------------------
# Utility Functions
# -------------------------
def split_oversized_sentence(sentence: str, max_words: int) -> List[str]:
    words = sentence.split()
    return [sentence] if len(words) <= max_words else [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def get_last_sentences(text: str, num_sentences: int) -> str:
    doc = nlp(text)
    sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return " ".join(sents[-num_sentences:]) if len(sents) > num_sentences else text

def add_context_overlap(chunks: List[str], overlap_sentences: int) -> List[str]:
    if overlap_sentences <= 0 or len(chunks) < 2:
        return chunks
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            overlap = get_last_sentences(chunks[i - 1], overlap_sentences)
            chunk = f"{overlap} {chunk}"
        final_chunks.append(chunk)
    return final_chunks

def generate_blocks(text: str, max_chars: int) -> List[str]:
    blocks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            next_period = text.rfind('.', start, end)
            if next_period > start:
                end = next_period + 1
        blocks.append(text[start:end].strip())
        start = end
    return blocks

# -------------------------
# Chunking Logic
# -------------------------
def semantic_chunking_with_tables(text: str, embedding_model) -> List[Dict]:
    tables = extract_tables(text)
    clean_text = remove_tables(text, tables)

    doc = nlp(clean_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    processed = []
    for sent in sentences:
        processed.extend(split_oversized_sentence(sent, CONFIG['MAX_CHUNK_SIZE'] // 2))

    if not processed:
        return [{"text": t, "type": "table"} for t in tables]

    try:
        embs = embedding_model.encode(processed, convert_to_tensor=True)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1).cpu().numpy()
    except Exception as e:
        logger.warning(f"Embedding error: {e}")
        return [{"text": " ".join(processed[i:i + CONFIG['TARGET_CHUNK_SIZE']]), "type": "text"}
                for i in range(0, len(processed), CONFIG['TARGET_CHUNK_SIZE'])]

    chunks, temp_chunk, idxs, word_count = [], [], [], 0
    i = 0
    while i < len(processed):
        sent = processed[i]
        words = len(sent.split())
        if word_count + words > CONFIG['MAX_CHUNK_SIZE'] and word_count >= CONFIG['MIN_CHUNK_SIZE']:
            chunks.append(" ".join(temp_chunk))
            temp_chunk, idxs, word_count = [sent], [i], words
            i += 1
            continue
        if words > CONFIG['MAX_CHUNK_SIZE']:
            if temp_chunk and word_count >= CONFIG['MIN_CHUNK_SIZE']:
                chunks.append(" ".join(temp_chunk))
            chunks.append(sent)
            temp_chunk, idxs, word_count = [], [], 0
            i += 1
            continue

        temp_chunk.append(sent)
        idxs.append(i)
        word_count += words
        j = i + 1
        while j < len(processed):
            sim = cosine_similarity(np.mean(embs[idxs], axis=0).reshape(1, -1), embs[j].reshape(1, -1))[0][0]
            threshold = CONFIG['SIMILARITY_THRESHOLD']
            if word_count >= CONFIG['TARGET_CHUNK_SIZE']: threshold += 0.02
            if word_count > CONFIG['MAX_CHUNK_SIZE']: threshold += 0.1
            if sim >= threshold:
                temp_chunk.append(processed[j])
                idxs.append(j)
                word_count += len(processed[j].split())
                j += 1
            else:
                break
        i = j if j > i else i + 1

    if temp_chunk and word_count >= CONFIG['MIN_CHUNK_SIZE']:
        chunks.append(" ".join(temp_chunk))

    text_chunks = add_context_overlap(chunks, CONFIG['OVERLAP_SENTENCES'])
    return [{"text": c, "type": "text"} for c in text_chunks] + [{"text": t, "type": "table"} for t in tables]

# -------------------------
# Main Processing
# -------------------------
def process_all_files():
    model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])
    client = chromadb.PersistentClient(path=CONFIG['CHROMADB_PATH'])
    collection = client.get_or_create_collection(CONFIG['CHROMA_COLLECTION'])

    total_chunks = 0
    for file_path in CONFIG['TEXT_FILES']:
        if not Path(file_path).exists():
            logger.warning(f"❌ File not found: {file_path}")
            continue

        logger.info(f"📄 Processing {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        blocks = generate_blocks(full_text, CONFIG['SPACY_BLOCK_CHAR_LIMIT'])

        for block_id, block in enumerate(tqdm(blocks, desc=f"🧠 Blocks in {file_path}")):
            chunks = semantic_chunking_with_tables(block, model)
            for i, c in enumerate(chunks):
                try:
                    emb = model.encode(c["text"], convert_to_tensor=True)
                    emb = torch.nn.functional.normalize(emb.squeeze(), p=2, dim=0)
                    chunk_id = f"{Path(file_path).stem}_block{block_id}_chunk{i:03d}"

                    collection.add(
                        ids=[chunk_id],
                        embeddings=[emb.tolist()],
                        metadatas=[{
                            "source_file": Path(file_path).name,
                            "type": c["type"],
                            "chunk": c["text"][:1000],
                            "chunk_size": len(c["text"].split()),
                            "chunk_method": "semantic_with_table"
                        }]
                    )
                    total_chunks += 1
                except Exception as e:
                    logger.warning(f"⚠️ Embedding failed: {e}")

    logger.info(f"✅ Total chunks stored: {total_chunks}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    process_all_files()
