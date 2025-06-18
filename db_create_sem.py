# Updated vector db creation code with semantic chunking
 
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
 
TEXT_FOLDER = "ril_pdf_pages"
COLLECTION_NAME = Path(TEXT_FOLDER).name
 
# Load the spaCy model
nlp = spacy.load('en_core_web_md')
 
# -------------------------
# Semantic Chunking using Embeddings
# -------------------------
def semantic_chunking(text, embedding_model, max_chunk_size=100, similarity_threshold=0.75):
    """
    Split text into semantically coherent chunks using sentence embeddings.
   
    Args:
        text: Input text to chunk
        embedding_model: SentenceTransformer model for embeddings
        max_chunk_size: Maximum number of words per chunk
        similarity_threshold: Cosine similarity threshold for grouping sentences
   
    Returns:
        chunks: List of text chunks
        sentence_indices: List of sentence indices for each chunk
    """
    # Extract sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
   
    if not sentences:
        return [], []
   
    # Generate embeddings for all sentences
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
   
    # Convert to numpy for sklearn compatibility
    embeddings_np = sentence_embeddings.cpu().numpy()
   
    chunks = []
    chunk_sentence_indices = []
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
            chunk_sentence_indices.append(current_indices.copy())
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
        chunk_sentence_indices.append(current_indices)
   
    return chunks, chunk_sentence_indices
 
# -------------------------
# Alternative: Breakpoint-based Semantic Chunking
# -------------------------
def semantic_chunking_breakpoints(text, embedding_model, max_chunk_size=300, percentile_threshold=35):
    """
    Alternative semantic chunking method using similarity breakpoints.
    Identifies natural breakpoints where similarity drops significantly.
    """
    # Extract sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
   
    if len(sentences) <= 1:
        return sentences, [[0]] if sentences else []
   
    # Generate embeddings for all sentences
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    embeddings_np = sentence_embeddings.cpu().numpy()
   
    # Calculate similarities between consecutive sentences
    similarities = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(
            embeddings_np[i].reshape(1, -1),
            embeddings_np[i + 1].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
   
    # Find breakpoints where similarity drops significantly
    threshold = np.percentile(similarities, percentile_threshold)
    breakpoints = [0]  # Start with first sentence
   
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i + 1)
   
    breakpoints.append(len(sentences))  # End with last sentence
   
    # Create chunks based on breakpoints, respecting max_chunk_size
    chunks = []
    chunk_sentence_indices = []
   
    for i in range(len(breakpoints) - 1):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1]
       
        # Create chunk from sentences in this segment
        segment_sentences = sentences[start_idx:end_idx]
        segment_indices = list(range(start_idx, end_idx))
       
        # If segment is too large, split it further
        current_chunk = []
        current_indices = []
        current_word_count = 0
       
        for j, sentence in enumerate(segment_sentences):
            sentence_word_count = len(sentence.split())
           
            if current_word_count + sentence_word_count > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                chunk_sentence_indices.append(current_indices)
                current_chunk = []
                current_indices = []
                current_word_count = 0
           
            current_chunk.append(sentence)
            current_indices.append(segment_indices[j])
            current_word_count += sentence_word_count
       
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            chunk_sentence_indices.append(current_indices)
   
    return chunks, chunk_sentence_indices
 
# -------------------------
# Load Embedding Model
# -------------------------
embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
 
# -------------------------
# Init ChromaDB Collection
# -------------------------
client = chromadb.PersistentClient(path="chromadb")
collection_name = f"{COLLECTION_NAME}_semantic"
collection = client.get_or_create_collection(collection_name)
 
# -------------------------
# Get Text Files
# -------------------------
txt_files = sorted(
    [f for f in os.listdir(TEXT_FOLDER) if f.startswith("page_") and f.endswith(".txt")],
    key=lambda x: int(re.search(r'\d+', x).group())
)
 
# -------------------------
# Process Each Page
# -------------------------
total_files = len(txt_files)
total_chunks_processed = 0
 
with tqdm(total=total_files, desc="Processing Pages") as page_bar:
    for txt_file in txt_files:
        page_path = os.path.join(TEXT_FOLDER, txt_file)
        page_number = int(re.search(r'\d+', txt_file).group())
 
        with open(page_path, 'r', encoding='utf-8') as file:
            content = file.read()
 
        # Use semantic chunking (you can switch between methods)
        # Method 1: Similarity-based chunking
        chunks, chunk_sentence_indices = semantic_chunking(
            content,
            embedding_model,
            max_chunk_size=400,
            similarity_threshold=0.7
        )
       
        # Method 2: Breakpoint-based chunking (alternative)
        # chunks, chunk_sentence_indices = semantic_chunking_breakpoints(
        #     content,
        #     embedding_model,
        #     max_chunk_size=400,
        #     percentile_threshold=95
        # )
 
        if not chunks:
            page_bar.update(1)
            continue
 
        # Generate embeddings for chunks in batches
        batch_size = 32
        all_embeddings = []
       
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embedding_model.encode(batch_chunks, convert_to_tensor=True)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.extend(batch_embeddings.tolist())
 
        # Add chunks and embeddings to ChromaDB
        with tqdm(total=len(chunks), desc=f"Page {page_number}", leave=False) as chunk_bar:
            for i, (chunk, embedding, sentence_indices) in enumerate(zip(chunks, all_embeddings, chunk_sentence_indices)):
                # Create unique ID for each chunk
                chunk_id = f"{collection_name}_p{page_number}_c{i}"
               
                # Convert sentence indices to comma-separated string
                sentence_indices_str = ",".join(map(str, sentence_indices))
               
                # Metadata for the chunk
                metadata = {
                    "chunk": chunk,
                    "page": page_number,
                    "document": COLLECTION_NAME,
                    "sentence_indices": sentence_indices_str,
                    "chunk_method": "semantic",
                    "chunk_size": len(chunk.split())
                }
               
                # Add to ChromaDB
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
               
                chunk_bar.update(1)
                total_chunks_processed += 1
 
        # Update the page progress bar
        page_bar.update(1)
 
print(f"Semantic chunking completed!")
print(f"Total pages processed: {total_files}")
print(f"Total chunks created: {total_chunks_processed}")
print(f"All text pages from '{TEXT_FOLDER}' processed and stored in ChromaDB under collection '{collection_name}'.")
print(f"Collection uses semantic chunking with embedding model: all-mpnet-base-v2")