# Updated vector db creation code

import os
import re
import torch
import chromadb
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
import spacy

TEXT_FOLDER = "ril_pdf_pages"
COLLECTION_NAME = Path(TEXT_FOLDER).name

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# -------------------------
# Split Text Into Chunks using spaCy for Sentence Segmentation
# -------------------------
def split_text_by_sentences_spacy(text, chunk_size=400):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    sentence_indices = []

    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence.split())  # Use word count to estimate sentence length
        if current_chunk_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            sentence_indices.append(i)  # Store sentence index
            current_chunk_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            sentence_indices_str = ",".join(map(str, sentence_indices))  # Store as comma-separated string
            sentence_indices = [i]  # Reset for the next chunk
            current_chunk = [sentence]
            current_chunk_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        sentence_indices_str = ",".join(map(str, sentence_indices))  # Store as comma-separated string
    
    return chunks, sentence_indices_str, sentences  # Return sentence indices as a string

# -------------------------
# Load Embedding Model - Changed to all-mpnet-base- qv2 for better quality and compatibility
# -------------------------
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# -------------------------
# Init ChromaDB Collection
# -------------------------
client = chromadb.PersistentClient(path="chromadb")
# If you had an existing collection with the previous model, consider recreating it with a new name
# to avoid mixing embeddings from different models
collection_name = f"{COLLECTION_NAME}_mpnet"
collection = client.get_or_create_collection(collection_name)

# -------------------------
# Get Text Files
# -------------------------
txt_files = sorted(
    [f for f in os.listdir(TEXT_FOLDER) if f.startswith("page_") and f.endswith(".txt")],
    key=lambda x: int(re.search(r'\d+', x).group())
)

# -------------------------
# Track total chunks for overall progress bar
# -------------------------
total_chunks = sum(len(split_text_by_sentences_spacy(open(os.path.join(TEXT_FOLDER, txt_file), 'r', encoding='utf-8').read(), chunk_size=400)[0]) for txt_file in txt_files)

# ----------    ---------------
# Process Each Page
# -------------------------
with tqdm(total=total_chunks, desc="Processing Chunks") as total_chunk_bar:
    with tqdm(total=len(txt_files), desc="Processing Pages", leave=False) as page_bar:
        for txt_file in txt_files:
            page_path = os.path.join(TEXT_FOLDER, txt_file)
            page_number = int(re.search(r'\d+', txt_file).group())

            with open(page_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Split the content into chunks using spaCy for sentence tokenization
            chunks, sentence_indices_str, sentences = split_text_by_sentences_spacy(content, chunk_size=400)

            # Encode the raw chunks (no preprocessing)
            # Process in smaller batches to avoid memory issues with the larger model
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_embeddings = embedding_model.encode(batch_chunks, convert_to_tensor=True)
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.extend(batch_embeddings.tolist())

            # Add chunks and embeddings to ChromaDB
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                # Create unique ID for each chunk
                id = f"{collection_name}_p{page_number}_c{i}"
                
                # Metadata for the chunk
                metadata = {
                    "chunk": chunk,
                    "page": page_number,
                    "document": COLLECTION_NAME,
                    "sentence_indices": sentence_indices_str
                }
                
                # Add to ChromaDB
                collection.add(
                    ids=[id],
                    embeddings=[embedding],  # Now already converted to list
                    metadatas=[metadata]
                )

                # Update the progress bar for the total chunks
                total_chunk_bar.update(1)

            # Update the page progress bar
            page_bar.update(1)

print(f"All text pages from '{TEXT_FOLDER}' processed and stored in ChromaDB under collection '{collection_name}'.")