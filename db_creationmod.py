# Updated vector db creation code with heading-aware chunking

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
# Extract Hierarchical Headings from Text
# -------------------------
def extract_headings(text):
    """
    Extract headings with their hierarchy levels and positions.
    Returns list of (heading_text, level, start_pos, end_pos)
    """
    headings = []
    lines = text.split('\n')
    current_pos = 0
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped:
            # Detect headings based on common patterns
            # Level 1: All caps or title case with specific patterns
            if (line_stripped.isupper() and len(line_stripped) > 3) or \
               (re.match(r'^[A-Z][A-Z\s]+$', line_stripped) and len(line_stripped) > 5):
                headings.append((line_stripped, 1, current_pos, current_pos + len(line)))
            # Level 2: Bold indicators or numbered sections
            elif re.match(r'^\*\*.*\*\*$', line_stripped) or \
                 re.match(r'^\d+\.\s+[A-Z]', line_stripped) or \
                 re.match(r'^[A-Z][a-z]+:$', line_stripped):
                headings.append((line_stripped, 2, current_pos, current_pos + len(line)))
            # Level 3: Sub-numbered sections or bullet points with caps
            elif re.match(r'^\d+\.\d+\s+[A-Z]', line_stripped) or \
                 re.match(r'^[a-z]\)\s+[A-Z]', line_stripped) or \
                 (line_stripped.startswith('•') and line_stripped[1:].strip()[0].isupper()):
                headings.append((line_stripped, 3, current_pos, current_pos + len(line)))
        
        current_pos += len(line) + 1  # +1 for newline
    
    return headings

# -------------------------
# Get Heading Context for Position
# -------------------------
def get_heading_context(position, headings):
    """
    Get the parent and grandparent headings for a given text position.
    Returns (parent_heading, grandparent_heading)
    """
    parent_heading = None
    grandparent_heading = None
    
    # Find the most recent headings before this position
    recent_headings = {1: None, 2: None, 3: None}
    
    for heading_text, level, start_pos, end_pos in headings:
        if start_pos <= position:
            recent_headings[level] = heading_text
            # Clear lower level headings when we encounter a higher level heading
            for lower_level in range(level + 1, 4):
                recent_headings[lower_level] = None
        else:
            break
    
    # Determine parent and grandparent based on available headings
    if recent_headings[3]:  # If we have a level 3 heading
        parent_heading = recent_headings[3]
        grandparent_heading = recent_headings[2] or recent_headings[1]
    elif recent_headings[2]:  # If we have a level 2 heading
        parent_heading = recent_headings[2]
        grandparent_heading = recent_headings[1]
    elif recent_headings[1]:  # If we have a level 1 heading
        parent_heading = recent_headings[1]
        grandparent_heading = None
    
    return parent_heading, grandparent_heading

# -------------------------
# Split Text Into Heading-Aware Chunks using spaCy
# -------------------------
def split_text_by_headings_and_sentences(text, chunk_size=400):
    """
    Split text into chunks that respect heading boundaries and sentence structure.
    All sentences in a chunk will belong to the same heading section.
    """
    # Extract headings first
    headings = extract_headings(text)
    
    # Process with spaCy
    doc = nlp(text)
    sentences = [(sent.text.strip(), sent.start_char, sent.end_char) for sent in doc.sents]
    
    chunks = []
    chunk_metadata = []
    
    current_chunk = []
    current_chunk_length = 0
    current_heading_context = (None, None)
    sentence_indices = []
    
    for i, (sentence, start_char, end_char) in enumerate(sentences):
        # Get heading context for this sentence
        sentence_heading_context = get_heading_context(start_char, headings)
        
        # Check if we need to start a new chunk due to heading change
        if (current_heading_context != sentence_heading_context and current_chunk) or \
           (current_chunk_length + len(sentence.split()) > chunk_size and current_chunk):
            
            # Finalize current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                chunk_metadata.append({
                    'parent_heading': current_heading_context[0],
                    'grandparent_heading': current_heading_context[1],
                    'sentence_indices': ",".join(map(str, sentence_indices))
                })
            
            # Start new chunk
            current_chunk = [sentence]
            current_chunk_length = len(sentence.split())
            current_heading_context = sentence_heading_context
            sentence_indices = [i]
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            sentence_indices.append(i)
            current_chunk_length += len(sentence.split())
            
            # Update heading context if this is the first sentence
            if not current_heading_context[0] and not current_heading_context[1]:
                current_heading_context = sentence_heading_context
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        chunk_metadata.append({
            'parent_heading': current_heading_context[0],
            'grandparent_heading': current_heading_context[1],
            'sentence_indices': ",".join(map(str, sentence_indices))
        })
    
    return chunks, chunk_metadata, sentences

# -------------------------
# Load Embedding Model
# -------------------------
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# -------------------------
# Init ChromaDB Collection
# -------------------------
client = chromadb.PersistentClient(path="chromadb")
collection_name = f"{COLLECTION_NAME}_mpnet_heading_aware"
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
total_chunks = 0
for txt_file in txt_files:
    with open(os.path.join(TEXT_FOLDER, txt_file), 'r', encoding='utf-8') as file:
        content = file.read()
    chunks, _, _ = split_text_by_headings_and_sentences(content, chunk_size=400)
    total_chunks += len(chunks)

# -------------------------
# Process Each Page
# -------------------------
with tqdm(total=total_chunks, desc="Processing Chunks") as total_chunk_bar:
    with tqdm(total=len(txt_files), desc="Processing Pages", leave=False) as page_bar:
        for txt_file in txt_files:
            page_path = os.path.join(TEXT_FOLDER, txt_file)
            page_number = int(re.search(r'\d+', txt_file).group())

            with open(page_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Split the content into heading-aware chunks
            chunks, chunk_metadata, sentences = split_text_by_headings_and_sentences(content, chunk_size=400)

            # Encode the chunks in batches
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_embeddings = embedding_model.encode(batch_chunks, convert_to_tensor=True)
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.extend(batch_embeddings.tolist())

            # Add chunks and embeddings to ChromaDB
            for i, (chunk, embedding, metadata) in enumerate(zip(chunks, all_embeddings, chunk_metadata)):
                # Create unique ID for each chunk
                chunk_id = f"{collection_name}_p{page_number}_c{i}"
                
                # Enhanced metadata for the chunk with heading information
                enhanced_metadata = {
                    "chunk": chunk,
                    "page": page_number,
                    "document": COLLECTION_NAME,
                    "sentence_indices": metadata['sentence_indices'],
                    "parent_heading": metadata['parent_heading'] or "",
                    "grandparent_heading": metadata['grandparent_heading'] or "",
                    "heading_path": f"{metadata['grandparent_heading'] or ''} > {metadata['parent_heading'] or ''}".strip(' > ')
                }
                
                # Add to ChromaDB
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[enhanced_metadata]
                )

                # Update the progress bar for the total chunks
                total_chunk_bar.update(1)

            # Update the page progress bar
            page_bar.update(1)

print(f"All text pages from '{TEXT_FOLDER}' processed and stored in ChromaDB under collection '{collection_name}'.")
print("Chunks are now organized by heading hierarchy with parent and grandparent heading information.")