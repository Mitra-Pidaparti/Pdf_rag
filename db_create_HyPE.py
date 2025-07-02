# HyPE-Enhanced Vector Database Creation
# Using OpenAI GPT-4o mini and semantic chunking with Hypothetical Passage Embeddings

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
from openai import OpenAI
import json
import logging
import time
from typing import List


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#OPENAI_API_KEY:

import os
os.environ["OPENAI_API_KEY"] = 



# Configuration
TEXT_FOLDER = "ril_pdf_pages"
COLLECTION_NAME = Path(TEXT_FOLDER).name
OPENAI_MODEL = "gpt-4o-mini"

# Initialize OpenAI client
client = OpenAI(api_key='your-api-key')

# Load the spaCy model
nlp = spacy.load('en_core_web_md')

# -------------------------
# HyPE Generator
# -------------------------
class HyPEGenerator:
    """
    HyPE (Hypothetical Passage Embeddings) generator using OpenAI GPT-4o mini
    Generates hypothetical questions/queries that would be answered by given passages
    """
    
    def __init__(self, model_name: str = OPENAI_MODEL):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_hypothetical_questions(self, passage: str, num_questions: int = 3) -> List[str]:
        """
        Generate hypothetical questions that would be answered by the given passage
        """
        
        system_prompt = """You are an expert at generating questions that would be answered by given text passages.

Given a text passage, generate diverse, specific questions that this passage would directly answer.
The questions should:
- Be natural and realistic (questions a user might actually ask)
- Cover different aspects of the passage content
- Vary in specificity (some general, some detailed)
- Be answerable primarily from the given passage
- Use different question types (what, how, why, when, where, etc.)

Return only the questions, one per line, without numbering."""
        
        user_prompt = f"""Generate {num_questions} different questions that would be answered by this passage:

"{passage}"

Focus on creating questions that capture the main information and key details in the passage."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            questions = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating hypothetical questions: {e}")
            return []

    def generate_hypothetical_queries(self, passage: str, num_queries: int = 2) -> List[str]:
        """
        Generate hypothetical search queries that would retrieve this passage
        """
        
        system_prompt = """You are an expert at generating search queries that would retrieve specific text passages.

Given a text passage, generate search queries that someone would use to find this information.
The queries should:
- Be concise (2-8 words typically)
- Use keywords that appear in or relate to the passage
- Represent realistic search behavior
- Cover different ways someone might search for this information

Return only the search queries, one per line, without numbering."""
        
        user_prompt = f"""Generate {num_queries} different search queries that would help retrieve this passage:

"{passage}"

Focus on key terms and concepts that someone would search for."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                max_tokens=150
            )
            
            queries = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            return queries[:num_queries]
            
        except Exception as e:
            logger.error(f"Error generating hypothetical queries: {e}")
            return []

# -------------------------
# Semantic Chunking
# -------------------------
def semantic_chunking(text, embedding_model, max_chunk_size=400, similarity_threshold=0.7):
    """
    Split text into semantically coherent chunks using sentence embeddings.
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
# HyPE-Enhanced Vector Database Creation
# -------------------------
def create_hype_enhanced_vectordb():
    """
    Create vector database with HyPE enhancements for improved embeddings
    """
    
    print("Starting HyPE-enhanced vector database creation...")
    
    # Load Embedding Model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    # Initialize HyPE Generator
    print("Initializing HyPE generator...")
    hype_generator = HyPEGenerator()
    
    # Init ChromaDB Collection
    print("Setting up ChromaDB collection...")
    chroma_client = chromadb.PersistentClient(path="chromadb")
    collection_name = f"{COLLECTION_NAME}_hype_semantic"
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        print(f"No existing collection found, creating new: {collection_name}")
    
    collection = chroma_client.create_collection(collection_name)
    
    # Get Text Files
    print("Finding text files...")
    txt_files = sorted(
        [f for f in os.listdir(TEXT_FOLDER) if f.startswith("page_") and f.endswith(".txt")],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    
    print(f"Found {len(txt_files)} text files to process")
    
    # Process Each Page
    total_files = len(txt_files)
    total_chunks_processed = 0
    
    with tqdm(total=total_files, desc="Processing Pages with HyPE") as page_bar:
        for txt_file in txt_files:
            page_path = os.path.join(TEXT_FOLDER, txt_file)
            page_number = int(re.search(r'\d+', txt_file).group())
            
            # Read page content
            with open(page_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Skip empty pages
            if not content.strip():
                page_bar.set_description(f"Skipping empty page {page_number}")
                page_bar.update(1)
                continue
            
            # Perform semantic chunking
            chunks, chunk_sentence_indices = semantic_chunking(
                content,
                embedding_model,
                max_chunk_size=400,
                similarity_threshold=0.7
            )
            
            if not chunks:
                page_bar.set_description(f"No chunks created for page {page_number}")
                page_bar.update(1)
                continue
            
            # Process chunks with HyPE enhancement
            page_bar.set_description(f"HyPE processing page {page_number} ({len(chunks)} chunks)")
            
            chunk_data = []
            
            for i, (chunk, sentence_indices) in enumerate(zip(chunks, chunk_sentence_indices)):
                try:
                    # Generate hypothetical questions and queries for this chunk
                    hypothetical_questions = hype_generator.generate_hypothetical_questions(chunk, num_questions=3)
                    hypothetical_queries = hype_generator.generate_hypothetical_queries(chunk, num_queries=2)
                    
                    # Combine all text for embedding
                    all_text_for_embedding = [chunk] + hypothetical_questions + hypothetical_queries
                    
                    # Generate embeddings for all text variations
                    embeddings = embedding_model.encode(all_text_for_embedding, convert_to_tensor=True)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Create weighted average embedding
                    # Original chunk gets highest weight, then hypothetical questions, then queries
                    num_questions = len(hypothetical_questions)
                    num_queries = len(hypothetical_queries)
                    total_hypotheticals = num_questions + num_queries
                    
                    if total_hypotheticals > 0:
                        # Distribute weights: 50% original, 30% questions, 20% queries
                        question_weight = 0.2 / max(num_questions, 1) if num_questions > 0 else 0
                        query_weight = 0.1 / max(num_queries, 1) if num_queries > 0 else 0
                        
                        weights = [7] + [question_weight] * num_questions + [query_weight] * num_queries
                    else:
                        weights = [1.0]  # Only original chunk
                    
                    # Ensure weights sum to 1
                    weights = np.array(weights[:len(embeddings)])
                    weights = weights / weights.sum()
                    
                    # Create weighted embedding
                    weighted_embedding = np.average(embeddings.cpu().numpy(), axis=0, weights=weights)
                    
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
                        "chunk_method": "hype_semantic",
                        "chunk_size": len(chunk.split()),
                        "hypothetical_questions": json.dumps(hypothetical_questions),
                        "hypothetical_queries": json.dumps(hypothetical_queries),
                        "num_hypotheticals": len(hypothetical_questions) + len(hypothetical_queries),
                        "hype_weights": json.dumps(weights.tolist())
                    }
                    
                    chunk_data.append({
                        'id': chunk_id,
                        'embedding': weighted_embedding.tolist(),
                        'metadata': metadata
                    })
                    
                    # Small delay to avoid OpenAI rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i} on page {page_number}: {e}")
                    # Continue with original embedding if HyPE fails
                    original_embedding = embedding_model.encode([chunk], convert_to_tensor=True)
                    original_embedding = torch.nn.functional.normalize(original_embedding, p=2, dim=1)
                    
                    chunk_id = f"{collection_name}_p{page_number}_c{i}"
                    sentence_indices_str = ",".join(map(str, sentence_indices))
                    
                    metadata = {
                        "chunk": chunk,
                        "page": page_number,
                        "document": COLLECTION_NAME,
                        "sentence_indices": sentence_indices_str,
                        "chunk_method": "semantic_fallback",
                        "chunk_size": len(chunk.split()),
                        "hypothetical_questions": "[]",
                        "hypothetical_queries": "[]",
                        "num_hypotheticals": 0,
                        "hype_weights": "[1.0]"
                    }
                    
                    chunk_data.append({
                        'id': chunk_id,
                        'embedding': original_embedding[0].tolist(),
                        'metadata': metadata
                    })
            
            # Batch add to ChromaDB
            if chunk_data:
                try:
                    ids = [item['id'] for item in chunk_data]
                    embeddings = [item['embedding'] for item in chunk_data]
                    metadatas = [item['metadata'] for item in chunk_data]
                    
                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                    
                    total_chunks_processed += len(chunk_data)
                    
                except Exception as e:
                    logger.error(f"Error adding chunks to ChromaDB for page {page_number}: {e}")
            
            # Update progress
            page_bar.set_description(f"Completed page {page_number}")
            page_bar.update(1)
    
    # Final summary
    print(f"\n{'='*60}")
    print("HyPE-ENHANCED VECTOR DATABASE CREATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total pages processed: {total_files}")
    print(f"Total chunks created: {total_chunks_processed}")
    print(f"Collection name: {collection_name}")
    print(f"Embedding model: BAAI/bge-base-en-v1.5")
    print(f"Enhancement method: HyPE (Hypothetical Passage Embeddings)")
    print(f"Database location: chromadb/")
    print(f"{'='*60}")
    
    return collection_name

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Check if text folder exists
    if not os.path.exists(TEXT_FOLDER):
        print(f"ERROR: Text folder '{TEXT_FOLDER}' not found")
        print(f"Please ensure your text files are in the '{TEXT_FOLDER}' directory")
        exit(1)
    
    # Create the HyPE-enhanced vector database
    try:
        collection_name = create_hype_enhanced_vectordb()
        print(f"\nSUCCESS: HyPE-enhanced vector database created successfully!")
        print(f"Collection name: {collection_name}")
        print(f"You can now use this collection for enhanced retrieval in your RAG system.")
        
    except Exception as e:
        print(f"ERROR: Failed to create vector database: {e}")
        exit(1)