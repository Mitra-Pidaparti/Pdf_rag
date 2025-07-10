import re, os, csv
from sentence_transformers import SentenceTransformer, util
import torch
import chromadb
import json
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
import time
# -------------------------
# Configuration
# -------------------------
INITIAL_POOL_SIZE = 300 # Large pool from BM25/TF-IDF
DENSE_RERANK_SIZE = 200 # Top candidates after dense reranking
FINAL_CHUNKS = 50    # Final chunks after sentence extraction
DB_PATH = "chromadb"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'

# -------------------------
# Initialize Components
# -------------------------
print("[INFO] Initializing components...")
client = OpenAI(api_key=
model = SentenceTransformer(MODEL_NAME)
client_chroma = chromadb.PersistentClient(path=DB_PATH)
nlp = spacy.load("en_core_web_md")

# -------------------------
# Heading Analysis Functions
# -------------------------

def is_heading(line: str, next_line: Optional[str] = None) -> bool:
    """
    Determine if a line is a true section heading based on multiple criteria.
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
    
    return True

def is_mostly_capitalized(text: str) -> bool:
    """
    Check if text is mostly capitalized (good indicator of headings).
    """
    letters_only = ''.join(c for c in text if c.isalpha())
    if not letters_only:
        return False
    
    uppercase_count = sum(1 for c in letters_only if c.isupper())
    return uppercase_count / len(letters_only) > 0.7

def extract_heading_text(line: str) -> str:
    """
    Extract the actual heading text from a line with '#' markers.
    """
    return re.sub(r'^#+\s*', '', line.strip())

def extract_headings_from_text(text: str) -> List[str]:
    """
    Extract all headings from a text chunk.
    """
    lines = text.split('\n')
    headings = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if is_heading(line, next_line):
            heading_text = extract_heading_text(line)
            headings.append(heading_text)
    
    return headings

def calculate_heading_relevance_score(query: str, chunk_text: str, chunk_heading: str = None) -> float:
    """
    Calculate relevance score based on heading-query alignment.
    """
    # Extract headings from chunk text
    extracted_headings = extract_headings_from_text(chunk_text)
    
    # Include the chunk heading metadata if available
    all_headings = []
    if chunk_heading and chunk_heading.lower() != "introduction":
        all_headings.append(chunk_heading)
    all_headings.extend(extracted_headings)
    
    if not all_headings:
        return 0.0
    
    # Process query
    query_doc = nlp(query)
    query_tokens = set([token.lemma_.lower() for token in query_doc 
                       if not token.is_stop and not token.is_punct])
    query_entities = set([ent.text.lower() for ent in query_doc.ents])
    
    max_heading_score = 0.0
    
    for heading in all_headings:
        heading_doc = nlp(heading)
        heading_tokens = set([token.lemma_.lower() for token in heading_doc 
                             if not token.is_stop and not token.is_punct])
        heading_entities = set([ent.text.lower() for ent in heading_doc.ents])
        
        # Token overlap score
        token_overlap = len(query_tokens.intersection(heading_tokens))
        token_overlap_ratio = token_overlap / len(query_tokens) if query_tokens else 0
        
        # Entity overlap score
        entity_overlap = len(query_entities.intersection(heading_entities))
        entity_overlap_ratio = entity_overlap / len(query_entities) if query_entities else 0
        
        # Semantic similarity using embeddings
        try:
            query_embedding = model.encode(query, convert_to_tensor=True)
            heading_embedding = model.encode(heading, convert_to_tensor=True)
            semantic_sim = util.pytorch_cos_sim(query_embedding, heading_embedding).item()
        except:
            semantic_sim = 0.0
        
        # Combined heading score
        heading_score = (
            token_overlap_ratio * 0.3 +
            entity_overlap_ratio * 0.3 +
            semantic_sim * 0.4
        )
        
        max_heading_score = max(max_heading_score, heading_score)
    
    return max_heading_score

def calculate_heading_hierarchy_bonus(query: str, chunk_text: str) -> float:
    """
    Calculate bonus based on heading hierarchy relevance.
    """
    lines = chunk_text.split('\n')
    heading_levels = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # Count the number of '#' symbols
            level = len(line) - len(line.lstrip('#'))
            heading_text = extract_heading_text(line)
            if heading_text:
                heading_levels[level] = heading_levels.get(level, []) + [heading_text]
    
    # Higher-level headings (fewer #'s) get more weight
    hierarchy_score = 0.0
    for level, headings in heading_levels.items():
        level_weight = 1.0 / level if level > 0 else 1.0  # Higher weight for higher levels
        
        for heading in headings:
            heading_relevance = calculate_heading_relevance_score(query, "", heading)
            hierarchy_score += heading_relevance * level_weight
    
    return min(hierarchy_score, 1.0)  # Cap at 1.0

# -------------------------
# OpenAI Sentence Extraction
# -------------------------
def extract_relevant_sentences_with_openai(query, context_text):
    """Extract all relevant sentences from context using OpenAI."""
    system_prompt = """You are an expert text analyzer. Your task is to extract all that is relevant to answering the user’s query.

Instructions:

Classify the chunk context whether its relevant to query(be lenient)
  -If it is
    -identify the parts/sections which are more relevant to the query and output it verbatim, that is, only and exactly the same wording.
  
  -If not,return "NONE"

."""


    user_prompt = f"Query: {query}\n\nContext: {context_text}\n\nExtract relevant sentences:"

    try:
        response = client.chat.completions.create(
           # model="gpt-4.1",
            model='o3',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            #temperature=0.2,
            #max_tokens=1500
        )
        
        extracted = response.choices[0].message.content.strip()
        return extracted if extracted != "NONE" else ""
        
    except Exception as e:
        print(f"[ERROR] OpenAI extraction failed: {e}")
        return ""

# -------------------------
# Semantic Similarity Scoring
# -------------------------
def calculate_semantic_similarity(query, sentence_group):
    """Calculate semantic similarity between query and sentence group using sentence transformers."""
    if not sentence_group or sentence_group == "No relevant sentences found":
        return 0.0
    
    try:
        # Encode query and sentence group
        query_embedding = model.encode(query, convert_to_tensor=True)
        sentence_embedding = model.encode(sentence_group, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(query_embedding, sentence_embedding)
        return float(similarity.cpu().numpy()[0][0])
        
    except Exception as e:
        print(f"[ERROR] Semantic similarity calculation failed: {e}")
        return 0.0

# -------------------------
# Query Processing
# -------------------------
def preprocess_query(query):
    """Process query to extract key terms."""
    doc = nlp(query)
    key_terms = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop]
    return {
        "original": query,
        "key_terms": key_terms
    }

# -------------------------
# Initial Retrieval (BM25 + TF-IDF Pool)
# -------------------------
def retrieve_initial_pool(query, collection_name, pool_size=INITIAL_POOL_SIZE):
    """
    Retrieve initial large pool using BM25 and TF-IDF fusion.
    More efficient than dense retrieval for large pools.
    """
    print(f"[INFO] Retrieving initial pool of {pool_size} candidates using BM25 + TF-IDF...")
    
    collection = client_chroma.get_collection(collection_name)
    data = collection.get()
    
    # Prepare data
    chunks = [meta.get("chunk", "") for meta in data["metadatas"]]
    pages = [meta.get("page", "Unknown") for meta in data["metadatas"]]
    headings = [meta.get("heading", "") for meta in data["metadatas"]]
    doc_ids = data.get("ids", ["unknown"] * len(chunks))
    
    print(f"[INFO] Processing {len(chunks)} total chunks...")
    
    # -------------------------
    # BM25 Scoring
    # -------------------------
    print("[INFO] Computing BM25 scores...")
    
    # Tokenize for BM25
    tokenized = []
    for chunk in chunks:
        doc = nlp(chunk)
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and len(token.text) > 1]
        tokenized.append(tokens)
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized)
    
    # Process query for BM25
    query_doc = nlp(query)
    query_terms = [token.lemma_.lower() for token in query_doc 
                   if not token.is_stop and not token.is_punct and len(token.text) > 1]
    
    # Get BM25 scores
    bm25_scores = bm25.get_scores(query_terms)
    
    # -------------------------
    # TF-IDF Scoring
    # -------------------------
    print("[INFO] Computing TF-IDF scores...")
    
    # Process texts for TF-IDF
    processed_texts = []
    for chunk in chunks:
        doc = nlp(chunk)
        terms = [token.lemma_.lower() for token in doc 
                if not token.is_stop and not token.is_punct and len(token.text) > 1]
        processed_texts.append(" ".join(terms))
    
    # Build TF-IDF index
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Process query for TF-IDF
    query_terms_tfidf = [token.lemma_.lower() for token in query_doc 
                        if not token.is_stop and not token.is_punct and len(token.text) > 1]
    query_text = " ".join(query_terms_tfidf)
    
    # Get TF-IDF scores
    query_vector = vectorizer.transform([query_text])
    tfidf_scores = (query_vector * tfidf_matrix.T).toarray().flatten()
    
    # -------------------------
    # Score Fusion and Ranking
    # -------------------------
    print("[INFO] Fusing BM25 and TF-IDF scores...")
    
    # Normalize scores to [0, 1] range
    bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
    tfidf_scores_norm = (tfidf_scores - np.min(tfidf_scores)) / (np.max(tfidf_scores) - np.min(tfidf_scores) + 1e-8)
    
    # Combine scores (weighted average)
    combined_scores = 0.7 * bm25_scores_norm + 0.3 * tfidf_scores_norm
    
    # Get top candidates
    top_indices = np.argsort(combined_scores)[-pool_size:][::-1]
    
    # Prepare results
    pool_candidates = []
    for idx in top_indices:
        pool_candidates.append({
            "chunk": chunks[idx],
            "page": pages[idx],
            "heading": headings[idx],
            "document": doc_ids[idx],
            "bm25_score": bm25_scores[idx],
            "tfidf_score": tfidf_scores[idx],
            "combined_score": combined_scores[idx],
            "index": idx
        })
    
    print(f"[INFO] Retrieved {len(pool_candidates)} candidates for dense reranking")
    return pool_candidates

# -------------------------
# Dense Reranking
# -------------------------
def dense_rerank_candidates(query, candidates, rerank_size=DENSE_RERANK_SIZE):
    """
    Apply dense semantic reranking to the candidate pool.
    More efficient than computing dense embeddings for all chunks.
    """
    print(f"[INFO] Dense reranking top {rerank_size} candidates...")
    
    if len(candidates) == 0:
        return []
    
    # Encode query once
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
    
    # Encode all candidate chunks
    candidate_texts = [candidate["chunk"] for candidate in candidates]
    print(f"[INFO] Encoding {len(candidate_texts)} candidate chunks...")
    
    candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True, batch_size=32)
    candidate_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)
    
    # Calculate semantic similarities
    similarities = util.pytorch_cos_sim(query_embedding.unsqueeze(0), candidate_embeddings).cpu().numpy()[0]
    
    # Add semantic scores to candidates
    for i, candidate in enumerate(candidates):
        candidate["semantic_score"] = float(similarities[i])
        # Combine with lexical score (weighted)
        candidate["final_score"] = 0.3 * candidate["combined_score"] + 0.7 * candidate["semantic_score"]
    
    # Sort by final score and return top candidates
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top_candidates = candidates[:rerank_size]
    
    print(f"[INFO] Reranked to top {len(top_candidates)} candidates")
    return top_candidates

# -------------------------
# Enhanced Context Enhancement with Heading Boost
# -------------------------
def enhance_with_context_features(chunks, query):
    """Add context-aware features including heading-based scoring to enhance ranking."""
    print("[INFO] Adding context-aware features with heading boost...")
    
    query_doc = nlp(query)
    query_tokens = set([token.lemma_.lower() for token in query_doc 
                       if not token.is_stop and not token.is_punct])
    query_entities = set([ent.text.lower() for ent in query_doc.ents])
    
    for chunk in chunks:
        context = chunk["chunk"]
        context_doc = nlp(context)
        
        # Token overlap
        context_tokens = set([token.lemma_.lower() for token in context_doc 
                             if not token.is_stop and not token.is_punct])
        token_overlap = len(query_tokens.intersection(context_tokens))
        
        # Entity matching
        context_entities = set([ent.text.lower() for ent in context_doc.ents])
        entity_match = len(query_entities.intersection(context_entities))
        
        # Basic context bonus
        context_bonus = (token_overlap * 0.1) + (entity_match * 0.2)
        
        # -------------------------
        # NEW: Heading-based scoring boost
        # -------------------------
        
        # 1. Heading relevance score
        heading_relevance = calculate_heading_relevance_score(
            query, context, chunk.get("heading", "")
        )
        
        # 2. Heading hierarchy bonus
        hierarchy_bonus = calculate_heading_hierarchy_bonus(query, context)
        
        # 3. Combined heading boost
        heading_boost = (heading_relevance * 0.6) + (hierarchy_bonus * 0.4)
        
        # Store individual scores
        chunk["context_score"] = context_bonus
        chunk["heading_relevance"] = heading_relevance
        chunk["hierarchy_bonus"] = hierarchy_bonus
        chunk["heading_boost"] = heading_boost
        
        # Enhanced score with heading boost
        chunk["enhanced_score"] = (
            chunk["final_score"] + 
            context_bonus + 
            (heading_boost * 0.4)  # Heading boost weight
        )
    
    return chunks

# -------------------------
# Main Optimized Hybrid Search
# -------------------------
def optimized_hybrid_search_with_dense_reranking(query, collection_name):
    """
    Optimized hybrid search pipeline with heading-based scoring:
    1. Retrieve large pool (200) using BM25 + TF-IDF fusion
    2. Dense reranking on pool to get top 100 candidates
    3. Enhanced context features with heading boost
    4. Sentence extraction on top candidates
    5. Final semantic reranking for top 30
    """
    print(f"\n[INFO] Starting optimized hybrid search with heading boost for query: {query}")
    print(f"[INFO] Pipeline: {INITIAL_POOL_SIZE} pool → {DENSE_RERANK_SIZE} dense rerank → {FINAL_CHUNKS} final")
    
    # Process query
    query_rep = preprocess_query(query)
    
    # STEP 1: Retrieve initial large pool using BM25 + TF-IDF
    print(f"\n=== STEP 1: INITIAL POOL RETRIEVAL ({INITIAL_POOL_SIZE} candidates) ===")
    pool_candidates = retrieve_initial_pool(query, collection_name, INITIAL_POOL_SIZE)
    
    if not pool_candidates:
        print("[WARNING] No candidates found in initial pool")
        return []
    
    # STEP 2: Dense reranking on the pool
    print(f"\n=== STEP 2: DENSE SEMANTIC RERANKING (top {DENSE_RERANK_SIZE}) ===")
    top_candidates = dense_rerank_candidates(query, pool_candidates, DENSE_RERANK_SIZE)
    
    # STEP 3: Enhanced context features with heading boost
    print(f"\n=== STEP 3: CONTEXT ENHANCEMENT WITH HEADING BOOST ===")
    enhanced_candidates = enhance_with_context_features(top_candidates, query)
    
    # Re-sort by enhanced score
    enhanced_candidates.sort(key=lambda x: x["enhanced_score"], reverse=True)
    
    # STEP 4: Sentence extraction and final ranking
    print(f"\n=== STEP 4: SENTENCE EXTRACTION AND FINAL RANKING ===")
    final_candidates = []
    
    for i, candidate in enumerate(enhanced_candidates):
        print(f"[INFO] Processing candidate {i+1}/{len(enhanced_candidates)}")
        
        # Extract relevant sentences
        extracted_sentences = extract_relevant_sentences_with_openai(query, candidate["chunk"])
        
        # Calculate sentence-level semantic similarity
        sentence_similarity = calculate_semantic_similarity(query, extracted_sentences)
        
        # Create final result
        final_candidate = candidate.copy()
        final_candidate["extracted_sentences"] = extracted_sentences if extracted_sentences else "No relevant sentences found"
        final_candidate["sentence_similarity"] = sentence_similarity
        final_candidate["ultimate_score"] = candidate["enhanced_score"] * (1 + sentence_similarity)
        
        final_candidates.append(final_candidate)
    
    # STEP 5: Final ranking by sentence similarity
    print(f"\n=== STEP 5: FINAL RANKING (top {FINAL_CHUNKS}) ===")
    final_candidates.sort(key=lambda x: x["sentence_similarity"], reverse=True)
    top_final = final_candidates[:FINAL_CHUNKS]
    
    # STEP 6: Prepare results
    final_results = []
    for candidate in top_final:
        result = {
            "query": query,
            "extracted_sentences": candidate["extracted_sentences"],
            "chunk_context": candidate["chunk"],
            "page": candidate["page"],
            "heading": candidate.get("heading", ""),
            "document": candidate["document"],
            "bm25_score": candidate["bm25_score"],
            "tfidf_score": candidate["tfidf_score"],
            "combined_lexical_score": candidate["combined_score"],
            "semantic_score": candidate["semantic_score"],
            "final_score": candidate["final_score"],
            "context_score": candidate["context_score"],
            "heading_relevance": candidate["heading_relevance"],
            "hierarchy_bonus": candidate["hierarchy_bonus"],
            "heading_boost": candidate["heading_boost"],
            "enhanced_score": candidate["enhanced_score"],
            "sentence_similarity": candidate["sentence_similarity"],
            "ultimate_score": candidate["ultimate_score"]
        }
        final_results.append(result)
    
    print(f"[INFO] Completed processing. Final results: {len(final_results)} chunks")
    return final_results

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Test OpenAI connection
    try:
        test_response = client.chat.completions.create(
            #model="gpt-4.1",
            model='o3',
            messages=[{"role": "user", "content": "Test"}],
            #max_tokens=1
        )
        print("[INFO] OpenAI API connection successful")
    except Exception as e:
        print(f"[ERROR] OpenAI API connection failed: {e}")
        exit(1)
    
    # Load queries
    queries = []
    with open('question.txt', 'r', encoding='utf-8') as file:
        for line in file:
            cleaned = line.strip()
            if cleaned:
                queries.append(cleaned)
    
    print(f"[INFO] Loaded {len(queries)} queries")
    
    # Process each query
    collection_name = "ril_pdf_pages_heading_semantic"
    
    # Initialize CSV file
    csv_filename = "RIL_ExhaustiveQ6.csv"
    csv_headers = [
        "User Query", "Extracted Sentences", "Chunk Context", "Page", "Heading",
        "Document", "BM25 Score", "TF-IDF Score", "Combined Lexical Score",
        "Semantic Score", "Final Score", "Context Score", "Heading Relevance",
        "Hierarchy Bonus", "Heading Boost", "Enhanced Score", "Sentence Similarity", "Ultimate Score"
    ]
    
    # Write CSV header
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
    
    # Process queries
    for query_idx, query in enumerate(queries):
        print("\n" + "="*80)
        print(f"Processing query {query_idx+1}/{len(queries)}: {query}")
        print("="*80)
        
        # Run optimized hybrid search
        results = optimized_hybrid_search_with_dense_reranking(query, collection_name)
        
        # Save results to CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            for result in results:
                writer.writerow([
                    result["query"],
                    result["extracted_sentences"],
                    result["chunk_context"],
                    result["page"],
                    result["heading"],
                    result["document"],
                    result["bm25_score"],
                    result["tfidf_score"],
                    result["combined_lexical_score"],
                    result["semantic_score"],
                    result["final_score"],
                    result["context_score"],
                    result["heading_relevance"],
                    result["hierarchy_bonus"],
                    result["heading_boost"],
                    result["enhanced_score"],
                    result["sentence_similarity"],
                    result["ultimate_score"]
                ])
        
        print(f"[INFO] Saved {len(results)} results for query: {query}")
        time.sleep(2)
    print(f"\n[INFO] Processing complete!")
    print(f"[INFO] Results saved to: {csv_filename}")
    print(f"[INFO] Total rows: {len(queries) * FINAL_CHUNKS} ({len(queries)} queries × {FINAL_CHUNKS} chunks each)")