import re, os, csv
import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder
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
#from db_create_heading_singlepage import is_heading
import ast  # For parsing string representations of lists
from dotenv import load_dotenv
import ast  # For safely evaluating string representations of lists

# -------------------------
# Configuration
# -------------------------
INITIAL_POOL_SIZE = 250 # Large pool from BM25/TF-IDF
DENSE_RERANK_SIZE = 150 # Candidates after dense reranking
NEURAL_RERANK_SIZE = 75 # Candidates after neural reranking
FINAL_CHUNKS = 60 # Final chunks after sentence extraction
DB_PATH = "chromadb"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
# Best cross-encoder for reranking - you can also try 'ms-marco-MiniLM-L-12-v2' for faster processing
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
load_dotenv()

# -------------------------
# Excel Results Storage
# -------------------------
class IntermediateResultsStorage:
    def __init__(self, filename):
        self.filename = filename
        self.sparse_results = []
        self.dense_results = []
        self.neural_results = []
        self.final_results = []
    
    def add_sparse_results(self, query, keywords, candidates):
        """Add sparse retrieval results"""
        for i, candidate in enumerate(candidates[:50]):  # Top 50 for space efficiency
            self.sparse_results.append({
                'Query': query,
                'Keywords': '; '.join(keywords) if keywords else '',
                'Rank': i + 1,
                'Page': candidate.get('page', ''),
                'Heading': candidate.get('heading', ''),
                'Document': candidate.get('document', ''),
                'Chunk_Preview': candidate['chunk'][:500] + '...' if len(candidate['chunk']) > 500 else candidate['chunk']
            })
    
    def add_dense_results(self, query, keywords, candidates):
        """Add dense reranking results"""
        for i, candidate in enumerate(candidates[:50]):  # Top 50 for space efficiency
            self.dense_results.append({
                'Query': query,
                'Keywords': '; '.join(keywords) if keywords else '',
                'Rank': i + 1,
                'Page': candidate.get('page', ''),
                'Heading': candidate.get('heading', ''),
                'Document': candidate.get('document', ''),
                'Chunk_Preview': candidate['chunk'][:200] + '...' if len(candidate['chunk']) > 200 else candidate['chunk']
            })
    
    def add_neural_results(self, query, keywords, candidates):
        """Add neural reranking results"""
        for i, candidate in enumerate(candidates):
            self.neural_results.append({
                'Query': query,
                'Keywords': '; '.join(keywords) if keywords else '',
                'Rank': i + 1,
                'Page': candidate.get('page', ''),
                'Heading': candidate.get('heading', ''),
                'Document': candidate.get('document', ''),
                'Chunk_Preview': candidate['chunk'][:200] + '...' if len(candidate['chunk']) > 200 else candidate['chunk']
            })
    
    def add_final_results(self, query, keywords, results):
        """Add final results with extracted sentences"""
        for i, result in enumerate(results):
            self.final_results.append({
                'Query': query,
                'Keywords': '; '.join(keywords) if keywords else '',
                'Rank': i + 1,
                'Page': result.get('page', ''),
                'Heading': result.get('heading', ''),
                'Document': result.get('document', ''),
                'Extracted_Sentences': result.get('extracted_sentences', ''),
                'Chunk_Context': result.get('chunk_context', '')[:300] + '...' if len(result.get('chunk_context', '')) > 300 else result.get('chunk_context', '')
            })
    
    def save_to_excel(self):
        """Save all intermediate results to Excel with multiple sheets"""
        print(f"[INFO] Saving intermediate results to {self.filename}")
        
        with pd.ExcelWriter(self.filename, engine='openpyxl') as writer:
            # Sparse Retrieval Results
            if self.sparse_results:
                df_sparse = pd.DataFrame(self.sparse_results)
                df_sparse.to_excel(writer, sheet_name='1_Sparse_Retrieval', index=False)
                print(f"[INFO] Saved {len(self.sparse_results)} sparse retrieval results")
            
            # Dense Reranking Results
            if self.dense_results:
                df_dense = pd.DataFrame(self.dense_results)
                df_dense.to_excel(writer, sheet_name='2_Dense_Reranking', index=False)
                print(f"[INFO] Saved {len(self.dense_results)} dense reranking results")
            
            # Neural Reranking Results
            if self.neural_results:
                df_neural = pd.DataFrame(self.neural_results)
                df_neural.to_excel(writer, sheet_name='3_Neural_Reranking', index=False)
                print(f"[INFO] Saved {len(self.neural_results)} neural reranking results")
            
            # Final Results
            if self.final_results:
                df_final = pd.DataFrame(self.final_results)
                df_final.to_excel(writer, sheet_name='4_Final_Results', index=False)
                print(f"[INFO] Saved {len(self.final_results)} final results")
        
        print(f"[SUCCESS] All intermediate results saved to {self.filename}")


# -------------------------
# Remove duplicates
# -------------------------
def deduplicate_candidates(candidates, similarity_threshold=0.9):
    """
    Remove duplicate chunks based on text similarity.
    
    Args:
        candidates: List of candidate dictionaries with 'chunk' key
        similarity_threshold: Float threshold for considering chunks as duplicates
    
    Returns:
        List of deduplicated candidates
    """
    if len(candidates) <= 1:
        return candidates
    
    print(f"[INFO] Deduplicating {len(candidates)} candidates (threshold: {similarity_threshold})")
    
    # Extract chunk texts for comparison
    chunk_texts = [candidate["chunk"] for candidate in candidates]
    
    try:
        # Use TF-IDF vectorizer for efficient similarity computation
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(chunk_texts)
        
        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicates
        to_remove = set()
        for i in range(len(candidates)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(candidates)):
                if j in to_remove:
                    continue
                if similarity_matrix[i][j] >= similarity_threshold:
                    # Keep the one with higher score (candidates are sorted by relevance)
                    to_remove.add(j)
                    print(f"[DEBUG] Removing duplicate chunk {j} (similarity: {similarity_matrix[i][j]:.3f})")
        
        # Remove duplicates
        deduplicated = [candidates[i] for i in range(len(candidates)) if i not in to_remove]
        
        print(f"[INFO] Removed {len(candidates) - len(deduplicated)} duplicates. Final count: {len(deduplicated)}")
        return deduplicated
        
    except Exception as e:
        print(f"[WARNING] Deduplication failed: {e}. Returning original candidates.")
        return candidates

# -------------------------
# Initialize Components
# -------------------------
print("[INFO] Initializing components...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer(MODEL_NAME)
# Initialize cross-encoder for neural reranking
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
client_chroma = chromadb.PersistentClient(path=DB_PATH)
nlp = spacy.load("en_core_web_md")

# -------------------------
# Keywords Loading and Management
# -------------------------
def load_keywords_from_excel(excel_path):
    """Original function with minimal changes for debugging"""
    try:
        df = pd.read_excel(excel_path)
        keywords_dict = {}
        df.columns = [col.strip() for col in df.columns]
        for _, row in df.iterrows():
            question = row['Question'].strip()
            keywords_raw = row['Keywords']
            
            # Parse keywords - handle both string representation of list and actual list
            if isinstance(keywords_raw, str):
                try:
                    # Try to parse as literal (for string representation of lists)
                    keywords = ast.literal_eval(keywords_raw)
                    if not isinstance(keywords, list):
                        # If it's not a list, split by common delimiters
                        keywords = [k.strip() for k in str(keywords).split(',')]
                except:
                    # Fallback: split by commas
                    keywords = [k.strip() for k in keywords_raw.split(',')]
            elif isinstance(keywords_raw, list):
                keywords = keywords_raw
            else:
                keywords = []
            
            # Clean and normalize keywords
            keywords = [k.strip().strip("'\"") for k in keywords if k.strip()]
            keywords_dict[question] = keywords
        
        print(f"[INFO] Loaded keywords for {len(keywords_dict)} questions")
        return keywords_dict
        
    except Exception as e:
        print(f"[ERROR] Failed to load keywords from Excel: {e}")
        return {}

def find_matching_keywords(query, keywords_dict):
    """
    Enhanced version of find_matching_keywords with detailed debugging
    """
    print(f"\n[DEBUG] Finding keywords for query: '{query}'")
    print(f"[DEBUG] Keywords dict has {len(keywords_dict)} entries")
    
    query_lower = query.lower().strip()
    print(f"[DEBUG] Normalized query: '{query_lower}'")
    
    # First, try exact match
    print(f"[DEBUG] Trying exact match...")
    for question, keywords in keywords_dict.items():
        question_lower = question.lower().strip()
        if query_lower == question_lower:
            print(f"[SUCCESS] Found exact match!")
            print(f"[DEBUG] Original question: '{question}'")
            print(f"[DEBUG] Keywords: {keywords}")
            return keywords
    
    print(f"[DEBUG] No exact match found, trying partial matching...")
    
    # Then try partial matching
    query_words = set(query_lower.split())
    print(f"[DEBUG] Query words: {query_words}")
    
    best_match = None
    best_score = 0
    
    for question, keywords in keywords_dict.items():
        question_lower = question.lower().strip()
        question_words = set(question_lower.split())
        
        # Calculate overlap
        intersection = query_words.intersection(question_words)
        union_size = max(len(query_words), len(question_words))
        
        if union_size > 0:
            overlap_ratio = len(intersection) / union_size
            print(f"[DEBUG] Question: '{question[:50]}...' -> Overlap: {overlap_ratio:.3f}")
            
            if overlap_ratio > 0.8 and overlap_ratio > best_score:
                best_match = (question, keywords)
                best_score = overlap_ratio
    
    if best_match:
        print(f"[SUCCESS] Found partial match with score {best_score:.3f}")
        print(f"[DEBUG] Matched question: '{best_match[0]}'")
        print(f"[DEBUG] Keywords: {best_match[1]}")
        return best_match[1]
    
    print(f"[WARNING] No matching keywords found for query: '{query}'")
    return []

# -------------------------
# OpenAI Sentence Extraction
# -------------------------
def extract_relevant_sentences_with_openai(query, context_text):
    """Extract all relevant sentences from context using OpenAI."""
    system_prompt = """You are an expert text analyzer. Extract content that could help answer the user's query.

IMPORTANT: Be VERY lenient in your relevance assessment. Include content that:
- Directly answers the query
- Provides context or background information
- Contains related concepts, terms, or examples
- Mentions the same entities, topics, or domains

Instructions:
1. If ANY part of the text relates to the query (even tangentially), extract the relevant portions verbatim
2. Only return "NONE" if the text is completely unrelated to the query topic at all, that is, if it is in a complete different domain
3. When in doubt, extract the content - it's better to include potentially relevant information

Extract relevant content exactly as written:"""

    user_prompt = f"Query: {query}\n\nContext: {context_text}\n\nExtract relevant sentences:"

    try:
        response = client.chat.completions.create(
            model='o3',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
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
    if not sentence_group or sentence_group == "No relevant sentences found":
        return 0.0

    try:
        # Ensure sentence_group is always a list
        if isinstance(sentence_group, str):
            sentence_group = [sentence_group]
        query_embedding = model.encode([query], convert_to_tensor=True)
        sentence_embedding = model.encode(sentence_group, convert_to_tensor=True)

        # Normalize embeddings along the batch dimension
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

        # Compute cosine similarities and return the highest score
        similarity = util.pytorch_cos_sim(query_embedding, sentence_embedding)[0]
        return float(similarity.max().cpu().numpy())
    except Exception as e:
        print(f"[ERROR] Semantic similarity calculation failed: {e}")
        return 0.0

# -------------------------
# Enhanced Initial Retrieval with Keywords
# -------------------------
def retrieve_initial_pool_with_keywords(query, keywords_list, collection_name, pool_size=INITIAL_POOL_SIZE):
    """
    Retrieve initial large pool using BM25 and TF-IDF fusion enhanced with keywords.
    """
    print(f"[INFO] Retrieving initial pool of {pool_size} candidates using BM25 + TF-IDF + Keywords...")
    print(f"[INFO] Query keywords: {keywords_list}")
    
    collection = client_chroma.get_collection(collection_name)
    data = collection.get()
    
    # Prepare data
    chunks = [meta.get("chunk", "") for meta in data["metadatas"]]
    pages = [meta.get("page", "Unknown") for meta in data["metadatas"]]
    headings = [meta.get("heading", "") for meta in data["metadatas"]]
    doc_ids = data.get("ids", ["unknown"] * len(chunks))
    

    print(f"[INFO] Processing {len(chunks)} total chunks...")
    
    # -------------------------
    # Enhanced BM25 Scoring with Keywords
    # -------------------------
    print("[INFO] Computing enhanced BM25 scores with keywords...")

    # --- 1. Prepare phrases ---
    def normalize_phrase(phrase):
        return phrase.lower().strip().replace(" ", "_")

    # Lowercase + normalize phrases from keywords_list
    phrase_keywords = [kw.lower() for kw in keywords_list if isinstance(kw, str) and " " in kw]
    phrase_tokens = [normalize_phrase(kw) for kw in phrase_keywords]

    # --- 2. Tokenize and lemmatize chunks, inject phrase tokens ---
    tokenized = []
    for chunk in chunks:
        text = chunk.lower()
        doc = nlp(text)
        
        tokens = [token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct and len(token.text) > 1]

        for phrase, phrase_token in zip(phrase_keywords, phrase_tokens):
            if phrase in text:
                tokens.append(phrase_token)  # phrase token injection

        tokenized.append(tokens)

    # --- 3. Build BM25 index ---
    bm25 = BM25Okapi(tokenized)

    # --- 4. Process query ---
    query_text = query.lower()
    query_doc = nlp(query_text)
    query_terms = [token.lemma_ for token in query_doc 
                if not token.is_stop and not token.is_punct and len(token.text) > 1]

    # Inject phrase tokens from keywords if present in query
    for phrase, phrase_token in zip(phrase_keywords, phrase_tokens):
        if phrase in query_text:
            query_terms.append(phrase_token)

    # --- 5. Process keyword tokens ---
    keyword_terms = set()
    for kw in keywords_list:
        if isinstance(kw, str) and len(kw) > 1:
            doc = nlp(kw.lower())
            keyword_terms.update([
                token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct and len(token.text) > 1
            ])

    keyword_terms.update(set(phrase_tokens))  # Add phrase tokens to keyword terms

    # --- 6. Combine and score ---
    enhanced_query_terms = query_terms*3+ list(set(keyword_terms))
    bm25_scores = bm25.get_scores(enhanced_query_terms)

    
    # -------------------------
    # Enhanced TF-IDF Scoring with Keywords
    # -------------------------
    print("[INFO] Computing enhanced TF-IDF scores with keywords...")
    
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
    
    # Process enhanced query for TF-IDF
    enhanced_query_text = " ".join(query_terms*4+ list(keyword_terms))
    
    # Get TF-IDF scores
    query_vector = vectorizer.transform([enhanced_query_text])
    tfidf_scores = (query_vector * tfidf_matrix.T).toarray().flatten()
    
    # -------------------------
    # Score Fusion and Ranking
    # -------------------------
    print("[INFO] Fusing BM25 and TF-IDF scores...")
    
    # Normalize scores to [0, 1] range
    bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
    tfidf_scores_norm = (tfidf_scores - np.min(tfidf_scores)) / (np.max(tfidf_scores) - np.min(tfidf_scores) + 1e-8)
    
    # Combine scores (weighted average)
    combined_scores = (0.7 * bm25_scores_norm + 0.3 * tfidf_scores_norm)
    
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
        candidate["dense_score"] = (0.3 * candidate["combined_score"] + 
                                   0.7 * candidate["semantic_score"])
    
    # Sort by dense score and return top candidates
    candidates.sort(key=lambda x: x["dense_score"], reverse=True)
    top_candidates = candidates[:rerank_size]
    
    print(f"[INFO] Dense reranked to top {len(top_candidates)} candidates")
    return top_candidates

# -------------------------
# Neural Reranking with Cross-Encoder
# -------------------------
def neural_rerank_candidates(query, candidates, rerank_size=NEURAL_RERANK_SIZE):
    """
    Apply neural reranking using a cross-encoder model.
    Cross-encoders are specifically designed for reranking tasks and typically
    provide better relevance scores than bi-encoders.
    """
    print(f"[INFO] Neural reranking top {rerank_size} candidates using cross-encoder...")
    
    if len(candidates) == 0:
        return []
    
    # Prepare query-document pairs for cross-encoder
    query_doc_pairs = []
    for candidate in candidates:
        # Truncate long documents to avoid token limits (cross-encoders have token limits)
        chunk_text = candidate["chunk"]
        if len(chunk_text) > 4000:  # Rough character limit
            chunk_text = chunk_text[:4000] + "..."
        query_doc_pairs.append([query, chunk_text])
    
    print(f"[INFO] Computing cross-encoder scores for {len(query_doc_pairs)} pairs...")
    
    # Get cross-encoder scores (these are relevance scores, not similarities)
    try:
        cross_encoder_scores = cross_encoder.predict(query_doc_pairs, batch_size=16)
        
        # Add neural scores to candidates
        for i, candidate in enumerate(candidates):
            candidate["neural_score"] = float(cross_encoder_scores[i])
            # Combine with previous scores
            candidate["neural_enhanced_score"] = (
                0.2 * candidate["combined_score"] +  # Lexical
                0.2 * candidate["semantic_score"] +  # Dense semantic
                0.6 * candidate["neural_score"]      # Neural reranking (highest weight)
            )
        
        # Sort by neural enhanced score and return top candidates
        candidates.sort(key=lambda x: x["neural_enhanced_score"], reverse=True)
        top_candidates = candidates[:rerank_size]
        
        print(f"[INFO] Neural reranked to top {len(top_candidates)} candidates")
        
        # Print top 3 scores for debugging
        print("[DEBUG] Top 3 neural reranking scores:")
        for i, candidate in enumerate(top_candidates[:3]):
            print(f"  {i+1}. Neural: {candidate['neural_score']:.4f}, "
                  f"Combined: {candidate['neural_enhanced_score']:.4f}")
        
        return top_candidates
        
    except Exception as e:
        print(f"[ERROR] Neural reranking failed: {e}")
        print("[INFO] Falling back to dense reranking results...")
        return candidates[:rerank_size]

# -------------------------
# Enhanced Context Enhancement (Simplified - No Heading Boost)
# -------------------------
def enhance_with_basic_context_features(chunks, query):
    """Add basic context-aware features to enhance ranking."""
    print("[INFO] Adding basic context-aware features...")
    
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
        
        # Store context score
        chunk["context_score"] = context_bonus
        
        # Enhanced score with context bonus
        chunk["final_enhanced_score"] = chunk["neural_enhanced_score"] + context_bonus*0.2
    
    return chunks

# -------------------------
# Main Enhanced Hybrid Search with Neural Reranking and Intermediate Results
# -------------------------
def optimized_hybrid_search_with_neural_reranking(query, keywords_list, collection_name, results_storage):
    """
    Enhanced hybrid search pipeline with neural reranking and intermediate results storage:
    1. Retrieve large pool using BM25 + TF-IDF + Keywords fusion
    2. Dense reranking on pool to get top candidates
    3. Neural reranking using cross-encoder
    4. Basic context enhancement
    5. Final sentence extraction and ranking
    """
    print(f"\n[INFO] Starting enhanced hybrid search with neural reranking for query: {query}")
    print(f"[INFO] Keywords: {keywords_list}")
    print(f"[INFO] Pipeline: {INITIAL_POOL_SIZE} pool → {DENSE_RERANK_SIZE} dense → {NEURAL_RERANK_SIZE} neural → {FINAL_CHUNKS} final")
    
    # STEP 1: Retrieve initial large pool using BM25 + TF-IDF + Keywords
    print(f"\n=== STEP 1: INITIAL POOL RETRIEVAL WITH KEYWORDS ({INITIAL_POOL_SIZE} candidates) ===")
    pool_candidates = retrieve_initial_pool_with_keywords(query, keywords_list, collection_name, INITIAL_POOL_SIZE)
    
    if not pool_candidates:
        print("[WARNING] No candidates found in initial pool")
        return []
    
    # STEP 1.5: Deduplicate initial pool
    print(f"\n=== STEP 1.5: DEDUPLICATING INITIAL POOL ===")
    pool_candidates = deduplicate_candidates(pool_candidates, similarity_threshold=0.9)
    
    # Store sparse retrieval results
    results_storage.add_sparse_results(query, keywords_list, pool_candidates)
    
    # STEP 2: Dense reranking on the pool
    print(f"\n=== STEP 2: DENSE SEMANTIC RERANKING (top {DENSE_RERANK_SIZE}) ===")
    dense_candidates = dense_rerank_candidates(query, pool_candidates, DENSE_RERANK_SIZE)
    
    # Store dense reranking results
    results_storage.add_dense_results(query, keywords_list, dense_candidates)
    
    # STEP 3: Neural reranking using cross-encoder
    print(f"\n=== STEP 3: NEURAL RERANKING (top {NEURAL_RERANK_SIZE}) ===")
    neural_candidates = neural_rerank_candidates(query, dense_candidates, NEURAL_RERANK_SIZE)
    
    # Store neural reranking results
    results_storage.add_neural_results(query, keywords_list, neural_candidates)
    
    # STEP 4: Basic context enhancement
    print(f"\n=== STEP 4: BASIC CONTEXT ENHANCEMENT ===")
    enhanced_candidates = enhance_with_basic_context_features(neural_candidates, query)
    
    # Re-sort by final enhanced score
    enhanced_candidates.sort(key=lambda x: x["final_enhanced_score"], reverse=True)
    
    # STEP 5: Sentence extraction and final ranking
    print(f"\n=== STEP 5: SENTENCE EXTRACTION AND FINAL RANKING ===")
    final_candidates = []
    
    for i, candidate in enumerate(enhanced_candidates):
        print(f"[INFO] Processing candidate {i+1}/{len(enhanced_candidates)}")
        
        # Extract relevant sentences
        extracted_sentences = extract_relevant_sentences_with_openai(
            query, candidate["chunk"])
        
        # Calculate sentence-level semantic similarity
        sentence_similarity = calculate_semantic_similarity(query, extracted_sentences)
        
        # Create final result
        final_candidate = candidate.copy()
        final_candidate["extracted_sentences"] = extracted_sentences if extracted_sentences else "No relevant sentences found"
        final_candidate["sentence_similarity"] = sentence_similarity
        final_candidate["ultimate_score"] = (candidate["final_enhanced_score"] * 
                                            (1 + sentence_similarity*0.2))
        
        final_candidates.append(final_candidate)
    
    # STEP 6: Final ranking by ultimate score
    print(f"\n=== STEP 6: FINAL RANKING (top {FINAL_CHUNKS}) ===")
    final_candidates.sort(key=lambda x: x["ultimate_score"], reverse=True)
    top_final = final_candidates[:FINAL_CHUNKS]
    
    # STEP 7: Prepare results
    final_results = []
    for candidate in top_final:
        result = {
            "query": query,
            "keywords": keywords_list,
            "extracted_sentences": candidate["extracted_sentences"],
            "chunk_context": candidate["chunk"],
            "page": candidate["page"],
            "heading": candidate.get("heading", ""),
            "document": candidate["document"],
            "bm25_score": candidate["bm25_score"],
            "tfidf_score": candidate["tfidf_score"],
            "combined_lexical_score": candidate["combined_score"],
            "semantic_score": candidate["semantic_score"],
            "dense_score": candidate["dense_score"],
            "neural_score": candidate["neural_score"],
            "neural_enhanced_score": candidate["neural_enhanced_score"],
            "context_score": candidate["context_score"],
            "final_enhanced_score": candidate["final_enhanced_score"],
            "sentence_similarity": candidate["sentence_similarity"],
            "ultimate_score": candidate["ultimate_score"]
        }
        final_results.append(result)
    
    # Store final results
    results_storage.add_final_results(query, keywords_list, final_results)
    
    print(f"[INFO] Completed processing. Final results: {len(final_results)} chunks")
    return final_results

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Test OpenAI connection
    try:
        test_response = client.chat.completions.create(
            model='o3',
            messages=[{"role": "user", "content": "Test"}],
        )
        print("[INFO] OpenAI API connection successful")
    except Exception as e:
        print(f"[ERROR] OpenAI API connection failed: {e}")
        exit(1)
    
    # Initialize intermediate results storage
    intermediate_excel_filename = "Novartis_intermediate_results_neural_rerankQ3.xlsx"
    results_storage = IntermediateResultsStorage(intermediate_excel_filename)
    
    # Load keywords from Excel
    keywords_excel_path = "novartis_keywords_by_question.xlsx" #keywords file path
    keywords_dict = load_keywords_from_excel(keywords_excel_path)
    
    # Load queries
    queries = []
    with open('question2.txt', 'r', encoding='utf-8') as file:
        for line in file:
            cleaned = line.strip()
            if cleaned:
                queries.append(cleaned)
    
    print(f"[INFO] Loaded {len(queries)} queries")
    
    # Process each query
    collection_name = "novartis_combined_chunks_docx"
    
    # Initialize CSV file with updated headers (keeping original CSV functionality)
    csv_filename = "Novartis_intermediateQ3.csv"
    csv_headers = [
        "User Query", "Keywords", "Extracted Sentences", "Chunk Context", "Page", "Heading",
        "Document", "BM25 Score", "TF-IDF Score", "Combined Lexical Score",
        "Semantic Score", "Dense Score", "Neural Score", "Neural Enhanced Score", 
        "Context Score", "Final Enhanced Score", "Sentence Similarity", "Ultimate Score"
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

        query_keywords = find_matching_keywords(query, keywords_dict)
        
        try:
            # Run enhanced hybrid search with neural reranking
            results = optimized_hybrid_search_with_neural_reranking(
                query, query_keywords, collection_name, results_storage)
            
            if results:
                print(f"[SUCCESS] Found {len(results)} results for query: {query}")
                
                # Write results to CSV (keeping original functionality)
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    for result in results:
                        writer.writerow([
                            result["query"],
                            "; ".join(result["keywords"]),
                            result["extracted_sentences"],
                            result["chunk_context"],
                            result["page"],
                            result["heading"],
                            result["document"],
                            round(result["bm25_score"], 4),
                            round(result["tfidf_score"], 4),
                            round(result["combined_lexical_score"], 4),
                            round(result["semantic_score"], 4),
                            round(result["dense_score"], 4),
                            round(result["neural_score"], 4),
                            round(result["neural_enhanced_score"], 4),
                            round(result["context_score"], 4),
                            round(result["final_enhanced_score"], 4),
                            round(result["sentence_similarity"], 4),
                            round(result["ultimate_score"], 4)
                        ])
                
                # Print top 3 results for verification
                print(f"\n[TOP 3 RESULTS FOR]: {query}")
                print(f"[KEYWORDS]: {query_keywords}")
                for i, result in enumerate(results[:3]):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Page: {result['page']}")
                    print(f"Heading: {result['heading']}")
                    print(f"Neural Score: {result['neural_score']:.4f}")
                    print(f"Ultimate Score: {result['ultimate_score']:.4f}")
                    print(f"Extracted: {result['extracted_sentences'][:200]}...")
            
            else:
                print(f"[WARNING] No results found for query: {query}")
                # Write empty row to maintain structure
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([query, "; ".join(query_keywords), "No results found"] + [""] * 15)
        
        except Exception as e:
            print(f"[ERROR] Failed to process query '{query}': {e}")
            # Write error row
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query, "; ".join(query_keywords), f"Error: {str(e)}"] + [""] * 15)
        
        # Small delay between queries to avoid overwhelming the API
        time.sleep(1)
    
    # Save all intermediate results to Excel
    print(f"\n[INFO] Saving intermediate results to Excel...")
    results_storage.save_to_excel()
    
    print(f"\n[COMPLETION] Processing completed!")
    print(f"[SUMMARY] Processed {len(queries)} queries with neural reranking hybrid search")
    print(f"[OUTPUT FILES]:")
    print(f"  - Final results CSV: {csv_filename}")
    print(f"  - Intermediate results Excel: {intermediate_excel_filename}")
    print("[INFO] Excel file contains 4 sheets:")
    print("  1. Sparse Retrieval - Results after BM25+TF-IDF fusion")
    print("  2. Dense Reranking - Results after semantic dense reranking") 
    print("  3. Neural Reranking - Results after cross-encoder neural reranking")
    print("  4. Final Results - Results after sentence extraction and final ranking")