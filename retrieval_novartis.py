import re, os, csv
import pandas as pd
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
from db_create_heading_singlepage import is_heading
import ast  # For parsing string representations of lists
from dotenv import load_dotenv
import ast  # For safely evaluating string representations of lists

# -------------------------
# Configuration
# -------------------------
INITIAL_POOL_SIZE = 300#rge pool from BM25/TF-IDF
DENSE_RERANK_SIZE =150#tes after dense reranking
FINAL_CHUNKS = 50 # Final chunks after sentence extraction
DB_PATH = "chromadb"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
load_dotenv()

# -------------------------
# Initialize Components
# -------------------------
print("[INFO] Initializing components...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer(MODEL_NAME)
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
# Heading Analysis Functions
# -------------------------

#imported from db_create_heading_singlepage.py

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
    system_prompt = """You are an expert text analyzer. Extract content that could help answer the user's query.

IMPORTANT: Be VERY lenient in your relevance assessment. Include content that:
- Directly answers the query
- Provides context or background information
- Contains related concepts, terms, or examples
- Mentions the same entities, topics, or domains

Instructions:
1. If ANY part of the text relates to the query (even tangentially), extract the relevant portions verbatim
2. Only return "NONE" if the text is completely unrelated to the query topic
3. When in doubt, extract the content - it's better to include potentially relevant information

Extract relevant content exactly as written:"""


    user_prompt = f"Query: {query}\n\nContext: {context_text}\n\nExtract relevant sentences:"

    try:
        response = client.chat.completions.create(
            #model="gpt-4o",
            model='o3',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
           # temperature=0.1,
            #max_tokens=1500
        )
        
        extracted = response.choices[0].message.content.strip()
        return extracted if extracted != "NONE" else ""
        
    except Exception as e:
        print(f"[ERROR] OpenAI extraction failed: {e}")
        return ""
    

'''

# -------------------------
# Keywords-Enhanced Scoring
# -------------------------
def calculate_keyword_match_score(text: str, keywords_list: List[str]) -> float:
    """
    Calculate how well the text matches the provided keywords.
    """
    if not keywords_list or not text:
        return 0.0
    
    text_lower = text.lower()
    matches = 0
    total_keywords = len(keywords_list)
    
    # Direct keyword matching
    for keyword in keywords_list:
        keyword_lower = keyword.lower().strip()
        if keyword_lower in text_lower:
            matches += 1

    # Calculate match ratio
    direct_match_ratio = matches / total_keywords if total_keywords > 0 else 0.0
    
    # Semantic similarity with keywords (average)
    if keywords_list:
        try:
            text_embedding = model.encode(text, convert_to_tensor=True)
            keyword_embeddings = model.encode(keywords_list, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(text_embedding, keyword_embeddings).cpu().numpy()[0]
            semantic_match_score = np.mean(similarities)
        except:
            semantic_match_score = 0.0
    else:
        semantic_match_score = 0.0
    
    # Combined score
    keyword_score = (direct_match_ratio * 0.6) + (semantic_match_score * 0.4)
    return keyword_score

'''
    

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
# Query Processing with Keywords
# -------------------------
def preprocess_query_with_keywords(query, keywords_list):
    """Process query to extract key terms and incorporate keywords."""
    doc = nlp(query)
    key_terms = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop]
    
    # Add keywords to key terms
    enhanced_terms = key_terms + keywords_list
    
    return {
        "original": query,
        "key_terms": key_terms,
        "keywords": keywords_list,
        "enhanced_terms": enhanced_terms
    }

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
    
    # Tokenize for BM25
    tokenized = []
    for chunk in chunks:
        doc = nlp(chunk)
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and len(token.text) > 1]
        tokenized.append(tokens)
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized)
    
    # Process query + keywords for BM25
    query_doc = nlp(query)
    query_terms = [token.lemma_.lower() for token in query_doc 
                   if not token.is_stop and not token.is_punct and len(token.text) > 1]
    
    # Add keyword terms
    keyword_terms = []
    for keyword in keywords_list:
        keyword_doc = nlp(keyword)
        kw_terms = [token.lemma_.lower() for token in keyword_doc 
                   if not token.is_stop and not token.is_punct and len(token.text) > 1]
        keyword_terms.extend(kw_terms)
    
    # Combine query and keyword terms
    enhanced_query_terms = query_terms + keyword_terms
    
    # Get BM25 scores
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
    enhanced_query_text = " ".join(query_terms + keyword_terms)
    
    # Get TF-IDF scores
    query_vector = vectorizer.transform([enhanced_query_text])
    tfidf_scores = (query_vector * tfidf_matrix.T).toarray().flatten()
    
    # -------------------------
    # Keyword Matching Scores
    # -------------------------

    '''
    print("[INFO] Computing keyword matching scores...")
    keyword_scores = []
    for chunk in chunks:
        keyword_score = calculate_keyword_match_score(chunk, keywords_list)
        keyword_scores.append(keyword_score)
    
    keyword_scores = np.array(keyword_scores)
    '''
    # -------------------------
    # Score Fusion and Ranking
    # -------------------------
    print("[INFO] Fusing BM25, TF-IDF, and keyword scores...")
    
    # Normalize scores to [0, 1] range
    bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
    tfidf_scores_norm = (tfidf_scores - np.min(tfidf_scores)) / (np.max(tfidf_scores) - np.min(tfidf_scores) + 1e-8)
    #keyword_scores_norm = (keyword_scores - np.min(keyword_scores)) / (np.max(keyword_scores) - np.min(keyword_scores) + 1e-8)
    
    # Combine scores (weighted average)
    combined_scores = (0.7 * bm25_scores_norm + 
                      0.3* tfidf_scores_norm)
    
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
            #"keyword_score": keyword_scores[idx],
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
        # Combine with lexical + keyword score (weighted)
        candidate["final_score"] = (0.3 * candidate["combined_score"] + 
                                   0.7 * candidate["semantic_score"])
    
    # Sort by final score and return top candidates
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top_candidates = candidates[:rerank_size]
    
    print(f"[INFO] Reranked to top {len(top_candidates)} candidates")
    return top_candidates

# -------------------------
# Enhanced Context Enhancement with Heading Boost and Keywords
# -------------------------
def enhance_with_context_features_and_keywords(chunks, query, keywords_list):
    """Add context-aware features including heading-based scoring and keyword matching to enhance ranking."""
    print("[INFO] Adding context-aware features with heading boost and keyword enhancement...")
    
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
        # Heading-based scoring boost
        # -------------------------
        
        # 1. Heading relevance score
        heading_relevance = calculate_heading_relevance_score(
            query, context, chunk.get("heading", "")
        )
        
        # 2. Heading hierarchy bonus
        hierarchy_bonus = calculate_heading_hierarchy_bonus(query, context)
        
        # 3. Combined heading boost
        heading_boost = (heading_relevance * 0.6) + (hierarchy_bonus * 0.4)
        
        # -------------------------
        # Enhanced keyword matching
        # -------------------------
        #enhanced_keyword_score = calculate_keyword_match_score(context, keywords_list)
        
        # Store individual scores
        chunk["context_score"] = context_bonus
        chunk["heading_relevance"] = heading_relevance
        chunk["hierarchy_bonus"] = hierarchy_bonus
        chunk["heading_boost"] = heading_boost
        #chunk["enhanced_keyword_score"] = enhanced_keyword_score
        
        # Enhanced score with heading boost and keyword enhancement
        chunk["enhanced_score"] = (
            chunk["final_score"] + 
            context_bonus + 
            (heading_boost * 0.3)  # Heading boost weight
  )
    
    return chunks

# -------------------------
# Main Enhanced Hybrid Search with Keywords
# -------------------------
def optimized_hybrid_search_with_keywords(query, keywords_list, collection_name):
    """
    Enhanced hybrid search pipeline with keyword integration:
    1. Retrieve large pool using BM25 + TF-IDF + Keywords fusion
    2. Dense reranking on pool to get top candidates
    3. Enhanced context features with heading boost
    5. Final semantic reranking
    """
    print(f"\n[INFO] Starting enhanced hybrid search with keywords for query: {query}")
    print(f"[INFO] Keywords: {keywords_list}")
    print(f"[INFO] Pipeline: {INITIAL_POOL_SIZE} pool → {DENSE_RERANK_SIZE} dense rerank → {FINAL_CHUNKS} final")
    
    # Process query with keywords
    query_rep = preprocess_query_with_keywords(query, keywords_list)
    
    # STEP 1: Retrieve initial large pool using BM25 + TF-IDF + Keywords
    print(f"\n=== STEP 1: INITIAL POOL RETRIEVAL WITH KEYWORDS ({INITIAL_POOL_SIZE} candidates) ===")
    pool_candidates = retrieve_initial_pool_with_keywords(query, keywords_list, collection_name, INITIAL_POOL_SIZE)
    
    if not pool_candidates:
        print("[WARNING] No candidates found in initial pool")
        return []
    
    # STEP 2: Dense reranking on the pool
    print(f"\n=== STEP 2: DENSE SEMANTIC RERANKING (top {DENSE_RERANK_SIZE}) ===")
    top_candidates = dense_rerank_candidates(query, pool_candidates, DENSE_RERANK_SIZE)
    
    # STEP 3: Enhanced context features with heading boost and keywords
    print(f"\n=== STEP 3: CONTEXT ENHANCEMENT WITH HEADING BOOST AND KEYWORDS ===")
    enhanced_candidates = enhance_with_context_features_and_keywords(top_candidates, query, keywords_list)
    
    # Re-sort by enhanced score
    enhanced_candidates.sort(key=lambda x: x["enhanced_score"], reverse=True)
    
    # STEP 4: Keyword-aware sentence extraction and final ranking
    print(f"\n=== STEP 4:SENTENCE EXTRACTION AND FINAL RANKING ===")
    final_candidates = []
    
    for i, candidate in enumerate(enhanced_candidates):
        print(f"[INFO] Processing candidate {i+1}/{len(enhanced_candidates)}")
        
        # Extract relevant sentences with keyword guidance
        extracted_sentences = extract_relevant_sentences_with_openai(
            query, candidate["chunk"])
        
        # Calculate sentence-level semantic similarity
        sentence_similarity = calculate_semantic_similarity(query, extracted_sentences)
        
        # Calculate sentence-level keyword matching
        #sentence_keyword_score = calculate_keyword_match_score(extracted_sentences, keywords_list)
        
        # Create final result
        final_candidate = candidate.copy()
        final_candidate["extracted_sentences"] = extracted_sentences if extracted_sentences else "No relevant sentences found"
        final_candidate["sentence_similarity"] = sentence_similarity
        #final_candidate["sentence_keyword_score"] = sentence_keyword_score
        final_candidate["ultimate_score"] = (candidate["enhanced_score"] * 
                                            (1 + sentence_similarity))
        
        final_candidates.append(final_candidate)
    
    # STEP 5: Final ranking by ultimate score
    print(f"\n=== STEP 5: FINAL RANKING (top {FINAL_CHUNKS}) ===")
    final_candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
    top_final = final_candidates[:FINAL_CHUNKS]
    
    # STEP 6: Prepare results
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
           # "keyword_score": candidate["keyword_score"],
            "combined_lexical_score": candidate["combined_score"],
            "semantic_score": candidate["semantic_score"],
            "final_score": candidate["final_score"],
            "context_score": candidate["context_score"],
            "heading_relevance": candidate["heading_relevance"],
            "hierarchy_bonus": candidate["hierarchy_bonus"],
            "heading_boost": candidate["heading_boost"],
           # "enhanced_keyword_score": candidate["enhanced_keyword_score"],
            "enhanced_score": candidate["enhanced_score"],
            "sentence_similarity": candidate["sentence_similarity"],
           # "sentence_keyword_score": candidate["sentence_keyword_score"],
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
            model='o3',
            messages=[{"role": "user", "content": "Test"}],
        )
        print("[INFO] OpenAI API connection successful")
    except Exception as e:
        print(f"[ERROR] OpenAI API connection failed: {e}")
        exit(1)
    
    # Load keywords from Excel
    keywords_excel_path = "novartis_keywords_by_question.xlsx" #keywords file path
    keywords_dict = load_keywords_from_excel(keywords_excel_path)
    
    # Load queries
    queries = []
    with open('question.txt', 'r', encoding='utf-8') as file:
        for line in file:
            cleaned = line.strip()
            if cleaned:
                queries.append(cleaned)
    
    print(f"[INFO] Loaded {len(queries)} queries")
    
    # Process each query
    collection_name = "combined_novartis_heading_semantic"
    
    # Initialize CSV file with enhanced headers
    csv_filename = "Novartis_extractionQ1v3.csv"
    csv_headers = [
        "User Query", "Keywords", "Extracted Sentences", "Chunk Context", "Page", "Heading",
        "Document", "BM25 Score", "TF-IDF Score", "Keyword Score", "Combined Lexical Score",
        "Semantic Score", "Final Score", "Context Score", "Heading Relevance",
        "Hierarchy Bonus", "Heading Boost", "Enhanced Keyword Score", "Enhanced Score", 
        "Sentence Similarity", "Sentence Keyword Score", "Ultimate Score"
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
            # Run enhanced hybrid search with keywords
            results = optimized_hybrid_search_with_keywords(query, query_keywords, collection_name)
            
            if results:
                print(f"[SUCCESS] Found {len(results)} results for query: {query}")
                
                # Write results to CSV
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
                           # round(result["keyword_score"], 4),
                            round(result["combined_lexical_score"], 4),
                            round(result["semantic_score"], 4),
                            round(result["final_score"], 4),
                            round(result["context_score"], 4),
                            round(result["heading_relevance"], 4),
                            round(result["hierarchy_bonus"], 4),
                            round(result["heading_boost"], 4),
                           # round(result["enhanced_keyword_score"], 4),
                            round(result["enhanced_score"], 4),
                            round(result["sentence_similarity"], 4),
                           # round(result["sentence_keyword_score"], 4),
                            round(result["ultimate_score"], 4)
                        ])
                
                # Print top 3 results for verification
                print(f"\n[TOP 3 RESULTS FOR]: {query}")
                print(f"[KEYWORDS]: {query_keywords}")
                for i, result in enumerate(results[:3]):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Page: {result['page']}")
                    print(f"Heading: {result['heading']}")
                    print(f"Ultimate Score: {result['ultimate_score']:.4f}")
                    print(f"Extracted: {result['extracted_sentences'][:200]}...")
            
            else:
                print(f"[WARNING] No results found for query: {query}")
                # Write empty row to maintain structure
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([query, "; ".join(query_keywords), "No results found"] + [""] * 19)
        
        except Exception as e:
            print(f"[ERROR] Failed to process query '{query}': {e}")
            # Write error row
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query, "; ".join(query_keywords), f"Error: {str(e)}"] + [""] * 19)
        
        # Small delay between queries to avoid overwhelming the API
        time.sleep(1)
    
    print(f"\n[COMPLETION] Processing completed! Results saved to: {csv_filename}")
    print(f"[SUMMARY] Processed {len(queries)} queries with keyword-enhanced hybrid search")
    print("[INFO] CSV file contains detailed scoring metrics for analysis")

