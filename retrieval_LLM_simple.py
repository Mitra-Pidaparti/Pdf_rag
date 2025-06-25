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

# -------------------------
# Configuration
# -------------------------
CHUNKS_PER_METHOD = 100  # Retrieve 100 chunks from each method (300 total)
TOP_CHUNKS_AFTER_FUSION = 60  # Select top 60 chunks after fusion
TOP_CHUNKS_FINAL = 30    # Final top 30 chunks after sentence similarity reranking
DB_PATH = "chromadb"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'

# -------------------------
# Initialize Components
# -------------------------
print("[INFO] Initializing components...")
client = OpenAI(api_key="")
model = SentenceTransformer(MODEL_NAME)
client_chroma = chromadb.PersistentClient(path=DB_PATH)
nlp = spacy.load("en_core_web_md")

# -------------------------
# OpenAI Sentence Extraction
# -------------------------
def extract_relevant_sentences_with_openai(query, context_text):
    """Extract all relevant sentences from context using OpenAI."""
    system_prompt = """You are an expert text analyzer. Extract ALL sentences from the context that are relevant to answering the user's query.

Instructions:
1. Return ONLY exact sentences from the context (verbatim)
2. If multiple sentences are relevant, separate them with " | " 
3. If no sentences are relevant, return "NONE"
4. Do not modify, paraphrase, or summarize sentences"""

    user_prompt = f"Query: {query}\n\nContext: {context_text}\n\nExtract relevant sentences:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
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
# BM25 Retrieval
# -------------------------
def retrieve_bm25_chunks(query, collection_name):
    """Retrieve top chunks using BM25."""
    print(f"[INFO] Retrieving {CHUNKS_PER_METHOD} BM25 chunks...")
    
    collection = client_chroma.get_collection(collection_name)
    data = collection.get()
    
    # Prepare data
    chunks = [meta.get("chunk", "") for meta in data["metadatas"]]
    pages = [meta.get("page", "Unknown") for meta in data["metadatas"]]
    doc_ids = data.get("ids", ["unknown"] * len(chunks))
    
    # Tokenize for BM25
    tokenized = []
    for chunk in chunks:
        doc = nlp(chunk)
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and len(token.text) > 1]
        tokenized.append(tokens)
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized)
    
    # Process query
    query_doc = nlp(query)
    query_terms = [token.lemma_.lower() for token in query_doc 
                   if not token.is_stop and not token.is_punct and len(token.text) > 1]
    
    # Get scores and top results
    scores = bm25.get_scores(query_terms)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CHUNKS_PER_METHOD]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "page": pages[idx],
            "document": doc_ids[idx],
            "score": scores[idx],
            "method": "BM25"
        })
    
    return results

# -------------------------
# TF-IDF Retrieval
# -------------------------
def retrieve_tfidf_chunks(query, collection_name):
    """Retrieve top chunks using TF-IDF."""
    print(f"[INFO] Retrieving {CHUNKS_PER_METHOD} TF-IDF chunks...")
    
    collection = client_chroma.get_collection(collection_name)
    data = collection.get()
    
    # Prepare data
    chunks = [meta.get("chunk", "") for meta in data["metadatas"]]
    pages = [meta.get("page", "Unknown") for meta in data["metadatas"]]
    doc_ids = data.get("ids", ["unknown"] * len(chunks))
    
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
    
    # Process query
    query_doc = nlp(query)
    query_terms = [token.lemma_.lower() for token in query_doc 
                   if not token.is_stop and not token.is_punct and len(token.text) > 1]
    query_text = " ".join(query_terms)
    
    # Get scores
    query_vector = vectorizer.transform([query_text])
    scores = (query_vector * tfidf_matrix.T).toarray().flatten()
    top_indices = scores.argsort()[-CHUNKS_PER_METHOD:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "page": pages[idx],
            "document": doc_ids[idx],
            "score": scores[idx],
            "method": "TF-IDF"
        })
    
    return results

# -------------------------
# Dense Embedding Retrieval
# -------------------------
def retrieve_dense_chunks(query, collection_name):
    """Retrieve top chunks using dense embeddings."""
    print(f"[INFO] Retrieving {CHUNKS_PER_METHOD} dense embedding chunks...")
    
    collection = client_chroma.get_collection(collection_name)
    
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0).tolist()
    
    # Query collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=CHUNKS_PER_METHOD
    )
    
    if not results["distances"] or not results["metadatas"]:
        return []
    
    # Process results
    chunk_results = []
    for i, meta in enumerate(results["metadatas"][0]):
        # Convert distance to similarity score
        distance = results["distances"][0][i]
        similarity = 1 - distance  # Assuming cosine distance
        
        chunk_results.append({
            "chunk": meta["chunk"],
            "page": meta["page"],
            "document": meta["document"],
            "score": similarity,
            "method": "Dense"
        })
    
    return chunk_results

# -------------------------
# Context-Aware Reranking
# -------------------------
def context_aware_reranking(chunks, query):
    """Apply context-aware reranking to chunks."""
    print("[INFO] Applying context-aware reranking...")
    
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
        
        # Calculate context score
        context_score = token_overlap + entity_match * 2
        chunk["context_score"] = context_score
        chunk["reranked_score"] = chunk["score"] * (1 + 0.2 * context_score)
    
    return chunks

# -------------------------
# Reciprocal Rank Fusion
# -------------------------
def reciprocal_rank_fusion(bm25_chunks, tfidf_chunks, dense_chunks, k=60):
    """Combine results using Reciprocal Rank Fusion."""
    print("[INFO] Applying reciprocal rank fusion...")
    
    fused_scores = {}
    
    # Process each method's results
    for method_name, chunks in [("BM25", bm25_chunks), ("TF-IDF", tfidf_chunks), ("Dense", dense_chunks)]:
        for rank, chunk in enumerate(chunks):
            # Create unique key for chunk
            chunk_key = f"{chunk['document']}_p{chunk['page']}_{hash(chunk['chunk'][:100])}"
            
            if chunk_key not in fused_scores:
                fused_scores[chunk_key] = {
                    "chunk_data": chunk,
                    "fused_score": 0,
                    "methods": []
                }
            
            # Add RRF score
            fused_scores[chunk_key]["fused_score"] += 1.0 / (rank + k)
            fused_scores[chunk_key]["methods"].append(method_name)
    
    # Create final ranked list
    fused_chunks = []
    for chunk_key, data in fused_scores.items():
        chunk = data["chunk_data"].copy()
        chunk["fused_score"] = data["fused_score"]
        chunk["methods"] = ", ".join(data["methods"])
        fused_chunks.append(chunk)
    
    # Sort by fused score
    fused_chunks.sort(key=lambda x: x["fused_score"], reverse=True)
    
    return fused_chunks

# -------------------------
# Main Hybrid Search Function
# -------------------------
def hybrid_search_with_sentence_semantic_reranking(query, collection_name):
    """
    Enhanced hybrid search function:
    1. Retrieve 300 chunks (100 per method)
    2. Apply initial reranking and fusion
    3. Select top 60 chunks
    4. Extract sentence groups from top 60 chunks using OpenAI
    5. Rerank based on semantic similarity between query and sentence groups
    6. Select final top 30 chunks
    """
    print(f"\n[INFO] Starting hybrid search for query: {query}")
    print(f"[INFO] Pipeline: {CHUNKS_PER_METHOD * 3} total → {TOP_CHUNKS_AFTER_FUSION} fusion → {TOP_CHUNKS_FINAL} final")
    
    # Process query
    query_rep = preprocess_query(query)
    
    # STEP 1: Retrieve chunks from each method (300 total)
    print("\n=== STEP 1: RETRIEVING CHUNKS (300 TOTAL) ===")
    bm25_chunks = retrieve_bm25_chunks(query, collection_name)
    tfidf_chunks = retrieve_tfidf_chunks(query, collection_name)
    dense_chunks = retrieve_dense_chunks(query, collection_name)
    
    print(f"[INFO] Retrieved {len(bm25_chunks)} BM25, {len(tfidf_chunks)} TF-IDF, {len(dense_chunks)} dense chunks")
    
    # STEP 2: Apply context-aware reranking to all chunks
    print("\n=== STEP 2: INITIAL RERANKING ===")
    bm25_chunks = context_aware_reranking(bm25_chunks, query)
    tfidf_chunks = context_aware_reranking(tfidf_chunks, query)
    dense_chunks = context_aware_reranking(dense_chunks, query)
    
    # STEP 3: Apply reciprocal rank fusion
    print("\n=== STEP 3: FUSING RANKINGS ===")
    fused_chunks = reciprocal_rank_fusion(bm25_chunks, tfidf_chunks, dense_chunks)
    print(f"[INFO] Fused to {len(fused_chunks)} unique chunks")
    
    # STEP 4: Select top 60 chunks
    print(f"\n=== STEP 4: SELECTING TOP {TOP_CHUNKS_AFTER_FUSION} CHUNKS ===")
    top_60_chunks = fused_chunks[:TOP_CHUNKS_AFTER_FUSION]
    print(f"[INFO] Selected top {len(top_60_chunks)} chunks for sentence extraction")
    
    # STEP 5: Extract sentence groups from top 60 chunks
    print(f"\n=== STEP 5: EXTRACTING SENTENCE GROUPS FROM TOP {TOP_CHUNKS_AFTER_FUSION} ===")
    chunks_with_sentences = []
    
    for i, chunk in enumerate(top_60_chunks):
        print(f"[INFO] Extracting sentences from chunk {i+1}/{len(top_60_chunks)}")
        
        # Extract relevant sentences using OpenAI
        extracted_sentences = extract_relevant_sentences_with_openai(query, chunk["chunk"])
        
        # Calculate semantic similarity between query and sentence group
        semantic_similarity = calculate_semantic_similarity(query, extracted_sentences)
        
        # Create enhanced chunk data
        enhanced_chunk = chunk.copy()
        enhanced_chunk["extracted_sentences"] = extracted_sentences if extracted_sentences else "No relevant sentences found"
        enhanced_chunk["semantic_similarity"] = semantic_similarity
        enhanced_chunk["final_score"] = chunk["fused_score"] * (1 + semantic_similarity)  # Combine scores
        
        chunks_with_sentences.append(enhanced_chunk)
    
    # STEP 6: Rerank based on semantic similarity and select top 30
    print(f"\n=== STEP 6: SEMANTIC RERANKING AND SELECTING TOP {TOP_CHUNKS_FINAL} ===")
    chunks_with_sentences.sort(key=lambda x: x["semantic_similarity"], reverse=True)
    top_30_chunks = chunks_with_sentences[:TOP_CHUNKS_FINAL]
    
    print(f"[INFO] Selected final top {len(top_30_chunks)} chunks based on sentence semantic similarity")
    
    # STEP 7: Prepare final results
    final_results = []
    for chunk in top_30_chunks:
        result = {
            "query": query,
            "extracted_sentences": chunk["extracted_sentences"],
            "chunk_context": chunk["chunk"],
            "page": chunk["page"],
            "document": chunk["document"],
            "methods": chunk.get("methods", chunk["method"]),
            "fused_score": chunk["fused_score"],
            "semantic_similarity": chunk["semantic_similarity"],
            "final_score": chunk["final_score"],
            "original_score": chunk["score"],
            "context_score": chunk.get("context_score", 0)
        }
        final_results.append(result)
    
    print(f"[INFO] Completed processing for {len(final_results)} final chunks")
    return final_results

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Test OpenAI connection
    try:
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1
        )
        print("[INFO] OpenAI API connection successful")
    except Exception as e:
        print(f"[ERROR] OpenAI API connection failed: {e}")
        exit(1)
    
    # Load queries
    queries = []
    with open('questions.txt', 'r', encoding='utf-8') as file:
        for line in file:
            cleaned = line.strip()
            if cleaned:
                queries.append(cleaned)
    
    print(f"[INFO] Loaded {len(queries)} queries")
    
    # Process each query
    collection_name = "ril_pdf_pages_semantic"
    
    # Initialize CSV file
    csv_filename = "extracted_contexts_semantic_sentence_reranked.csv"
    csv_headers = [
        "User Query", "Extracted Sentences", "Chunk Context", "Page", 
        "Document", "Methods", "Fused Score", "Semantic Similarity", "Final Score", 
        "Original Score", "Context Score"
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
        
        # Run hybrid search with semantic sentence reranking
        results = hybrid_search_with_sentence_semantic_reranking(query, collection_name)
        
        # Save results to CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            for result in results:
                writer.writerow([
                    result["query"],
                    result["extracted_sentences"],
                    result["chunk_context"],
                    result["page"],
                    result["document"],
                    result["methods"],
                    result["fused_score"],
                    result["semantic_similarity"],
                    result["final_score"],
                    result["original_score"],
                    result["context_score"]
                ])
        
        print(f"[INFO] Saved {len(results)} results for query: {query}")
    
    print(f"\n[INFO] Processing complete!")
    print(f"[INFO] Results saved to: {csv_filename}")
    print(f"[INFO] Total rows: {len(queries) * TOP_CHUNKS_FINAL} ({len(queries)} queries × {TOP_CHUNKS_FINAL} chunks each)")