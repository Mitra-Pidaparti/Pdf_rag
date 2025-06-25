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
CHUNKS_PER_METHOD = 100  # Retrieve 100 chunks from each method (300 total per subquery)
TOP_CHUNKS_AFTER_FUSION = 70  # Select top 70 chunks after fusion
TOP_CHUNKS_FINAL = 30    # Final top 30 chunks after sentence similarity reranking
DB_PATH = "chromadb"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'

# -------------------------
# Initialize Components
# -------------------------
print("[INFO] Initializing components...")
client = OpenAI(api_key="")  # Replace with your OpenAI API key
client_chroma = chromadb.PersistentClient(path=DB_PATH)
nlp = spacy.load("en_core_web_md")

# -------------------------
# Subquery Generation
# -------------------------
def generate_subqueries(main_query):
    """Generate 3 subqueries that capture different meanings and angles of the main query."""
    system_prompt = """You are an expert query analyst. Given a main query, generate exactly 3 subqueries that capture slightly different perspectives or aspects of the original query, but remain closely related in wording and terminology.

Instructions:
1. Generate exactly 3 subqueries.
2. Each subquery should focus on a different facet or angle of the main query.
3. Avoid introducing unrelated topics too far from the main query's meaning.
4. The subqueries should be concise, clear, and to maximize retrieval effectiveness with hybrid retrieval system.
5. Return only the 3 subqueries, one per line, without numbering or extra formatting."""

    user_prompt = f"Main Query: {main_query}\n\nGenerate 3 diverse subqueries:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        subqueries_text = response.choices[0].message.content.strip()
        subqueries = [sq.strip() for sq in subqueries_text.split('\n') if sq.strip()]
        
        # Ensure we have exactly 3 subqueries
        if len(subqueries) >= 3:
            return subqueries[:3]
        else:
            # If we don't get 3, pad with variations of the original
            while len(subqueries) < 3:
                subqueries.append(main_query)
            return subqueries[:3]
        
    except Exception as e:
        print(f"[ERROR] Subquery generation failed: {e}")
        # Fallback: return the original query 3 times
        return [main_query, main_query, main_query]

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
def reciprocal_rank_fusion(all_chunks_by_method, k=60):
    """Combine results using Reciprocal Rank Fusion across all subqueries and methods."""
    print("[INFO] Applying reciprocal rank fusion across all subqueries and methods...")
    
    fused_scores = {}
    
    # Process each subquery's results
    for subquery_idx, methods_results in enumerate(all_chunks_by_method):
        for method_name, chunks in methods_results.items():
            for rank, chunk in enumerate(chunks):
                # Create unique key for chunk
                chunk_key = f"{chunk['document']}_p{chunk['page']}_{hash(chunk['chunk'][:100])}"
                
                if chunk_key not in fused_scores:
                    fused_scores[chunk_key] = {
                        "chunk_data": chunk,
                        "fused_score": 0,
                        "methods": set(),
                        "subqueries": set()
                    }
                
                # Add RRF score
                fused_scores[chunk_key]["fused_score"] += 1.0 / (rank + k)
                fused_scores[chunk_key]["methods"].add(method_name)
                fused_scores[chunk_key]["subqueries"].add(f"SQ{subquery_idx+1}")
    
    # Create final ranked list
    fused_chunks = []
    for chunk_key, data in fused_scores.items():
        chunk = data["chunk_data"].copy()
        chunk["fused_score"] = data["fused_score"]
        chunk["methods"] = ", ".join(sorted(data["methods"]))
        chunk["subqueries"] = ", ".join(sorted(data["subqueries"]))
        fused_chunks.append(chunk)
    
    # Sort by fused score
    fused_chunks.sort(key=lambda x: x["fused_score"], reverse=True)
    
    return fused_chunks

# -------------------------
# Main Hybrid Search Function
# -------------------------
def hybrid_search_with_subqueries_and_fallback(main_query, collection_name):
    """
    Enhanced hybrid search function with subqueries and fallback:
    1. Generate 3 subqueries from the main query
    2. Retrieve chunks for each subquery using all 3 methods
    3. Apply fusion across all results
    4. Extract sentences with fallback for empty results
    5. Return top 30 chunks with relevant sentences
    """
    print(f"\n[INFO] Starting hybrid search for main query: {main_query}")
    
    # STEP 1: Generate subqueries
    print("\n=== STEP 1: GENERATING SUBQUERIES ===")
    subqueries = generate_subqueries(main_query)
    print(f"[INFO] Generated subqueries:")
    for i, sq in enumerate(subqueries, 1):
        print(f"  {i}. {sq}")
    
    # STEP 2: Retrieve chunks for each subquery
    print(f"\n=== STEP 2: RETRIEVING CHUNKS FOR ALL SUBQUERIES ===")
    all_chunks_by_method = []
    
    for sq_idx, subquery in enumerate(subqueries):
        print(f"\n[INFO] Processing subquery {sq_idx+1}: {subquery}")
        
        # Retrieve using all methods for this subquery
        bm25_chunks = retrieve_bm25_chunks(subquery, collection_name)
        tfidf_chunks = retrieve_tfidf_chunks(subquery, collection_name)
        dense_chunks = retrieve_dense_chunks(subquery, collection_name)
        
        # Apply context-aware reranking
        bm25_chunks = context_aware_reranking(bm25_chunks, subquery)
        tfidf_chunks = context_aware_reranking(tfidf_chunks, subquery)
        dense_chunks = context_aware_reranking(dense_chunks, subquery)
        
        # Store results for this subquery
        subquery_results = {
            "BM25": bm25_chunks,
            "TF-IDF": tfidf_chunks,
            "Dense": dense_chunks
        }
        all_chunks_by_method.append(subquery_results)
        
        print(f"[INFO] Subquery {sq_idx+1} retrieved: {len(bm25_chunks)} BM25, {len(tfidf_chunks)} TF-IDF, {len(dense_chunks)} Dense")
    
    # STEP 3: Apply reciprocal rank fusion across all results
    print(f"\n=== STEP 3: FUSING ALL RESULTS ===")
    fused_chunks = reciprocal_rank_fusion(all_chunks_by_method)
    print(f"[INFO] Fused to {len(fused_chunks)} unique chunks")

    # STEP 4: Extract sentences and rerank
    print(f"\n=== STEP 4: EXTRACTING SENTENCES AND RERANKING ===")
    final_results = []
    chunk_index = 0

    for chunk in fused_chunks[:TOP_CHUNKS_AFTER_FUSION]:
        chunk_index += 1
        extracted_sentences = extract_relevant_sentences_with_openai(main_query, chunk["chunk"])
        if extracted_sentences:
            semantic_similarity = calculate_semantic_similarity(main_query, extracted_sentences)
            result = {
                "query": main_query,
                "subqueries": " | ".join(subqueries),
                "extracted_sentences": extracted_sentences,
                "chunk_context": chunk["chunk"],
                "page": chunk["page"],
                "document": chunk["document"],
                "methods": chunk.get("methods", chunk.get("method", "")),
                "subqueries_used": chunk.get("subqueries", ""),
                "fused_score": chunk["fused_score"],
                "semantic_similarity": semantic_similarity,
                "final_score": chunk["fused_score"] * (1 + semantic_similarity),
                "original_score": chunk.get("score", 0),
                "context_score": chunk.get("context_score", 0)
            }
            final_results.append(result)
            print(f"[INFO] ✓ Chunk accepted ({len(final_results)}/{TOP_CHUNKS_FINAL}) - Semantic similarity: {semantic_similarity:.3f}")
            if len(final_results) >= TOP_CHUNKS_FINAL:
                break
        else:
            print(f"[INFO] ✗ Chunk skipped - No relevant sentences found")

    # Sort final results by semantic similarity
    final_results.sort(key=lambda x: x["semantic_similarity"], reverse=True)

    print(f"\n[INFO] Completed processing. Final results: {len(final_results)} chunks")
    print(f"[INFO] Processed {chunk_index} chunks to get {len(final_results)} valid results")

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
    csv_filename = "extracted_contexts_LLM_simple_mq.csv"
    csv_headers = [
        "User Query", "Subqueries", "Extracted Sentences", "Chunk Context", "Page", 
        "Document", "Methods", "Subqueries Used", "Fused Score", "Semantic Similarity", 
        "Final Score", "Original Score", "Context Score"
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
        
        # Run enhanced hybrid search
        results = hybrid_search_with_subqueries_and_fallback(query, collection_name)
        
        # Save results to CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            for result in results:
                writer.writerow([
                    result["query"],
                    result["subqueries"],
                    result["extracted_sentences"],
                    result["chunk_context"],
                    result["page"],
                    result["document"],
                    result["methods"],
                    result["subqueries_used"],
                    result["fused_score"],
                    result["semantic_similarity"],
                    result["final_score"],
                    result["original_score"],
                    result["context_score"]
                ])
        
        print(f"[INFO] Saved {len(results)} results for query: {query}")
    
    print(f"\n[INFO] Processing complete!")
    print(f"[INFO] Results saved to: {csv_filename}")
    print(f"[INFO] Enhanced pipeline with subqueries and fallback mechanism completed successfully")