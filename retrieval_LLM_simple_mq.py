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
CHUNKS_PER_METHOD = 80   # Retrieve 80 chunks from each method per subquery
NUM_SUBQUERIES = 7       # Generate 7 diverse subqueries
TOP_CHUNKS_AFTER_FUSION = 100  # Select top 100 chunks after fusion
TOP_CHUNKS_MAIN_MATCH = 50     # Top 50 chunks after main query matching
TOP_CHUNKS_FINAL = 30    # Final top 30 chunks with extracted sentences
DB_PATH = "chromadb"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'

# -------------------------
# Initialize Components
# -------------------------
print("[INFO] Initializing components...")
client = OpenAI(api_key="")  # Replace with your OpenAI API key
client_chroma = chromadb.PersistentClient(path=DB_PATH)
nlp = spacy.load("en_core_web_md")
model = SentenceTransformer(MODEL_NAME)

# -------------------------
# Enhanced Subquery Generation
# -------------------------
def generate_diverse_subqueries(main_query, num_subqueries=NUM_SUBQUERIES):
    """Generate diverse subqueries using different perspectives and decomposition strategies."""
    system_prompt = f"""You are an expert query decomposition specialist. Generate exactly {num_subqueries} diverse subqueries that explore different aspects, perspectives, and semantic angles of the main query.

Use these strategies to create diverse subqueries:
1. **Definitional**: What is/are the core concepts?
2. **Causal**: What causes/leads to/results from this?
3. **Comparative**: How does this compare to alternatives?
4. **Procedural**: How is this done/implemented/executed?
5. **Contextual**: When/where/why does this apply?
6. **Problem-solving**: What challenges/solutions are involved?
7. **Analytical**: What are the components/factors/implications?

Instructions:
- Generate exactly {num_subqueries} subqueries
- Each should explore a different semantic angle
- Use varied vocabulary while maintaining relevance
- Make them specific enough for targeted retrieval
- Avoid redundancy between subqueries
- Return only the subqueries, one per line, no numbering"""

    user_prompt = f"Main Query: {main_query}\n\nGenerate {num_subqueries} diverse subqueries using different perspectives:"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using better model for subquery generation
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # Higher temperature for more diversity
            max_tokens=400
        )
        
        subqueries_text = response.choices[0].message.content.strip()
        subqueries = [sq.strip() for sq in subqueries_text.split('\n') if sq.strip()]
        
        # Clean up any numbering that might have been added
        cleaned_subqueries = []
        for sq in subqueries:
            # Remove leading numbers, dots, dashes
            cleaned = re.sub(r'^\d+[\.\-\)\s]*', '', sq).strip()
            if cleaned and len(cleaned) > 10:  # Ensure meaningful length
                cleaned_subqueries.append(cleaned)
        
        # Ensure we have the right number of subqueries
        if len(cleaned_subqueries) >= num_subqueries:
            return cleaned_subqueries[:num_subqueries]
        else:
            # If we don't get enough, generate variations of the original
            while len(cleaned_subqueries) < num_subqueries:
                # Add semantic variations
                variations = [
                    f"What are the key aspects of {main_query}?",
                    f"How to understand {main_query}?",
                    f"What factors relate to {main_query}?",
                    f"What is the context of {main_query}?",
                    f"What are the implications of {main_query}?"
                ]
                for var in variations:
                    if len(cleaned_subqueries) < num_subqueries:
                        cleaned_subqueries.append(var)
            return cleaned_subqueries[:num_subqueries]
        
    except Exception as e:
        print(f"[ERROR] Subquery generation failed: {e}")
        # Enhanced fallback with semantic variations
        fallback_subqueries = [
            f"What is {main_query}?",
            f"How does {main_query} work?",
            f"What are the benefits of {main_query}?",
            f"What are the challenges with {main_query}?",
            f"What factors affect {main_query}?",
            f"What is the process of {main_query}?",
            f"What are the implications of {main_query}?"
        ]
        return fallback_subqueries[:num_subqueries]

# -------------------------
# Enhanced Semantic Similarity with Main Query
# -------------------------
def calculate_main_query_similarity(main_query, chunk_text):
    """Calculate semantic similarity between main query and chunk text."""
    if not chunk_text:
        return 0.0
    
    try:
        # Encode both texts
        main_query_embedding = model.encode(main_query, convert_to_tensor=True)
        chunk_embedding = model.encode(chunk_text, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(main_query_embedding, chunk_embedding)
        return float(similarity.cpu().numpy()[0][0])
        
    except Exception as e:
        print(f"[ERROR] Main query similarity calculation failed: {e}")
        return 0.0

# -------------------------
# Enhanced OpenAI Sentence Extraction
# -------------------------
def extract_relevant_sentences_with_openai(query, context_text, max_retries=2):
    """Extract all relevant sentences from context using OpenAI with retry logic."""
    system_prompt = """You are an expert text analyzer. Extract ALL sentences from the context that are directly relevant to answering the user's query.

CRITICAL INSTRUCTIONS:
1. Return ONLY exact sentences from the context (verbatim, word-for-word)
2. If multiple sentences are relevant, separate them with " | "
3. If no sentences are directly relevant, return exactly "NONE"
4. Do not modify, paraphrase, summarize, or add any words
5. Focus on sentences that directly answer or relate to the query"""

    user_prompt = f"Query: {query}\n\nContext: {context_text}\n\nExtract all relevant sentences (exact text only):"

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Better model for extraction
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1200
            )
            
            extracted = response.choices[0].message.content.strip()
            
            # Enhanced validation
            if extracted == "NONE" or not extracted:
                return ""
            
            # Check if the extracted text seems to be actual sentences from context
            if len(extracted) < 20:  # Too short to be meaningful
                if attempt < max_retries:
                    continue
                return ""
            
            return extracted
            
        except Exception as e:
            print(f"[ERROR] OpenAI extraction failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            return ""
    
    return ""

# -------------------------
# Query Processing
# -------------------------
def preprocess_query(query):
    """Enhanced query processing to extract key terms and entities."""
    doc = nlp(query)
    
    # Extract different types of important terms
    key_terms = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and not token.is_stop]
    entities = [ent.text for ent in doc.ents]
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
    
    return {
        "original": query,
        "key_terms": key_terms,
        "entities": entities,
        "lemmas": lemmas
    }

# -------------------------
# BM25 Retrieval
# -------------------------
def retrieve_bm25_chunks(query, collection_name):
    """Retrieve top chunks using BM25."""
    print(f"[INFO] Retrieving {CHUNKS_PER_METHOD} BM25 chunks for: {query[:50]}...")
    
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
    print(f"[INFO] Retrieving {CHUNKS_PER_METHOD} TF-IDF chunks for: {query[:50]}...")
    
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
    print(f"[INFO] Retrieving {CHUNKS_PER_METHOD} dense embedding chunks for: {query[:50]}...")
    
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
# Enhanced Context-Aware Reranking
# -------------------------
def context_aware_reranking(chunks, query):
    """Apply enhanced context-aware reranking to chunks."""
    print(f"[INFO] Applying context-aware reranking for query: {query[:50]}...")
    
    query_processed = preprocess_query(query)
    query_tokens = set([term.lower() for term in query_processed["key_terms"]])
    query_entities = set([ent.lower() for ent in query_processed["entities"]])
    query_lemmas = set(query_processed["lemmas"])
    
    for chunk in chunks:
        context = chunk["chunk"]
        context_doc = nlp(context)
        
        # Enhanced token overlap
        context_tokens = set([token.text.lower() for token in context_doc 
                             if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and not token.is_stop])
        token_overlap = len(query_tokens.intersection(context_tokens))
        
        # Entity matching
        context_entities = set([ent.text.lower() for ent in context_doc.ents])
        entity_match = len(query_entities.intersection(context_entities))
        
        # Lemma overlap
        context_lemmas = set([token.lemma_.lower() for token in context_doc 
                             if not token.is_stop and not token.is_punct])
        lemma_overlap = len(query_lemmas.intersection(context_lemmas))
        
        # Calculate enhanced context score
        context_score = (token_overlap * 2) + (entity_match * 3) + (lemma_overlap * 1)
        chunk["context_score"] = context_score
        chunk["reranked_score"] = chunk["score"] * (1 + 0.3 * context_score)
    
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
                        "subqueries": set(),
                        "method_scores": {}
                    }
                
                # Add RRF score
                rrf_score = 1.0 / (rank + k)
                fused_scores[chunk_key]["fused_score"] += rrf_score
                fused_scores[chunk_key]["methods"].add(method_name)
                fused_scores[chunk_key]["subqueries"].add(f"SQ{subquery_idx+1}")
                
                # Track method scores for analysis
                if method_name not in fused_scores[chunk_key]["method_scores"]:
                    fused_scores[chunk_key]["method_scores"][method_name] = []
                fused_scores[chunk_key]["method_scores"][method_name].append(rrf_score)
    
    # Create final ranked list
    fused_chunks = []
    for chunk_key, data in fused_scores.items():
        chunk = data["chunk_data"].copy()
        chunk["fused_score"] = data["fused_score"]
        chunk["methods"] = ", ".join(sorted(data["methods"]))
        chunk["subqueries"] = ", ".join(sorted(data["subqueries"]))
        chunk["method_count"] = len(data["methods"])
        chunk["subquery_count"] = len(data["subqueries"])
        fused_chunks.append(chunk)
    
    # Sort by fused score
    fused_chunks.sort(key=lambda x: x["fused_score"], reverse=True)
    
    return fused_chunks

# -------------------------
# Main Query Matching and Reranking
# -------------------------
def rerank_by_main_query_similarity(chunks, main_query, top_k=TOP_CHUNKS_MAIN_MATCH):
    """Rerank chunks based on semantic similarity with main query."""
    print(f"[INFO] Reranking top {len(chunks)} chunks by main query similarity...")
    
    # Calculate similarity with main query for each chunk
    for chunk in chunks:
        main_similarity = calculate_main_query_similarity(main_query, chunk["chunk"])
        chunk["main_query_similarity"] = main_similarity
        # Combine fused score with main query similarity
        chunk["combined_score"] = chunk["fused_score"] * (1 + main_similarity)
    
    # Sort by combined score
    chunks.sort(key=lambda x: x["combined_score"], reverse=True)
    
    print(f"[INFO] Selected top {min(top_k, len(chunks))} chunks based on main query similarity")
    return chunks[:top_k]

# -------------------------
# Enhanced Main Hybrid Search Function
# -------------------------
def enhanced_hybrid_search_with_subqueries(main_query, collection_name):
    """
    Enhanced hybrid search function with improved subquery generation and processing:
    1. Generate diverse subqueries from the main query
    2. Retrieve chunks for each subquery using all 3 methods
    3. Apply fusion across all results
    4. Rerank by main query similarity
    5. Extract sentences with fallback mechanism
    6. Return top 30 chunks with relevant sentences
    """
    print(f"\n[INFO] Starting enhanced hybrid search for main query: {main_query}")
    
    # STEP 1: Generate diverse subqueries
    print("\n=== STEP 1: GENERATING DIVERSE SUBQUERIES ===")
    subqueries = generate_diverse_subqueries(main_query, NUM_SUBQUERIES)
    print(f"[INFO] Generated {len(subqueries)} diverse subqueries:")
    for i, sq in enumerate(subqueries, 1):
        print(f"  {i}. {sq}")
    
    # STEP 2: Retrieve chunks for each subquery (NOT main query)
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
    
    # STEP 4: Rerank by main query similarity
    print(f"\n=== STEP 4: RERANKING BY MAIN QUERY SIMILARITY ===")
    top_chunks = fused_chunks[:TOP_CHUNKS_AFTER_FUSION]
    reranked_chunks = rerank_by_main_query_similarity(top_chunks, main_query, TOP_CHUNKS_MAIN_MATCH)
    
    # STEP 5: Extract sentences with fallback mechanism
    print(f"\n=== STEP 5: EXTRACTING SENTENCES WITH FALLBACK ===")
    final_results = []
    processed_chunks = 0
    skipped_chunks = 0
    
    for chunk in reranked_chunks:
        processed_chunks += 1
        
        # Extract sentences using OpenAI
        extracted_sentences = extract_relevant_sentences_with_openai(main_query, chunk["chunk"])
        
        if extracted_sentences:  # If we found relevant sentences
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
                "main_query_similarity": chunk["main_query_similarity"],
                "combined_score": chunk["combined_score"],
                "original_score": chunk.get("score", 0),
                "context_score": chunk.get("context_score", 0),
                "method_count": chunk.get("method_count", 1),
                "subquery_count": chunk.get("subquery_count", 1)
            }
            final_results.append(result)
            print(f"[INFO] ✓ Chunk accepted ({len(final_results)}/{TOP_CHUNKS_FINAL}) - Main similarity: {chunk['main_query_similarity']:.3f}")
            
            if len(final_results) >= TOP_CHUNKS_FINAL:
                break
        else:
            skipped_chunks += 1
            print(f"[INFO] ✗ Chunk skipped ({skipped_chunks}) - No relevant sentences found")
    
    # Final sort by main query similarity
    final_results.sort(key=lambda x: x["main_query_similarity"], reverse=True)
    
    print(f"\n[INFO] Completed processing. Final results: {len(final_results)} chunks")
    print(f"[INFO] Processed {processed_chunks} chunks, skipped {skipped_chunks} chunks")
    print(f"[INFO] Success rate: {len(final_results)}/{processed_chunks} ({len(final_results)/processed_chunks*100:.1f}%)")
    
    return final_results

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Test OpenAI connection
    try:
        test_response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    csv_filename = "extracted_contexts_enhanced_subquery_v2.csv"
    csv_headers = [
        "User Query", "Subqueries", "Extracted Sentences", "Chunk Context", "Page", 
        "Document", "Methods", "Subqueries Used", "Fused Score", "Main Query Similarity", 
        "Combined Score", "Original Score", "Context Score", "Method Count", "Subquery Count"
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
        results = enhanced_hybrid_search_with_subqueries(query, collection_name)
        
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
                    result["main_query_similarity"],
                    result["combined_score"],
                    result["original_score"],
                    result["context_score"],
                    result["method_count"],
                    result["subquery_count"]
                ])
        
        print(f"[INFO] Saved {len(results)} results for query: {query}")
    
    print(f"\n[INFO] Processing complete!")
    print(f"[INFO] Results saved to: {csv_filename}")
    print(f"[INFO] Enhanced pipeline with improved subquery generation and fallback mechanism completed successfully")