##LARGER MODELS 

import re ,os ,csv
from sentence_transformers import SentenceTransformer, util
import torch
import chromadb
import json
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np
import requests
from transformers import pipeline
import openai
from openai import OpenAI

# -------------------------
# Initialize OpenAI Client
# -------------------------
# Initialize OpenAI client - make sure to set your API key
# Option 1: Set environment variable OPENAI_API_KEY
# Option 2: Pass key directly (not recommended for production)
client = OpenAI(
  api_key= ""  )# Replace with your OpenAI API key or set as environment variable)


# -------------------------
# Initialize Chroma Client & Model
# -------------------------
DB_PATH = "chromadb"
#MODEL_NAME = "all-mpnet-base-v2"  # Upgraded model from MiniLM
MODEL_NAME='BAAI/bge-base-en-v1.5'

print("[INFO] Loading Sentence Transformer model...")
model = SentenceTransformer(MODEL_NAME)

print("[INFO] Connecting to ChromaDB...")
client_chroma = chromadb.PersistentClient(path=DB_PATH)
collections = client_chroma.list_collections()

print("[INFO] Loading spaCy model for NLP tasks...")
nlp = spacy.load("en_core_web_md")

# -------------------------
# OpenAI Sentence Extraction
# -------------------------
def extract_relevant_sentences_with_openai(query, context_text, model_name="gpt-3.5-turbo"):
    """
    Use OpenAI API to extract all relevant sentences from context that answer the query.
    Returns a list of verbatim sentences from the context.
    """
    system_prompt = """You are an expert text analyzer. Your task is to identify and extract ALL sentences from the given context that are relevant to answering the user's query.

Instructions:
1. Read the user's query carefully
2. Examine the provided context text
3. Extract ALL sentences that contain information relevant to answering the query
4. Return ONLY the exact sentences as they appear in the context (verbatim)
5. If multiple sentences are relevant, separate them with " | " (pipe separator)
6. If no sentences are relevant, return "NO_RELEVANT_SENTENCES"
7. Do not paraphrase, summarize, or modify the sentences in any way

Example:
Query: "What is the company's revenue?"
Context: "The company reported strong growth last year. Revenue increased to $50 million in 2023. The CEO was pleased with the results. Operating expenses also rose during this period."
Output: "Revenue increased to $50 million in 2023."
"""

    user_prompt = f"""Query: {query}

Context: {context_text}

Extract all relevant sentences:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=1000,  # Adjust based on expected response length
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        if extracted_text == "NO_RELEVANT_SENTENCES":
            return []
        
        # Split by pipe separator and clean up
        sentences = [sent.strip() for sent in extracted_text.split(" | ") if sent.strip()]
        return sentences
        
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        # Fallback to original sentence extraction if OpenAI fails
        return [context_text[:200] + "..." if len(context_text) > 200 else context_text]

# -------------------------
# Query Preprocessing
# -------------------------
def preprocess_query(query):
    """Improve query without hardcoding specific expansions."""
    # Clean query
    query = query.strip()
    
    # Parse with spaCy for NLP features
    doc = nlp(query)
    
    # Extract relevant entities and noun chunks
    entities = [ent.text for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    
    # Extract key terms (nouns, proper nouns, adjectives)
    key_terms = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop]
    
    # Create a query representation
    query_rep = {
        "original": query,
        "entities": entities,
        "noun_chunks": noun_chunks,
        "key_terms": key_terms
    }
    
    return query_rep

# -------------------------
# Helper: Retrieve Context
# -------------------------
def get_context_from_indices(matched_index, all_sentences, context_range):
    """Get surrounding context sentences based on the matched sentence."""
    start = max(0, matched_index - context_range)
    end = min(len(all_sentences), matched_index + context_range + 1)
    return " ".join(all_sentences[start:end])

# -------------------------
# Sentence Splitter with Improved Context Handling
# -------------------------
def split_text_into_sentences(text):
    """Use spaCy to split text into sentences with better handling."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Handle cases where spaCy might not split sentences properly
    if len(sentences) <= 1 and len(text) > 100:
        # Fallback splitting by periods with context
        rough_sentences = text.split('.')
        sentences = [s.strip() + '.' for s in rough_sentences if s.strip()]
    
    return sentences

# -------------------------
# BM25 Indexing with Improved Preprocessing
# -------------------------
def build_bm25_index(collection_name):
    collection = client_chroma.get_collection(collection_name)
    data = collection.get()
    all_sentences = [meta.get("chunk", "") for meta in data["metadatas"]]
    page_numbers = [meta.get("page", "Unknown") for meta in data["metadatas"]]
    sentence_indices = [meta.get("sentence_indices", "") for meta in data["metadatas"]]
    doc_ids = data.get("ids", ["unknown"] * len(all_sentences))

    # Improved tokenization with lemmatization
    tokenized = []
    for sentence in all_sentences:
        doc = nlp(sentence)
        # Extract meaningful tokens with lemmatization
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and len(token.text) > 1]
        tokenized.append(tokens)
    
    bm25 = BM25Okapi(tokenized)
    return bm25, all_sentences, page_numbers, sentence_indices, doc_ids, collection, tokenized

# -------------------------
# TF-IDF Indexing with Improvements
# -------------------------
def build_tfidf_index(collection_name):
    collection = client_chroma.get_collection(collection_name)
    data = collection.get()
    all_sentences = [meta.get("chunk", "") for meta in data["metadatas"]]
    page_numbers = [meta.get("page", "Unknown") for meta in data["metadatas"]]
    sentence_indices = [meta.get("sentence_indices", "") for meta in data["metadatas"]]
    doc_ids = data.get("ids", ["unknown"] * len(all_sentences))

    # Process text for TF-IDF
    processed_texts = []
    for sentence in all_sentences:
        doc = nlp(sentence)
        # Keep important terms with lemmatization
        important_terms = [token.lemma_.lower() for token in doc 
                          if not token.is_stop and not token.is_punct and len(token.text) > 1]
        processed_texts.append(" ".join(important_terms))

    # Improved TF-IDF with better parameters
    vectorizer = TfidfVectorizer(
        min_df=1,                 # Include rare terms
        max_df=0.9,               # Ignore very common terms
        sublinear_tf=True,        # Apply sublinear tf scaling
        use_idf=True,             # Enable IDF
        ngram_range=(1, 2)        # Include both unigrams and bigrams
    )
    
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    return vectorizer, tfidf_matrix, all_sentences, page_numbers, sentence_indices, doc_ids, collection, processed_texts

# -------------------------
# Maximum Marginal Relevance for Diversity
# -------------------------
def mmr(query_embedding, candidate_embeddings, candidates, lambda_param=0.5, top_k=5):
    """Apply Maximum Marginal Relevance to ensure diversity."""
    if len(candidates) <= top_k:
        return list(range(len(candidates)))
    
    device = query_embedding.device
    candidate_embeddings = candidate_embeddings.to(device)   
     
    # Calculate similarity between query and candidates
    similarities = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]
    
    # Initialize selected indices and remaining indices
    selected_indices = []
    remaining_indices = list(range(len(candidates)))
    
    # Select the first document with highest similarity
    best_idx = max(remaining_indices, key=lambda idx: similarities[idx].item())
    selected_indices.append(best_idx)
    remaining_indices.remove(best_idx)
    
    # Select the rest using MMR
    for _ in range(min(top_k - 1, len(remaining_indices))):
        # Calculate the MMR score for each remaining document
        mmr_scores = []
        
        for idx in remaining_indices:
            # Calculate similarity with query
            sim_query = similarities[idx].item()
            
            # Calculate maximum similarity with selected documents
            max_sim_selected = 0
            for sel_idx in selected_indices:
                sim = util.pytorch_cos_sim(
                    candidate_embeddings[idx].unsqueeze(0), 
                    candidate_embeddings[sel_idx].unsqueeze(0)
                )[0][0].item()
                max_sim_selected = max(max_sim_selected, sim)
            
            # Calculate MMR score
            mmr_score = lambda_param * sim_query - (1 - lambda_param) * max_sim_selected
            mmr_scores.append(mmr_score)
        
        # Select document with highest MMR score
        best_mmr_idx = mmr_scores.index(max(mmr_scores))
        best_idx = remaining_indices[best_mmr_idx]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
    return selected_indices

# -------------------------
# Hybrid Query Processing
# -------------------------
def process_query_for_hybrid_search(query_rep, tokenized_docs, vectorizer, processed_texts):
    """Process query for multiple retrieval methods."""
    
    # For BM25: extract and lemmatize important terms from the query
    doc = nlp(query_rep["original"])
    bm25_query_terms = [token.lemma_.lower() for token in doc 
                        if not token.is_stop and not token.is_punct and len(token.text) > 1]
    
    # For TF-IDF: prepare query in the same way as the documents
    tfidf_query_terms = " ".join(bm25_query_terms)
    
    # For Dense: using original query performs better with transformer models
    dense_query = query_rep["original"]
    
    # Add key terms for better matching
    if query_rep["key_terms"]:
        # Combine with original query for dense search
        dense_query_expanded = query_rep["original"] + " " + " ".join(query_rep["key_terms"])
        dense_query = dense_query_expanded
    
    return bm25_query_terms, tfidf_query_terms, dense_query

# -------------------------
# Context-Aware Reranking
# -------------------------
def context_aware_reranking(results, query):
    """Rerank results based on context relevance to query."""
    if not results:
        return results
    
    # Parse query
    query_doc = nlp(query)
    query_key_tokens = set([token.lemma_.lower() for token in query_doc 
                            if not token.is_stop and not token.is_punct])
    
    # Score each result based on context match
    for result in results:
        context = result["context"]
        context_doc = nlp(context)
        
        # Calculate key token overlap
        context_key_tokens = set([token.lemma_.lower() for token in context_doc 
                                 if not token.is_stop and not token.is_punct])
        token_overlap = len(query_key_tokens.intersection(context_key_tokens))
        
        # Calculate named entity match
        query_entities = set([ent.text.lower() for ent in query_doc.ents])
        context_entities = set([ent.text.lower() for ent in context_doc.ents])
        entity_match = len(query_entities.intersection(context_entities))
        
        # Calculate adjusted score
        context_score = token_overlap + entity_match * 2  # Entities matter more
        
        # Update score with context relevance
        result["rerank_score"] = result.get("score", 0) * (1 + 0.2 * context_score)
    
    # Sort by reranked score
    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return results

# -------------------------
# Reciprocal Rank Fusion
# -------------------------
def reciprocal_rank_fusion(result_lists, k=60):
    """Combine multiple result lists using Reciprocal Rank Fusion."""
    fused_scores = {}
    
    # Process each result list
    for method_name, results in result_lists.items():
        # For each result in the ranked list
        for rank, result in enumerate(results):
            # Use a combination of sentence and document as ID to avoid conflicts
            doc_id = f"{result['document']}_p{result['page']}_{hash(result['matched_sentence'])}"
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"result": result, "score": 0}
            
            # RRF formula: 1 / (rank + k)
            fused_scores[doc_id]["score"] += 1.0 / (rank + k)
    
    # Create fused results list
    fused_results = [item["result"] for item in fused_scores.values()]
    
    # Sort by fused score
    fused_results.sort(key=lambda x: fused_scores[f"{x['document']}_p{x['page']}_{hash(x['matched_sentence'])}"]['score'], reverse=True)
    
    # Add fused score to results
    for result in fused_results:
        doc_id = f"{result['document']}_p{result['page']}_{hash(result['matched_sentence'])}"
        result["fused_score"] = fused_scores[doc_id]["score"]
        
    return fused_results

#---------------
#deduplication and merging of chunks
#---------------
def deduplicate_and_merge_results(results):
    #""Remove duplicate chunks by merging their matched sentences."""
        chunk_groups = {}
        
        for result in results:
            # Create a unique key based on the chunk content
            chunk_key = result.get("context", "").strip().lower()
            
            if chunk_key not in chunk_groups:
                chunk_groups[chunk_key] = result.copy()
                chunk_groups[chunk_key]["matched_sentences"] = [result["matched_sentence"]]
            else:
                # Merge matched sentences, avoiding duplicates
                existing_sentences = chunk_groups[chunk_key]["matched_sentences"]
                new_sentence = result["matched_sentence"]
                
                # Check if this sentence is already in the list (case-insensitive)
                sentence_exists = any(
                    new_sentence.strip().lower() == existing.strip().lower() 
                    for existing in existing_sentences
                )
                
                if not sentence_exists:
                    existing_sentences.append(new_sentence)
                
                # Keep the best score
                if result.get("score", 0) > chunk_groups[chunk_key].get("score", 0):
                    chunk_groups[chunk_key]["score"] = result["score"]
        
        # Convert back to list format with merged sentences
        unique_results = []
        for chunk_data in chunk_groups.values():
            # Combine all matched sentences with separator
            chunk_data["matched_sentence"] = " | ".join(chunk_data["matched_sentences"])
            del chunk_data["matched_sentences"]  # Remove temporary field
            unique_results.append(chunk_data)
        
        return unique_results



# -------------------------
# Enhanced Semantic Search with OpenAI Extraction
# -------------------------
def enhanced_semantic_search(query, collection, query_embedding, top_k=10):
    """Perform enhanced semantic search with OpenAI sentence extraction."""
    # Initial retrieval
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2  # Retrieve more for reranking
    )
    
    if not results["distances"] or not results["metadatas"]:
        return []
    
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    
    # Prepare embeddings for MMR
    chunk_embeddings = []
    for i, meta in enumerate(metadatas):
        chunk = meta.get("chunk", "")
        chunk_embedding = model.encode(chunk, convert_to_tensor=True)
        chunk_embeddings.append(chunk_embedding)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    query_embedding_tensor = torch.tensor(query_embedding, device=device)
    # Convert to tensor
    chunk_embeddings_tensor = torch.stack(chunk_embeddings).to(device)
    
    # Apply MMR for diversity
    query_embedding_tensor = torch.tensor(query_embedding)
    mmr_indices = mmr(
        query_embedding_tensor.unsqueeze(0),
        chunk_embeddings_tensor,
        metadatas,
        lambda_param=0.7,
        top_k=top_k
    )
    
    
    
    # Create results with MMR ranking and OpenAI extraction
    semantic_results = []
    for idx in mmr_indices:
        meta = metadatas[idx]
        chunk = meta["chunk"]
        page = meta["page"]
        docname = meta["document"]
        
        # Use OpenAI to extract relevant sentences from the chunk
        print(f"[INFO] Extracting sentences from chunk using OpenAI...")
        extracted_sentences = extract_relevant_sentences_with_openai(query, chunk)
        
        if not extracted_sentences:
            continue
        
        # For each extracted sentence, create a separate result
        for sentence in extracted_sentences:
            # Get expanded context around this sentence
            sentences = split_text_into_sentences(chunk)
            
            # Find the sentence in the chunk for context
            best_match_idx = 0
            best_similarity = 0
            for i, chunk_sentence in enumerate(sentences):
                # Simple similarity check
                similarity = len(set(sentence.lower().split()) & set(chunk_sentence.lower().split()))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i
            
            context = get_context_from_indices(best_match_idx, sentences, context_range=3)
            
            # Calculate relevance score for the extracted sentence
            sentence_embedding = model.encode(sentence, convert_to_tensor=True)
            query_emb = model.encode(query, convert_to_tensor=True)
            relevance_score = util.pytorch_cos_sim(query_emb, sentence_embedding)[0][0].item()
            
            semantic_results.append({
                "matched_sentence": sentence,
                "context": context,
                "page": page,
                "document": docname,
                "score": relevance_score,
                "source": "Dense Embeddings + OpenAI Extraction"
            })
    
    return semantic_results

# -------------------------
# Process Results with OpenAI Extraction
# -------------------------
def process_results_with_openai_extraction(query, results, method_name):
    """Process retrieved results using OpenAI to extract relevant sentences."""
    processed_results = []
    
    for result in results:
        context = result.get("context", "")
        
        # Use OpenAI to extract relevant sentences
        print(f"[INFO] Extracting sentences using OpenAI for {method_name}...")
        extracted_sentences = extract_relevant_sentences_with_openai(query, context)
        
        if not extracted_sentences:
            continue
        
        # Create a result for each extracted sentence
        for sentence in extracted_sentences:
            processed_result = result.copy()
            processed_result["matched_sentence"] = sentence
            processed_result["source"] = f"{method_name} + OpenAI Extraction"
            processed_results.append(processed_result)
    
    return processed_results

# -------------------------
# Improved Hybrid Search with OpenAI Integration
# -------------------------
def improved_hybrid_search(query, collection_name, top_k=5, context_range=3):
    """Flexible hybrid search with OpenAI sentence extraction."""
    if not query.strip():
        return json.dumps({"error": "Empty query"}, indent=4)
    
    print(f"[INFO] Starting hybrid search with OpenAI extraction for query: {query}")
    
    # Process query
    query_rep = preprocess_query(query)
    
    # Prepare indices
    bm25, all_sentences, bm25_pages, bm25_sentence_indices, bm25_doc_ids, collection, tokenized_docs = build_bm25_index(collection_name)
    vectorizer, tfidf_matrix, _, tfidf_pages, tfidf_sentence_indices, tfidf_doc_ids, _, processed_texts = build_tfidf_index(collection_name)
    
    # Process query for each retrieval method
    bm25_query_terms, tfidf_query_terms, dense_query = process_query_for_hybrid_search(
        query_rep, tokenized_docs, vectorizer, processed_texts
    )
    
    # BM25 Search
    bm25_scores = bm25.get_scores(bm25_query_terms)
    top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k*2]
    
    # TF-IDF Search
    tfidf_query = vectorizer.transform([tfidf_query_terms])
    tfidf_scores = (tfidf_query * tfidf_matrix.T).toarray().flatten()
    top_tfidf = tfidf_scores.argsort()[-top_k*2:][::-1]
    
    # Dense Embedding Search with OpenAI extraction
    query_embedding = model.encode(dense_query, convert_to_tensor=True)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0).tolist()
    
    dense_results = enhanced_semantic_search(
        dense_query, 
        collection, 
        query_embedding, 
        top_k=top_k
    )
    
    # Prepare results containers
    all_results = {
        "bm25": [],
        "tfidf": [],
        "dense": []
    }
    
    # Process BM25 results with OpenAI extraction
    print("[INFO] Processing BM25 results with OpenAI extraction...")
    for i, idx in enumerate(top_bm25):
        if idx >= len(all_sentences):
            continue
            
        chunk = all_sentences[idx]
        context = get_context_from_indices(0, split_text_into_sentences(chunk), context_range)
        
        # Use OpenAI to extract relevant sentences
        extracted_sentences = extract_relevant_sentences_with_openai(query, chunk)
        
        for sentence in extracted_sentences:
            all_results["bm25"].append({
                "matched_sentence": sentence,
                "context": context,
                "page": bm25_pages[idx],
                "document": bm25_doc_ids[idx] if idx < len(bm25_doc_ids) else collection_name,
                "score": bm25_scores[idx],
                "source": "BM25 + OpenAI Extraction"
            })
    
    # Process TF-IDF results with OpenAI extraction
    print("[INFO] Processing TF-IDF results with OpenAI extraction...")
    for i, idx in enumerate(top_tfidf):
        if idx >= len(all_sentences):
            continue
            
        chunk = all_sentences[idx]
        context = get_context_from_indices(0, split_text_into_sentences(chunk), context_range)
        
        # Use OpenAI to extract relevant sentences
        extracted_sentences = extract_relevant_sentences_with_openai(query, chunk)
        
        for sentence in extracted_sentences:
            all_results["tfidf"].append({
                "matched_sentence": sentence,
                "context": context, 
                "page": tfidf_pages[idx],
                "document": tfidf_doc_ids[idx] if idx < len(tfidf_doc_ids) else collection_name,
                "score": tfidf_scores[idx],
                "source": "TF-IDF + OpenAI Extraction"
            })
    
    # Dense results already processed with OpenAI extraction
    all_results["dense"] = dense_results
    
    # Apply context-aware reranking to each result set
    for method in all_results:
        all_results[method] = context_aware_reranking(all_results[method], query)
    
    # Combine results using reciprocal rank fusion
    fused_results = reciprocal_rank_fusion(all_results)
    
    # Remove duplicate chunks and merge their matched sentences
    unique_results = deduplicate_and_merge_results(fused_results)

    # If we need more results to reach top_k, we can adjust
    final_results = unique_results[:top_k] if len(unique_results) >= top_k else unique_results

    return json.dumps(final_results, indent=4)

#---------------------------------------------------------------------
# -------------------------
#LLM Final_Layer (Modified for OpenAI-extracted sentences)
#--------------------------
def llm_pre_final_layer(query, results, top=30):
    """
    Rerank results that already have OpenAI-extracted sentences.
    Since sentences are already extracted by OpenAI, we just need to score them.
    """
    url = "http://10.101.240.7/ollama/api/generate"  # Ollama API endpoint for Gemma
    headers = {"Content-Type": "application/json"}
    scored_results = []

    for result in results:
        # Sentences are already extracted by OpenAI, so we just score them
        prompt = (
            f"Query: {query}\n"
            f"Candidate: {result['matched_sentence']}\n"
            f"Context: {result['context']}\n"
            "Score the candidate's relevance to the query based on how well it answers or relates to the query, "
            "on a scale from 0 (not relevant) to 1 (highly relevant). "
            "The candidate sentence has already been pre-filtered for relevance. "
            "Respond with only the score as a float."
        )

        data = {"model": "gemma", "prompt": prompt}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            if response.status_code == 200:
                score_str = response.json().get("response", "0").strip()
                try:
                    llm_score = float(score_str)
                except ValueError:
                    llm_score = 0.0
            else:
                llm_score = 0.0
        except Exception as e:
            llm_score = 0.0

        result['llm_score'] = llm_score
        scored_results.append(result)

    # Sort by LLM score descending and return top N
    scored_results.sort(key=lambda x: x['llm_score'], reverse=True)
    return scored_results[:top]

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Check if OpenAI API key is set
    try:
        # Test OpenAI connection
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1
        )
        print("[INFO] OpenAI API connection successful")
    except Exception as e:
        print(f"[ERROR] OpenAI API connection failed: {e}")
        print("[ERROR] Please set your OPENAI_API_KEY environment variable or configure the API key in the code")
        exit(1)
    
    queries = []
    with open('questions.txt', 'r', encoding='utf-8') as file:
        for line in file:
            cleaned = line.strip()
            if cleaned:  # Skip empty lines
                queries.append(cleaned)
    
    for query in queries:
        print("\n============================")
        print(f"Processing query: {query}")
        print("============================")
        user_query = query
        collection_name = "ril_pdf_pages_semantic"
        top_k = 50
        context_range = 2
        print(f"[INFO] Extracting from collection: {collection_name}")
        results_json = improved_hybrid_search(user_query, collection_name, top_k, context_range)
        results = json.loads(results_json)

        # LLM Screening (optional since OpenAI already filtered)
        results_final = llm_pre_final_layer(user_query, results, top=30)
        
        print("\nTop Results (OpenAI Extracted + Ranked by Relevance):")
        # Handle context saving in the extracted_contexts.csv
        context_file_exists = os.path.isfile("extracted_contexts_openai.csv")
        with open("extracted_contexts_openai.csv", mode="a", newline="") as context_file:
            context_writer = csv.writer(context_file)
            if not context_file_exists:
                context_writer.writerow(["User Query", "Matched Sentence", "Context", "Page", "Document", "Source"])

            # Process the results_final
            for i, result in enumerate(results_final):
                print(f"{i + 1}. Source: {result.get('source', 'N/A')} - Score: {result.get('fused_score', result.get('rerank_score', 0)):.4f}")
                print(f"   OpenAI Extracted Sentence: {result['matched_sentence']}")
                print(f"   Context: {result['context']}")
                print(f"   Document: {result['document']}, Page: {result['page']}")
                print("-" * 80)

                # Save context to CSV
                context_writer.writerow([
                    user_query,
                    result['matched_sentence'],
                    result['context'],
                    result['page'],
                    result['document'],
                    result.get('source', 'N/A')
                ])

    print(f"Processed all queries with OpenAI sentence extraction")