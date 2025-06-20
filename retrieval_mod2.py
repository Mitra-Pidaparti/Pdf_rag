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

# -------------------------
# Initialize Chroma Client & Model
# -------------------------
DB_PATH = "chromadb"
#MODEL_NAME = "all-mpnet-base-v2"  # Upgraded model from MiniLM
MODEL_NAME='BAAI/bge-base-en-v1.5'

print("[INFO] Loading Sentence Transformer model...")
model = SentenceTransformer(MODEL_NAME)

print("[INFO] Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
collections = client.list_collections()

print("[INFO] Loading spaCy model for NLP tasks...")
nlp = spacy.load("en_core_web_md")

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
# Helper: Get Full Chunk
# -------------------------
def get_full_chunk(metadata):
    """Get the full chunk text from metadata."""
    return metadata.get("chunk", "")

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
    collection = client.get_collection(collection_name)
    data = collection.get()
    all_sentences = [meta.get("chunk", "") for meta in data["metadatas"]]
    page_numbers = [meta.get("page", "Unknown") for meta in data["metadatas"]]
    sentence_indices = [meta.get("sentence_indices", "") for meta in data["metadatas"]]
    doc_ids = data.get("ids", ["unknown"] * len(all_sentences))
    metadatas = data["metadatas"]  # Keep full metadata for chunk retrieval

    # Improved tokenization with lemmatization
    tokenized = []
    for sentence in all_sentences:
        doc = nlp(sentence)
        # Extract meaningful tokens with lemmatization
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and len(token.text) > 1]
        tokenized.append(tokens)
    
    bm25 = BM25Okapi(tokenized)
    return bm25, all_sentences, page_numbers, sentence_indices, doc_ids, collection, tokenized, metadatas

# -------------------------
# TF-IDF Indexing with Improvements
# -------------------------
def build_tfidf_index(collection_name):
    collection = client.get_collection(collection_name)
    data = collection.get()
    all_sentences = [meta.get("chunk", "") for meta in data["metadatas"]]
    page_numbers = [meta.get("page", "Unknown") for meta in data["metadatas"]]
    sentence_indices = [meta.get("sentence_indices", "") for meta in data["metadatas"]]
    doc_ids = data.get("ids", ["unknown"] * len(all_sentences))
    metadatas = data["metadatas"]  # Keep full metadata for chunk retrieval

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
    return vectorizer, tfidf_matrix, all_sentences, page_numbers, sentence_indices, doc_ids, collection, processed_texts, metadatas

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
# Context-Aware Reranking (Modified to work with chunks)
# -------------------------
def context_aware_reranking(results, query):
    """Rerank results based on chunk relevance to query."""
    if not results:
        return results
    
    # Parse query
    query_doc = nlp(query)
    query_key_tokens = set([token.lemma_.lower() for token in query_doc 
                            if not token.is_stop and not token.is_punct])
    
    # Score each result based on chunk match
    for result in results:
        chunk = result["chunk"]
        chunk_doc = nlp(chunk)
        
        # Calculate key token overlap
        chunk_key_tokens = set([token.lemma_.lower() for token in chunk_doc 
                               if not token.is_stop and not token.is_punct])
        token_overlap = len(query_key_tokens.intersection(chunk_key_tokens))
        
        # Calculate named entity match
        query_entities = set([ent.text.lower() for ent in query_doc.ents])
        chunk_entities = set([ent.text.lower() for ent in chunk_doc.ents])
        entity_match = len(query_entities.intersection(chunk_entities))
        
        # Calculate adjusted score
        chunk_score = token_overlap + entity_match * 2  # Entities matter more
        
        # Update score with chunk relevance
        result["rerank_score"] = result.get("score", 0) * (1 + 0.2 * chunk_score)
    
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

# -------------------------
# Enhanced Semantic Search (Modified to return chunks)
# -------------------------
def enhanced_semantic_search(query, collection, query_embedding, top_k=10):
    """Perform enhanced semantic search with reranking."""
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
    
    # Create results with MMR ranking
    semantic_results = []
    for idx in mmr_indices:
        meta = metadatas[idx]
        chunk = meta["chunk"]
        page = meta["page"]
        docname = meta["document"]
        
        # Split into sentences for better context identification
        sentences = split_text_into_sentences(chunk)
        if not sentences:
            continue

        # Find best matching sentence
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True) 
        query_emb = model.encode(query, convert_to_tensor=True)
        sentence_scores = util.pytorch_cos_sim(query_emb, sentence_embeddings)[0]
        
        best_sent_idx = torch.argmax(sentence_scores).item()
        matched_sentence = sentences[best_sent_idx]
        
        semantic_results.append({
            "matched_sentence": matched_sentence,
            "chunk": chunk,  # Store full chunk instead of context
            "page": page,
            "document": docname,
            "score": sentence_scores[best_sent_idx].item(),
            "source": "Dense Embedding (MMR)"
        })
    
    return semantic_results

# -------------------------
# Improved Hybrid Search (Modified to return chunks)
# -------------------------
def improved_hybrid_search(query, collection_name, top_k=5, context_range=3):
    """Flexible hybrid search without hardcoded query expansions."""
    if not query.strip():
        return json.dumps({"error": "Empty query"}, indent=4)
    
    # Process query
    query_rep = preprocess_query(query)
    
    # Prepare indices
    bm25, all_sentences, bm25_pages, bm25_sentence_indices, bm25_doc_ids, collection, tokenized_docs, bm25_metadatas = build_bm25_index(collection_name)
    vectorizer, tfidf_matrix, _, tfidf_pages, tfidf_sentence_indices, tfidf_doc_ids, _, processed_texts, tfidf_metadatas = build_tfidf_index(collection_name)
    
    # Process query for each retrieval method
    bm25_query_terms, tfidf_query_terms, dense_query = process_query_for_hybrid_search(
        query_rep, tokenized_docs, vectorizer, processed_texts
    )
    
    # BM25 Search
    bm25_scores = bm25.get_scores(bm25_query_terms)
    top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    
    # TF-IDF Search
    tfidf_query = vectorizer.transform([tfidf_query_terms])
    tfidf_scores = (tfidf_query * tfidf_matrix.T).toarray().flatten()
    top_tfidf = tfidf_scores.argsort()[-top_k:][::-1]
    
    # Dense Embedding Search
    query_embedding = model.encode(dense_query, convert_to_tensor=True)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0).tolist()
    
    # Enhanced semantic search
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
    # Process BM25 results
    for i, idx in enumerate(top_bm25):
        if idx >= len(all_sentences):
            continue
            
        chunk = all_sentences[idx]
        sentences = split_text_into_sentences(chunk)
        if not sentences:
            continue
        
        # Find best matching sentence
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        query_emb = model.encode(dense_query, convert_to_tensor=True)
        sentence_scores = util.pytorch_cos_sim(query_emb, sentence_embeddings)[0]
        
        best_sent_idx = torch.argmax(sentence_scores).item()
        matched_sentence = sentences[best_sent_idx]
        
        # Get full chunk from metadata
        full_chunk = get_full_chunk(bm25_metadatas[idx])
        
        all_results["bm25"].append({
            "matched_sentence": matched_sentence,
            "chunk": full_chunk,  # Store full chunk instead of context
            "page": bm25_pages[idx],
            "document": bm25_doc_ids[idx] if idx < len(bm25_doc_ids) else collection_name,
            "score": bm25_scores[idx],
            "source": "BM25"
        })
    

    # Process TF-IDF results
    for i, idx in enumerate(top_tfidf):
        if idx >= len(all_sentences):
            continue
            
        chunk = all_sentences[idx]
        sentences = split_text_into_sentences(chunk)
        if not sentences:
            continue
        
        # Find best matching sentence
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        query_emb = model.encode(dense_query, convert_to_tensor=True)
        sentence_scores = util.pytorch_cos_sim(query_emb, sentence_embeddings)[0]
        
        best_sent_idx = torch.argmax(sentence_scores).item()
        matched_sentence = sentences[best_sent_idx]
        
        # Get full chunk from metadata
        full_chunk = get_full_chunk(tfidf_metadatas[idx])
        
        all_results["tfidf"].append({
            "matched_sentence": matched_sentence,
            "chunk": full_chunk,  # Store full chunk instead of context
            "page": tfidf_pages[idx],
            "document": tfidf_doc_ids[idx] if idx < len(tfidf_doc_ids) else collection_name,
            "score": tfidf_scores[idx],
            "source": "TF-IDF"
        })
    
    # Add dense results directly
    all_results["dense"] = dense_results
    
    # Apply context-aware reranking to each result set
    for method in all_results:
        all_results[method] = context_aware_reranking(all_results[method], query)
    
    # Combine results using reciprocal rank fusion
    fused_results = reciprocal_rank_fusion(all_results)
    
    # Remove duplicates (based on chunk similarity)
    seen_chunks = set()
    unique_results = []
    
    for result in fused_results:
        # Create a simplified representation of the chunk
        chunk_simplified = ' '.join(re.findall(r'\b\w+\b', result["chunk"].lower()))
        chunk_hash = hash(chunk_simplified)
        
        if chunk_hash not in seen_chunks:
            seen_chunks.add(chunk_hash)
            unique_results.append(result)
    
    return json.dumps(unique_results[:top_k], indent=4)


#---------------------------------------------------------------------
# -------------------------
#LLM Pre_Final_Layer
#--------------------------
def llm_pre_final_layer(query, results, top=30):
    """
    Rerank results using the Gemma LLM via Ollama API.
    Returns the top N results based on LLM scoring.
    """
    url = "http://10.101.240.7/ollama/api/generate"  # Ollama API endpoint for Gemma
    headers = {"Content-Type": "application/json"}
    scored_results = []

    for result in results:
        prompt = (
            f"Query: {query}\n"
            f"Candidate: {result['matched_sentence']}\n"
            f"Chunk: {result['chunk']}\n"
            "Score the candidate's relevance to the query and how well it answers the query on a scale from 0 (not relevant) to 1 (highly relevant). "
            "Respond with only the score as a float."
            "Also if the sentence is not meaningful or is a single phrase or proper noun assign a score of 0. "
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

'''

# Load the reranker pipeline
pipe = pipeline("feature-extraction", model="BAAI/bge-reranker-large", device=0)  # Use device=-1 for CPU

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def llm_pre_final_layer(query, results, top=30):
    """
    Rerank results using BGE reranker by computing embedding similarity.
    Returns the top N results based on cosine similarity of [CLS] embeddings.
    """
    scored_results = []

    for result in results:
        input_text = f"{query} [SEP] {result['matched_sentence']}"
        # Get [CLS] embedding (assumed to be the first token vector)
        with torch.no_grad():
            output = pipe(input_text)
        cls_embedding = output[0][0]  # First token of first sequence

        if not isinstance(cls_embedding, (list, np.ndarray)):
            cls_embedding = cls_embedding.tolist()

        #result['llm_score'] = cosine_similarity(cls_embedding, cls_embedding)  # self-sim for debug
        scored_results.append(result)

    # Rerun with actual query embedding for proper ranking
    with torch.no_grad():
        query_vec = pipe(query)[0][0]

    for result in scored_results:
        candidate_vec = pipe(f"{query} [SEP] {result['matched_sentence']}")[0][0]
        result['llm_score'] = cosine_similarity(query_vec, candidate_vec)

    scored_results.sort(key=lambda x: x['llm_score'], reverse=True)
    return scored_results[:top]

'''




#---------------------------------------------

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # queries = ["What is the organization's purpose/vision/mission?","To what extent and how does the company manage external contractors / non-permanent /temporary employees?",
    #               "How does the organization interact with its different stakeholders (customers, users, suppliers, employees, regulators, investors, government, society)?",
    #               "To what extent is the organization leveraging different disruptive and emerging technologies?","What are the key Metrics / KPIs / key performance indicators being tracked related to the innovation portfolio and overall business performance?",
    #               "Does the organization tolerate failure and encourage risk-taking?"]
    #queries = ["Growth is Life"]

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
        print(f"[INFO] Extracting from collection: {collection_name}")  # <-- Add this line
        results_json = improved_hybrid_search(user_query, collection_name, top_k, context_range)
        results = json.loads(results_json)

        ##LLM Screening
        results_final = llm_pre_final_layer(user_query, results, top=30)
        
        print("\nTop Results (Ranked by Relevance):")
        # Handle context saving in the extracted_contexts.csv
        context_file_exists = os.path.isfile("extracted_contexts12.csv")
        with open("extracted_contexts12.csv", mode="a", newline="") as context_file:
            context_writer = csv.writer(context_file)
            if not context_file_exists:
                # Modified header to show 'Chunk' instead of 'Context'
                context_writer.writerow(["User Query", "Matched Sentence", "Chunk", "Page", "Document"])

            # Now, process the results_final and extract up to 30 matched sentences for each subquery
            for i, result in enumerate(results_final):
                print(f"{i + 1}. Source: {result.get('source', 'N/A')} - Score: {result.get('fused_score', result.get('rerank_score', 0))}")
                print(f"   Matched Sentence: {result['matched_sentence']}")
                print(f"   Chunk: {result['chunk']}")  # Display chunk instead of context
                print(f"   Document: {result['document']}, Page: {result['page']}")
                print("-" * 80)

                # Save chunk to CSV
                context_writer.writerow([
                    user_query,
                    result['matched_sentence'],
                    result['chunk'],  # Save chunk instead of context
                    result['page'],
                    result['document']
                ])


        print(f"Processed all the queries")
