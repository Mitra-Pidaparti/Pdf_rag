import re ,os ,csv
from sentence_transformers import SentenceTransformer, util
import torch
import chromadb
import json
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np

# -------------------------
# Initialize Chroma Client & Model
# -------------------------
DB_PATH = "chromadb"
MODEL_NAME = "all-mpnet-base-v2"  # Upgraded model from MiniLM

print("[INFO] Loading Sentence Transformer model...")
model = SentenceTransformer(MODEL_NAME)

print("[INFO] Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
collections = client.list_collections()

print("[INFO] Loading spaCy model for NLP tasks...")
nlp = spacy.load("en_core_web_sm")

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
    collection = client.get_collection(collection_name)
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
    collection = client.get_collection(collection_name)
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

# -------------------------
# Enhanced Semantic Search
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
        
        # Extract sentence indices from metadata
        sentence_indices_str = meta.get("sentence_indices", "")
        sentence_indices = list(map(int, sentence_indices_str.split(","))) if sentence_indices_str else []
        
        matched_global_idx = sentence_indices[best_sent_idx] if best_sent_idx < len(sentence_indices) else -1
        
        # Get expanded context
        context = get_context_from_indices(best_sent_idx, sentences, context_range=3)
        
        semantic_results.append({
            "matched_sentence": matched_sentence,
            "context": context,
            "page": page,
            "document": docname,
            "score": sentence_scores[best_sent_idx].item(),
            "source": "Dense Embedding (MMR)"
        })
    
    return semantic_results

# -------------------------
# Improved Hybrid Search
# -------------------------
def improved_hybrid_search(query, collection_name, top_k=5, context_range=3):
    """Flexible hybrid search without hardcoded query expansions."""
    if not query.strip():
        return json.dumps({"error": "Empty query"}, indent=4)
    
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
        
        # Extract sentence indices
        sentence_indices = []
        if bm25_sentence_indices[idx]:
            try:
                sentence_indices = list(map(int, bm25_sentence_indices[idx].split(",")))
            except:
                sentence_indices = []
        
        matched_global_idx = sentence_indices[best_sent_idx] if best_sent_idx < len(sentence_indices) else -1
        context = get_context_from_indices(best_sent_idx, sentences, context_range)
        
        all_results["bm25"].append({
            "matched_sentence": matched_sentence,
            "context": context,
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
        
        # Extract sentence indices
        sentence_indices = []
        if tfidf_sentence_indices[idx]:
            try:
                sentence_indices = list(map(int, tfidf_sentence_indices[idx].split(",")))
            except:
                sentence_indices = []
        
        matched_global_idx = sentence_indices[best_sent_idx] if best_sent_idx < len(sentence_indices) else -1
        context = get_context_from_indices(best_sent_idx, sentences, context_range)
        
        all_results["tfidf"].append({
            "matched_sentence": matched_sentence,
            "context": context, 
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
    
    # Remove duplicates (based on context similarity)
    seen_contexts = set()
    unique_results = []
    
    for result in fused_results:
        # Create a simplified representation of the context
        context_simplified = ' '.join(re.findall(r'\b\w+\b', result["context"].lower()))
        context_hash = hash(context_simplified)
        
        if context_hash not in seen_contexts:
            seen_contexts.add(context_hash)
            unique_results.append(result)
    
    return json.dumps(unique_results[:top_k], indent=4)

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # queries = ["What is the organization’s purpose/vision/mission?","To what extent and how does the company manage external contractors / non-permanent /temporary employees?",
    #               "How does the organization interact with its different stakeholders (customers, users, suppliers, employees, regulators, investors, government, society)?",
    #               "To what extent is the organization leveraging different disruptive and emerging technologies?","What are the key Metrics / KPIs / key performance indicators being tracked related to the innovation portfolio and overall business performance?",
    #               "Does the organization tolerate failure and encourage risk-taking?"]
    queries = ["Growth is Life"]

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
        top_k = 30
        context_range = 3
        print(f"[INFO] Extracting from collection: {collection_name}")  # <-- Add this line
        results_json = improved_hybrid_search(user_query, collection_name, top_k, context_range)
        results = json.loads(results_json)
        
        print("\nTop Results (Ranked by Relevance):")
        # Handle context saving in the extracted_contexts.csv
        context_file_exists = os.path.isfile("extracted_contexts2.csv")
        with open("extracted_contexts2.csv", mode="a", newline="") as context_file:
            context_writer = csv.writer(context_file)
            if not context_file_exists:
                context_writer.writerow(["User Query", "Matched Sentence", "Context", "Page", "Document"])

            # Now, process the results and extract up to 10 matched sentences for each subquery
            for i, result in enumerate(results):
                print(f"{i + 1}. Source: {result['source']} - Score: {result.get('fused_score', result.get('rerank_score', 0))}")
                print(f"   Matched Sentence: {result['matched_sentence']}")
                print(f"   Context: {result['context']}")
                print(f"   Document: {result['document']}, Page: {result['page']}")
                print("-" * 80)

                # Save context to CSV
                context_writer.writerow([
                    user_query,
                    result['matched_sentence'],
                    result['context'],
                    result['page'],
                    result['document']
                ])
                
    
    print(f"Processed all the queries")