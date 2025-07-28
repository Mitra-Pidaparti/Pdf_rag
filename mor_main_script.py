import re, os, csv
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import chromadb
import json
import spacy
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
import time
import ast
from dotenv import load_dotenv

os.environ["OMP_NUM_THREADS"] = "7"


# Import MoR components
from mor_retrievers import MoRRetrieverPool
from mor_fusion import MoRPipeline, MoRScoreFusion, integrate_mor_with_existing_pipeline, find_matching_keywords_enhanced

# -------------------------
# Configuration
# -------------------------
INITIAL_POOL_SIZE = 300 # Large pool from each retriever
MOR_RERANK_SIZE = 150   # Candidates after MoR fusion  
NEURAL_RERANK_SIZE = 100 # Candidates after neural reranking
FINAL_CHUNKS = 60 # Final chunks after sentence extraction
DB_PATH = "chromadb"
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
load_dotenv()

# -------------------------
# Initialize Components
# -------------------------
print("[INFO] Initializing enhanced components with MoR...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sentence_model = SentenceTransformer(MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
client_chroma = chromadb.PersistentClient(path=DB_PATH)
nlp = spacy.load("en_core_web_md")

# -------------------------
# Enhanced Keywords Loading
# -------------------------
def load_keywords_from_excel(excel_path):
    """Load keywords with enhanced preprocessing"""
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
                    keywords = ast.literal_eval(keywords_raw)
                    if not isinstance(keywords, list):
                        keywords = [k.strip() for k in str(keywords).split(',')]
                except:
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

# -------------------------
# OpenAI Sentence Extraction (Enhanced)
# -------------------------
def extract_relevant_sentences_with_openai(query, context_text, keywords=None):
    """Enhanced sentence extraction with keyword awareness"""
    
    keyword_context = ""
    if keywords:
        keyword_context = f"\nKey terms to consider: {', '.join(keywords)}"
    
    system_prompt = f"""You are an expert text analyzer. Extract content that could help answer the user's query.{keyword_context}

IMPORTANT: Be VERY lenient in your relevance assessment. Include content that:
- Directly answers the query
- Provides context or background information  
- Contains related concepts, terms, or examples
- Mentions the same entities, topics, or domains
- Relates to any of the key terms provided

Instructions:
1. If ANY part of the text relates to the query (even tangentially), extract the relevant portions verbatim
2. Only return "NONE" if the text is completely unrelated to the query topic
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
# Enhanced Semantic Similarity
# -------------------------
def calculate_enhanced_semantic_similarity(query, sentence_group, keywords=None):
    """Enhanced semantic similarity with keyword boosting"""
    if not sentence_group or sentence_group == "No relevant sentences found":
        return 0.0

    try:
        # Enhance query with keywords
        enhanced_query = query
        if keywords:
            enhanced_query = f"{query} {' '.join(keywords)}"
        
        if isinstance(sentence_group, str):
            sentence_group = [sentence_group]
            
        query_embedding = sentence_model.encode([enhanced_query], convert_to_tensor=True)
        sentence_embedding = sentence_model.encode(sentence_group, convert_to_tensor=True)

        # Normalize embeddings
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

        # Compute cosine similarities
        from sentence_transformers import util
        similarity = util.pytorch_cos_sim(query_embedding, sentence_embedding)[0]
        return float(similarity.max().cpu().numpy())
        
    except Exception as e:
        print(f"[ERROR] Enhanced semantic similarity calculation failed: {e}")
        return 0.0

# -------------------------
# MoR-Enhanced Hybrid Search
# -------------------------
def mor_enhanced_hybrid_search(query, keywords_list, collection_name, mor_pipeline):
    """
    MoR-enhanced hybrid search pipeline:
    1. Use MoR pipeline for initial retrieval and fusion
    2. Apply sentence extraction and final ranking
    """
    print(f"\n[INFO] Starting MoR-enhanced hybrid search for query: {query}")
    print(f"[INFO] Keywords: {keywords_list}")
    
    # STEP 1: MoR Pipeline Search
    print(f"\n=== STEP 1: MoR PIPELINE SEARCH ===")
    mor_results = mor_pipeline.search(
        query=query,
        keywords=keywords_list,
        k=50,  # candidates per retriever
        final_k=MOR_RERANK_SIZE,
        use_reranking=True
    )
    
    if not mor_results:
        print("[WARNING] No results from MoR pipeline")
        return []
    
    # STEP 2: Sentence extraction and final enhancement
    print(f"\n=== STEP 2: SENTENCE EXTRACTION AND FINAL RANKING ===")
    final_candidates = []
    
    for i, result in enumerate(mor_results):
        print(f"[INFO] Processing MoR result {i+1}/{len(mor_results)}")
        
        # Extract relevant sentences with keyword awareness
        extracted_sentences = extract_relevant_sentences_with_openai(
            query, result['content'], keywords_list)
        
        # Calculate enhanced sentence-level semantic similarity
        sentence_similarity = calculate_enhanced_semantic_similarity(
            query, extracted_sentences, keywords_list)
        
        # Create enhanced final result
        final_candidate = {
            'query': query,
            'keywords': keywords_list,
            'extracted_sentences': extracted_sentences if extracted_sentences else "No relevant sentences found",
            'chunk_context': result['content'],
            'page': result['metadata'].get('page', 'Unknown'),
            'heading': result['metadata'].get('heading', ''),
            'document': result['metadata'].get('doc_id', 'Unknown'),
            
            # MoR scores
            'mor_fused_score': result.get('fused_score', 0.0),
            'cross_encoder_score': result.get('cross_encoder_score', 0.0),
            'final_mor_score': result.get('final_score', result.get('fused_score', 0.0)),
            
            # Retriever contributions
            'contributing_retrievers': result.get('contributing_retrievers', 0),
            'retriever_scores': result.get('retriever_scores', {}),
            'retriever_weights': result.get('retriever_weights', {}),
            
            # Enhanced scores
            'sentence_similarity': sentence_similarity,
            'ultimate_score': result.get('final_score', result.get('fused_score', 0.0)) * (1 + sentence_similarity * 0.3),
            
            # Metadata
            'rank': result.get('rank', i+1),
            'original_index': result['metadata'].get('original_index', -1)
        }
        
        final_candidates.append(final_candidate)
    
    # STEP 3: Final ranking by ultimate score
    print(f"\n=== STEP 3: FINAL RANKING (top {FINAL_CHUNKS}) ===")
    final_candidates.sort(key=lambda x: x['ultimate_score'], reverse=True)
    top_final = final_candidates[:FINAL_CHUNKS]
    
    print(f"[INFO] MoR-enhanced search completed. Final results: {len(top_final)} chunks")
    return top_final

# -------------------------
# Results Analysis and Debugging
# -------------------------
def analyze_mor_results(results):
    """Analyze MoR results for debugging"""
    if not results:
        return
    
    print(f"\n[MoR ANALYSIS] Analyzing {len(results)} results...")
    
    # Retriever contribution analysis
    retriever_contributions = {}
    total_contributing = 0
    
    for result in results:
        contrib_count = result.get('contributing_retrievers', 0)
        total_contributing += contrib_count
        
        retriever_scores = result.get('retriever_scores', {})
        for retriever_name in retriever_scores.keys():
            if retriever_name not in retriever_contributions:
                retriever_contributions[retriever_name] = 0
            retriever_contributions[retriever_name] += 1
    
    print(f"\n[RETRIEVER CONTRIBUTIONS]")
    for retriever, count in retriever_contributions.items():
        percentage = (count / len(results)) * 100
        print(f"  {retriever}: {count}/{len(results)} ({percentage:.1f}%)")
    
    print(f"\nAverage contributing retrievers per result: {total_contributing/len(results):.2f}")
    
    # Score distribution analysis
    if results:
        mor_scores = [r.get('final_mor_score', 0) for r in results]
        ultimate_scores = [r.get('ultimate_score', 0) for r in results]
        
        print(f"\n[SCORE ANALYSIS]")
        print(f"MoR Scores - Min: {min(mor_scores):.4f}, Max: {max(mor_scores):.4f}, Avg: {np.mean(mor_scores):.4f}")
        print(f"Ultimate Scores - Min: {min(ultimate_scores):.4f}, Max: {max(ultimate_scores):.4f}, Avg: {np.mean(ultimate_scores):.4f}")
    
    # Top results preview
    print(f"\n[TOP 3 RESULTS PREVIEW]")
    for i, result in enumerate(results[:3]):
        print(f"\n--- Rank {i+1} ---")
        print(f"Page: {result['page']}, Heading: {result['heading']}")
        print(f"MoR Score: {result.get('final_mor_score', 0):.4f}")
        print(f"Ultimate Score: {result['ultimate_score']:.4f}")
        print(f"Contributing Retrievers: {result.get('contributing_retrievers', 0)}")
        print(f"Content: {result['chunk_context'][:150]}...")

# -------------------------
# Main Execution with MoR Integration
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
    keywords_excel_path = "novartis_keywords_by_question.xlsx"
    keywords_dict = load_keywords_from_excel(keywords_excel_path)
    
    # Initialize MoR Pipeline
    print("\n" + "="*60)
    print("INITIALIZING MoR PIPELINE")
    print("="*60)
    
    collection_name = "combined_novartis_heading_semantic"
    
    try:
        mor_pipeline = integrate_mor_with_existing_pipeline(
            collection_name=collection_name,
            keywords_dict=keywords_dict,
            cross_encoder=cross_encoder,
            sentence_model=sentence_model
        )
        print("[SUCCESS] MoR pipeline initialized successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize MoR pipeline: {e}")
        exit(1)
    
    # Load queries
    queries = []
    with open('question2.txt', 'r', encoding='utf-8') as file:
        for line in file:
            cleaned = line.strip()
            if cleaned:
                queries.append(cleaned)
    
    print(f"[INFO] Loaded {len(queries)} queries for processing")
    
    # Initialize CSV file with MoR-enhanced headers
    csv_filename = "Novartis_MoR_enhanced_extractionQall.csv"
    csv_headers = [
        "User Query", "Keywords", "Extracted Sentences", "Chunk Context", "Page", "Heading",
        "Document", "MoR Fused Score", "Cross Encoder Score", "Final MoR Score",
        "Contributing Retrievers", "Sentence Similarity", "Ultimate Score",
        "Rank", "Retriever Contributions"
    ]
    
    # Write CSV header
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)

    # Process queries with MoR-enhanced pipeline
    print("\n" + "="*60)
    print("PROCESSING QUERIES WITH MoR PIPELINE")
    print("="*60)
    
    for query_idx, query in enumerate(queries):
        print("\n" + "="*80)
        print(f"Processing query {query_idx+1}/{len(queries)}: {query}")
        print("="*80)

        # Find matching keywords using enhanced function
        query_keywords = find_matching_keywords_enhanced(query, keywords_dict)
        
        try:
            # Run MoR-enhanced hybrid search
            results = mor_enhanced_hybrid_search(query, query_keywords, collection_name, mor_pipeline)
            
            if results:
                print(f"[SUCCESS] Found {len(results)} results for query: {query}")
                
                # Analyze results
                analyze_mor_results(results)
                
                # Write results to CSV
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    for result in results:
                        # Format retriever contributions for CSV
                        retriever_contribs = "; ".join([f"{k}:{v:.3f}" for k, v in result.get('retriever_scores', {}).items()])
                        
                        writer.writerow([
                            result["query"],
                            "; ".join(result["keywords"]),
                            result["extracted_sentences"],
                            result["chunk_context"],
                            result["page"],
                            result["heading"],
                            result["document"],
                            round(result.get("mor_fused_score", 0), 4),
                            round(result.get("cross_encoder_score", 0), 4),
                            round(result.get("final_mor_score", 0), 4),
                            result.get("contributing_retrievers", 0),
                            round(result["sentence_similarity"], 4),
                            round(result["ultimate_score"], 4),
                            result["rank"],
                            retriever_contribs
                        ])
                
            else:
                print(f"[WARNING] No results found for query: {query}")
                # Write empty row to maintain structure
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([query, "; ".join(query_keywords), "No results found"] + [""] * 12)
        
        except Exception as e:
            print(f"[ERROR] Failed to process query '{query}': {e}")
            import traceback
            traceback.print_exc()
            
            # Write error row
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query, "; ".join(query_keywords), f"Error: {str(e)}"] + [""] * 12)
        
        # Small delay between queries
        time.sleep(1)
    
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETED!")
    print("="*60)
    print(f"[COMPLETION] Results saved to: {csv_filename}")
    print(f"[SUMMARY] Processed {len(queries)} queries with MoR-enhanced hybrid search")
    print("[INFO] CSV contains detailed MoR scoring metrics and retriever contributions")