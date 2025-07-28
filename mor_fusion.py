import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import torch
from sentence_transformers import util

class MoRScoreFusion:
    """
    Score fusion module for Mixture-of-Retrievers
    Handles weighted combination and deduplication of results
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.scaler = MinMaxScaler()
        
    def normalize_scores(self, candidates_dict: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Normalize scores within each retriever to [0, 1] range
        """
        normalized_dict = {}
        
        for retriever_name, candidates in candidates_dict.items():
            if not candidates:
                normalized_dict[retriever_name] = []
                continue
                
            # Extract scores
            scores = np.array([cand['score'] for cand in candidates]).reshape(-1, 1)
            
            # Normalize scores
            if len(scores) > 1 and np.std(scores) > 0:
                normalized_scores = self.scaler.fit_transform(scores).flatten()
            else:
                # If all scores are the same, assign equal normalized scores
                normalized_scores = np.ones(len(scores)) * 0.5
            
            # Update candidates with normalized scores
            normalized_candidates = []
            for i, cand in enumerate(candidates):
                new_cand = cand.copy()
                new_cand['normalized_score'] = float(normalized_scores[i])
                normalized_candidates.append(new_cand)
            
            normalized_dict[retriever_name] = normalized_candidates
            
        return normalized_dict

    def weighted_score_fusion(self, 
                            candidates_dict: Dict[str, List[Dict]], 
                            retriever_weights: Dict[str, float]) -> List[Dict]:
        """
        Combine scores from all retrievers using weights
        """
        print("[INFO] Performing weighted score fusion...")
        
        # Normalize scores first
        normalized_candidates = self.normalize_scores(candidates_dict)
        
        # Collect all unique documents with their weighted scores
        document_scores = defaultdict(list)
        document_info = {}
        
        for retriever_name, candidates in normalized_candidates.items():
            weight = retriever_weights.get(retriever_name, 0.0)
            
            for cand in candidates:
                doc_idx = cand['doc_idx']
                weighted_score = weight * cand['normalized_score']
                
                document_scores[doc_idx].append(weighted_score)
                
                # Store document info (use first occurrence)
                if doc_idx not in document_info:
                    document_info[doc_idx] = {
                        'content': cand['content'],
                        'metadata': cand['metadata'],
                        'retriever_scores': {}
                    }
                
                # Store individual retriever score
                document_info[doc_idx]['retriever_scores'][retriever_name] = cand['normalized_score']
        
        # Combine scores for each document
        fused_candidates = []
        for doc_idx, score_list in document_scores.items():
            # Aggregate scores (sum of weighted scores)
            final_score = sum(score_list)
            
            fused_candidates.append({
                'doc_idx': doc_idx,
                'fused_score': final_score,
                'content': document_info[doc_idx]['content'],
                'metadata': document_info[doc_idx]['metadata'],
                'retriever_scores': document_info[doc_idx]['retriever_scores'],
                'contributing_retrievers': len(score_list),
                'score_components': score_list
            })
        
        # Sort by fused score
        fused_candidates.sort(key=lambda x: x['fused_score'], reverse=True)
        
        print(f"[INFO] Fused {len(document_scores)} unique documents from all retrievers")
        return fused_candidates

    def deduplicate_results(self, candidates: List[Dict], model=None) -> List[Dict]:
        """
        Remove duplicate documents based on content similarity
        """
        if len(candidates) <= 1:
            return candidates
            
        print(f"[INFO] Deduplicating {len(candidates)} candidates...")
        
        # Extract content for similarity computation
        contents = [cand['content'] for cand in candidates]
        
        try:
            if model is not None:
                # Use semantic similarity with provided model
                embeddings = model.encode(contents, convert_to_tensor=True)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
            else:
                # Fallback to TF-IDF similarity
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                vectorizer = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(contents)
                similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            to_remove = set()
            for i in range(len(candidates)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(candidates)):
                    if j in to_remove:
                        continue
                    if similarity_matrix[i][j] >= self.similarity_threshold:
                        # Keep the one with higher fused score
                        if candidates[i]['fused_score'] >= candidates[j]['fused_score']:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break
            
            # Remove duplicates
            deduplicated = [candidates[i] for i in range(len(candidates)) if i not in to_remove]
            
            print(f"[INFO] Removed {len(candidates) - len(deduplicated)} duplicates")
            return deduplicated
            
        except Exception as e:
            print(f"[WARNING] Deduplication failed: {e}. Returning original candidates.")
            return candidates


class MoRPipeline:
    """
    Complete MoR pipeline integrating retriever pool and score fusion
    """
    
    def __init__(self, retriever_pool, cross_encoder=None, sentence_model=None):
        self.retriever_pool = retriever_pool
        self.score_fusion = MoRScoreFusion()
        self.cross_encoder = cross_encoder
        self.sentence_model = sentence_model
        
    def search(self, query: str, 
               keywords: List[str] = None,
               k: int = 50, 
               final_k: int = 20,
               use_reranking: bool = True) -> List[Dict]:
        """
        Complete MoR search pipeline
        
        Args:
            query: Search query
            keywords: Optional keywords to boost query
            k: Number of candidates to retrieve per retriever
            final_k: Final number of results to return
            use_reranking: Whether to apply cross-encoder reranking
        
        Returns:
            List of ranked results with MoR scores
        """
        print(f"\n[MoR PIPELINE] Starting search for: '{query}'")
        if keywords:
            print(f"[MoR PIPELINE] Keywords: {keywords}")
        
        # Enhance query with keywords if provided
        enhanced_query = self._enhance_query_with_keywords(query, keywords)
        
        # Step 1: Retrieve candidates from all retrievers
        print("\n=== STEP 1: MULTI-RETRIEVER CANDIDATE RETRIEVAL ===")
        candidates_dict = self.retriever_pool.retrieve_candidates(enhanced_query, k)
        
        # Filter out empty retrievers
        candidates_dict = {name: cands for name, cands in candidates_dict.items() if cands}
        
        if not candidates_dict:
            print("[WARNING] No candidates retrieved from any retriever")
            return []
        
        # Step 2: Compute retriever weights using MoR signals
        print("\n=== STEP 2: COMPUTING MoR RETRIEVER WEIGHTS ===")
        retriever_weights = self.retriever_pool.compute_retriever_weights(enhanced_query, candidates_dict)
        
        # Step 3: Score fusion
        print("\n=== STEP 3: WEIGHTED SCORE FUSION ===")
        fused_candidates = self.score_fusion.weighted_score_fusion(candidates_dict, retriever_weights)
        
        # Step 4: Deduplication
        print("\n=== STEP 4: DEDUPLICATION ===")
        deduplicated_candidates = self.score_fusion.deduplicate_results(
            fused_candidates, model=self.sentence_model)
        
        # Step 5: Optional cross-encoder reranking
        if use_reranking and self.cross_encoder and len(deduplicated_candidates) > 1:
            print("\n=== STEP 5: CROSS-ENCODER RERANKING ===")
            reranked_candidates = self._cross_encoder_rerank(query, deduplicated_candidates)
        else:
            reranked_candidates = deduplicated_candidates
        
        # Step 6: Prepare final results
        final_results = reranked_candidates[:final_k]
        
        # Add search metadata
        for i, result in enumerate(final_results):
            result['rank'] = i + 1
            result['query'] = query
            result['keywords'] = keywords or []
            result['retriever_weights'] = retriever_weights
        
        print(f"\n[MoR PIPELINE] Completed. Returning {len(final_results)} results")
        self._print_top_results(final_results[:3])
        
        return final_results
    
    def _enhance_query_with_keywords(self, query: str, keywords: List[str]) -> str:
        """Enhance query with keywords"""
        if not keywords:
            return query
        
        # Simple keyword integration - boost important terms
        keyword_str = " ".join(keywords)
        enhanced = f"{query} {keyword_str}"
        
        print(f"[DEBUG] Enhanced query: '{enhanced}'")
        return enhanced
    
    def _cross_encoder_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Apply cross-encoder reranking"""
        print(f"[INFO] Cross-encoder reranking {len(candidates)} candidates...")
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for cand in candidates:
                content = cand['content']
                # Truncate if too long
                if len(content) > 4000:
                    content = content[:4000] + "..."
                query_doc_pairs.append([query, content])
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(query_doc_pairs, batch_size=16)
            
            # Update candidates with cross-encoder scores
            for i, cand in enumerate(candidates):
                cand['cross_encoder_score'] = float(cross_scores[i])
                # Combine with MoR fused score
                cand['final_score'] = (0.3 * cand['fused_score'] + 
                                     0.7 * cand['cross_encoder_score'])
            
            # Re-sort by final score
            candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            print(f"[INFO] Cross-encoder reranking completed")
            return candidates
            
        except Exception as e:
            print(f"[ERROR] Cross-encoder reranking failed: {e}")
            return candidates
    
    def _print_top_results(self, results: List[Dict]):
        """Print top results for debugging"""
        print("\n[TOP RESULTS]")
        for i, result in enumerate(results):
            print(f"\n--- Rank {i+1} ---")
            print(f"MoR Fused Score: {result.get('fused_score', 0):.4f}")
            if 'cross_encoder_score' in result:
                print(f"Cross-Encoder Score: {result['cross_encoder_score']:.4f}")
                print(f"Final Score: {result['final_score']:.4f}")
            print(f"Contributing Retrievers: {result.get('contributing_retrievers', 0)}")
            print(f"Content Preview: {result['content'][:150]}...")
            
            # Show retriever contributions
            retriever_scores = result.get('retriever_scores', {})
            if retriever_scores:
                print("Retriever Contributions:")
                for ret_name, score in retriever_scores.items():
                    print(f"  {ret_name}: {score:.3f}")


def integrate_mor_with_existing_pipeline(collection_name: str, 
                                       keywords_dict: Dict[str, List[str]],
                                       cross_encoder=None,
                                       sentence_model=None) -> MoRPipeline:
    """
    Integration function to set up MoR pipeline with existing ChromaDB collection
    """
    print("[INFO] Integrating MoR with existing pipeline...")
    
    # Import required components
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    # Initialize ChromaDB client
    client_chroma = chromadb.PersistentClient(path="chromadb")
    collection = client_chroma.get_collection(collection_name)
    
    # Get all documents and metadata
    print("[INFO] Loading documents from ChromaDB...")
    data = collection.get()
    
    documents = []
    doc_metadata = []
    
    for i, metadata in enumerate(data["metadatas"]):
        chunk = metadata.get("chunk", "")
        if chunk.strip():  # Only include non-empty chunks
            documents.append(chunk)
            doc_metadata.append({
                'page': metadata.get("page", "Unknown"),
                'heading': metadata.get("heading", ""),
                'doc_id': data["ids"][i] if i < len(data["ids"]) else f"doc_{i}",
                'original_index': i
            })
    
    print(f"[INFO] Loaded {len(documents)} documents for MoR initialization")
    
    # Initialize MoR components
    from mor_retrievers import MoRRetrieverPool
    
    # Create retriever pool
    retriever_pool = MoRRetrieverPool(cache_dir="mor_cache")
    
    # Initialize with documents
    retriever_pool.initialize_retrievers(documents, doc_metadata)
    
    # Create MoR pipeline
    mor_pipeline = MoRPipeline(
        retriever_pool=retriever_pool,
        cross_encoder=cross_encoder,
        sentence_model=sentence_model
    )
    
    print("[INFO] MoR pipeline integration completed!")
    return mor_pipeline


# Utility functions for existing code integration
def find_matching_keywords_enhanced(query: str, keywords_dict: Dict[str, List[str]]) -> List[str]:
    """
    Enhanced keyword matching function compatible with existing code
    """
    import spacy
    
    nlp = spacy.load("en_core_web_md")
    
    print(f"\n[DEBUG] Finding keywords for query: '{query}'")
    
    query_lower = query.lower().strip()
    
    # First, try exact match
    for question, keywords in keywords_dict.items():
        if query_lower == question.lower().strip():
            print(f"[SUCCESS] Found exact match!")
            return keywords
    
    # Then try partial matching with semantic similarity
    query_doc = nlp(query_lower)
    query_words = set(query_lower.split())
    
    best_match = None
    best_score = 0
    
    for question, keywords in keywords_dict.items():
        question_lower = question.lower().strip()
        question_words = set(question_lower.split())
        
        # Word overlap score
        intersection = query_words.intersection(question_words)
        union_size = max(len(query_words), len(question_words))
        overlap_ratio = len(intersection) / union_size if union_size > 0 else 0
        
        # Semantic similarity score
        question_doc = nlp(question_lower)
        semantic_sim = query_doc.similarity(question_doc)
        
        # Combined score
        combined_score = 0.6 * overlap_ratio + 0.4 * semantic_sim
        
        if combined_score > 0.7 and combined_score > best_score:
            best_match = (question, keywords)
            best_score = combined_score
    
    if best_match:
        print(f"[SUCCESS] Found semantic match with score {best_score:.3f}")
        return best_match[1]
    
    print(f"[WARNING] No matching keywords found for query: '{query}'")
    return []