import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import pickle
import os
from typing import Dict, List, Tuple, Any
import spacy
from collections import defaultdict

class MoRRetrieverPool:
    """
    Mixture-of-Retrievers (MoR) Pool Implementation
    Manages multiple retrievers with automated weighting
    """
    
    def __init__(self, cache_dir="mor_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize retrievers
        self.retrievers = {}
        self.embeddings = {}
        self.centroids = {}
        self.cluster_labels = {}
        self.nlp = spacy.load("en_core_web_md")
        
        # Hyperparameters for signal combination
        self.alpha = 0.4  # Pre-retrieval signal weight
        self.beta = 0.3   # Post-retrieval signal weight  
        self.gamma = 0.3  # Moran's I weight
        
        print("[INFO] MoR Retriever Pool initialized")

    def initialize_retrievers(self, documents: List[str], doc_metadata: List[Dict]):
        """
        Initialize and setup all retrievers with document corpus
        """
        print("[INFO] Initializing MoR retriever pool...")
        
        # Store documents and metadata
        self.documents = documents
        self.doc_metadata = doc_metadata
        
        # 1. Initialize Sparse Retrievers
        print("[INFO] Setting up sparse retrievers...")
        self._setup_sparse_retrievers(documents)
        
        # 2. Initialize Dense Retrievers
        print("[INFO] Setting up dense retrievers...")
        self._setup_dense_retrievers(documents)
        
        print(f"[INFO] MoR pool initialized with {len(self.retrievers)} retrievers")
        
    def _setup_sparse_retrievers(self, documents: List[str]):
        """Setup BM25 and TF-IDF retrievers"""
        
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in documents:
            doc_nlp = self.nlp(doc.lower())
            tokens = [token.lemma_ for token in doc_nlp 
                     if not token.is_stop and not token.is_punct and len(token.text) > 1]
            tokenized_docs.append(tokens)
        
        # BM25 Retriever
        self.retrievers['bm25'] = BM25Okapi(tokenized_docs)
        
        # TF-IDF Retriever
        processed_docs = []
        for doc in documents:
            doc_nlp = self.nlp(doc)
            terms = [token.lemma_.lower() for token in doc_nlp 
                    if not token.is_stop and not token.is_punct and len(token.text) > 1]
            processed_docs.append(" ".join(terms))
        
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        
        self.retrievers['tfidf'] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix
        }
        
        print(f"[INFO] Sparse retrievers initialized: BM25, TF-IDF")

    def _setup_dense_retrievers(self, documents: List[str]):
        """Setup dense retrievers with precomputed embeddings and clusters"""
        
        dense_models = {
            'bge': 'BAAI/bge-base-en-v1.5',
            'dpr': 'facebook-dpr-ctx_encoder-single-nq-base',
            'mpnet': 'sentence-transformers/all-mpnet-base-v2',
            'minilm': 'sentence-transformers/all-MiniLM-L6-v2'
        }
        
        for model_name, model_path in dense_models.items():
            cache_file = os.path.join(self.cache_dir, f"{model_name}_embeddings.pkl")
            
            try:
                # Try to load cached embeddings
                if os.path.exists(cache_file):
                    print(f"[INFO] Loading cached embeddings for {model_name}...")
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        self.retrievers[model_name] = cache_data['model']
                        self.embeddings[model_name] = cache_data['embeddings']
                        self.centroids[model_name] = cache_data['centroids']
                        self.cluster_labels[model_name] = cache_data['labels']
                else:
                    # Compute and cache embeddings
                    print(f"[INFO] Computing embeddings for {model_name}...")
                    model = SentenceTransformer(model_path)
                    embeddings = model.encode(documents, convert_to_tensor=False, batch_size=32)
                    
                    # Compute clusters
                    n_clusters = max(2, int(np.sqrt(len(documents) / 4)))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    centroids = kmeans.cluster_centers_
                    
                    # Store
                    self.retrievers[model_name] = model
                    self.embeddings[model_name] = embeddings
                    self.centroids[model_name] = centroids
                    self.cluster_labels[model_name] = labels
                    
                    # Cache
                    cache_data = {
                        'model': model,
                        'embeddings': embeddings,
                        'centroids': centroids,
                        'labels': labels
                    }
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    
                    print(f"[INFO] {model_name} embeddings computed and cached")
                    
            except Exception as e:
                print(f"[WARNING] Failed to initialize {model_name}: {e}")
                continue
        
        print(f"[INFO] Dense retrievers initialized: {list(self.retrievers.keys())}")

    def compute_pre_retrieval_signals(self, query: str) -> Dict[str, float]:
        """
        Compute pre-retrieval signals (V_pre) for each retriever
        Based on query-to-corpus centroid distances
        """
        signals = {}
        
        for retriever_name in self.retrievers.keys():
            if retriever_name in ['bm25', 'tfidf']:
                # For sparse retrievers, use a simple heuristic
                signals[retriever_name] = self._compute_sparse_pre_signal(query, retriever_name)
            else:
                # For dense retrievers, compute centroid-based signal
                signals[retriever_name] = self._compute_dense_pre_signal(query, retriever_name)
        
        return signals

    def _compute_sparse_pre_signal(self, query: str, retriever_name: str) -> float:
        """Compute pre-retrieval signal for sparse retrievers"""
        
        if retriever_name == 'bm25':
            # Query term coverage in corpus
            query_doc = self.nlp(query.lower())
            query_terms = [token.lemma_ for token in query_doc 
                          if not token.is_stop and not token.is_punct]
            
            # Simple heuristic: ratio of query terms that appear in corpus
            corpus_terms = set()
            for doc in self.documents[:100]:  # Sample for efficiency
                doc_nlp = self.nlp(doc.lower())
                corpus_terms.update([token.lemma_ for token in doc_nlp 
                                   if not token.is_stop and not token.is_punct])
            
            coverage = sum(1 for term in query_terms if term in corpus_terms) / max(1, len(query_terms))
            return coverage
            
        elif retriever_name == 'tfidf':
            # TF-IDF based signal
            query_doc = self.nlp(query)
            query_terms = [token.lemma_.lower() for token in query_doc 
                          if not token.is_stop and not token.is_punct]
            query_text = " ".join(query_terms)
            
            try:
                query_vector = self.retrievers['tfidf']['vectorizer'].transform([query_text])
                # Average TF-IDF weight as signal
                return float(np.mean(query_vector.toarray()))
            except:
                return 0.5  # Default signal
        
        return 0.5  # Default signal

    def _compute_dense_pre_signal(self, query: str, retriever_name: str) -> float:
        """Compute pre-retrieval signal for dense retrievers using centroid distances"""
        
        try:
            model = self.retrievers[retriever_name]
            centroids = self.centroids[retriever_name]
            labels = self.cluster_labels[retriever_name]
            
            # Encode query
            query_embedding = model.encode([query], convert_to_tensor=False)[0]
            
            # Compute distances to all centroids
            distances = []
            contributions = []
            
            for i, centroid in enumerate(centroids):
                distance = np.linalg.norm(query_embedding - centroid)
                cluster_size = np.sum(labels == i)
                
                # Weight by inverse distance and cluster size
                if distance > 0:
                    contribution = (cluster_size / len(centroids)) * (1.0 / distance)
                    contributions.append(contribution)
            
            # Aggregate contributions
            if contributions:
                signal = np.linalg.norm(contributions)
                return float(signal)
            else:
                return 0.5
                
        except Exception as e:
            print(f"[WARNING] Failed to compute pre-retrieval signal for {retriever_name}: {e}")
            return 0.5

    def retrieve_candidates(self, query: str, k: int = 50) -> Dict[str, List[Dict]]:
        """
        Retrieve top-k candidates from each retriever
        """
        all_candidates = {}
        
        for retriever_name in self.retrievers.keys():
            try:
                if retriever_name == 'bm25':
                    candidates = self._retrieve_bm25(query, k)
                elif retriever_name == 'tfidf':
                    candidates = self._retrieve_tfidf(query, k)
                else:
                    candidates = self._retrieve_dense(query, retriever_name, k)
                
                all_candidates[retriever_name] = candidates
                print(f"[INFO] {retriever_name}: Retrieved {len(candidates)} candidates")
                
            except Exception as e:
                print(f"[WARNING] Retrieval failed for {retriever_name}: {e}")
                all_candidates[retriever_name] = []
        
        return all_candidates

    def _retrieve_bm25(self, query: str, k: int) -> List[Dict]:
        """Retrieve using BM25"""
        query_doc = self.nlp(query.lower())
        query_terms = [token.lemma_ for token in query_doc 
                      if not token.is_stop and not token.is_punct and len(token.text) > 1]
        
        scores = self.retrievers['bm25'].get_scores(query_terms)
        top_indices = np.argsort(scores)[-k:][::-1]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                'doc_idx': int(idx),
                'score': float(scores[idx]),
                'content': self.documents[idx],
                'metadata': self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
            })
        
        return candidates

    def _retrieve_tfidf(self, query: str, k: int) -> List[Dict]:
        """Retrieve using TF-IDF"""
        query_doc = self.nlp(query)
        query_terms = [token.lemma_.lower() for token in query_doc 
                      if not token.is_stop and not token.is_punct]
        query_text = " ".join(query_terms)
        
        vectorizer = self.retrievers['tfidf']['vectorizer']
        tfidf_matrix = self.retrievers['tfidf']['matrix']
        
        query_vector = vectorizer.transform([query_text])
        scores = (query_vector * tfidf_matrix.T).toarray().flatten()
        
        top_indices = np.argsort(scores)[-k:][::-1]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                'doc_idx': int(idx),
                'score': float(scores[idx]),
                'content': self.documents[idx],
                'metadata': self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
            })
        
        return candidates

    def _retrieve_dense(self, query: str, retriever_name: str, k: int) -> List[Dict]:
        """Retrieve using dense embeddings"""
        model = self.retrievers[retriever_name]
        embeddings = self.embeddings[retriever_name]
        
        query_embedding = model.encode([query], convert_to_tensor=False)[0]
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                'doc_idx': int(idx),
                'score': float(similarities[idx]),
                'content': self.documents[idx],
                'metadata': self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
            })
        
        return candidates

    def compute_post_retrieval_signals(self, query: str, candidates_dict: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Compute post-retrieval signals (V_post) for each retriever
        """
        signals = {}
        
        for retriever_name, candidates in candidates_dict.items():
            if not candidates:
                signals[retriever_name] = 0.0
                continue
                
            try:
                # Extract retrieved documents
                retrieved_docs = [cand['content'] for cand in candidates]
                
                if retriever_name in ['bm25', 'tfidf']:
                    # For sparse retrievers, use document diversity
                    signals[retriever_name] = self._compute_document_diversity(retrieved_docs)
                else:
                    # For dense retrievers, compute V_post using centroids
                    signals[retriever_name] = self._compute_dense_post_signal(retrieved_docs, retriever_name)
                    
            except Exception as e:
                print(f"[WARNING] Failed to compute post-retrieval signal for {retriever_name}: {e}")
                signals[retriever_name] = 0.0
        
        return signals

    def _compute_document_diversity(self, documents: List[str]) -> float:
        """Compute document diversity as post-retrieval signal"""
        if len(documents) < 2:
            return 0.0
        
        try:
            # Use TF-IDF to compute document similarities
            vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
            doc_vectors = vectorizer.fit_transform(documents)
            
            # Compute pairwise similarities
            similarities = cosine_similarity(doc_vectors)
            
            # Average pairwise similarity (lower = more diverse)
            n_docs = len(documents)
            total_sim = 0
            count = 0
            
            for i in range(n_docs):
                for j in range(i+1, n_docs):
                    total_sim += similarities[i][j]
                    count += 1
            
            avg_similarity = total_sim / max(1, count)
            diversity = 1.0 - avg_similarity  # Convert to diversity score
            
            return max(0.0, diversity)
            
        except:
            return 0.5  # Default diversity

    def _compute_dense_post_signal(self, documents: List[str], retriever_name: str) -> float:
        """Compute V_post for dense retrievers"""
        try:
            model = self.retrievers[retriever_name]
            centroids = self.centroids[retriever_name]
            
            # Encode retrieved documents
            doc_embeddings = model.encode(documents, convert_to_tensor=False)
            
            # For each document, compute its "query-like" signal against centroids
            doc_signals = []
            for doc_emb in doc_embeddings:
                contributions = []
                for centroid in centroids:
                    distance = np.linalg.norm(doc_emb - centroid)
                    if distance > 0:
                        contributions.append(1.0 / distance)
                
                if contributions:
                    doc_signal = np.linalg.norm(contributions)
                    doc_signals.append(doc_signal)
            
            # Aggregate document signals
            if doc_signals:
                return float(np.linalg.norm(doc_signals))
            else:
                return 0.0
                
        except Exception as e:
            print(f"[WARNING] Failed to compute dense post-signal: {e}")
            return 0.0

    def compute_moran_i_signals(self, candidates_dict: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Compute Moran's I spatial autocorrelation signals
        """
        signals = {}
        
        for retriever_name, candidates in candidates_dict.items():
            if len(candidates) < 3:  # Need minimum documents for Moran's I
                signals[retriever_name] = 0.0
                continue
                
            try:
                if retriever_name in ['bm25', 'tfidf']:
                    # Use TF-IDF embeddings for sparse retrievers
                    signals[retriever_name] = self._compute_sparse_moran_i(candidates)
                else:
                    # Use dense embeddings
                    signals[retriever_name] = self._compute_dense_moran_i(candidates, retriever_name)
                    
            except Exception as e:
                print(f"[WARNING] Failed to compute Moran's I for {retriever_name}: {e}")
                signals[retriever_name] = 0.0
        
        return signals

    def _compute_sparse_moran_i(self, candidates: List[Dict]) -> float:
        """Compute Moran's I for sparse retrievers"""
        try:
            documents = [cand['content'] for cand in candidates]
            
            # Create TF-IDF embeddings
            vectorizer = TfidfVectorizer(min_df=1, stop_words='english', max_features=100)
            doc_vectors = vectorizer.fit_transform(documents).toarray()
            
            return self._moran_i_calculation(doc_vectors)
            
        except:
            return 0.0

    def _compute_dense_moran_i(self, candidates: List[Dict], retriever_name: str) -> float:
        """Compute Moran's I for dense retrievers"""
        try:
            doc_indices = [cand['doc_idx'] for cand in candidates]
            embeddings = self.embeddings[retriever_name]
            
            # Get embeddings for retrieved documents
            selected_embeddings = embeddings[doc_indices]
            
            return self._moran_i_calculation(selected_embeddings)
            
        except:
            return 0.0

    def _moran_i_calculation(self, embeddings: np.ndarray) -> float:
        """Calculate Moran's I statistic"""
        try:
            n = len(embeddings)
            if n < 3:
                return 0.0
            
            # Compute similarity matrix as weights
            similarity_matrix = cosine_similarity(embeddings)
            
            # Remove diagonal (self-similarities)
            np.fill_diagonal(similarity_matrix, 0)
            
            # Compute Moran's I
            W = similarity_matrix
            W_sum = np.sum(W)
            
            if W_sum == 0:
                return 0.0
            
            # Use similarity scores as the attribute
            x = np.sum(embeddings, axis=1)  # Simple aggregation of embedding dimensions
            x_mean = np.mean(x)
            
            numerator = 0
            for i in range(n):
                for j in range(n):
                    numerator += W[i,j] * (x[i] - x_mean) * (x[j] - x_mean)
            
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator == 0:
                return 0.0
            
            moran_i = (n / W_sum) * (numerator / denominator)
            
            # Normalize to [0, 1] range
            return max(0.0, min(1.0, (moran_i + 1) / 2))
            
        except:
            return 0.0

    def compute_retriever_weights(self, query: str, candidates_dict: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Compute final retriever weights using combined signals
        """
        print("[INFO] Computing retriever weights using MoR signals...")
        
        # Compute all signals
        pre_signals = self.compute_pre_retrieval_signals(query)
        post_signals = self.compute_post_retrieval_signals(query, candidates_dict)
        moran_signals = self.compute_moran_i_signals(candidates_dict)
        
        # Combine signals for each retriever
        retriever_scores = {}
        for retriever_name in self.retrievers.keys():
            pre_score = pre_signals.get(retriever_name, 0.0)
            post_score = post_signals.get(retriever_name, 0.0)
            moran_score = moran_signals.get(retriever_name, 0.0)
            
            # Combined score
            combined_score = (self.alpha * pre_score + 
                            self.beta * post_score + 
                            self.gamma * moran_score)
            
            retriever_scores[retriever_name] = combined_score
            
            print(f"[DEBUG] {retriever_name}: pre={pre_score:.3f}, post={post_score:.3f}, "
                  f"moran={moran_score:.3f} -> combined={combined_score:.3f}")
        
        # Normalize weights
        total_score = sum(retriever_scores.values())
        if total_score > 0:
            weights = {name: score/total_score for name, score in retriever_scores.items()}
        else:
            # Fallback to equal weights
            weights = {name: 1.0/len(retriever_scores) for name in retriever_scores.keys()}
        
        print(f"[INFO] Final retriever weights: {weights}")
        return weights