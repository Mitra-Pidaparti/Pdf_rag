#!/usr/bin/env python3
"""
Enhanced chunk-to-chunk & sentence-level comparator (2025-07-29, revised)
- Adds substring fallback after exact match
- Precomputes system embeddings (optional) and TF-IDF for fast top-K candidate selection
- Limits O(N^2) pair unions to top-K candidates
- Keeps negations out of stopwords to avoid meaning flips
"""

import os, re, string, logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Any, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK for stopwords and tokenization
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Optional: sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_EMBEDDINGS = True
except ImportError:
    SENTENCE_EMBEDDINGS = False

# -------- CONFIG --------
MIN_OVERLAP_RATIO = 0.80             # word overlap coverage
FRAGMENT_MIN_COVERAGE = 0.80         # fragment overlap coverage
PARAPHRASE_SIM_THRESHOLD = 0.80      # cosine sim (with normalized embeddings)
MULTI_SENT_KEYWORD_COVERAGE = 0.80   # coverage by union of two candidates
KEYWORD_BACKSTOP_COVERAGE = 0.90     # stricter threshold for global keyword backstop
TOP_K_CANDIDATES = 50                # cap for candidate set per sentence (limits pairwise unions)

# --------- TOOLS ---------
STOPWORDS_SET = set(stopwords.words("english"))
# Keep negations & related terms (don’t let stopwording remove meaning)
STOPWORDS_SET -= {"no", "not", "nor", "without", "never"}

PUNCT_TABLE = str.maketrans('', '', string.punctuation)

def preprocess(text: str) -> str:
    """Lowercase, remove punctuation and compress spaces."""
    text = str(text)
    text = text.lower().translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    """Word tokenize and remove stopwords (preserving negations kept above)."""
    return [w for w in word_tokenize(text) if w not in STOPWORDS_SET]

def get_keywords(text: str) -> Set[str]:
    """Keywords = tokens after preprocess & stopword removal."""
    return set(tokenize(preprocess(text)))

def split_fragments(sentence: str) -> List[str]:
    """Split a sentence into fragments (by , ; and/but/or/although/though)."""
    parts = re.split(r",|;|\sand\s|\sbut\s|\sor\s|\salt hough\s|\sthough\s", sentence, flags=re.IGNORECASE)
    return [preprocess(p) for p in parts if p and p.strip()]

# --------- LOADING ---------
def read_column(path: str) -> List[str]:
    df = pd.read_excel(path)
    col = df.columns[0]
    vals = []
    for x in df[col].tolist():
        s = str(x).strip() if pd.notna(x) else ""
        if s and s.lower() not in ("nan", "null"):
            vals.append(s)
    return vals

# --------- MAIN LOGIC ---------
class SentenceComparator:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") if SENTENCE_EMBEDDINGS else None

    # ---- Precompute system data once ----
    def _build_system_context(self, system_chunks: List[str]) -> Dict[str, Any]:
        sys_prep = [preprocess(s) for s in system_chunks]
        set_sys_prep = set(sys_prep)  # for O(1) equality checks

        sys_tokens_list = [set(tokenize(s)) for s in system_chunks]  # overlap calc
        # Global keyword union only used as last-resort backstop
        sys_keywords_union: Set[str] = set()
        for s in system_chunks:
            sys_keywords_union |= get_keywords(s)

        # TF-IDF (L2-normalized by default) for fast top-K lexical candidates
        tfidf = TfidfVectorizer(stop_words="english")
        sys_matrix = tfidf.fit_transform(sys_prep)  # shape: (N_docs, N_terms)

        # Optional sentence embeddings (pre-normalized for cosine via dot)
        sys_embs = None
        if self.embedding_model:
            sys_embs = self.embedding_model.encode(system_chunks, normalize_embeddings=True)

        return {
            "system": system_chunks,
            "sys_prep": sys_prep,
            "set_sys_prep": set_sys_prep,
            "sys_tokens_list": sys_tokens_list,
            "sys_keywords_union": sys_keywords_union,
            "tfidf": tfidf,
            "sys_matrix": sys_matrix,
            "sys_embs": sys_embs,  # np.ndarray [N, d] or None
        }

    def _topk_candidate_indices(self, bench_sent: str, ctx: Dict[str, Any], k: int) -> List[int]:
        """Select top-K candidates by TF-IDF cosine; if embeddings available, mix in top-K by semantic."""
        k = max(1, k)
        # TF-IDF sims
        q_vec = ctx["tfidf"].transform([preprocess(bench_sent)])  # (1, V)
        sims_lex = (ctx["sys_matrix"] @ q_vec.T).toarray().ravel()  # cosine if TF-IDF normalized
        top_lex_idx = np.argsort(-sims_lex)[: max(1, k // (2 if ctx["sys_embs"] is not None else 1))]

        # Semantic sims (optional)
        top_sem_idx = np.array([], dtype=int)
        if ctx["sys_embs"] is not None:
            bench_emb = self.embedding_model.encode([bench_sent], normalize_embeddings=True)[0]
            sims_sem = ctx["sys_embs"] @ bench_emb  # cosine via dot (normalized)
            top_sem_idx = np.argsort(-sims_sem)[: k - len(top_lex_idx)]

        # Union (preserve order roughly by lexical then semantic)
        cand = list(dict.fromkeys(list(top_lex_idx) + list(top_sem_idx)))
        return cand[:k]

    def sentence_covered(self, bench_sent: str, ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """Returns (covered, method) for a single benchmark sentence."""
        p_bench = preprocess(bench_sent)
        bench_tokens = set(tokenize(bench_sent))
        bench_keywords = get_keywords(bench_sent)

        # Pre-select candidate indices for heavier checks
        cand_idx = self._topk_candidate_indices(bench_sent, ctx, TOP_K_CANDIDATES)

        # 1. Exact or substring match (fast path)
        if p_bench in ctx["set_sys_prep"]:
            return True, "exact"
        for i in cand_idx:
            if p_bench and p_bench in ctx["sys_prep"][i]:
                return True, "substring"

        # 2. High word overlap (use candidates)
        if bench_tokens:
            for i in cand_idx:
                sys_toks = ctx["sys_tokens_list"][i]
                if len(bench_tokens & sys_toks) / len(bench_tokens) >= MIN_OVERLAP_RATIO:
                    return True, "overlap"

        # 3. Fragment/phrase matching (use candidates)
        bench_frags = split_fragments(bench_sent)
        for frag in bench_frags:
            if not frag:
                continue
            frag_tokens = set(tokenize(frag))
            for i in cand_idx:
                sys_p = ctx["sys_prep"][i]
                if frag in sys_p:
                    return True, "fragment"
                if frag_tokens:
                    sys_toks = ctx["sys_tokens_list"][i]
                    if len(frag_tokens & sys_toks) / len(frag_tokens) >= FRAGMENT_MIN_COVERAGE:
                        return True, "fragment"

        # 4. Paraphrase/semantic match (use candidates)
        if ctx["sys_embs"] is not None:
            bench_emb = self.embedding_model.encode([bench_sent], normalize_embeddings=True)[0]
            sims = ctx["sys_embs"][cand_idx] @ bench_emb
            if sims.max(initial=-1.0) >= PARAPHRASE_SIM_THRESHOLD:
                return True, "semantic"

        # 5. Multi-sentence coverage via union of two candidate chunks
        if bench_tokens:
            L = len(cand_idx)
            # Small cap to prevent worst O(K^2) blowups (K already limited)
            for a in range(L):
                sys1 = ctx["sys_tokens_list"][cand_idx[a]]
                for b in range(a + 1, L):
                    sys2 = ctx["sys_tokens_list"][cand_idx[b]]
                    if len(bench_tokens & (sys1 | sys2)) / len(bench_tokens) >= MULTI_SENT_KEYWORD_COVERAGE:
                        return True, "multi-sent"

        # 6. Keyword backstop against global union (strict threshold)
        if bench_keywords:
            if len(bench_keywords & ctx["sys_keywords_union"]) / len(bench_keywords) >= KEYWORD_BACKSTOP_COVERAGE:
                return True, "keywords"

        # Not covered
        return False, "none"

    def compare(self, benchmark: List[str], system: List[str], print_each: bool = True):
        # Build system-side context once
        ctx = self._build_system_context(system)

        bench_sentence_counts: List[int] = []
        bench_sentence_cov_counts: List[int] = []
        method_stats = Counter()

        for idx, bench_chunk in enumerate(tqdm(benchmark, desc="Benchmark chunks")):
            bench_sents = sent_tokenize(bench_chunk)
            covered = 0

            for sent in bench_sents:
                is_covered, method = self.sentence_covered(sent, ctx)
                if is_covered:
                    covered += 1
                method_stats[method] += 1

            total = len(bench_sents) or 1
            pct = 100 * covered / total
            bench_sentence_counts.append(total)
            bench_sentence_cov_counts.append(covered)
            if print_each:
                print(f"Benchmark {idx+1}: {covered}/{total} sentences covered ({pct:.1f}%)")

        avg_pct = 100 * sum(bench_sentence_cov_counts) / max(1, sum(bench_sentence_counts))
        print("=" * 60)
        print(f"TOTAL benchmark chunks: {len(benchmark)}")
        print(f"Average sentence coverage: {avg_pct:.2f}%")
        print(f"Method breakdown: {dict(method_stats)}")
        print("=" * 60)


def main():
    BENCHMARK_FILE = "Compare_outputs/benchmark.xlsx"
    SYSTEM_FILE = "Compare_outputs/system.xlsx"

    benchmark = read_column(BENCHMARK_FILE)
    system = read_column(SYSTEM_FILE)

    comp = SentenceComparator()
    comp.compare(benchmark, system, print_each=True)

if __name__ == "__main__":
    main()
