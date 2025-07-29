#!/usr/bin/env python3
"""
Enhanced chunk-to-chunk & sentence-level comparator (2025-07-29)
Implements matching strategies 1–8, less strict but robust.
"""

import os, re, string, logging
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Any, Union, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK for stopwords and lemmatization
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab",quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Optional: sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SENTENCE_EMBEDDINGS = True
except ImportError:
    SENTENCE_EMBEDDINGS = False

# -------- CONFIG --------
MIN_OVERLAP_RATIO = 0.75     # For word overlap coverage (step 2)
FRAGMENT_MIN_COVERAGE = 0.75  # For fragment matching
PARAPHRASE_SIM_THRESHOLD = 0.75
MULTI_SENT_KEYWORD_COVERAGE = 1

# --------- TOOLS ---------
STOPWORDS_SET = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

def preprocess(text: str) -> str:
    """Lowercase, remove punctuation and excess spaces."""
    text = str(text).lower().translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    """Word tokenize and remove stopwords."""
    return [w for w in word_tokenize(text) if w not in STOPWORDS_SET]

def get_keywords(text: str) -> Set[str]:
    """Get keywords by removing stopwords/punctuation."""
    return set(tokenize(preprocess(text)))

def split_fragments(sentence: str) -> List[str]:
    """Split a sentence into fragments (by , ; and/but)."""
    parts = re.split(r",|;| and | but | or | although | though ", sentence)
    return [preprocess(p) for p in parts if p.strip()]

# --------- LOADING ---------
def read_column(path: str) -> List[str]:
    df = pd.read_excel(path)
    col = df.columns[0]
    return [str(x).strip() for x in df[col].fillna("").tolist() if str(x).strip() and str(x).strip().lower() not in ("nan", "null")]

# --------- MAIN LOGIC ---------
class SentenceComparator:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") if SENTENCE_EMBEDDINGS else None

    def sentence_covered(self, bench_sent: str, system_chunks: List[str]) -> Tuple[bool, str]:
        """Returns (covered, method) for a single benchmark sentence."""
        # Preprocessing
        p_bench = preprocess(bench_sent)
        bench_tokens = set(tokenize(bench_sent))
        bench_keywords = get_keywords(bench_sent)

        # 1. Exact match
        for sys_chunk in system_chunks:
            if p_bench == preprocess(sys_chunk):
                return True, "exact"

        # 2. High word overlap
        for sys_chunk in system_chunks:
            sys_tokens = set(tokenize(sys_chunk))
            if len(bench_tokens) > 0 and len(bench_tokens & sys_tokens) / len(bench_tokens) >= MIN_OVERLAP_RATIO:
                return True, "overlap"

        # 3. Fragment/phrase matching
        bench_frags = split_fragments(bench_sent)
        for frag in bench_frags:
            for sys_chunk in system_chunks:
                if frag and frag in preprocess(sys_chunk):
                    return True, "fragment"
                # partial overlap within fragment
                frag_tokens = set(tokenize(frag))
                sys_tokens = set(tokenize(sys_chunk))
                if len(frag_tokens) > 0 and len(frag_tokens & sys_tokens) / len(frag_tokens) > FRAGMENT_MIN_COVERAGE:
                    return True, "fragment"

        # 4. Paraphrase/semantic match
        if self.embedding_model:
            bench_emb = self.embedding_model.encode([bench_sent])[0]
            sys_embs = self.embedding_model.encode(system_chunks)
            for sys_emb in sys_embs:
                sim = np.dot(bench_emb, sys_emb) / (np.linalg.norm(bench_emb) * np.linalg.norm(sys_emb))
                if sim >= PARAPHRASE_SIM_THRESHOLD:
                    return True, "semantic"

        # 5. Multi-sentence coverage: can we find 2+ system sentences/fragments whose union covers this benchmark?
        for i in range(len(system_chunks)):
            for j in range(i+1, len(system_chunks)):
                sys1, sys2 = system_chunks[i], system_chunks[j]
                sys1_toks, sys2_toks = set(tokenize(sys1)), set(tokenize(sys2))
                union = sys1_toks | sys2_toks
                if len(bench_tokens) > 0 and len(bench_tokens & union) / len(bench_tokens) >= MULTI_SENT_KEYWORD_COVERAGE:
                    return True, "multi-sent"

        # 6. Keyword coverage: are 80% of main words anywhere in system chunks?
        all_sys_keywords = set()
        for sys_chunk in system_chunks:
            all_sys_keywords |= get_keywords(sys_chunk)
        if len(bench_keywords) > 0 and len(bench_keywords & all_sys_keywords) / len(bench_keywords) >= MULTI_SENT_KEYWORD_COVERAGE:
            return True, "keywords"

        # Not covered
        return False, "none"

    def compare(self, benchmark: List[str], system: List[str], print_each: bool = True):
        bench_sentence_counts = []
        bench_sentence_cov_counts = []
        method_stats = Counter()

        for idx, bench_chunk in enumerate(tqdm(benchmark, desc="Benchmark chunks")):
            bench_sents = sent_tokenize(bench_chunk)
            covered = 0
            methods = []

            for sent in bench_sents:
                is_covered, method = self.sentence_covered(sent, system)
                if is_covered:
                    covered += 1
                methods.append(method)
                method_stats[method] += 1

            pct = 100 * covered / len(bench_sents) if bench_sents else 0
            bench_sentence_counts.append(len(bench_sents))
            bench_sentence_cov_counts.append(covered)
            if print_each:
                print(f"Benchmark {idx+1}: {covered}/{len(bench_sents)} sentences covered ({pct:.1f}%)")

        avg_pct = 100 * sum(bench_sentence_cov_counts) / sum(bench_sentence_counts)
        print("="*60)
        print(f"TOTAL benchmark chunks: {len(benchmark)}")
        print(f"Average sentence coverage: {avg_pct:.2f}%")
        print(f"Method breakdown: {dict(method_stats)}")
        print("="*60)


def main():
    BENCHMARK_FILE = "Compare_outputs/benchmark.xlsx"
    SYSTEM_FILE = "Compare_outputs/system.xlsx"

    benchmark = read_column(BENCHMARK_FILE)
    system = read_column(SYSTEM_FILE)

    comp = SentenceComparator()
    comp.compare(benchmark, system, print_each=True)

if __name__ == "__main__":
    main()
