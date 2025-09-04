#!/usr/bin/env python3
"""
Multi-question comparator (v6): balanced coverage & attribution

Goals:
- Keep the report layout identical to v4/v5.
- Avoid over-attribution (5–6+ IDs) while not losing true contributors.
- Per-sentence exclusive credit (best single system chunk per sentence).
- Greedy, alpha-coverage selection with a soft cap + hard cap:
  - Hit target coverage (alpha, default 95%) using as few chunks as possible.
  - Prefer ≤ soft_k (default 3), but allow up to hard_k (default 5) if needed.

Output sheets:
  Summary + per-question numeric sheets with:
  [System Chunks] | [ ] | [Benchmark Chunks] | [Covering System Chunk #s] | [Coverage (%)]
"""

import argparse
import re
import string
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

# NLTK
import nltk
nltk.download("punkt", quiet=True)
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Optional semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_EMBEDDINGS = True
except Exception:
    SENTENCE_EMBEDDINGS = False


# ==================== Configuration ====================
@dataclass
class Config:
    # sentence-level thresholds
    min_overlap_ratio: float = 0.95
    fragment_min_coverage: float = 0.95
    paraphrase_sim_threshold: float = 0.95

    # attribution / selection
    coverage_target_ratio: float = 0.95  # alpha coverage for each benchmark chunk
    soft_k: int = 3                      # preferred max #system chunks per benchmark chunk
    hard_k: int = 5                      # absolute max if needed to reach alpha
    allow_exceed_soft_k: bool = True     # allow going beyond soft_k (up to hard_k) to hit alpha
    min_chunk_contribution_ratio: float = 0.30  # candidate filter
    min_chunk_contribution_sents: int = 1

    # execution
    max_workers: Optional[int] = None
    batch_size: int = 100


CONFIG = Config()
STOPWORDS_SET = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


# ==================== Data Classes ====================
@dataclass
class ChunkCoverage:
    chunk_id: int
    bench_chunk_text: str
    total_sentences: int
    covered_sentences: int
    coverage_percentage: float
    covering_sys_ids: List[int] = field(default_factory=list)


@dataclass
class QuestionReport:
    question_index: int
    question_name: str
    system_chunks: List[str]
    chunk_coverages: List[ChunkCoverage]
    average_coverage: float
    total_chunks: int


# ==================== Text Processing ====================
class TextProcessor:
    @staticmethod
    def preprocess(text: str) -> str:
        text = str(text).lower().translate(PUNCT_TABLE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [w for w in word_tokenize(text) if w not in STOPWORDS_SET]

    @staticmethod
    def get_keywords(text: str) -> Set[str]:
        return set(TextProcessor.tokenize(TextProcessor.preprocess(text)))

    @staticmethod
    def split_fragments(sentence: str) -> List[str]:
        parts = re.split(r",|;| and | but | or | although | though ", sentence)
        return [TextProcessor.preprocess(p) for p in parts if p.strip()]


# ==================== Data Loading ====================
class DataLoader:
    @staticmethod
    def load_questions_data(filepath: str) -> Tuple[Dict[str, List[str]], List[str]]:
        df = pd.read_excel(filepath)
        questions_data: Dict[str, List[str]] = {}
        ordered_questions: List[str] = []

        for col in df.columns:
            question_name = str(col).strip()
            chunks = [
                str(x).strip()
                for x in df[col].fillna("").tolist()
                if str(x).strip() and str(x).strip().lower() not in ("nan", "null", "")
            ]
            if chunks:
                questions_data[question_name] = chunks
                ordered_questions.append(question_name)

        return questions_data, ordered_questions


# ==================== Sentence Comparator ====================
class SentenceComparator:
    """Per-sentence best-match scoring + balanced attribution"""

    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.text_processor = TextProcessor()
        self.embedding_model = None
        self._init_embedding_model()

        # caches
        self._system_cache: Dict[int, Dict[str, object]] = {}
        self._embedding_cache: Dict[int, np.ndarray] = {}
        self._system_chunks_list: List[str] = []

    def _init_embedding_model(self):
        if SENTENCE_EMBEDDINGS:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_model.eval()
            except Exception as e:
                print(f"Warning: Could not initialize embedding model: {e}")
                self.embedding_model = None

    def _precompute_system_data(self, system_chunks: List[str]):
        self._system_cache.clear()
        self._embedding_cache.clear()
        self._system_chunks_list = list(system_chunks)

        for i, chunk in enumerate(system_chunks):
            self._system_cache[i] = {
                'original': chunk,
                'processed': self.text_processor.preprocess(chunk),
                'tokens': set(self.text_processor.tokenize(chunk)),
                'keywords': self.text_processor.get_keywords(chunk)
            }

        if self.embedding_model and system_chunks:
            self._compute_embeddings_batch(system_chunks)

    def _compute_embeddings_batch(self, texts: List[str]):
        if not self.embedding_model:
            return
        batch_size = self.config.batch_size
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, convert_to_numpy=True)
            all_embeddings.extend(embeddings)
        for i, emb in enumerate(all_embeddings):
            self._embedding_cache[i] = emb

    # ----- sentence match scoring (0..1) per system chunk -----
    def _sentence_match_score(self, bench_sent: str, sid: int) -> float:
        data = self._system_cache.get(sid)
        if not data:
            return 0.0

        p_bench = self.text_processor.preprocess(bench_sent)
        bench_tokens = set(self.text_processor.tokenize(bench_sent))
        best = 0.0

        # 1) Exact
        if p_bench and p_bench == data['processed']:
            return 1.0  # strongest possible

        # 2) High token overlap
        if bench_tokens and data['tokens']:
            overlap_ratio = len(bench_tokens & data['tokens']) / max(1, len(bench_tokens))
            if overlap_ratio >= self.config.min_overlap_ratio:
                best = max(best, overlap_ratio)  # 0.8..1.0 typically

        # 3) Fragment
        bench_frags = self.text_processor.split_fragments(bench_sent)
        for frag in bench_frags:
            if not frag:
                continue
            frag_processed = self.text_processor.preprocess(frag)
            frag_tokens = set(self.text_processor.tokenize(frag))

            if frag_processed and frag_processed in data['processed']:
                best = max(best, 0.9)  # strong evidence
                continue
            if frag_tokens:
                overlap = len(frag_tokens & data['tokens']) / max(1, len(frag_tokens))
                if overlap > self.config.fragment_min_coverage:
                    best = max(best, overlap)

        # 4) Semantic
        if self.embedding_model and self._embedding_cache:
            bench_emb = self.embedding_model.encode([bench_sent], convert_to_numpy=True)[0]
            denom = np.linalg.norm(bench_emb)
            if denom > 0:
                sys_emb = self._embedding_cache.get(sid)
                if sys_emb is not None:
                    denom2 = np.linalg.norm(sys_emb)
                    if denom2 > 0:
                        sim = float(np.dot(bench_emb, sys_emb) / (denom * denom2))
                        if sim >= self.config.paraphrase_sim_threshold:
                            best = max(best, sim)  # ~0.82..1.0

        return best

    # ----- benchmark chunk processing with alpha coverage & soft/hard caps -----
    def _process_chunk(self, args: Tuple[int, str]) -> ChunkCoverage:
        idx, bench_chunk = args
        bench_sents = [s for s in sent_tokenize(bench_chunk) if s.strip()]
        total_sents = len(bench_sents)

        if total_sents == 0:
            return ChunkCoverage(idx + 1, bench_chunk, 0, 0, 0.0, [])

        num_sys = len(self._system_chunks_list)

        # Exclusive best-match assignment: each sentence -> best system chunk (single sid)
        # If all scores are 0 for a sentence, it remains uncovered.
        sys_to_sentidxs: Dict[int, Set[int]] = {sid: set() for sid in range(num_sys)}
        for si, sent in enumerate(bench_sents):
            best_sid = None
            best_score = 0.0
            for sid in range(num_sys):
                score = self._sentence_match_score(sent, sid)
                if score > best_score:
                    best_score = score
                    best_sid = sid
            if best_sid is not None and best_score > 0.0:
                sys_to_sentidxs[best_sid].add(si)

        # Candidate filter: only highly contributing system chunks
        candidates: List[int] = []
        for sid, covered in sys_to_sentidxs.items():
            contrib = len(covered)
            if contrib == 0:
                continue
            ratio = contrib / total_sents
            if ratio >= self.config.min_chunk_contribution_ratio or contrib >= self.config.min_chunk_contribution_sents:
                candidates.append(sid)

        # No coverage at all
        if not candidates:
            return ChunkCoverage(idx + 1, bench_chunk, total_sents, 0, 0.0, [])

        # Greedy selection to reach alpha coverage with minimal chunks
        target = int(np.ceil(self.config.coverage_target_ratio * total_sents))
        selected: List[int] = []
        covered: Set[int] = set()

        # Order candidates by descending contribution first (fast start)
        cand_order = sorted(candidates, key=lambda s: len(sys_to_sentidxs[s]), reverse=True)

        # First pass: try to hit target within soft_k
        for sid in cand_order:
            if len(selected) >= self.config.soft_k:
                break
            gain = len(sys_to_sentidxs[sid] - covered)
            if gain <= 0:
                continue
            selected.append(sid)
            covered |= sys_to_sentidxs[sid]
            if len(covered) >= target:
                break

        # If still below target and allowed, extend up to hard_k
        if self.config.allow_exceed_soft_k and len(covered) < target and len(selected) < self.config.hard_k:
            for sid in cand_order:
                if sid in selected:
                    continue
                if len(selected) >= self.config.hard_k:
                    break
                gain = len(sys_to_sentidxs[sid] - covered)
                if gain <= 0:
                    continue
                selected.append(sid)
                covered |= sys_to_sentidxs[sid]
                if len(covered) >= target:
                    break

        covered_count = len(covered)
        coverage_pct = 100.0 * covered_count / total_sents

        return ChunkCoverage(
            chunk_id=idx + 1,
            bench_chunk_text=bench_chunk,
            total_sentences=total_sents,
            covered_sentences=covered_count,
            coverage_percentage=coverage_pct,
            covering_sys_ids=sorted([sid + 1 for sid in selected])  # 1-based
        )

    def compare_question(self, benchmark_chunks: List[str], system_chunks: List[str]) -> Tuple[List[ChunkCoverage], List[str]]:
        self._precompute_system_data(system_chunks)

        chunk_coverages: List[ChunkCoverage] = []
        max_workers = self.config.max_workers or mp.cpu_count()
        chunk_args = list(enumerate(benchmark_chunks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_chunk, args) for args in chunk_args]
            for future in as_completed(futures):
                chunk_coverages.append(future.result())

        chunk_coverages.sort(key=lambda x: x.chunk_id)
        return chunk_coverages, list(self._system_chunks_list)


# ==================== Multi-Question Comparator ====================
class MultiQuestionComparator:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.comparator = SentenceComparator(config)

    def compare_all_questions(
        self,
        benchmark_data: Dict[str, List[str]],
        system_data: Dict[str, List[str]],
        ordered_questions: List[str]
    ) -> List[QuestionReport]:
        reports: List[QuestionReport] = []

        for q_idx, question_name in enumerate(ordered_questions, start=1):
            bench_chunks = benchmark_data.get(question_name, [])
            if not bench_chunks:
                continue
            sys_chunks = system_data.get(question_name, [])

            if not sys_chunks:
                chunk_coverages = [
                    ChunkCoverage(
                        chunk_id=i+1,
                        bench_chunk_text=chunk,
                        total_sentences=len(sent_tokenize(chunk)),
                        covered_sentences=0,
                        coverage_percentage=0.0,
                        covering_sys_ids=[]
                    )
                    for i, chunk in enumerate(bench_chunks)
                ]
                sys_raw = []
            else:
                chunk_coverages, sys_raw = self.comparator.compare_question(bench_chunks, sys_chunks)

            avg_coverage = float(np.mean([cc.coverage_percentage for cc in chunk_coverages])) if chunk_coverages else 0.0
            reports.append(QuestionReport(
                question_index=q_idx,
                question_name=question_name,
                system_chunks=sys_raw,
                chunk_coverages=chunk_coverages,
                average_coverage=avg_coverage,
                total_chunks=len(chunk_coverages)
            ))
        return reports


# ==================== Report Generator (same layout as v4) ====================
ILLEGAL_SHEET_CHARS = r'[:\\/?*\[\]]'
MAX_SHEET_LEN = 31

def _unique_sheet_name(base: str, used: set) -> str:
    base = base[:MAX_SHEET_LEN]
    cand = base
    i = 2
    while cand.lower() in used:
        suffix = f" ({i})"
        cand = base[:MAX_SHEET_LEN - len(suffix)] + suffix
        i += 1
    used.add(cand.lower())
    return cand

class ReportGenerator:
    @staticmethod
    def generate_excel_report(reports: List[QuestionReport], output_file: str = "multi_question_coverage_report.xlsx"):
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            reports_sorted = sorted(reports, key=lambda r: r.question_index)
            # Summary
            summary_data = [{
                'Question #': r.question_index,
                'Original Title': r.question_name,
                'Total Benchmark Chunks': r.total_chunks,
                'Average Coverage (%)': float(f"{r.average_coverage:.2f}")
            } for r in reports_sorted]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            used_names = {"summary"}
            for r in reports_sorted:
                sys_display = [f"[{i+1}] {txt}" for i, txt in enumerate(r.system_chunks)]
                bench_display = [cc.bench_chunk_text for cc in r.chunk_coverages]
                covering_ids_col = [", ".join(map(str, cc.covering_sys_ids)) if cc.covering_sys_ids else ""
                                    for cc in r.chunk_coverages]
                coverage_col = [float(f"{cc.coverage_percentage:.2f}") for cc in r.chunk_coverages]

                n_rows = max(len(sys_display), len(bench_display))
                df = pd.DataFrame({
                    'System Chunks': sys_display + [""] * (n_rows - len(sys_display)),
                    ' ': [""] * n_rows,
                    'Benchmark Chunks': bench_display + [""] * (n_rows - len(bench_display)),
                    'Covering System Chunk #s': covering_ids_col + [""] * (n_rows - len(covering_ids_col)),
                    'Coverage (%)': coverage_col + [""] * (n_rows - len(coverage_col))
                })

                sheet_name = _unique_sheet_name(str(r.question_index), used_names)
                df.to_excel(writer, sheet_name=sheet_name, index=False)


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="Compare_outputs/benchmark_zurich.xlsx")
    parser.add_argument("--system",    default="Compare_outputs/system_zurich.xlsx")
    parser.add_argument("--output",    default="multi_question_coverage_report_zurich.xlsx")

    parser.add_argument("--alpha", type=float, default=CONFIG.coverage_target_ratio,
                        help="Target coverage (0..1) to achieve per benchmark chunk (default: 0.95)")
    parser.add_argument("--soft_k", type=int, default=CONFIG.soft_k,
                        help="Preferred max number of system chunks per benchmark chunk (default: 3)")
    parser.add_argument("--hard_k", type=int, default=CONFIG.hard_k,
                        help="Absolute max number of system chunks per benchmark chunk (default: 5)")
    parser.add_argument("--no_exceed", action="store_true",
                        help="Do NOT exceed soft_k even if needed to hit alpha")

    parser.add_argument("--min_ratio", type=float, default=CONFIG.min_chunk_contribution_ratio,
                        help="Min per-chunk contribution ratio (default: 0.30)")
    parser.add_argument("--min_sents", type=int, default=CONFIG.min_chunk_contribution_sents,
                        help="Min per-chunk contribution sentences (default: 1)")

    args = parser.parse_args()

    # apply overrides
    CONFIG.coverage_target_ratio = max(0.0, min(1.0, args.alpha))
    CONFIG.soft_k = max(1, args.soft_k)
    CONFIG.hard_k = max(CONFIG.soft_k, args.hard_k)
    CONFIG.allow_exceed_soft_k = not args.no_exceed
    CONFIG.min_chunk_contribution_ratio = max(0.0, min(1.0, args.min_ratio))
    CONFIG.min_chunk_contribution_sents = max(1, args.min_sents)

    loader = DataLoader()
    try:
        benchmark_data, bench_order = loader.load_questions_data(args.benchmark)
        system_data, _ = loader.load_questions_data(args.system)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    multi = MultiQuestionComparator(CONFIG)
    reports = multi.compare_all_questions(benchmark_data, system_data, bench_order)
    if not reports:
        print("No valid comparisons.")
        return

    ReportGenerator.generate_excel_report(reports, args.output)
    print(f"✅ Report saved to: {args.output}")


if __name__ == "__main__":
    main()

#python QC_tool_v1.py --benchmark "Compare_outputs/benchmark_zurich.xlsx" --system "Compare_outputs/system_zurich.xlsx" --output "multi_question_coverage_report_zurich.xlsx" --alpha 0.95 --soft_k 2 --hard_k 4