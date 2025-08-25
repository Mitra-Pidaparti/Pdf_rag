#!/usr/bin/env python3
"""
Agentic multi-pass retrieval that delegates to Retrieval_vf and stops at **neural reranking**.
- No LLM sentence extraction (outputs full chunks directly).
- On rounds 2 and 3, use **one-third** of the round-1 pool/rerank sizes to cut latency.
- Enrich with metadata so filename/page are present in CSV.

Run (Windows cmd):
  python agentic_retrieve_keywords_lastneural.py ^
    --query "What is the nature and focus of the organization's 'Purpose'?" ^
    --keywords_excel novartis_keywords_by_question.xlsx ^
    --collection novartis_collection ^
    --chroma_path chromadb ^
    --csv keyword_multipass_resultsQ1.csv ^
    --jsonl trace.jsonl
"""
from __future__ import annotations

import os
import re
import csv
import time
import json
import math
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

# === Your retrieval pipeline (BM25+TFIDF → dense → cross‑encoder neural rerank) ===
import Retrieval_vf as RVF

# Direct Chroma access for metadata enrichment
import chromadb

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("agentic_lastneural")

# -------------------------
# Config
# -------------------------
@dataclass
class AgentConfig:
    chroma_path: str = "chromadb"
    collection_name: str = "novartis_collection"

    # Agent loop
    max_iters: int = 5
    coverage_stop_ratio: float = 0.90
    min_new_evidence: int = 4
    min_keyword_gain: int = 2
    sleep_between_rounds_sec: float = 0.25

    # Sizing controls
    shrink_factor_rounds_2plus: float = 1.0/2.0  # one-third sizes after round 1
    top_k_per_round: int = 60

    # Query refinement (optional LLM; safe fallback if key missing)
    propose_llm_queries: bool = True
    llm_model: str = "gpt-5-mini"
    llm_num_queries: int = 4

    # Outputs
    out_csv: str = "agentic_results.csv"
    out_jsonl: Optional[str] = None

# -------------------------
# Chroma accessor (for robust metadata)
# -------------------------
class ChromaAccessor:
    def __init__(self, path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_collection(collection_name)
        self._md_cache: Dict[str, Dict] = {}

    def get_metadata_for_ids(self, ids: List[str]) -> Dict[str, Dict]:
        missing = [cid for cid in ids if cid not in self._md_cache]
        if missing:
            try:
                res = self.collection.get(ids=missing, include=["metadatas"])
                got_ids = res.get("ids", []) or []
                mds = res.get("metadatas", []) or []
                for i, cid in enumerate(got_ids):
                    self._md_cache[cid] = mds[i] or {}
            except Exception as e:
                log.warning(f"Chroma get() failed for {len(missing)} ids: {e}")
        return {cid: self._md_cache.get(cid, {}) for cid in ids}

# -------------------------
# Utilities
# -------------------------

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _candidate_uid(c: Dict) -> str:
    md = c.get("metadata") or {}
    return str(c.get("chunk_id") or md.get("chunk_id") or c.get("id") or c.get("document") or id(c))


def extract_keywords_present(text: str, keywords: List[str]) -> Set[str]:
    t = _norm_text(text)
    covered: Set[str] = set()
    for kw in keywords:
        if not kw:
            continue
        k = kw.strip().lower()
        if len(k.split()) > 1:
            if k in t:
                covered.add(kw)
        else:
            if re.search(rf"\b{re.escape(k)}\b", t):
                covered.add(kw)
    return covered


def measure_round_coverage(cands: List[Dict], query: str, keywords: List[str]) -> Tuple[Set[str], int]:
    seen_kw: Set[str] = set()
    hits = 0
    for c in cands:
        blob = " ".join([
            c.get("chunk") or "",
            (c.get("metadata") or {}).get("chunk_text") or "",
            c.get("heading") or "",
        ])
        covered = extract_keywords_present(blob, keywords)
        if covered:
            hits += 1
            seen_kw.update(covered)
    seen_kw.update(extract_keywords_present(query, keywords))
    return seen_kw, hits

# -------------------------
# Query refinement (optional)
# -------------------------

def _heuristic_rewrites(query: str, missing_keywords: List[str], n: int = 4) -> List[str]:
    base = _norm_text(query)
    kws = [k for k in missing_keywords if k]
    random.shuffle(kws)
    bundles = [kws[i:i+3] for i in range(0, min(len(kws), 12), 3)]
    rewrites: List[str] = []
    for b in bundles[:max(1, n)]:
        rewrites.append(base + " " + " ".join(b))
    out, seen = [], set()
    for r in rewrites:
        if r and r not in seen and r != base:
            out.append(r)
            seen.add(r)
    return out[:n]


def propose_refined_queries(query: str,
                            top_candidates: List[Dict],
                            missing_keywords: List[str],
                            cfg: AgentConfig) -> List[str]:
    # keep optional LLM rewrites; otherwise heuristic
    context_snips: List[str] = []
    for c in top_candidates[:12]:
        ctx = c.get("chunk") or (c.get("metadata") or {}).get("chunk_text") or ""
        if ctx:
            context_snips.append(ctx[:600])
    context = "\n\n---\n\n".join(context_snips)

    rewrites: List[str] = []
    if cfg.propose_llm_queries and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            sys_prompt = (
                "You are a search query strategist. Propose focused sub-queries (<=12 tokens) "
                "that recover missing aspects given the original query, missing keywords, and sample context."
            )
            user_prompt = json.dumps({
                "original_query": query,
                "missing_keywords": missing_keywords[:20],
                "sample_context": context[:6000],
                "num": cfg.llm_num_queries,
            }, ensure_ascii=False)
            resp = client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            for line in text.splitlines():
                line = line.strip("-•* \t")
                if line:
                    rewrites.append(line)
        except Exception as e:
            log.warning(f"[WARN] LLM refinement failed: {e}. Falling back to heuristics.")

    if not rewrites:
        rewrites = _heuristic_rewrites(query, missing_keywords, n=cfg.llm_num_queries)
    rewrites = [r for r in rewrites if r and r.lower() != _norm_text(query)]
    seen, uniq = set(), []
    for r in rewrites:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq[: cfg.llm_num_queries]

# -------------------------
# Metadata enrichment
# -------------------------

def _merge_metadata(c: Dict, md: Dict) -> Dict:
    c = dict(c)
    meta = dict(c.get("metadata") or {})
    meta.update({k: v for k, v in (md or {}).items() if v is not None})
    c["metadata"] = meta
    c.setdefault("chunk_id", meta.get("chunk_id") or c.get("id") or c.get("document"))
    c["page"] = meta.get("page", c.get("page"))
    c["document"] = meta.get("filename") or meta.get("filepath") or c.get("document")
    if not c.get("chunk") and meta.get("chunk_text"):
        c["chunk"] = meta.get("chunk_text")
    return c


def enrich_with_metadata(candidates: List[Dict], accessor: ChromaAccessor) -> List[Dict]:
    ids: List[str] = []
    for c in candidates:
        md = c.get("metadata") or {}
        cid = c.get("chunk_id") or md.get("chunk_id") or c.get("id") or c.get("document")
        if cid:
            ids.append(str(cid))
    ids = list(dict.fromkeys(ids))
    id2md = accessor.get_metadata_for_ids(ids) if ids else {}
    out: List[Dict] = []
    for c in candidates:
        md = c.get("metadata") or {}
        cid = c.get("chunk_id") or md.get("chunk_id") or c.get("id") or c.get("document")
        full = id2md.get(str(cid), {}) if cid else md
        out.append(_merge_metadata(c, full))
    return out

# -------------------------
# One agentic round (no LLM extraction; sizes can shrink)
# -------------------------

def _sized(n_base: int, round_idx: int, cfg: AgentConfig) -> int:
    if round_idx == 0:
        return max(1, int(n_base))
    return max(1, int(math.ceil(n_base * cfg.shrink_factor_rounds_2plus)))


def single_round(query: str,
                 keywords: List[str],
                 cfg: AgentConfig,
                 accessor: ChromaAccessor,
                 already_seen: Set[str],
                 round_idx: int) -> List[Dict]:
    # Dynamic sizes per round
    pool_n   = _sized(RVF.INITIAL_POOL_SIZE,   round_idx, cfg)
    dense_n  = _sized(RVF.DENSE_RERANK_SIZE,   round_idx, cfg)
    neural_n = _sized(RVF.NEURAL_RERANK_SIZE,  round_idx, cfg)
    log.info(f"[SIZES] round={round_idx+1} pool={pool_n} dense={dense_n} neural={neural_n}")

    # STEP 1: Initial pool via RVF (BM25+TFIDF+keywords)
    try:
        pool = RVF.retrieve_initial_pool_with_keywords(query, keywords, cfg.collection_name, pool_size=pool_n)
    except Exception as e:
        log.warning(f"[WARN] initial pool failed: {e}")
        pool = []
    if not pool:
        log.warning("[WARN] No candidates after initial stage; returning empty round.")
        return []

    # STEP 1.5: Deduplicate
    pool = RVF.deduplicate_candidates(pool, similarity_threshold=0.9)

    # STEP 2: Dense (bi‑encoder)
    dense = RVF.dense_rerank_candidates(query, pool, rerank_size=dense_n)

    # STEP 3: Neural rerank (cross‑encoder)
    neural = RVF.neural_rerank_candidates(query, dense, rerank_size=neural_n)

    # STEP 4: Light context features (RVF)
    enhanced = RVF.enhance_with_basic_context_features(neural, query)

    # STEP 4.5: Enrich with full metadata from Chroma
    enhanced = enrich_with_metadata(enhanced, accessor)

    # FINAL: No LLM extraction; output chunks directly
    out: List[Dict] = []
    for c in enhanced:
        cc = dict(c)
        # Choose final score (keep RVF's enhanced score if present; else neural score)
        cc["ultimate_score"] = cc.get("final_enhanced_score", cc.get("neural_score", 0.0))
        out.append(cc)

    # Dedup by uid across rounds & trim
    filtered: List[Dict] = []
    for c in sorted(out, key=lambda x: x.get("ultimate_score", 0.0), reverse=True):
        uid = _candidate_uid(c)
        if uid in already_seen:
            continue
        already_seen.add(uid)
        filtered.append(c)
    return filtered[: cfg.top_k_per_round]

# -------------------------
# Agentic controller
# -------------------------

def agentic_search(query: str, keywords: List[str], cfg: AgentConfig) -> Dict:
    log.info(f"[AGENT] Query: {query}")
    log.info(f"[AGENT] Keywords: {keywords}")

    # Ensure Retrieval_vf uses the same Chroma path
    try:
        RVF.client_chroma = chromadb.PersistentClient(path=cfg.chroma_path)
        log.info(f"[AGENT] Bound Retrieval_vf to Chroma at: {cfg.chroma_path}")
    except Exception as e:
        log.warning(f"[WARN] Could not override Retrieval_vf client: {e}")

    accessor = ChromaAccessor(cfg.chroma_path, cfg.collection_name)

    seen_uids: Set[str] = set()
    all_results: List[Dict] = []
    covered_keywords: Set[str] = set()

    round_queries: List[str] = [query]
    round_idx = 0

    while round_idx < cfg.max_iters and round_queries:
        q = round_queries.pop(0)
        log.info(f"[AGENT] === Round {round_idx+1}/{cfg.max_iters}: {q} ===")

        round_results = single_round(q, keywords, cfg, accessor, already_seen=seen_uids, round_idx=round_idx)
        all_results.extend(round_results)

        # JSONL trace (no extracted sentences fields)
        if cfg.out_jsonl and round_results:
            with open(cfg.out_jsonl, "a", encoding="utf-8") as jf:
                for c in round_results:
                    md = c.get("metadata") or {}
                    jf.write(json.dumps({
                        "round": round_idx + 1,
                        "query": q,
                        "uid": _candidate_uid(c),
                        "filename": md.get("filename"),
                        "filepath": md.get("filepath"),
                        "page": md.get("page"),
                        "chunk_index": md.get("chunk_index"),
                        "chunk_id": md.get("chunk_id"),
                        "score": c.get("ultimate_score"),
                    }) + "\n")

        # Coverage & stopping
        round_cov, _ = measure_round_coverage(round_results, q, keywords)
        prev = set(covered_keywords)
        covered_keywords.update(round_cov)
        new_gain = len(covered_keywords) - len(prev)
        log.info(f"[AGENT] New results: {len(round_results)} | New keyword coverage +{new_gain} (covered {len(covered_keywords)}/{len(keywords)})")

        ratio = (len(covered_keywords) / max(1, len(keywords))) if keywords else 1.0
        if ratio >= cfg.coverage_stop_ratio:
            log.info(f"[AGENT] Stop: coverage {ratio:.2%} ≥ {cfg.coverage_stop_ratio:.0%}")
            break
        if len(round_results) < cfg.min_new_evidence and new_gain < cfg.min_keyword_gain:
            log.info(f"[AGENT] Stop: low gain (new={len(round_results)} < {cfg.min_new_evidence} & gain={new_gain} < {cfg.min_keyword_gain})")
            break

        missing = [kw for kw in keywords if kw not in covered_keywords]
        refinements = propose_refined_queries(q, round_results, missing, cfg)
        refinements = [r for r in refinements if r.lower() != q.lower()]
        round_queries.extend(refinements)

        round_idx += 1
        time.sleep(cfg.sleep_between_rounds_sec)

    # Fuse by best score per uid
    def _rank_key(x: Dict) -> float:
        return float(x.get("ultimate_score", 0.0))

    best_by_uid: Dict[str, Dict] = {}
    for c in sorted(all_results, key=_rank_key, reverse=True):
        uid = _candidate_uid(c)
        if uid not in best_by_uid:
            best_by_uid[uid] = c
    fused = list(best_by_uid.values())
    fused.sort(key=_rank_key, reverse=True)

    return {
        "query": query,
        "keywords": keywords,
        "iterations": round_idx + 1,
        "covered_keywords": sorted(list(covered_keywords)),
        "coverage_ratio": (len(covered_keywords) / max(1, len(keywords))) if keywords else 1.0,
        "total_unique_results": len(fused),
        "results": fused,
    }

# -------------------------
# CSV writer (no LLM extraction columns)
# -------------------------

def write_csv(results: Dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = [
        "query","keywords","rounds","coverage_ratio","covered_keywords_count",
        "chunk_id","chunk_index","page","filename","filepath",
        "bm25_score","tfidf_score","combined_lexical_score",
        "semantic_score","dense_score","neural_score","neural_enhanced_score",
        "context_score","final_enhanced_score","ultimate_score",
        "chunk_text_preview"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for c in results["results"]:
            md = c.get("metadata") or {}
            filepath = md.get("filepath")
            filename = md.get("filename") or (os.path.basename(filepath) if filepath else None) or c.get("document")
            page     = md.get("page") if md.get("page") is not None else c.get("page")
            w.writerow([
                results["query"],
                "; ".join(results["keywords"]),
                results["iterations"],
                round(float(results["coverage_ratio"]), 4),
                len(results.get("covered_keywords", [])),
                md.get("chunk_id") or c.get("chunk_id"),
                md.get("chunk_index"),
                page,
                filename,
                filepath,
                round(float(c.get("bm25_score",0.0)),4),
                round(float(c.get("tfidf_score",0.0)),4),
                round(float(c.get("combined_score", c.get("combined_lexical_score", 0.0))),4),
                round(float(c.get("semantic_score",0.0)),4),
                round(float(c.get("dense_score",0.0)),4),
                round(float(c.get("neural_score",0.0)),4),
                round(float(c.get("neural_enhanced_score", c.get("final_enhanced_score",0.0))),4),
                round(float(c.get("context_score",0.0)),4),
                round(float(c.get("final_enhanced_score",0.0)),4),
                round(float(c.get("ultimate_score",0.0)),4),
                (md.get("chunk_text") or c.get("chunk") or "")[:1200],
            ])

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Agentic multi‑round retrieval (stops at neural rerank; 1/3 sizes on rounds 2+)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--query", required=True, help="User question")
    ap.add_argument("--keywords_excel", default="novartis_keywords_by_question.xlsx", help="Excel with Question/Keywords columns")
    ap.add_argument("--collection", default="novartis_collection", help="Chroma collection name")
    ap.add_argument("--chroma_path", default="chromadb", help="Chroma persistent path")
    ap.add_argument("--csv", default="agentic_results.csv", help="Output CSV path")
    ap.add_argument("--jsonl", default=None, help="Optional JSONL trace path")
    ap.add_argument("--max_iters", type=int, default=3, help="Maximum agentic rounds")

    args, unknown = ap.parse_known_args()
    if unknown:
        log.warning(f"[WARN] Ignoring unknown args: {unknown}")

    cfg = AgentConfig(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        max_iters=args.max_iters,
        out_csv=args.csv,
        out_jsonl=args.jsonl,
    )

    # Bind Chroma path for RVF
    try:
        RVF.client_chroma = chromadb.PersistentClient(path=cfg.chroma_path)
    except Exception as e:
        log.warning(f"[WARN] Could not override Retrieval_vf client: {e}")

    # Keywords
    keywords_dict = RVF.load_keywords_from_excel(args.keywords_excel)
    kw = RVF.find_matching_keywords(args.query, keywords_dict)

    report = agentic_search(args.query, kw, cfg)

    log.info(json.dumps({
        "iterations": report["iterations"],
        "coverage_ratio": report["coverage_ratio"],
        "covered_keywords_count": len(report["covered_keywords"]),
        "total_unique_results": report["total_unique_results"],
    }, indent=2))

    write_csv(report, cfg.out_csv)
    log.info(f"[DONE] Saved CSV: {cfg.out_csv}")
