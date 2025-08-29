#!/usr/bin/env python3
"""
agentic_retrieve_doccheck_lastllm.py — latency-optimized

Changes vs previous:
1) Chunk selection no longer uses an LLM; we use Chroma distances + keyword assist.
2) We rely on distances returned by Chroma (no local re-embedding of chunks).
3) Query embedding is computed ONCE (if BGE available) and reused for all pages.
4) Page processing is parallelized (ThreadPoolExecutor).
5) JSONL tracing is optional — only written if --out_jsonl is provided (non-empty).

CLI remains compatible for orchestrator:
- All existing flags are accepted. `--filter_chunks_with_llm` is ignored with a warning.
- `--out_jsonl` is now optional (default empty -> no JSONL written).

Requires:
  pip install chromadb sentence-transformers openai python-dotenv pandas tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import chromadb

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("doccheck")

# ---------------- Env ----------------
load_dotenv()

# ---------------- Optional OpenAI (page-gate LLM) ----------------
try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


def get_openai_client() -> "OpenAI":
    if not _HAS_OPENAI:
        raise RuntimeError("OpenAI package not installed. `pip install openai` and set OPENAI_API_KEY.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

# ---------------- Optional SentenceTransformer (for ONE query embedding) ----------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _BGE_MODEL_NAME = os.getenv("BGE_MODEL", "BAAI/bge-base-en-v1.5")
    BGE_MODEL: Optional[SentenceTransformer] = SentenceTransformer(_BGE_MODEL_NAME)
    log.info(f"Loaded embedding model: {_BGE_MODEL_NAME}")
except Exception as e:
    BGE_MODEL = None
    log.warning(f"SentenceTransformer not available ({e}). Will rely on Chroma to embed query_texts server-side.")

# ---------------- Page splitting ----------------
PAGE_RX = re.compile(r"###\s*Page\s+(\d+)\s*###", re.IGNORECASE)


def split_pages(text: str) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text). If no markers, treat whole text as page 1."""
    parts: List[Tuple[int, int]] = []
    for m in PAGE_RX.finditer(text):
        pno = int(m.group(1))
        parts.append((pno, m.start()))
    if not parts:
        return [(1, text)]
    parts.append((10**9, len(text)))  # sentinel end
    out: List[Tuple[int, str]] = []
    for i in range(len(parts)-1):
        pno, start = parts[i]
        _, end = parts[i + 1]
        out.append((pno, text[start:end].strip()))
    return out

# ---------------- Utilities ----------------

def dedupe_list(xs: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        k = (x or "").strip()
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def keyword_prefilter(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    t = text.lower()
    for k in keywords:
        k2 = (k or "").strip().lower()
        if k2 and k2 in t:
            return True
    return False


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    if not keywords:
        return 0
    t = text.lower()
    hits = 0
    for k in keywords:
        k2 = (k or "").strip().lower()
        if k2 and k2 in t:
            hits += 1
    return hits

# ---------------- Prompts ----------------
SUMMARY_AND_GATE_PROMPT = """You are an expert page triager and summarizer for enterprise RAG.
Given:
- a user Query,
- an optional list of Keywords (hints),
- and the full text of ONE document page,

You must produce a compact JSON object with:
  - "summary": <= 150 words, objective, high-signal, faithful to the page
  - "relevance_score": float in [0,1]; 0=completely irrelevant, 1=related to the Query
  - "matched_keywords": list[str] of any provided keywords present or obviously implied
  - "missing_aspects": list[str], sub-points the page hints at but doesn’t fully address for the Query
  - "suggested_subqueries": list[str], concrete follow-ups to capture missed angles

Rules:
- Prioritize the Query over keywords. Keywords are hints, not the target.
- Count a page as relevant even if tangential but informative for the Query.
- Be strict JSON. No commentary.

Return ONLY JSON.

Query:
{query}

Keywords (may be empty):
{keywords}

Page Text (may be long; focus on signal):
{page_text}
"""

# ---------------- OpenAI helper (page gate only) ----------------

def call_openai_json(model: str, prompt: str) -> Dict[str, Any]:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception as e:
        log.warning(f"OpenAI JSON parse failed: {e}. Raw: {content[:200]}...")
        return {}

# ---------------- Chroma helpers ----------------

def build_page_where(filename_field: Optional[str], source_value: Optional[str], page_number: int) -> Dict[str, Any]:
    # Use $and to satisfy modern Chroma validators; we'll also defensively re-filter by page in code.
    preds: List[Dict[str, Any]] = []
    preds.append({"$or": [{"page": {"$eq": page_number}}, {"page": {"$eq": str(page_number)}}]})
    if filename_field and source_value:
        preds.append({filename_field: {"$eq": source_value}})
    if len(preds) == 1:
        return preds[0]
    return {"$and": preds}


def _collect_query_with_distances(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    ids = (res.get("ids") or [[]])[0]
    mds = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        md = mds[i] or {}
        dist = dists[i] if i < len(dists) else None
        sim = (1.0 - float(dist)) if dist is not None else None
        out.append({
            "id": ids[i],
            "chunk": md.get("chunk_text") or md.get("chunk") or "",
            "metadata": md,
            "distance": dist,
            "similarity": sim,        # <-- add this
        })
    return out



def semantic_rank_chunks_for_page(
    collection,
    query_text: str,
    query_embedding: Optional[List[float]],
    filename_field: Optional[str],
    source_value: Optional[str],
    page_number: int,
    n_results: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retrieve chunks for the given (file, page) ranked by semantic distance to the QUERY,
    using Chroma distances (no local re-embedding). We restrict 'where' by filename (and page),
    and additionally re-filter by page in-code for safety.
    """
    where_clause = build_page_where(filename_field, source_value, page_number)

    try:
        if query_embedding is not None:
            qemb = query_embedding
            if isinstance(qemb, list) and qemb and isinstance(qemb[0], float):
                qemb = [qemb]  # batch of 1
            res = collection.query(
                query_embeddings=qemb,
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "distances"],
            )
        else:
            res = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "distances"],
            )
        items = _collect_query_with_distances(res)
        # Redundant safety: keep only exact same page
        items = [r for r in items if str(r.get("metadata", {}).get("page")) in {str(page_number)}]
        # Sort by distance ascending (smaller is more similar)
        items.sort(key=lambda x: (x.get("distance") if x.get("distance") is not None else 1e9))
        return items
    except Exception as e:
        log.warning(f"Chroma query() failed for page {page_number}: {e}")
        return []

# ---------------- Data structures ----------------
@dataclass
class PageGate:
    page: int
    summary: str
    relevance_score: float
    matched_keywords: List[str]
    missing_aspects: List[str]
    suggested_subqueries: List[str]


# ---------------- Per-page worker ----------------
def process_one_page(
    query: str,
    supplied_keywords: List[str],
    doc_path: Path,
    source_value: str,
    pno: int,
    ptext: str,
    collection,
    filename_field: str,
    model: str,
    relevance_threshold: float,
    require_keyword_hit: bool,
    max_chunks_per_page: int,
    query_embedding_once: Optional[List[float]],
    jsonl_enabled: bool,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[Dict[str, Any]]]:
    """
    Returns: (csv_rows, text_lines, jsonl_record or None)
    """
    # 1) PAGE GATE (LLM) — may be skipped if OpenAI unavailable.
    gate_json: Dict[str, Any] = {}
    if _HAS_OPENAI:
        try:
            # keep page prompt reasonably bounded; 12k chars carried over for compatibility
            prompt = SUMMARY_AND_GATE_PROMPT.format(
                query=query,
                keywords=", ".join(supplied_keywords) if supplied_keywords else "",
                page_text=ptext[:12000],
            )
            gate_json = call_openai_json(model=model, prompt=prompt)
        except Exception as e:
            log.warning(f"LLM page gate failed on {source_value} p{pno}: {e}")
    # Fallback if LLM missing or failed
    if not gate_json:
        # simple heuristic: keep if any keyword literal hit
        keep_kw = any((k and k.lower() in ptext.lower()) for k in supplied_keywords)
        gate = PageGate(
            page=pno, summary="", relevance_score=1.0 if keep_kw else 0.0,
            matched_keywords=[], missing_aspects=[], suggested_subqueries=[]
        )
    else:
        gate = PageGate(
            page=pno,
            summary=gate_json.get("summary", ""),
            relevance_score=float(gate_json.get("relevance_score", 0.0) or 0.0),
            matched_keywords=dedupe_list(gate_json.get("matched_keywords", []) or []),
            missing_aspects=dedupe_list(gate_json.get("missing_aspects", []) or []),
            suggested_subqueries=dedupe_list(gate_json.get("suggested_subqueries", []) or []),
        )

    keep_page = gate.relevance_score >= relevance_threshold
    if require_keyword_hit and keep_page:
        keep_page = any(k and (k.lower() in ptext.lower()) for k in supplied_keywords)

    csv_rows: List[Dict[str, Any]] = []
    text_lines: List[str] = []
    jsonl_record: Optional[Dict[str, Any]] = None

    kept_chunks: List[Dict[str, Any]] = []
    if keep_page:
        # 2) SEMANTIC + KEYWORD ASSISTED CHUNK CHOOSING via Chroma distances
        #    We query for this (file, page) using the ONE cached query embedding (or server-side query_texts).
        n_res = max(50, max_chunks_per_page * 10) if max_chunks_per_page > 0 else 100
        ranked = semantic_rank_chunks_for_page(
            collection=collection,
            query_text=query,
            query_embedding=query_embedding_once,
            filename_field=filename_field,
            source_value=source_value,
            page_number=pno,
            n_results=n_res,
        )

        # Keyword assist: lightly boost items with hits by subtracting a small bonus from distance
        def adjusted_similarity(item: Dict[str, Any]) -> float:
            sim = item.get("similarity")
            if sim is None:
                return -1.0  # treat unknown as worst
            hits = count_keyword_hits(item.get("chunk") or "", supplied_keywords)
            return float(sim) + 0.05 * hits  # small boost per keyword hit

        ranked.sort(key=adjusted_similarity, reverse=True)  # higher similarity is better


        if max_chunks_per_page > 0:
            ranked = ranked[:max_chunks_per_page]

        kept_chunks = ranked

    # 3) Build outputs
    if jsonl_enabled:
        jsonl_record = {
            "source": source_value,
            "page": pno,
            "summary": gate.summary,
            "relevance_score": gate.relevance_score,
            "matched_keywords": gate.matched_keywords,
            "missing_aspects": gate.missing_aspects,
            "suggested_subqueries": gate.suggested_subqueries,
            "kept_chunks": [
                {"id": (c.get("metadata", {}) or {}).get("chunk_id") or c.get("id"), "text": (c.get("chunk") or "")}
                for c in kept_chunks
            ],
        }

    for c in kept_chunks:
        md = c.get("metadata", {}) or {}
        chunk_id = md.get("chunk_id") or c.get("id")
        chunk_text = (c.get("chunk") or "")
        csv_rows.append({
            "Source": source_value,
            "Page": md.get("page", pno),
            "ChunkID": chunk_id,
            "ChunkIndex": md.get("chunk_index"),
            # keep page-level score for reference
            "Score": gate.relevance_score,
            "ChunkText": chunk_text,
        })
        sim_str = ""
        if c.get("similarity") is not None:
            try:
                sim_str = f" | sim {c['similarity']:.3f}"
            except Exception:
                pass
        text_lines.append(f"===== {source_value} | Page {md.get('page', pno)} | Chunk {chunk_id}{sim_str} =====")    
        text_lines.append(chunk_text)
        text_lines.append("")

    return csv_rows, text_lines, jsonl_record


# ---------------- Pipeline ----------------
def process_documents(
    query: str,
    docs: List[Path],
    collection_name: str,
    chroma_path: str,
    filename_field: str = "filename",
    keywords: Optional[List[str]] = None,
    model: str = "gpt-5-mini",
    filter_chunks_with_llm: bool = False,  # accepted for compatibility; ignored
    relevance_threshold: float = 0.65,
    require_keyword_hit: bool = False,
    chunk_sim_threshold: float = 0.6,      # accepted for compatibility; not used (we rank by distance)
    max_chunks_per_page: int = 20,
    out_csv: str = "agentic_chunks.csv",
    out_jsonl: str = "",                   # empty -> JSONL disabled
    out_text: str = "agentic_chunks.txt",
    workers: int = 8,
) -> None:
    """Process docs and write ONLY kept chunks to CSV; optional per-page JSONL trace; TXT aggregation."""
    if filter_chunks_with_llm:
        log.warning("--filter_chunks_with_llm was provided but is ignored in this version (we use semantic+keywords).")

    # Connect to Chroma
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)

    supplied_keywords = keywords or []

    # Cache ONE query embedding (if BGE available), else let Chroma embed via query_texts
    query_embedding_once: Optional[List[float]] = None
    if BGE_MODEL is not None:
        try:
            q_emb = BGE_MODEL.encode([query])
            # unwrap to 1D list of floats
            if hasattr(q_emb, "tolist"):
                q_emb = q_emb.tolist()
            if isinstance(q_emb, list) and q_emb and isinstance(q_emb[0], list):
                q_emb = q_emb[0]
            query_embedding_once = q_emb  # type: ignore
            log.info("Computed query embedding once for reuse.")
        except Exception as e:
            log.warning(f"Failed to compute query embedding; will rely on Chroma's server-side embedding. {e}")
            query_embedding_once = None

    # Prepare outputs
    all_csv_rows: List[Dict[str, Any]] = []
    all_text_lines: List[str] = []
    jsonl_enabled = bool(out_jsonl and out_jsonl.strip())
    jsonl_records: List[Dict[str, Any]] = []

    # Process each document
    for doc_path in docs:
        text = Path(doc_path).read_text(encoding="utf-8", errors="ignore")
        pages = split_pages(text)
        source_value = Path(doc_path).name
        log.info(f"Loaded {doc_path} with {len(pages)} page(s)")

        # Parallelize per-page work
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for pno, ptext in pages:
                futures.append(ex.submit(
                    process_one_page,
                    query,
                    supplied_keywords,
                    doc_path,
                    source_value,
                    pno,
                    ptext,
                    collection,
                    filename_field,
                    model,
                    relevance_threshold,
                    require_keyword_hit,
                    max_chunks_per_page,
                    query_embedding_once,
                    jsonl_enabled,
                ))

            # Consume results with a progress bar
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Pages in {source_value}"):
                csv_rows, text_lines, jsonl_record = f.result()
                if csv_rows:
                    all_csv_rows.extend(csv_rows)
                if text_lines:
                    all_text_lines.extend(text_lines)
                if jsonl_enabled and jsonl_record:
                    jsonl_records.append(jsonl_record)

    # Write outputs
    pd.DataFrame(all_csv_rows).to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_text, "w", encoding="utf-8") as tf:
        tf.write("".join(all_text_lines))
    if jsonl_enabled:
        with open(out_jsonl, "w", encoding="utf-8") as jf:
            for rec in jsonl_records:
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"Done. Wrote kept chunks CSV: {out_csv}"
             f"{', page trace JSONL: ' + out_jsonl if jsonl_enabled else ''}"
             f", and TXT of final chunks: {out_text}")

# ---------------- CLI ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic page summarization + relevant chunk extraction (CSV-only kept chunks)")
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--docs", nargs="+", required=True, help="One or more .txt/.md files (with '### Page N ###' markers preferred)")
    ap.add_argument("--collection", required=True, help="Chroma collection name to pull verbatim chunks from")
    ap.add_argument("--chroma_path", default="chromadb", help="Path to Chroma persistent store")
    ap.add_argument("--filename_field", default="filename", help="Metadata field name that stores the source filename")
    ap.add_argument("--keywords", nargs="*", default=None, help="Optional keyword hints for summarization & chunk filtering")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI chat model for JSON outputs (page gate)")
    ap.add_argument("--filter_chunks_with_llm", action="store_true", help="(Ignored) keep for compatibility")
    ap.add_argument("--relevance_threshold", type=float, default=0.65, help="Keep pages with score >= threshold")
    ap.add_argument("--require_keyword_hit", action="store_true", help="Additionally require keyword literal hit in page text to keep the page")
    ap.add_argument("--chunk_sim_threshold", type=float, default=0.6, help="(Ignored) kept for compatibility")
    ap.add_argument("--max_chunks_per_page", type=int, default=20, help="Cap number of kept chunks per page (0 = no cap)")
    ap.add_argument("--out_csv", default="agentic_chunks.csv", help="Output CSV path (only kept chunks)")
    ap.add_argument("--out_jsonl", default="", help="(Optional) Output JSONL path for page traces; leave empty to skip")
    ap.add_argument("--out_text", default="agentic_chunks.txt", help="Output TXT path containing final kept chunks, grouped by page")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers for page processing")

    args = ap.parse_args()

    process_documents(
        query=args.query,
        docs=[Path(x) for x in args.docs],
        collection_name=args.collection,
        chroma_path=args.chroma_path,
        filename_field=args.filename_field,
        keywords=args.keywords,
        model=args.model,
        filter_chunks_with_llm=args.filter_chunks_with_llm,  # ignored but accepted
        relevance_threshold=args.relevance_threshold,
        require_keyword_hit=args.require_keyword_hit,
        chunk_sim_threshold=args.chunk_sim_threshold,        # ignored
        max_chunks_per_page=args.max_chunks_per_page,
        out_csv=args.out_csv,
        out_jsonl=args.out_jsonl,                            # optional
        out_text=args.out_text,
        workers=args.workers,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(130)


# python agentic_retrieve_doccheck_lastllm_optimized.py --query "What is the focus and nature of the organization's 'Purpose'?" --docs Novartis_docs/novartis_integrated_report_2024_cleaned_ocr2.txt Novartis_docs/novartis_annual_report_2024_cleaned_ocr2.txt --collection novartis_collection --chroma_path chromadb --out_csv results_doccheckQ1.csv --out_text results_doccheckQ1.txt --out_jsonl results_doccheckQ1.jsonl
