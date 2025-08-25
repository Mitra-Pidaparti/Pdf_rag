#!/usr/bin/env python3
"""
agentic_retrieve_doccheck_lastllm_v2.py

Goal (as requested):
1) For each input document with optional '### Page N ###' markers, summarize each page and score relevance to the user query.
2) If a page is above the relevance threshold (and optional keyword-hit rule), fetch ONLY that page's chunks from ChromaDB.
3) Filter those chunks to keep only the ones relevant to the query (with assistance from provided keywords) using an LLM when available, otherwise a local similarity/keyword heuristic.
4) Write ONLY the kept chunks to the CSV output (one row per kept chunk). Also write a JSONL record per processed page for debugging/traceability.
5) Additionally, write a human-readable TXT file that concatenates the **final kept chunks** grouped by Source and Page.

This script is compatible with a ChromaDB store created by your Create_db_vf.py, which:
- stores chunk text in metadata under 'chunk_text',
- includes 'filename', 'filepath', and integer 'page' fields,
- uses SentenceTransformer('BAAI/bge-base-en-v1.5') at ingestion time.

Notes:
- Chroma where-clauses follow modern validation rules: a single top-level operator OR a single predicate; page int/str handled with inner $or.
- Semantic fallback uses the same BGE model for query embeddings to avoid MiniLM downloads.
- CSV contains ONLY the filtered chunks (no page rows). Columns: Source, Page, ChunkID, ChunkIndex, Score, ChunkText.
- TXT contains the **exact final chunks chosen after filtering**, grouped by Source and Page.

Requirements:
  pip install chromadb sentence-transformers openai python-dotenv pandas tqdm
  export/set OPENAI_API_KEY for LLM-assisted steps (optional but recommended)

Example (Windows CMD one-liner):
  python agentic_retrieve_doccheck_lastllm_v2.py ^
    --query "What did the report say about ESG targets?" ^
    --docs "C:\\Users\\mitra\\Novartis_docs\\novartis_integrated_report_2024_cleaned_ocr2.txt" ^
    --collection novartis_collection ^
    --chroma_path "C:\\Users\\mitra\\chromadb" ^
    --keywords ESG sustainability emissions targets ^
    --filter_chunks_with_llm ^
    --out_csv "C:\\Users\\mitra\\agentic_chunks.csv"
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

import pandas as pd
from tqdm import tqdm

import chromadb
from dotenv import load_dotenv

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("doccheck")

# ---------------- Env ----------------
load_dotenv()

# ---------------- LLM client (optional) ----------------
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

# ---------------- Embedding model (optional but recommended) ----------------
# We try to load the same model used during ingestion to align similarity.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _BGE_MODEL_NAME = os.getenv("BGE_MODEL", "BAAI/bge-base-en-v1.5")
    BGE_MODEL: Optional[SentenceTransformer] = SentenceTransformer(_BGE_MODEL_NAME)
    log.info(f"Loaded embedding model: {_BGE_MODEL_NAME}")
except Exception as e:
    BGE_MODEL = None
    log.warning(f"SentenceTransformer not available ({e}). Will fallback to keyword-only / Chroma text queries.")

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
    for i in range(len(parts) - 1):
        pno, start = parts[i]
        _, end = parts[i + 1]
        out.append((pno, text[start:end].strip()))
    return out

# ---------------- Utilities ----------------

def dedupe_list(xs: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        k = x.strip()
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def listify_embedding(vec) -> List[List[float]]:
    """Ensure embeddings are a list-of-list (Chroma expects batch)."""
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if isinstance(vec, list) and vec and isinstance(vec[0], float):
        vec = [vec]
    return vec

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

CHUNK_FILTER_PROMPT = """You are selecting verbatim chunks that best answer the user's Query.
Use the given Keywords as hints. Return ONLY JSON with:
  - "keep": list[int] of indices to keep (from the provided list)

Query:
{query}

Keywords:
{keywords}

Chunks:
{indexed_chunks}
"""

# ---------------- OpenAI helper ----------------

def call_openai_json(model: str, prompt: str,) -> Dict[str, Any]:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        #temperature=temperature,
        response_format={"type": "json_object"},
        #max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception as e:
        log.warning(f"OpenAI JSON parse failed: {e}. Raw: {content[:200]}...")
        return {}

# ---------------- Chroma helpers ----------------

def _collect_from_get_response(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    ids = res.get("ids", []) or []
    mds = res.get("metadatas", []) or []
    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        md = mds[i] or {}
        out.append({
            "id": ids[i],
            "chunk": md.get("chunk_text") or md.get("chunk") or "",
            "metadata": md,
        })
    return out


def get_page_chunks_by_metadata_page(
    collection,
    page_number: int,
    filename_field: Optional[str],
    source_value: Optional[str],
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Exact page match via metadata['page'] AND same file.
    Uses a single get() with $or for page (int/str). Wrap in $and only when 2+ predicates.
    """
    preds: List[Dict[str, Any]] = []
    page_or = {"$or": [
        {"page": {"$eq": page_number}},
        {"page": {"$eq": str(page_number)}}
    ]}
    preds.append(page_or)
    if filename_field and source_value:
        preds.append({filename_field: {"$eq": source_value}})

    where_clause: Dict[str, Any]
    if len(preds) == 1:
        where_clause = preds[0]
    else:
        where_clause = {"$and": preds}

    try:
        res = collection.get(where=where_clause, limit=limit, include=["metadatas"])
        return _collect_from_get_response(res)
    except Exception as e:
        log.warning(f"Chroma get() failed for page {page_number}: {e}")
        return []


def semantic_chunks_for_page_fallback(
    collection,
    page_text: str,
    filename_field: Optional[str],
    source_value: Optional[str],
    n: int = 20,
) -> List[Dict[str, Any]]:
    """Semantic fallback constrained by same file (if provided).
    Uses BGE query embeddings when available; else lets Chroma embed from text.
    """
    where: Optional[Dict[str, Any]] = None
    if filename_field and source_value:
        # stick to simple equality to avoid validator edge-cases
        where = {filename_field: source_value}

    try:
        if BGE_MODEL is not None:
            query_emb = BGE_MODEL.encode([page_text])
            query_emb = listify_embedding(query_emb)
            res = collection.query(
                query_embeddings=query_emb,
                n_results=n,
                where=where,
                include=["metadatas", "distances"],
            )
        else:
            res = collection.query(
                query_texts=[page_text],
                n_results=n,
                where=where,
                include=["metadatas", "distances"],
            )

        ids = (res.get("ids") or [[]])[0]
        mds = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            md = mds[i] or {}
            out.append({
                "id": ids[i],
                "chunk": md.get("chunk_text") or md.get("chunk") or "",
                "metadata": md,
                "distance": dists[i] if i < len(dists) else None,
            })
        return out
    except Exception as e:
        log.warning(f"Chroma query() fallback failed: {e}")
        return []

# ---------------- Chunk filtering ----------------

def keyword_prefilter(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    t = text.lower()
    for k in keywords:
        k2 = (k or "").strip().lower()
        if not k2:
            continue
        if k2 in t:
            return True
    return False


def score_chunks_locally(query: str, keywords: List[str], chunks: List[str]) -> List[Tuple[int, float]]:
    """Return list of (index, score). If BGE available, cosine sim with query embedding.
    Otherwise, fall back to keyword hit count + token overlap heuristic.
    """
    scores: List[Tuple[int, float]] = []
    if BGE_MODEL is not None:
        try:
            q_emb = BGE_MODEL.encode([query])
            q_emb = q_emb[0] if hasattr(q_emb, "__getitem__") else q_emb
            # normalize
            if hasattr(q_emb, "tolist"):
                import numpy as np
                q = q_emb
                q = q / (np.linalg.norm(q) + 1e-8)
            else:
                q = q_emb
            # embed chunks in small batches
            batch: List[str] = []
            idxs: List[int] = []
            import numpy as np
            def flush():
                if not batch:
                    return
                c_emb = BGE_MODEL.encode(batch)
                # normalize
                if hasattr(c_emb, "shape"):
                    c = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-8)
                else:
                    c = c_emb
                sims = (c @ q).tolist() if hasattr(c, "__matmul__") else [0.0]*len(batch)
                for ii, s in zip(idxs, sims):
                    scores.append((ii, float(s)))
                batch.clear(); idxs.clear()
            for i, ch in enumerate(chunks):
                batch.append(ch)
                idxs.append(i)
                if len(batch) >= 32:
                    flush()
            flush()
            return sorted(scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            log.warning(f"BGE scoring failed, falling back to keyword heuristic: {e}")

    # keyword/token heuristic
    kws = [k.lower() for k in keywords if (k or '').strip()]
    q_tokens = set((query.lower()).split())
    for i, ch in enumerate(chunks):
        t = ch.lower()
        kw_hits = sum(1 for k in kws if k in t)
        overlap = len(q_tokens.intersection(set(t.split())))
        score = kw_hits * 2 + overlap * 0.05
        scores.append((i, float(score)))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def llm_filter_chunks(query: str, keywords: List[str], chunks: List[str], model: str, max_prompt_chars: int = 12000) -> List[int]:
    """Ask the LLM to choose chunk indices. Returns list of indices to keep.
    Falls back to local scoring if OpenAI is unavailable or errors.
    """
    if not _HAS_OPENAI:
        log.info("OpenAI not available; skipping LLM filter.")
        return [i for i, _ in score_chunks_locally(query, keywords, chunks)]

    # Compose truncated presentation to avoid over-token prompts
    indexed_lines: List[str] = []
    running = 0
    for i, ch in enumerate(chunks):
        snip = ch[:800]
        line = f"[{i}] {snip}"
        if running + len(line) > max_prompt_chars:
            break
        indexed_lines.append(line)
        running += len(line)
    prompt = CHUNK_FILTER_PROMPT.format(
        query=query,
        keywords=", ".join(keywords or []),
        indexed_chunks="\n".join(indexed_lines),
    )
    try:
        j = call_openai_json(model=model, prompt=prompt)
        ks = j.get("keep") or []
        idxs = []
        for x in ks:
            try:
                idxs.append(int(x))
            except Exception:
                continue
        if idxs:
            return sorted(set(i for i in idxs if 0 <= i < len(chunks)))
    except Exception as e:
        log.warning(f"LLM chunk filter failed: {e}")

    # fallback to local
    return [i for i, _ in score_chunks_locally(query, keywords, chunks)]

# ---------------- Pipeline ----------------
@dataclass
class PageGate:
    page: int
    summary: str
    relevance_score: float
    matched_keywords: List[str]
    missing_aspects: List[str]
    suggested_subqueries: List[str]


def process_documents(
    query: str,
    docs: List[Path],
    collection_name: str,
    chroma_path: str,
    filename_field: str = "filename",
    keywords: Optional[List[str]] = None,
    model: str = "gpt-5-mini",
    filter_chunks_with_llm: bool = True,
    relevance_threshold: float = 0.65,
    require_keyword_hit: bool = False,
    chunk_sim_threshold: float = 0.6,
    max_chunks_per_page: int = 20,
    out_csv: str = "agentic_chunks.csv",
    out_jsonl: str = "agentic_pages.jsonl",
    out_text: str = "agentic_chunks.txt",
) -> None:
    """Process docs and write ONLY kept chunks to CSV; write per-page JSONL for trace."""
    # Connect to Chroma
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)

    csv_rows: List[Dict[str, Any]] = []
    jsonl_f = open(out_jsonl, "w", encoding="utf-8")
    text_lines: List[str] = []

    supplied_keywords = keywords or []

    for doc_path in docs:
        text = Path(doc_path).read_text(encoding="utf-8", errors="ignore")
        pages = split_pages(text)
        log.info(f"Loaded {doc_path} with {len(pages)} page(s)")
        source_value = Path(doc_path).name

        for pno, ptext in tqdm(pages, desc=f"Pages in {doc_path}"):
            # 1) summarize & gate
            prompt = SUMMARY_AND_GATE_PROMPT.format(
                query=query,
                keywords=", ".join(supplied_keywords) if supplied_keywords else "",
                page_text=ptext[:12000],
            )
            j: Dict[str, Any] = {}
            try:
                j = call_openai_json(model=model, prompt=prompt)
            except Exception as e:
                log.warning(f"LLM call failed on page {pno}: {e}")

            gate = PageGate(
                page=pno,
                summary=j.get("summary", ""),
                relevance_score=float(j.get("relevance_score", 0.0) or 0.0),
                matched_keywords=dedupe_list(j.get("matched_keywords", []) or []),
                missing_aspects=dedupe_list(j.get("missing_aspects", []) or []),
                suggested_subqueries=dedupe_list(j.get("suggested_subqueries", []) or []),
            )

            keep_page = gate.relevance_score >= relevance_threshold
            if require_keyword_hit and keep_page:
                keep_page = any(k.lower() in ptext.lower() for k in supplied_keywords if k)

            kept_chunks: List[Dict[str, Any]] = []

            if keep_page:
                # 2) pull page chunks by exact metadata match; fallback semantic if none
                page_recs = get_page_chunks_by_metadata_page(
                    collection=collection,
                    page_number=pno,
                    filename_field=filename_field,
                    source_value=source_value,
                    limit=500,
                )
                if not page_recs:
                    page_recs = semantic_chunks_for_page_fallback(
                        collection=collection,
                        page_text=ptext,
                        filename_field=filename_field,
                        source_value=source_value,
                        n=50,
                    )

                # 3) choose only relevant chunks (keywords assist)
                chunk_texts = [r.get("chunk") or "" for r in page_recs]
                # keyword prefilter
                pre_idx = [i for i, t in enumerate(chunk_texts) if keyword_prefilter(t, supplied_keywords)]
                if not pre_idx:  # if nothing passes keyword prefilter, allow all
                    pre_idx = list(range(len(chunk_texts)))

                candidate_texts = [chunk_texts[i] for i in pre_idx]
                if filter_chunks_with_llm:
                    keep_local_idx = llm_filter_chunks(query, supplied_keywords, candidate_texts, model)
                else:
                    keep_local_idx = [i for i, _ in score_chunks_locally(query, supplied_keywords, candidate_texts)]

                # apply optional similarity threshold when using local scoring (BGE or heuristic)
                if not filter_chunks_with_llm and BGE_MODEL is not None:
                    ranked = score_chunks_locally(query, supplied_keywords, candidate_texts)
                    keep_local_idx = [i for i, s in ranked if s >= chunk_sim_threshold]
                    if not keep_local_idx and ranked:
                        keep_local_idx = [ranked[0][0]]  # at least top-1

                # map back to original indices and cap
                keep_abs = [pre_idx[i] for i in keep_local_idx if 0 <= i < len(pre_idx)]
                if max_chunks_per_page > 0:
                    keep_abs = keep_abs[:max_chunks_per_page]

                kept_chunks = [page_recs[i] for i in keep_abs]

            # write JSONL trace per page
            jsonl_f.write(json.dumps({
                "source": source_value,
                "page": pno,
                "summary": gate.summary,
                "relevance_score": gate.relevance_score,
                "matched_keywords": gate.matched_keywords,
                "missing_aspects": gate.missing_aspects,
                "suggested_subqueries": gate.suggested_subqueries,
                "kept_chunks": [
                    {"id": c.get("metadata", {}).get("chunk_id") or c.get("id"), "text": (c.get("chunk") or "")}
                    for c in kept_chunks
                ],
            }, ensure_ascii=False) + "")

            # append ONLY kept chunks to CSV rows
            for c in kept_chunks:
                md = c.get("metadata", {}) or {}
                chunk_id = md.get("chunk_id") or c.get("id")
                chunk_text = (c.get("chunk") or "")
                csv_rows.append({
                    "Source": source_value,
                    "Page": md.get("page", pno),
                    "ChunkID": chunk_id,
                    "ChunkIndex": md.get("chunk_index"),
                    "Score": gate.relevance_score,  # page-level score for reference
                    "ChunkText": chunk_text,
                })
                # also append to TXT aggregation
                text_lines.append(f"===== {source_value} | Page {md.get('page', pno)} | Chunk {chunk_id} =====")
                text_lines.append(chunk_text)
                text_lines.append("")

    jsonl_f.close()
    # Write ONLY the kept chunks
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_text, "w", encoding="utf-8") as tf:
        tf.write("".join(text_lines))
    log.info(f"Done. Wrote kept chunks CSV: {out_csv}, page trace JSONL: {out_jsonl}, and TXT of final chunks: {out_text}")

# ---------------- CLI ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic page summarization + relevant chunk extraction (CSV-only kept chunks)")
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--docs", nargs="+", required=True, help="One or more .txt/.md files (with '### Page N ###' markers preferred)")
    ap.add_argument("--collection", required=True, help="Chroma collection name to pull verbatim chunks from")
    ap.add_argument("--chroma_path", default="chromadb", help="Path to Chroma persistent store")
    ap.add_argument("--filename_field", default="filename", help="Metadata field name that stores the source filename")
    ap.add_argument("--keywords", nargs="*", default=None, help="Optional keyword hints for summarization & chunk filtering")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI chat model for JSON outputs")
    ap.add_argument("--filter_chunks_with_llm", action="store_true", help="Use LLM to choose relevant chunks (else local scoring)")
    ap.add_argument("--relevance_threshold", type=float, default=0.65, help="Keep pages with score >= threshold")
    ap.add_argument("--require_keyword_hit", action="store_true", help="Additionally require keyword literal hit in page text to keep the page")
    ap.add_argument("--chunk_sim_threshold", type=float, default=0.6, help="Similarity threshold for local scoring when not using LLM")
    ap.add_argument("--max_chunks_per_page", type=int, default=20, help="Cap number of kept chunks per page (0 = no cap)")
    ap.add_argument("--out_csv", default="agentic_chunks.csv", help="Output CSV path (only kept chunks)")
    ap.add_argument("--out_jsonl", default="agentic_pages.jsonl", help="Output JSONL path for page traces")
    ap.add_argument("--out_text", default="agentic_chunks.txt", help="Output TXT path containing final kept chunks, grouped by page")

    args = ap.parse_args()

    process_documents(
        query=args.query,
        docs=[Path(x) for x in args.docs],
        collection_name=args.collection,
        chroma_path=args.chroma_path,
        filename_field=args.filename_field,
        keywords=args.keywords,
        model=args.model,
        filter_chunks_with_llm=args.filter_chunks_with_llm,
        relevance_threshold=args.relevance_threshold,
        require_keyword_hit=args.require_keyword_hit,
        chunk_sim_threshold=args.chunk_sim_threshold,
        max_chunks_per_page=args.max_chunks_per_page,
        out_csv=args.out_csv,
        out_jsonl=args.out_jsonl,
        out_text=args.out_text,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(130)



#Usage
#python agentic_retrieve_doccheck_lastllm.py --query "To what extent are the organization's products and services information-based or information-enabled?" --docs "Novartis_docs\novartis_integrated_report_2024_cleaned_ocr2.txt" Novartis_docs\novartis_annual_report_2024_cleaned_ocr2.txt --collection novartis_collection --chroma_path "chromadb" --keywords --filter_chunks_with_llm --out_csv "agentic_chunks_Q2.csv" --out_text "agentic_chunks_Q2.txt"