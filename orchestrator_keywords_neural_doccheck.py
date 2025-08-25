#!/usr/bin/env python3
"""
Orchestrator: run agentic_retrieve_keywords_lastneural_optimized first, then apply
agentic_retrieve_doccheck_lastllm ONLY on the remaining (non-hit) pages, and finally
merge chunks from both into one CSV/TXT.

Pipeline:
  1) Run last-neural optimized retrieval (stops at cross‑encoder neural rerank; no LLM).
     → Collect (filename → pages used) from its results.
  2) Query Chroma to discover ALL pages per filename; compute remaining pages.
  3) Build temporary filtered documents that contain ONLY those remaining pages
     (by stitching together the desired "### Page N ###" segments).
  4) Run doccheck on these temp docs (page triage + relevant-chunk selection).
  5) Merge: (A) last‑neural chunks + (B) doccheck‑kept chunks → combined outputs.

Assumptions:
- Your ChromaDB was created by Create_db_vf.py and stores chunk text in metadata under 'chunk_text'
  along with 'filename', 'filepath', and integer 'page' fields.
- agentic_retrieve_keywords_lastneural_optimized exposes RVF.load_keywords_from_excel / find_matching_keywords
  and an agentic_search(query, keywords, cfg) with AgentConfig containing chroma_path & collection_name.
- agentic_retrieve_doccheck_lastllm exposes process_documents(...) that writes a CSV of kept chunks.

Install:
  pip install chromadb sentence-transformers pandas python-dotenv tqdm openai

Windows CMD example:
  python orchestrate_lastneural_plus_doccheck.py ^
    --query "What did the reports say about the organization's 'Purpose'?" ^
    --docs "C:\\Users\\mitra\\Novartis_docs\\novartis_integrated_report_2024_cleaned_ocr2.txt" "C:\\Users\\mitra\\Novartis_docs\\novartis_annual_report_2024_cleaned_ocr2.txt" ^
    --keywords_excel novartis_keywords_by_question.xlsx ^
    --collection novartis_collection ^
    --chroma_path chromadb ^
    --model gpt-5-mini ^
    --filter_chunks_with_llm ^
    --out_csv combined_chunks.csv ^
    --out_text combined_chunks.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import chromadb
import pandas as pd

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("orchestrator")

# ---------- Page splitting ----------
PAGE_RX = re.compile(r"###\s*Page\s+(\d+)\s*###", re.IGNORECASE)


def split_pages(text: str) -> List[Tuple[int, str]]:
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

# ---------- Helpers ----------

def as_filename(p: Path) -> str:
    return p.name


def collect_pages_for_files(collection, filenames: List[str], filename_field: str = "filename") -> Dict[str, Set[int]]:
    """Scan Chroma metadata to learn which page numbers exist per filename."""
    pages_by_file: Dict[str, Set[int]] = {fn: set() for fn in filenames}
    for fn in filenames:
        try:
            res = collection.get(where={filename_field: {"$eq": fn}}, include=["metadatas"], limit=200_000)
        except Exception as e:
            log.warning(f"Chroma get failed for {fn}: {e}")
            continue
        mds = res.get("metadatas", []) or []
        for md in mds:
            if not md:
                continue
            p = md.get("page")
            try:
                p_int = int(p)
                pages_by_file[fn].add(p_int)
            except Exception:
                continue
    return pages_by_file


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stitch_filtered_doc(original_path: Path, keep_pages: Set[int]) -> Optional[Path]:
    """Create a temp file that contains only the requested pages from original_path.
    Returns temp file path or None if no pages kept.
    """
    text = original_path.read_text(encoding="utf-8", errors="ignore")
    pages = split_pages(text)
    if not keep_pages:
        return None
    selected: List[str] = []
    for pno, ptext in pages:
        if pno in keep_pages:
            selected.append(f"### Page {pno} ###\n{ptext}\n")
    if not selected:
        return None
    tmpdir = Path(tempfile.mkdtemp(prefix="docfilter_"))
    outp = tmpdir / original_path.name
    outp.write_text("\n".join(selected), encoding="utf-8")
    return outp


def build_rows_from_lastneural(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in report.get("results", []):
        md = c.get("metadata") or {}
        filename = md.get("filename") or (Path(md.get("filepath")).name if md.get("filepath") else c.get("document"))
        page = md.get("page") if md.get("page") is not None else c.get("page")
        try:
            page = int(page)
        except Exception:
            page = None
        rows.append({
            "Source": filename,
            "Page": page,
            "ChunkID": md.get("chunk_id") or c.get("chunk_id") or c.get("id"),
            "ChunkIndex": md.get("chunk_index"),
            "Score": c.get("ultimate_score", c.get("final_enhanced_score", c.get("neural_score", 0.0))),
            "ChunkText": (md.get("chunk_text") or c.get("chunk") or ""),
        })
    return rows


def pages_used_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Set[int]]:
    by_file: Dict[str, Set[int]] = {}
    for r in rows:
        fn = r.get("Source")
        p = r.get("Page")
        if not fn or p is None:
            continue
        by_file.setdefault(fn, set()).add(int(p))
    return by_file


# ---------- Main orchestrator ----------

def orchestrate(
    query: str,
    docs: List[Path],
    keywords_excel: Optional[Path],
    collection_name: str,
    chroma_path: str,
    model: str = "gpt-5-mini",
    filter_chunks_with_llm: bool = True,
    out_csv: Path = Path("combined_chunks.csv"),
    out_text: Path = Path("combined_chunks.txt"),
    keep_intermediate: bool = False,
    filename_field: str = "filename",
    max_iters_lastneural: int = 3,
) -> None:
    # 0) Bind Chroma
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)

    # 1) --- Run last‑neural optimized retrieval ---
    import Retrieval_vf as RVF
    import agentic_retrieve_keywords_lastneural_optimized as LNO

    # Bind RVF to our Chroma path
    try:
        RVF.client_chroma = chromadb.PersistentClient(path=chroma_path)
        log.info(f"Bound Retrieval_vf to Chroma at: {chroma_path}")
    except Exception as e:
        log.warning(f"Could not override Retrieval_vf client: {e}")

    # Load keywords from Excel and match
    if keywords_excel and Path(keywords_excel).exists():
        kw_dict = RVF.load_keywords_from_excel(str(keywords_excel))
        kw_list = RVF.find_matching_keywords(query, kw_dict)
    else:
        kw_list = []

    # Configure and run
    cfg = getattr(LNO, "AgentConfig")(chroma_path=chroma_path, collection_name=collection_name, max_iters=max_iters_lastneural)
    report = LNO.agentic_search(query, kw_list, cfg)

    rows_last = build_rows_from_lastneural(report)
    used_pages_by_file = pages_used_from_rows(rows_last)

    # 2) --- Discover ALL pages per file via Chroma ---
    doc_filenames = [as_filename(p) for p in docs]
    all_pages_by_file = collect_pages_for_files(collection, doc_filenames, filename_field=filename_field)

    # 3) --- Compute remaining pages & build temp docs containing ONLY those pages ---
    temp_docs: List[Path] = []
    for p in docs:
        name = as_filename(p)
        all_pages = all_pages_by_file.get(name, set())
        used_pages = used_pages_by_file.get(name, set())
        remaining = all_pages - used_pages if all_pages else set()
        if not remaining:
            log.info(f"No remaining pages for {name}; skipping")
            continue
        tmp = stitch_filtered_doc(p, remaining)
        if tmp:
            temp_docs.append(tmp)
            log.info(f"Prepared filtered doc for {name}: pages {sorted(list(remaining))[:8]}{'...' if len(remaining)>8 else ''}")

    # 4) --- Run doccheck on filtered docs only ---
    rows_doc: List[Dict[str, Any]] = []
    if temp_docs:
        import agentic_retrieve_doccheck_lastllm as DOC
        tmp_dir = Path(tempfile.mkdtemp(prefix="doccheck_"))
        tmp_csv = tmp_dir / "doccheck_chunks.csv"
        tmp_jsonl = tmp_dir / "doccheck_pages.jsonl"
        tmp_txt = tmp_dir / "doccheck_chunks.txt"

        DOC.process_documents(
            query=query,
            docs=temp_docs,
            collection_name=collection_name,
            chroma_path=chroma_path,
            filename_field=filename_field,
            keywords=kw_list,
            model=model,
            filter_chunks_with_llm=filter_chunks_with_llm,
            relevance_threshold=0.65,
            require_keyword_hit=False,
            chunk_sim_threshold=0.6,
            max_chunks_per_page=20,
            out_csv=str(tmp_csv),
            out_jsonl=str(tmp_jsonl),
            out_text=str(tmp_txt),
        )
        # Load kept chunks from doccheck CSV
        if tmp_csv.exists():
            df = pd.read_csv(tmp_csv)
            for _, r in df.iterrows():
                rows_doc.append({
                    "Source": r.get("Source"),
                    "Page": int(r.get("Page")) if not pd.isna(r.get("Page")) else None,
                    "ChunkID": r.get("ChunkID"),
                    "ChunkIndex": r.get("ChunkIndex"),
                    "Score": r.get("Score", 0.0),
                    "ChunkText": r.get("ChunkText", ""),
                })
        if not keep_intermediate:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
    else:
        log.info("No filtered docs produced; skipping doccheck phase.")

    # 5) --- Merge outputs and write combined CSV/TXT ---
    combined_rows = rows_last + rows_doc

    # Sort by (Source, Page, Score desc)
    def _row_key(r: Dict[str, Any]):
        return (
            r.get("Source") or "",
            r.get("Page") if r.get("Page") is not None else -1,
            -float(r.get("Score") or 0.0),
        )

    combined_rows.sort(key=_row_key)

    # Write CSV
    ensure_dir(Path(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Page", "ChunkID", "ChunkIndex", "Score", "ChunkText"])
        for r in combined_rows:
            w.writerow([
                r.get("Source"),
                r.get("Page"),
                r.get("ChunkID"),
                r.get("ChunkIndex"),
                r.get("Score"),
                (r.get("ChunkText") or "")[:2000],
            ])
    log.info(f"Wrote combined CSV: {out_csv}")

    # Write TXT (grouped by Source→Page)
    ensure_dir(Path(out_text))
    with open(out_text, "w", encoding="utf-8") as tf:
        cur_src = None
        cur_page = None
        for r in combined_rows:
            src = r.get("Source")
            pg = r.get("Page")
            if src != cur_src or pg != cur_page:
                tf.write(f"\n===== {src} | Page {pg} =====\n")
                cur_src, cur_page = src, pg
            tf.write((r.get("ChunkText") or "") + "\n\n")
    log.info(f"Wrote combined TXT: {out_text}")


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Orchestrate last-neural retrieval + doccheck on remaining pages (merge outputs)")
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--docs", nargs="+", required=True, help="Text files with '### Page N ###' markers")
    ap.add_argument("--keywords_excel", default=None, help="Excel with Question/Keywords columns")
    ap.add_argument("--collection", required=True, help="Chroma collection name")
    ap.add_argument("--chroma_path", default="chromadb", help="Path to Chroma persistent store")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI chat model for doccheck JSON steps")
    ap.add_argument("--filter_chunks_with_llm", action="store_true", help="Use LLM to choose relevant chunks in doccheck phase")
    ap.add_argument("--out_csv", default="combined_chunks.csv", help="Combined CSV output path")
    ap.add_argument("--out_text", default="combined_chunks.txt", help="Combined TXT output path")
    ap.add_argument("--keep_intermediate", action="store_true", help="Keep temporary filtered docs and doccheck outputs")
    ap.add_argument("--max_iters_lastneural", type=int, default=3, help="Max agentic rounds for lastneural phase")

    args = ap.parse_args()

    orchestrate(
        query=args.query,
        docs=[Path(x) for x in args.docs],
        keywords_excel=Path(args.keywords_excel) if args.keywords_excel else None,
        collection_name=args.collection,
        chroma_path=args.chroma_path,
        model=args.model,
        filter_chunks_with_llm=args.filter_chunks_with_llm,
        out_csv=Path(args.out_csv),
        out_text=Path(args.out_text),
        keep_intermediate=args.keep_intermediate,
        max_iters_lastneural=args.max_iters_lastneural,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(130)



#python orchestrator_keywords_neural_doccheck.py --query "To what extent are the organization's products and services information-based or information-enabled?" --docs "Novartis_docs/novartis_integrated_report_2024_cleaned_ocr2.txt" "Novartis_docs/novartis_annual_report_2024_cleaned_ocr2.txt" --keywords_excel novartis_keywords_by_question.xlsx --collection novartis_collection --chroma_path chromadb --model gpt-5-mini --filter_chunks_with_llm --out_csv combined_chunksQ1.csv --out_text combined_chunksQ1.txt
