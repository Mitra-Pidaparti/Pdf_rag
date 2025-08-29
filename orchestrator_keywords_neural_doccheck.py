#!/usr/bin/env python3
"""
Final orchestrator (v2):
- Runs agentic_retrieve_keywords_lastneural_optimized first (no LLM extraction).
- Records (filename, page) actually used.
- Looks up ALL pages per file from Chroma.
- Computes strictly-remaining pages (robust to filename mismatches via normalization).
- Stitches temp docs containing ONLY those pages.
- Runs agentic_retrieve_doccheck_lastllm on the temp docs.
- Merges chunks from both phases into a single CSV/TXT.

Key improvements over v1:
- **Filename normalization** on both sides (basenames, lowercase) to prevent
  accidental all-pages runs when names differ by path/case.
- **Strict page int handling**: rows without valid page numbers are skipped for the
  remaining-pages computation (they still appear in output, but don’t skew filtering).
- Exposes doccheck knobs via CLI: relevance threshold, keyword gating, chunk limit, LLM-chunk filter.

Example (Windows cmd):
  python orchestrator_keywords_neural_doccheck_v2.py ^
    --query "What did the reports say about the organization's 'Purpose'?" ^
    --docs "C:\\path\\novartis_integrated_report_2024_cleaned_ocr2.txt" "C:\\path\\novartis_annual_report_2024_cleaned_ocr2.txt" ^
    --keywords_excel novartis_keywords_by_question.xlsx ^
    --collection novartis_collection ^
    --chroma_path chromadb ^
    --model gpt-5-mini ^
    --filter_chunks_with_llm ^
    --relevance_threshold 0.7 ^
    --require_keyword_hit ^
    --max_chunks_per_page 12 ^
    --out_csv combined_chunks.csv ^
    --out_text combined_chunks.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
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
log = logging.getLogger("orchestrator_v2")

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

def norm_name(s: str) -> str:
    """Normalize any file path/name to a stable key: basename, lowercase, stripped."""
    return Path(str(s)).name.strip().lower()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stitch_filtered_doc(original_path: Path, keep_pages: Set[int]) -> Optional[Path]:
    """Create a temp file with only the requested UNIQUE pages from original_path."""
    text = original_path.read_text(encoding="utf-8", errors="ignore")
    segments = split_pages(text)  # [(pno, text_segment), ...]
    if not keep_pages:
        return None

    # keep only pages that exist in the text, and only the FIRST occurrence per page
    existing = {p for p, _ in segments}
    target = set(keep_pages) & existing

    selected, seen = [], set()
    for pno, ptext in segments:
        if pno in target and pno not in seen:
            selected.append(ptext + "\n")
            seen.add(pno)

    if not selected:
        return None
    tmpdir = Path(tempfile.mkdtemp(prefix="docfilter_"))
    outp = tmpdir / original_path.name
    outp.write_text("".join(selected), encoding="utf-8")
    return outp

# ---------- Chroma access ----------

def collect_pages_for_files(collection, filenames: List[str], filename_field: str = "filename") -> Dict[str, Set[int]]:
    """Scan Chroma metadata to learn which page numbers exist per filename.
    Returns a dict keyed by **normalized** filename.
    """
    pages_by_file: Dict[str, Set[int]] = {norm_name(fn): set() for fn in filenames}
    for fn in filenames:
        try:
            res = collection.get(where={filename_field: {"$eq": fn}}, include=["metadatas"], limit=200_000)
        except Exception as e:
            log.warning(f"Chroma get failed for {fn}: {e}")
            continue
        mds = res.get("metadatas", []) or []
        key = norm_name(fn)
        for md in mds:
            if not md:
                continue
            p = md.get("page")
            try:
                pages_by_file[key].add(int(p))
            except Exception:
                continue
    return pages_by_file

#----------Truncation----------------
def _smart_truncate(s: str, limit: int = 2000) -> str:
    s = s or ""
    if len(s) <= limit:
        return s
    cut = s[:limit]
    # Prefer stopping at end of sentence near the end
    for p in (".", "!", "?", "…"):
        idx = cut.rfind(p)
        if idx >= int(limit * 0.7):  # only if reasonably close to the end
            return cut[:idx+1]
    # Else stop at last whitespace
    sp = cut.rfind(" ")
    if sp >= int(limit * 0.5):
        return cut[:sp]
    return cut  # fallback: hard cut (rare)



# ---------- Results shaping ----------

def build_rows_from_lastneural(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten last-neural results into rows we can merge later.
    Rows missing a valid integer page are kept for output but excluded from remaining-pages math.
    """
    rows: List[Dict[str, Any]] = []
    for c in report.get("results", []):
        md = c.get("metadata") or {}
        fn = md.get("filename") or (Path(md.get("filepath")).name if md.get("filepath") else c.get("document"))
        page_raw = md.get("page") if md.get("page") is not None else c.get("page")
        page_int: Optional[int]
        try:
            page_int = int(page_raw) if page_raw is not None else None
        except Exception:
            page_int = None
        rows.append({
            "Source": Path(fn).name if fn else None,
            "SourceNorm": norm_name(fn) if fn else None,
            "Page": page_int,  # may be None
            "ChunkID": md.get("chunk_id") or c.get("chunk_id") or c.get("id"),
            "ChunkIndex": md.get("chunk_index"),
            "Score": c.get("ultimate_score", c.get("final_enhanced_score", c.get("neural_score", 0.0))),
            "ChunkText": (md.get("chunk_text") or c.get("chunk") or ""),
        })
    return rows


def pages_used_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Set[int]]:
    """Return mapping of normalized filename -> set(page ints) for rows with page numbers."""
    by_file: Dict[str, Set[int]] = {}
    for r in rows:
        srcn = r.get("SourceNorm")
        p = r.get("Page")
        if not srcn or p is None:
            continue
        by_file.setdefault(srcn, set()).add(int(p))
    return by_file

# ---------- Orchestrator ----------

def orchestrate(
    query: str,
    docs: List[Path],
    keywords_excel: Optional[Path],
    collection_name: str,
    chroma_path: str,
    model: str = "gpt-5-mini",
    filter_chunks_with_llm: bool = False,
    relevance_threshold: float = 0.6,
    require_keyword_hit: bool = False,
    chunk_sim_threshold: float = 0.5,
    max_chunks_per_page: int = 12,
    out_csv: Path = Path("combined_chunks.csv"),
    out_text: Path = Path("combined_chunks.txt"),
    keep_intermediate: bool = False,
    filename_field: str = "filename",
    max_iters_lastneural: int = 5,
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
    # Optional: make last-neural not call any LLM for rewrites
    try:
        cfg.propose_llm_queries = False
    except Exception:
        pass

    report = LNO.agentic_search(query, kw_list, cfg)

    rows_last = build_rows_from_lastneural(report)
    used_pages_by_file = pages_used_from_rows(rows_last)  # normalized keys

    # 2) --- Discover ALL pages per file via Chroma (keys normalized to match) ---
    input_basenames = [Path(p).name for p in docs]
    all_pages_by_file = collect_pages_for_files(collection, input_basenames, filename_field=filename_field)

    # 3) --- Compute remaining pages & build temp docs containing ONLY those pages ---
    temp_docs: List[Path] = []
    total_all = 0
    total_used = 0
    total_remaining = 0

    for p in docs:
        name_base = Path(p).name
        key = norm_name(name_base)
        all_pages = all_pages_by_file.get(key, set())
        used_pages = used_pages_by_file.get(key, set())
        remaining = all_pages - used_pages if all_pages else set()

        total_all += len(all_pages)
        total_used += len(used_pages)
        total_remaining += len(remaining)

        if not remaining:
            log.info(f"No remaining pages for {name_base}; skipping doccheck for this file")
            continue
        tmp = stitch_filtered_doc(p, remaining)
        if tmp:
            temp_docs.append(tmp)
            preview = ", ".join(map(str, sorted(list(remaining))[:12]))
            more = "..." if len(remaining) > 12 else ""
            log.info(f"Prepared filtered doc for {name_base}: pages [{preview}{more}]")

    log.info(f"Pages summary — all:{total_all} used:{total_used} remaining:{total_remaining}")

    # 4) --- Run doccheck on filtered docs only ---
    rows_doc: List[Dict[str, Any]] = []
    if temp_docs:
        import agentic_retrieve_doccheck_lastllm_optimized as DOC
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
            relevance_threshold=relevance_threshold,
            require_keyword_hit=require_keyword_hit,
            chunk_sim_threshold=chunk_sim_threshold,
            max_chunks_per_page=max_chunks_per_page,
            out_csv=str(tmp_csv),
            out_jsonl=str(tmp_jsonl),
            out_text=str(tmp_txt),
        )
        # Load kept chunks from doccheck CSV
        
        # Load kept chunks from doccheck CSV (robust to empty/missing/unreadable file)
        if tmp_csv.exists() and tmp_csv.stat().st_size > 0:
            try:
                df = pd.read_csv(tmp_csv)
            except pd.errors.EmptyDataError:
                log.warning("Doccheck CSV is empty; proceeding with agentic results only.")
                df = None
            except Exception as e:
                log.warning(f"Failed to read doccheck CSV ({tmp_csv}): {e}. Proceeding without it.")
                df = None

            if df is not None and not df.empty:
                for _, r in df.iterrows():
                    # Normalize source for consistency in output ordering
                    src = r.get("Source")
                    page_val = r.get("Page")
                    try:
                        page_int = int(page_val) if pd.notna(page_val) else None
                    except Exception:
                        page_int = None
                    rows_doc.append({
                        "Source": Path(str(src)).name if src else None,
                        "SourceNorm": norm_name(src) if src else None,
                        "Page": page_int,
                        "ChunkID": r.get("ChunkID"),
                        "ChunkIndex": r.get("ChunkIndex"),
                        "Score": r.get("Score", 0.0),
                        "ChunkText": r.get("ChunkText", ""),
                    })
        else:
            log.info("Doccheck CSV missing or empty; proceeding with agentic results only.")

        if not keep_intermediate:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
    # 5) --- Merge outputs and write combined CSV/TXT ---
    combined_rows = rows_last + rows_doc

    # Sort by (SourceNorm, Page, Score desc)
    def _row_key(r: Dict[str, Any]):
        return (
            r.get("SourceNorm") or "",
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
                (r.get("ChunkText") or ""),
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
    ap = argparse.ArgumentParser(description="Final orchestrator: last-neural (remaining-pages aware) + doccheck only on remaining pages")
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--docs", nargs="+", required=True, help="Text files with '### Page N ###' markers")
    ap.add_argument("--keywords_excel", default=None, help="Excel with Question/Keywords columns")
    ap.add_argument("--collection", required=True, help="Chroma collection name")
    ap.add_argument("--chroma_path", default="chromadb", help="Path to Chroma persistent store")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI chat model for doccheck JSON steps")
    ap.add_argument("--filter_chunks_with_llm", action="store_true", help="Use LLM to choose relevant chunks in doccheck phase")
    ap.add_argument("--relevance_threshold", type=float, default=0.6, help="Minimum page relevance to keep (0..1)")
    ap.add_argument("--require_keyword_hit", action="store_true", help="Drop pages without any keyword matches at triage")
    ap.add_argument("--chunk_sim_threshold", type=float, default=0.5, help="Local similarity threshold for chunk filter")
    ap.add_argument("--max_chunks_per_page", type=int, default=12, help="Max chunks to keep per page in doccheck phase")
    ap.add_argument("--out_csv", default="combined_chunks.csv", help="Combined CSV output path")
    ap.add_argument("--out_text", default="combined_chunks.txt", help="Combined TXT output path")
    ap.add_argument("--keep_intermediate", action="store_true", help="Keep temporary filtered docs and doccheck outputs")
    ap.add_argument("--max_iters_lastneural", type=int, default=5, help="Max agentic rounds for lastneural phase")

    args = ap.parse_args()

    orchestrate(
        query=args.query,
        docs=[Path(x) for x in args.docs],
        keywords_excel=Path(args.keywords_excel) if args.keywords_excel else None,
        collection_name=args.collection,
        chroma_path=args.chroma_path,
        model=args.model,
        filter_chunks_with_llm=args.filter_chunks_with_llm,
        relevance_threshold=args.relevance_threshold,
        require_keyword_hit=args.require_keyword_hit,
        chunk_sim_threshold=args.chunk_sim_threshold,
        max_chunks_per_page=args.max_chunks_per_page,
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


#python orchestrator_keywords_neural_doccheck.py --query "To what extent does the organization use technologies for knowledge-sharing, communication, and workflow management?" --docs "Novartis_docs/novartis_integrated_report_2024_cleaned_ocr2.txt" "Novartis_docs/novartis_annual_report_2024_cleaned_ocr2.txt" --keywords_excel novartis_keywords_by_question.xlsx --collection novartis_collection --chroma_path chromadb --model gpt-5-mini --relevance_threshold 0.6 --max_chunks_per_page 12 --out_csv combined_chunks_optQ2.csv --out_text combined_chunksQ2_opt.txt