"""Chunking utilities and CLI.

Produces chunk JSON files used for vector loading. Output is intentionally minimal:
- chunk_id, paper_id, section, paragraph_index, text, year

This module also provides a higher-level ``chunk_documents`` helper that can
skip unchanged files based on file metadata and run chunking in parallel.
"""

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rag_utils.metrics import stage_timer, log_stage


try:
    import tiktoken

    def make_tiktoken_tokenizer(model_name: str = "gpt-4o-mini"):
        enc = tiktoken.encoding_for_model(model_name)
        return {
            "encode": lambda s: enc.encode(s),
            "decode": lambda tokens: enc.decode(tokens),
            "count": lambda s: len(enc.encode(s)),
        }


except Exception:
    tiktoken = None


def default_tokenizer():
    if tiktoken:
        return make_tiktoken_tokenizer()
    return {
        "encode": lambda s: s.split(),
        "decode": lambda tokens: " ".join(tokens),
        "count": lambda s: max(1, len(str(s or "").split())),
    }


def chunk_paragraphs(
    paragraphs: List[Dict[str, Any]],
    *,
    tokenizer: Optional[Dict] = None,
    min_tokens: int = 400,
    max_tokens: int = 1200,
    overlap_percent: float = 0.30,
) -> List[Dict[str, Any]]:
    """Combine consecutive paragraphs into bounded-size chunks.

    paragraphs items should include:
      - text (str)
      - section_heading (str) optional
      - paper_id (str) optional
      - year (int) optional

    Returns chunk dicts containing:
      - chunk_id, paper_id, section, paragraph_index, text, year, token_count
    """
    if tokenizer is None:
        tokenizer = default_tokenizer()

    assert 0 < min_tokens <= max_tokens
    assert 0.0 < overlap_percent <= 1.0

    all_chunks = []
    i = 0
    import time as _time
    start_ts = _time.perf_counter()
    while i < len(paragraphs):
        current_tokens = 0
        chunk_paras = []
        start_para_idx = i
        current_section = paragraphs[i].get("section_heading")

        while i < len(paragraphs) and current_tokens < min_tokens:
            para = paragraphs[i]
            para_text = str(para.get("text", "")).strip()
            if not para_text:
                i += 1
                continue

            para_tokens = int(tokenizer["count"](para_text))
            if current_tokens > 0 and current_tokens + para_tokens > max_tokens:
                break

            if current_section and para.get("section_heading") != current_section:
                if current_tokens >= min_tokens:
                    break
                current_section = para.get("section_heading")

            chunk_paras.append(para)
            current_tokens += para_tokens
            i += 1

        if current_tokens < min_tokens and i < len(paragraphs):
            continue

        if chunk_paras:
            chunk_text = "\n\n".join([str(p.get("text", "")).strip() for p in chunk_paras if str(p.get("text", "")).strip()])
            chunk_id = f"{chunk_paras[0].get('paper_id', 'paper')}_chunk-{len(all_chunks) + 1}"
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": chunk_paras[0].get("paper_id"),
                    "section": chunk_paras[0].get("section_heading"),
                    "paragraph_index": start_para_idx,
                    "text": chunk_text,
                    "year": chunk_paras[0].get("year"),
                    "token_count": current_tokens,
                }
                )

            duration_ms = (_time.perf_counter() - start_ts) * 1000.0
            log_stage(
            "chunking",
            duration_ms=duration_ms,
            num_items=len(all_chunks),
            extra={"num_paragraphs": len(paragraphs)},
            )
            return all_chunks


def extract_year_from_text(text: str) -> Optional[int]:
    m = re.search(r"\b(19|20)\d{2}\b", str(text or "")[:500])
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _parse_markdownish_sections(content: str, *, paper_id: str, year: Optional[int]) -> List[Dict[str, Any]]:
    paragraphs_raw = [p.strip() for p in str(content or "").split("\n\n") if p.strip()]
    current_section_heading: Optional[str] = None

    paragraphs: List[Dict[str, Any]] = []
    for idx, para_text in enumerate(paragraphs_raw):
        if para_text.startswith("##"):
            heading_match = re.match(r"^##\s*(\d+(?:\.\d+)*)\s+(.+)$", para_text)
            if heading_match:
                current_section_heading = heading_match.group(2).strip()
            else:
                current_section_heading = para_text.replace("##", "").strip() or None
            continue
        if para_text.startswith("#"):
            continue

        paragraphs.append(
            {
                "text": para_text,
                "section_heading": current_section_heading,
                "paper_id": paper_id,
                "paragraph_id": f"para_{idx}",
                "year": year,
            }
        )

    return paragraphs


@dataclass
class FileSignature:
    path: str
    mtime: float
    size: int

    @classmethod
    def from_path(cls, path: Path) -> "FileSignature":
        stat = path.stat()
        return cls(path=str(path), mtime=stat.st_mtime, size=stat.st_size)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "mtime": self.mtime, "size": self.size}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileSignature":
        return cls(path=data["path"], mtime=float(data["mtime"]), size=int(data["size"]))

    def key(self) -> Tuple[str, float, int]:
        return (self.path, self.mtime, self.size)


def _load_processed_file_cache(cache_path: Path) -> Dict[str, FileSignature]:
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        out: Dict[str, FileSignature] = {}
        for k, v in raw.items():
            try:
                out[k] = FileSignature.from_dict(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _save_processed_file_cache(cache_path: Path, cache: Dict[str, FileSignature]) -> None:
    serializable = {k: v.to_dict() for k, v in cache.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_text_hash(text: str) -> str:
    h = sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _chunk_single_file(
    txt_file: Path,
    *,
    min_tokens: int,
    max_tokens: int,
    overlap_percent: float,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Worker function executed in a separate process.

    Returns the path and the list of chunk dicts (including token_count).
    """
    content = txt_file.read_text(encoding="utf-8")
    doc_year = extract_year_from_text(content)
    paragraphs = _parse_markdownish_sections(content, paper_id=txt_file.stem, year=doc_year)

    chunks = chunk_paragraphs(
        paragraphs,
        tokenizer=default_tokenizer(),
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_percent=overlap_percent,
    )
    return txt_file, chunks


def chunk_documents(
    files: Iterable[Path],
    *,
    workspace_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_tokens: int = 400,
    max_tokens: int = 1200,
    overlap_percent: float = 0.30,
    max_workers: Optional[int] = None,
) -> None:
    """Chunk a collection of text files with caching and parallelism.

    Skips files whose (path, mtime, size) triple has not changed since the
    last run and for which an existing *_chunks.json file exists.
    """
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[1]
    if output_dir is None:
        output_dir = workspace_root / "OUTPUT" / "Chunked_text"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "_chunking_cache.json"
    cache = _load_processed_file_cache(cache_path)

    files = [Path(f) for f in files]
    to_process: List[Path] = []
    skipped: List[Path] = []

    for txt_file in files:
        if not txt_file.is_file():
            continue
        sig = FileSignature.from_path(txt_file)
        rel_key = os.path.relpath(str(txt_file), str(workspace_root))
        cached_sig = cache.get(rel_key)

        rel = txt_file
        if workspace_root in txt_file.parents:
            rel = txt_file.relative_to(workspace_root)
        out_parent = output_dir / rel.parent
        output_file = out_parent / f"{txt_file.stem}_chunks.json"

        if (
            cached_sig is not None
            and cached_sig.key() == sig.key()
            and output_file.exists()
            and output_file.stat().st_size > 0
        ):
            skipped.append(txt_file)
            continue

        cache[rel_key] = sig
        to_process.append(txt_file)

    if not to_process:
        log_stage("chunking", num_items=0, extra={"skipped_files": len(skipped)})
        return

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _chunk_single_file,
                txt_file,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap_percent=overlap_percent,
            )
            for txt_file in to_process
        ]

        for fut in as_completed(futures):
            try:
                txt_file, chunks = fut.result()
            except Exception as exc:
                print(f"Failed to process {fut}: {exc}")
                continue

            rel = txt_file
            if workspace_root in txt_file.parents:
                rel = txt_file.relative_to(workspace_root)
            out_parent = output_dir / rel.parent
            out_parent.mkdir(parents=True, exist_ok=True)
            output_file = out_parent / f"{txt_file.stem}_chunks.json"

            out_chunks = []
            for c in chunks:
                out_chunks.append(
                    {
                        "chunk_id": c.get("chunk_id"),
                        "paper_id": c.get("paper_id"),
                        "section": c.get("section"),
                        "paragraph_index": c.get("paragraph_index"),
                        "text": c.get("text"),
                        "year": c.get("year"),
                    }
                )

            output_file.write_text(json.dumps(out_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Written: {output_file} ({len(out_chunks)} chunks)")

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    log_stage(
        "chunking",
        duration_ms=duration_ms,
        num_items=len(to_process),
        extra={"skipped_files": len(skipped)},
    )


def run(
    *,
    workspace_root: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_tokens: int = 400,
    max_tokens: int = 1200,
    overlap_percent: float = 0.30,
) -> None:
    """Chunk all .txt files from input_dir and write *_chunks.json to output_dir."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[1]
    if input_dir is None:
        input_dir = workspace_root / "OUTPUT" / "text" / "main"
    if output_dir is None:
        output_dir = workspace_root / "OUTPUT" / "Chunked_text"

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_dir.rglob("*.txt"))
    chunk_documents(
        txt_files,
        workspace_root=workspace_root,
        output_dir=output_dir,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_percent=overlap_percent,
    )


if __name__ == "__main__":
    run()
