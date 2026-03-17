"""Utility to re-chunk the text corpus.

This script is a thin wrapper around Extractor.chunking.run that:
- optionally clears the previous chunking cache and chunk JSON files
- re-runs chunking with new parameters (min_tokens / max_tokens / overlap_percent)

Typical usage (from repo root):

    python -m Extractor.rechunk \
        --min-tokens 200 \
        --max-tokens 800 \
        --overlap-percent 0.3 \
        --reset-cache

After re-chunking, you should regenerate vectors and FAISS indexes, e.g.:

    python -m Extractor.generate_vectors

"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import chunking


def _clear_chunk_output(output_dir: Path) -> None:
    """Delete existing chunk JSONs and cache so all files are reprocessed.

    This does *not* touch your original text files or FAISS indexes.
    You will need to regenerate vectors after running this.
    """
    output_dir = output_dir.expanduser().resolve()
    if not output_dir.exists():
        return

    # Remove cache file so chunk_documents does not skip unchanged files
    cache_path = output_dir / "_chunking_cache.json"
    if cache_path.exists():
        cache_path.unlink()

    # Remove existing *_chunks.json files so new ones are written cleanly
    for json_path in output_dir.rglob("*_chunks.json"):
        try:
            json_path.unlink()
        except Exception:
            # Best-effort: continue even if some files cannot be removed
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-chunk text corpus for RAG database.")
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="Workspace root; default is project root inferred from this file.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory with source .txt files (default: OUTPUT/text/main under workspace root).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write *_chunks.json files (default: OUTPUT/Chunked_text under workspace root).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=400,
        help="Minimum tokens per chunk (default: 400).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Maximum tokens per chunk (default: 1200).",
    )
    parser.add_argument(
        "--overlap-percent",
        type=float,
        default=0.30,
        help="Fraction of overlap between consecutive chunks (0-1, default: 0.30).",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="If set, delete existing chunk JSONs and cache before re-chunking.",
    )

    args = parser.parse_args()

    # Resolve paths
    if args.workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[1]
    else:
        workspace_root = Path(args.workspace_root).expanduser().resolve()

    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else workspace_root / "OUTPUT" / "text" / "main"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else workspace_root / "OUTPUT" / "Chunked_text"

    if args.reset_cache:
        _clear_chunk_output(output_dir)

    # Delegate to existing chunking.run helper
    chunking.run(
        workspace_root=workspace_root,
        input_dir=input_dir,
        output_dir=output_dir,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_percent=args.overlap_percent,
    )


if __name__ == "__main__":
    main()
