import os
import shutil
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _safe_move(src: Path, dst: Path) -> Path:
    """Move src -> dst, avoiding overwrite by suffixing if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst

    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        candidate = dst.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
        i += 1


def _move_processed_pdfs(
    *,
    papers_dir: Path,
    completed_dir: Path,
    chunked_dir: Path,
) -> int:
    """Move PDFs that appear to have been successfully chunked."""
    moved = 0
    pdfs = sorted(papers_dir.glob("**/*.pdf"))
    for pdf in pdfs:
        try:
            rel = pdf.relative_to(papers_dir)
        except Exception:
            rel = Path(pdf.name)

        expected_chunk = chunked_dir / rel.parent / f"{pdf.stem}.tei_chunks.json"
        if not expected_chunk.exists():
            continue

        try:
            if expected_chunk.stat().st_size < 5:
                continue
        except Exception:
            continue

        dest = completed_dir / rel
        _safe_move(pdf, dest)
        moved += 1

    return moved


def _cleanup_output_dirs(*, workspace_root: Path) -> None:
    output_root = workspace_root / "OUTPUT"
    if not output_root.exists():
        return

    # These are intermediate artifacts; DB + Qdrant become the source of truth.
    # Keep OUTPUT/ itself (and any future logs) but remove heavy intermediates.
    to_remove = [
        output_root / "tei",
        output_root / "tei_main",
        output_root / "tei_references",
        output_root / "text",
        output_root / "Chunked_text",
    ]

    for p in to_remove:
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def main() -> None:
    workspace_root = Path(__file__).resolve().parent

    from Extractor import TextExtraction
    from Extractor import SeparateContentReferences
    from Extractor import convert_into_text
    from Extractor import convert_into_json
    from Extractor import chunking
    from Extractor import load_to_postgres
    from Extractor import generate_vectors

    grobid_server = os.getenv("GROBID_SERVER", "http://localhost:8070")

    papers_dir = Path(os.getenv("PAPERS_DIR", str(workspace_root / "Research Papers")))
    completed_dir = Path(os.getenv("COMPLETED_PAPERS_DIR", str(workspace_root / "Completed Papers")))
    move_processed_papers = _env_bool("MOVE_PROCESSED_PAPERS", True)
    clean_output = _env_bool("CLEAN_OUTPUT", False)

    postgres_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),
        "database": os.getenv("POSTGRES_DB", "research_papers"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
    }
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "research_papers")
    vector_batch_size = int(os.getenv("VECTOR_BATCH_SIZE", "10"))
    build_citation_graph = _env_bool("BUILD_CITATION_GRAPH", True)

    print("=" * 80)
    print("STEP 1: GROBID TEI + raw text extraction")
    print("=" * 80)
    TextExtraction.run(workspace_root=workspace_root, grobid_server=grobid_server, input_dir=papers_dir)

    print("\n" + "=" * 80)
    print("STEP 2: Split TEI into main vs references")
    print("=" * 80)
    SeparateContentReferences.run(workspace_root=workspace_root)

    print("\n" + "=" * 80)
    print("STEP 3: Convert main TEI -> text")
    print("=" * 80)
    convert_into_text.run(workspace_root=workspace_root)

    print("\n" + "=" * 80)
    print("STEP 4: Convert references TEI -> JSON")
    print("=" * 80)
    convert_into_json.run(workspace_root=workspace_root)

    print("\n" + "=" * 80)
    print("STEP 5: Chunk main text -> chunk JSON")
    print("=" * 80)
    chunking.run(workspace_root=workspace_root)

    print("\n" + "=" * 80)
    print("STEP 6: Load chunks + paper metadata into PostgreSQL")
    print("=" * 80)
    load_to_postgres.run(
        postgres_config=postgres_config,
        chunked_text_dir=workspace_root / "OUTPUT" / "Chunked_text",
        reference_dir=workspace_root / "OUTPUT" / "text" / "reference",
        build_citation_graph=build_citation_graph,
    )

    print("\n" + "=" * 80)
    print("STEP 7: Generate vectors + load to Qdrant")
    print("=" * 80)
    generate_vectors.run(
        postgres_config=postgres_config,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=qdrant_collection,
        batch_size=vector_batch_size,
    )

    if move_processed_papers:
        print("\n" + "=" * 80)
        print("FINAL STEP: Move processed PDFs to Completed Papers")
        print("=" * 80)
        moved = _move_processed_pdfs(
            papers_dir=papers_dir,
            completed_dir=completed_dir,
            chunked_dir=workspace_root / "OUTPUT" / "Chunked_text",
        )
        print(f"✓ Moved {moved} processed PDF(s) to: {completed_dir}")

    if clean_output:
        print("\n" + "=" * 80)
        print("FINAL STEP: Clean intermediate OUTPUT artifacts")
        print("=" * 80)
        _cleanup_output_dirs(workspace_root=workspace_root)
        print("✓ Removed intermediate folders under OUTPUT/")

    print("\n✓ Extraction + DB ingestion complete")


if __name__ == "__main__":
    main()
