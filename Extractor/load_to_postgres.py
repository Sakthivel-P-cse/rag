"""Load chunk JSON files into PostgreSQL (minimal schema).

This version intentionally keeps only required fields for RAG + vector generation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extras import execute_values


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


class PostgresLoader:
    def __init__(self, postgres_config: Dict[str, Any]):
        self.postgres_config = postgres_config
        self.conn = None

    def connect(self) -> None:
        self.conn = psycopg2.connect(
            host=self.postgres_config["host"],
            port=int(self.postgres_config.get("port", 5433)),
            database=self.postgres_config["database"],
            user=self.postgres_config["user"],
            password=self.postgres_config.get("password", ""),
        )
        print(f"✓ Connected to PostgreSQL database: {self.postgres_config['database']}")

    def create_schema(self) -> None:
        create_table_sql = """
        DROP TABLE IF EXISTS chunks CASCADE;
        DROP TABLE IF EXISTS papers CASCADE;

        CREATE TABLE chunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            paper_id VARCHAR(255) NOT NULL,
            section VARCHAR(500),
            paragraph_index INTEGER,
            chunk_text TEXT NOT NULL,
            year INTEGER,

            vector_generated BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
        CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);
        CREATE INDEX IF NOT EXISTS idx_chunks_vector_generated ON chunks(vector_generated);

        CREATE TABLE papers (
            paper_id VARCHAR(255) PRIMARY KEY,
            title TEXT,
            authors TEXT[],
            year INTEGER,
            arxiv_id VARCHAR(100),
            source VARCHAR(255),
            total_chunks INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
        CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id);
        """

        cursor = self.conn.cursor()
        cursor.execute(create_table_sql)
        self.conn.commit()
        cursor.close()
        print("✓ PostgreSQL schema created successfully (minimal)")

    def load_from_json(self, json_file_path: str) -> int:
        json_path = Path(json_file_path)
        if not json_path.exists():
            print(f"✗ File not found: {json_file_path}")
            return 0

        print(f"\nLoading chunks from: {json_path.name}")

        try:
            chunks = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(chunks, list):
                raise ValueError("Chunk JSON must be a list")

            insert_sql = """
            INSERT INTO chunks (
                chunk_id, paper_id, section, paragraph_index, chunk_text, year, vector_generated
            ) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                paper_id = EXCLUDED.paper_id,
                section = EXCLUDED.section,
                paragraph_index = EXCLUDED.paragraph_index,
                chunk_text = EXCLUDED.chunk_text,
                year = EXCLUDED.year,
                updated_at = CURRENT_TIMESTAMP
            """

            values = []
            for chunk in chunks:
                values.append(
                    (
                        chunk["chunk_id"],
                        chunk["paper_id"],
                        chunk.get("section"),
                        chunk.get("paragraph_index"),
                        chunk["text"],
                        chunk.get("year"),
                        False,
                    )
                )

            cursor = self.conn.cursor()
            execute_values(cursor, insert_sql, values)
            self.conn.commit()
            cursor.close()

            print(f"✓ Successfully loaded {len(chunks)} chunks into PostgreSQL")
            return len(chunks)

        except Exception as e:
            print(f"✗ Error loading chunks: {e}")
            import traceback

            traceback.print_exc()
            self.conn.rollback()
            return 0

    def load_paper_metadata(self, reference_json_dir: str) -> None:
        ref_dir = Path(reference_json_dir)
        if not ref_dir.exists():
            print(f"⚠ Reference directory not found: {reference_json_dir}")
            return

        json_files = list(ref_dir.glob("*.json"))
        if not json_files:
            print(f"⚠ No reference JSON files found in {reference_json_dir}")
            return

        print(f"\nLoading paper metadata from {len(json_files)} reference file(s)...")

        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))

                paper_id = data.get("paper_id", json_file.stem)
                title = data.get("title", "")
                arxiv_id = data.get("arxiv_id")

                authors = []
                for author in data.get("authors", []):
                    given = author.get("given", "")
                    family = author.get("family", "")
                    full_name = f"{given} {family}".strip()
                    if full_name:
                        authors.append(full_name)

                year = None
                # Best-effort year extraction from paper references (keeps schema lean)
                ref_years = [ref.get("year") for ref in data.get("references", []) if ref.get("year")]
                if ref_years:
                    try:
                        year = max([int(y) for y in ref_years if str(y).isdigit()])
                    except Exception:
                        year = None

                insert_sql = """
                INSERT INTO papers (
                    paper_id, title, authors, year, arxiv_id, source, total_chunks
                ) VALUES (%s, %s, %s, %s, %s, %s, 0)
                ON CONFLICT (paper_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    year = EXCLUDED.year,
                    arxiv_id = EXCLUDED.arxiv_id,
                    source = EXCLUDED.source
                """

                cursor = self.conn.cursor()
                cursor.execute(
                    insert_sql,
                    (
                        paper_id,
                        title,
                        authors,
                        year,
                        arxiv_id,
                        "arXiv" if arxiv_id else "Unknown",
                    ),
                )
                self.conn.commit()
                cursor.close()

                print(f"  ✓ Loaded metadata for: {title[:60]}...")

            except Exception as e:
                print(f"  ✗ Error loading metadata from {json_file.name}: {e}")
                continue

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE papers p
                SET total_chunks = (
                    SELECT COUNT(*) FROM chunks c WHERE c.paper_id = p.paper_id
                )
                """
            )
            self.conn.commit()
            cursor.close()
            print("\n✓ Updated chunk counts for all papers")
        except Exception as e:
            print(f"⚠ Failed to update chunk counts: {e}")

    def load_multiple_files(self, directory: str, pattern: str = "*_chunks.json") -> Dict[str, int]:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"✗ Directory not found: {directory}")
            return {}

        json_files = list(dir_path.glob(pattern))
        if not json_files:
            print(f"✗ No JSON files found matching '{pattern}' in {directory}")
            return {}

        print(f"\nFound {len(json_files)} JSON file(s) to process")

        results: Dict[str, int] = {}
        total_chunks = 0
        for json_file in json_files:
            count = self.load_from_json(str(json_file))
            results[json_file.name] = count
            total_chunks += count

        print("\n" + "=" * 80)
        print(f"SUMMARY: Loaded {total_chunks} total chunks from {len(json_files)} files")
        print("=" * 80)
        return results

    def get_stats(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks;")
        total_chunks = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE vector_generated = TRUE;")
        vectors_generated = cursor.fetchone()[0]
        cursor.close()

        print("\n" + "=" * 80)
        print("DATABASE STATISTICS")
        print("=" * 80)
        print(f"Total chunks: {total_chunks}")
        print(f"Vectors generated: {vectors_generated}/{total_chunks}")
        print("=" * 80)

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            print("✓ PostgreSQL connection closed")


def run(
    *,
    postgres_config: Dict[str, Any],
    chunked_text_dir: Optional[Path] = None,
    reference_dir: Optional[Path] = None,
    build_citation_graph: bool = True,
) -> None:
    loader = PostgresLoader(postgres_config)
    loader.connect()
    loader.create_schema()

    if chunked_text_dir is None:
        chunked_text_dir = WORKSPACE_ROOT / "OUTPUT" / "Chunked_text"
    if reference_dir is None:
        reference_dir = WORKSPACE_ROOT / "OUTPUT" / "text" / "reference"

    results = loader.load_multiple_files(directory=str(chunked_text_dir), pattern="*_chunks.json")
    loader.load_paper_metadata(str(reference_dir))
    loader.get_stats()

    if build_citation_graph:
        print("\n" + "=" * 70)
        print("STEP: Building Citation Graph")
        print("=" * 70)

        from Extractor.citation_graph import CitationGraphManager

        citation_manager = CitationGraphManager(postgres_config, use_bm25=True)
        citation_manager.connect()
        print("\n1. Creating citation graph schema...")
        citation_manager.create_citation_schema()
        print("\n2. Loading citation references...")
        citation_manager.load_all_references(Path(reference_dir))
        print("\n3. Resolving citations (hybrid: database + BM25)...")
        citation_manager.resolve_citations()
        print("\n4. Building citation graph edges...")
        citation_manager.build_citation_edges()
        print("\n5. Citation graph statistics:")
        citation_manager.get_stats()
        citation_manager.close()

    loader.close()

    if results:
        print("\n✓ Database load complete")
    else:
        print("\n⚠ No chunks loaded (check OUTPUT/Chunked_text)")


if __name__ == "__main__":
    postgres_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),
        "database": os.getenv("POSTGRES_DB", "research_papers"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
    }
    run(postgres_config=postgres_config)
