"""Generate vector embeddings from PostgreSQL chunks and load into Qdrant."""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import uuid


try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠ sentence-transformers not installed. Install with: pip install sentence-transformers")


def chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class VectorGenerator:
    def __init__(
        self,
        postgres_config: Dict[str, Any],
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self.postgres_config = postgres_config
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.pg_conn = None
        self.qdrant_client = None
        self.model_1 = None
        self.model_2 = None

    def connect_postgres(self) -> None:
        self.pg_conn = psycopg2.connect(
            host=self.postgres_config["host"],
            port=int(self.postgres_config.get("port", 5433)),
            database=self.postgres_config["database"],
            user=self.postgres_config["user"],
            password=self.postgres_config.get("password", ""),
        )
        print(f"✓ Connected to PostgreSQL database: {self.postgres_config['database']}")

    def connect_qdrant(self) -> None:
        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        print(f"✓ Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")

    def load_embedding_models(self) -> None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        print("\nLoading embedding models...")
        print("  Loading all-MiniLM-L6-v2 (384 dimensions)...")
        self.model_1 = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Loading all-mpnet-base-v2 (768 dimensions)...")
        self.model_2 = SentenceTransformer("all-mpnet-base-v2")
        print("✓ Embedding models loaded successfully")

    def create_qdrant_collection(self, collection_name: str = "research_papers") -> None:
        collections = self.qdrant_client.get_collections().collections
        if collection_name in [c.name for c in collections]:
            print(f"⚠ Collection '{collection_name}' already exists")
            return

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "vector_1": VectorParams(size=384, distance=Distance.COSINE),
                "vector_2": VectorParams(size=768, distance=Distance.COSINE),
            },
        )
        print(f"✓ Qdrant collection '{collection_name}' created")

    def get_chunks_without_vectors(self) -> List[Dict[str, Any]]:
        cursor = self.pg_conn.cursor()
        cursor.execute(
            """
            SELECT chunk_id, paper_id, chunk_text, section, year
            FROM chunks
            WHERE vector_generated = FALSE
            ORDER BY paper_id, chunk_id
            """
        )
        rows = cursor.fetchall()
        cursor.close()

        chunks = []
        for row in rows:
            chunks.append(
                {
                    "chunk_id": row[0],
                    "paper_id": row[1],
                    "chunk_text": row[2],
                    "section": row[3],
                    "year": row[4],
                }
            )
        return chunks

    def generate_embeddings(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        vector_1 = self.model_1.encode(text, convert_to_numpy=True, show_progress_bar=False)
        vector_2 = self.model_2.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vector_1.astype(np.float32), vector_2.astype(np.float32)

    def update_vector_status(self, chunk_ids: List[str]) -> None:
        cursor = self.pg_conn.cursor()
        cursor.execute(
            """
            UPDATE chunks
            SET vector_generated = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE chunk_id = ANY(%s)
            """,
            (chunk_ids,),
        )
        self.pg_conn.commit()
        cursor.close()

    def process_chunks(self, collection_name: str = "research_papers", batch_size: int = 10) -> None:
        chunks = self.get_chunks_without_vectors()
        if not chunks:
            print("\n✓ All chunks already have vectors generated!")
            return

        print(f"\nFound {len(chunks)} chunks to process")

        total_processed = 0
        points_batch: List[PointStruct] = []

        for chunk in chunks:
            try:
                vector_1, vector_2 = self.generate_embeddings(chunk["chunk_text"])

                points_batch.append(
                    PointStruct(
                        id=chunk_id_to_uuid(chunk["chunk_id"]),
                        vector={
                            "vector_1": vector_1.tolist(),
                            "vector_2": vector_2.tolist(),
                        },
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "paper_id": chunk["paper_id"],
                            "section": chunk.get("section"),
                            "year": chunk.get("year"),
                        },
                    )
                )

                if len(points_batch) >= batch_size:
                    self.qdrant_client.upsert(collection_name=collection_name, points=points_batch)
                    self.update_vector_status([p.payload["chunk_id"] for p in points_batch])
                    total_processed += len(points_batch)
                    print(f"  Processed {total_processed}/{len(chunks)} chunks...")
                    points_batch = []

            except Exception as e:
                print(f"✗ Error processing chunk {chunk.get('chunk_id')}: {e}")
                continue

        if points_batch:
            self.qdrant_client.upsert(collection_name=collection_name, points=points_batch)
            self.update_vector_status([p.payload["chunk_id"] for p in points_batch])
            total_processed += len(points_batch)

        print(f"\n✓ Successfully processed {total_processed} chunks")


def run(
    *,
    postgres_config: Dict[str, Any],
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "research_papers",
    batch_size: int = 10,
) -> None:
    vg = VectorGenerator(postgres_config, qdrant_host=qdrant_host, qdrant_port=qdrant_port)
    vg.connect_postgres()
    vg.connect_qdrant()
    vg.load_embedding_models()
    vg.create_qdrant_collection(collection_name=collection_name)
    vg.process_chunks(collection_name=collection_name, batch_size=batch_size)


if __name__ == "__main__":
    postgres_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),
        "database": os.getenv("POSTGRES_DB", "research_papers"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
    }
    run(
        postgres_config=postgres_config,
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "research_papers"),
        batch_size=int(os.getenv("VECTOR_BATCH_SIZE", "10")),
    )
