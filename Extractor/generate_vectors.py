"""Generate vector embeddings from PostgreSQL chunks and load into Qdrant."""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
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
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_client = None
        self.model_1 = None
        self.model_2 = None

    def connect_qdrant(self) -> None:
        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        print(f"✓ Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")

    def load_embedding_models(self) -> None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        print("\nLoading embedding models...")
        print("  Loading BAAI/bge-small-en-v1.5 (384 dimensions)...")
        self.model_1 = SentenceTransformer("BAAI/bge-small-en-v1.5")
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

    def process_chunks(self, chunks: List[Dict[str, Any]], collection_name: str = "research_papers", batch_size: int = 10) -> None:
        if not chunks:
            print("\n✓ No chunks to process!")
            return

        print(f"\nFound {len(chunks)} chunks to process")

        total_processed = 0
        points_batch: List[PointStruct] = []

        for chunk in chunks:
            try:
                # Handle potential key differences between DB schema and JSON format
                text = chunk.get("chunk_text") or chunk.get("text", "")
                if not text:
                    raise ValueError(f"No text found in chunk {chunk.get('chunk_id')}")
                vector_1 = self.model_1.encode(text, convert_to_numpy=True, show_progress_bar=False)
                vector_2 = self.model_2.encode(text, convert_to_numpy=True, show_progress_bar=False)
                vector_1 = vector_1.astype(np.float32)
                vector_2 = vector_2.astype(np.float32)

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
                            "paragraph_index": chunk.get("paragraph_index"),
                            "chunk_text": text,
                            "year": chunk.get("year"),
                        },
                    )
                )

                if len(points_batch) >= batch_size:
                    self.qdrant_client.upsert(collection_name=collection_name, points=points_batch)
                    total_processed += len(points_batch)
                    print(f"  Processed {total_processed}/{len(chunks)} chunks...")
                    points_batch = []

            except Exception as e:
                print(f"✗ Error processing chunk {chunk.get('chunk_id')}: {e}")
                continue

        if points_batch:
            self.qdrant_client.upsert(collection_name=collection_name, points=points_batch)
            total_processed += len(points_batch)

        print(f"\n✓ Successfully processed {total_processed} chunks")


import json
from pathlib import Path


def load_chunks_from_json(directory: Path, pattern: str = "*_chunks.json") -> List[Dict[str, Any]]:
    if not directory.exists():
        print(f"✗ Directory not found: {directory}")
        return []

    json_files = list(directory.glob(pattern))
    if not json_files:
        print(f"✗ No JSON files found matching '{pattern}' in {directory}")
        return []

    all_chunks = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                if isinstance(chunks, list):
                    all_chunks.extend(chunks)
        except Exception as e:
            print(f"✗ Failed to load {json_file}: {e}")
    return all_chunks


def run(
    *,
    chunked_text_dir: Path,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "research_papers",
    batch_size: int = 10,
) -> None:
    chunks = load_chunks_from_json(chunked_text_dir)
    vg = VectorGenerator(qdrant_host=qdrant_host, qdrant_port=qdrant_port)
    vg.connect_qdrant()
    vg.load_embedding_models()
    vg.create_qdrant_collection(collection_name=collection_name)
    vg.process_chunks(chunks=chunks, collection_name=collection_name, batch_size=batch_size)


if __name__ == "__main__":
    WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
    
    run(
        chunked_text_dir=WORKSPACE_ROOT / "OUTPUT" / "Chunked_text",
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "research_papers"),
        batch_size=int(os.getenv("VECTOR_BATCH_SIZE", "10")),
    )
