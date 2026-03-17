"""Generate vector embeddings from chunk JSON and load into FAISS.

This module now supports batched, device-aware embedding generation and an
optional SQLite-backed embedding cache to avoid recomputing vectors for
unchanged chunks.
"""

import os
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Extractor.database import DatabaseManager
from Extractor.embedding_cache import EmbeddingCache
from rag_utils.metrics import stage_timer, log_stage


try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠ sentence-transformers not installed. Install with: pip install sentence-transformers")


def chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class VectorGenerator:
    def __init__(self, *, index_dir: str | Path | None = None, collection_name: str = "research_papers"):
        self.index_dir = Path(index_dir or os.getenv("FAISS_INDEX_DIR") or Path(__file__).resolve().parents[1] / "faiss_store")
        self.collection_name = collection_name
        self.db = DatabaseManager(index_dir=self.index_dir, collection_name=collection_name)
        self.model_1 = None
        self.model_2 = None
        cache_path = self.index_dir / "embedding_cache.sqlite"
        self.cache = EmbeddingCache(cache_path)

    def connect_store(self) -> None:
        self.db.connect_faiss()

    def load_embedding_models(self) -> None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        print("\nLoading embedding models...")
        model_small = os.getenv("EMBED_MODEL_SMALL", "BAAI/bge-small-en-v1.5")
        # Use the same large encoder as retrieval (see RAG.py): 768-dim
        model_large = os.getenv("EMBED_MODEL_LARGE", "all-mpnet-base-v2")

        # Auto-select device: use CUDA if available
        device = os.getenv("EMBED_DEVICE")
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        print(f"  Loading {model_small} on {device} (small encoder)...")
        self.model_1 = SentenceTransformer(model_small, device=device)
        print(f"  Loading {model_large} on {device} (large encoder)...")
        self.model_2 = SentenceTransformer(model_large, device=device)
        print("✓ Embedding models loaded successfully")
    def _text_hash(self, text: str) -> str:
        h = sha256()
        h.update(text.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def generate_embeddings(self, chunks: Sequence[Dict[str, Any]], batch_size: int = 64) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks with caching and batching.

        Returns a list of enriched chunk dicts containing vector_1 and vector_2.
        """
        if not chunks:
            return []

        # Prepare cache lookup keys
        small_model_name = getattr(self.model_1, "__str__", lambda: "model_1")()
        large_model_name = getattr(self.model_2, "__str__", lambda: "model_2")()

        texts: List[str] = []
        keys_small: List[Tuple[str, str, str]] = []
        keys_large: List[Tuple[str, str, str]] = []
        for ch in chunks:
            text = ch.get("chunk_text") or ch.get("text", "")
            if not text:
                raise ValueError(f"No text found in chunk {ch.get('chunk_id')}")
            th = self._text_hash(text)
            cid = ch["chunk_id"]
            texts.append(text)
            keys_small.append((cid, th, small_model_name))
            keys_large.append((cid, th, large_model_name))

        # Cache lookup
        with stage_timer("embedding_cache_lookup", extra={"num_chunks": len(chunks)}):
            cache_small = self.cache.get_many(tuple(keys_small))
            cache_large = self.cache.get_many(tuple(keys_large))

        missing_indices_small = [i for i, k in enumerate(keys_small) if k not in cache_small]
        missing_indices_large = [i for i, k in enumerate(keys_large) if k not in cache_large]

        # Encode missing texts in batches
        def _encode_missing(model, indices: List[int], model_name: str):
            if not indices:
                return {}
            encoded: Dict[int, np.ndarray] = {}
            with stage_timer("embedding", extra={"model": model_name, "num_chunks": len(indices)}):
                for start in range(0, len(indices), batch_size):
                    batch_idx = indices[start : start + batch_size]
                    batch_texts = [texts[i] for i in batch_idx]
                    vecs = model.encode(batch_texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
                    vecs = np.asarray(vecs, dtype=np.float32)
                    for idx, vec in zip(batch_idx, vecs):
                        encoded[idx] = vec
            return encoded

        encoded_small = _encode_missing(self.model_1, missing_indices_small, small_model_name)
        encoded_large = _encode_missing(self.model_2, missing_indices_large, large_model_name)

        # Persist new embeddings into cache
        with stage_timer("embedding_cache_write", extra={"num_chunks": len(encoded_small) + len(encoded_large)}):
            for i, vec in encoded_small.items():
                cid, th, mn = keys_small[i]
                self.cache.put(cid, th, mn, vec)
            for i, vec in encoded_large.items():
                cid, th, mn = keys_large[i]
                self.cache.put(cid, th, mn, vec)

        # Build enriched chunks
        enriched: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            cid_s, th_s, mn_s = keys_small[i]
            cid_l, th_l, mn_l = keys_large[i]
            # Avoid using "or" with NumPy arrays; check None explicitly.
            v1 = cache_small.get((cid_s, th_s, mn_s))
            if v1 is None:
                v1 = encoded_small.get(i)
            v2 = cache_large.get((cid_l, th_l, mn_l))
            if v2 is None:
                v2 = encoded_large.get(i)
            if v1 is None or v2 is None:
                # This should not happen but guard defensively
                continue
            enriched.append(
                {
                    "chunk_id": ch["chunk_id"],
                    "paper_id": ch.get("paper_id", ""),
                    "paragraph_index": ch.get("paragraph_index"),
                    "chunk_text": texts[i],
                    "year": ch.get("year"),
                    "vector_1": np.asarray(v1, dtype=np.float32),
                    "vector_2": np.asarray(v2, dtype=np.float32),
                }
            )

        return enriched

    def process_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> None:
        if not chunks:
            print("\n✓ No chunks to process!")
            return

        print(f"\nFound {len(chunks)} chunks to process")

        with stage_timer("embedding_total", extra={"num_chunks": len(chunks), "batch_size": batch_size}):
            enriched = self.generate_embeddings(chunks, batch_size=batch_size)

        total = len(enriched)
        with stage_timer("faiss_insert", extra={"num_chunks": total}):
            # Insert in batches to keep memory usage bounded
            inserted = 0
            batch: List[Dict[str, Any]] = []
            for ch in enriched:
                batch.append(ch)
                if len(batch) >= batch_size:
                    self.db.insert_chunks_batch(batch)
                    inserted += len(batch)
                    print(f"  Inserted {inserted}/{total} chunks into FAISS...")
                    batch = []
            if batch:
                self.db.insert_chunks_batch(batch)

        print(f"\n✓ Successfully processed {total} chunks")


import json
from pathlib import Path


def load_chunks_from_json(directory: Path, pattern: str = "*_chunks.json") -> List[Dict[str, Any]]:
    if not directory.exists():
        print(f"✗ Directory not found: {directory}")
        return []

    # Search recursively so we pick up chunk files nested under subdirectories
    # like OUTPUT/Chunked_text/OUTPUT/text/main/...
    json_files = list(directory.rglob(pattern))
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
    faiss_index_dir: Path | None = None,
    collection_name: str = "research_papers",
    batch_size: int = 64,
) -> None:
    chunks = load_chunks_from_json(chunked_text_dir)
    vg = VectorGenerator(index_dir=faiss_index_dir, collection_name=collection_name)
    vg.connect_store()
    vg.load_embedding_models()
    vg.process_chunks(chunks=chunks, batch_size=batch_size)


if __name__ == "__main__":
    WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

    run(
        chunked_text_dir=WORKSPACE_ROOT / "OUTPUT" / "Chunked_text",
        faiss_index_dir=Path(os.getenv("FAISS_INDEX_DIR", str(WORKSPACE_ROOT / "faiss_store"))),
        collection_name=os.getenv("FAISS_COLLECTION", "research_papers"),
        batch_size=int(os.getenv("VECTOR_BATCH_SIZE", "10")),
    )
