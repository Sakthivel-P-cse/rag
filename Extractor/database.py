"""
Database Setup for Research Paper Chunking System
- FAISS: Stores dual-vector representations (2 vectors per chunk in FP32)
- Common key: chunk_id links vectors to metadata
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from rag_utils.metrics import stage_timer


def _normalize(vec: np.ndarray, dim: int) -> np.ndarray:
    """L2-normalize a vector for cosine similarity via inner product."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"Vector has dim {arr.size}, expected {dim}")
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


class DatabaseManager:
    """Manages FAISS indexes and metadata for chunked text storage."""

    def __init__(
        self,
        *,
        index_dir: str | Path | None = None,
        collection_name: str = "research_papers",
        vector_size_1: int = 384,
        vector_size_2: int = 768,
    ):
        self.collection_name = collection_name
        self.index_dir = Path(index_dir or os.getenv("FAISS_INDEX_DIR") or (Path(__file__).resolve().parents[1] / "faiss_store"))
        self.index_dir = self.index_dir.expanduser().resolve()
        self.vector_size_1 = int(vector_size_1)
        self.vector_size_2 = int(vector_size_2)

        # Index type can be configured via env var, default to flat for
        # correctness. Supported: "flat", "ivf", "hnsw".
        self.index_type = (os.getenv("FAISS_INDEX_TYPE") or "flat").lower()

        self.indexes: dict[str, faiss.IndexIDMap] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.id_to_chunk: dict[int, str] = {}
        self.chunk_to_id: dict[str, int] = {}
        self.next_id: int = 0

        self._meta_path = self.index_dir / "metadata.json"
        self._index_paths = {
            "vector_1": self.index_dir / "vector_1.index",
            "vector_2": self.index_dir / "vector_2.index",
        }

    def connect_postgres(self):
        """Deprecated stub kept for backward compatibility."""
        return None

    def connect_qdrant(self):
        """Alias kept for backward compatibility; initializes FAISS instead."""
        return self.connect_faiss()

    def connect_faiss(self):
        """Load or initialize FAISS indexes and metadata."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()
        self._load_or_init_index("vector_1", self.vector_size_1)
        self._load_or_init_index("vector_2", self.vector_size_2)
        print(f"✓ FAISS ready at {self.index_dir}")

    def setup_postgres_schema(self):
        """Deprecated stub kept for backward compatibility."""
        return None

    def setup_qdrant_collection(self, *_, **__):
        """No-op; FAISS indexes are created on connect."""
        return None

    def _build_base_index(self, dim: int) -> faiss.Index:
        """Create a FAISS index according to configured index_type.

        Defaults to IndexFlatIP for exact search. IVF/HNSW are available for
        larger collections when configured via FAISS_INDEX_TYPE.
        """
        if self.index_type == "ivf":
            # Use a heuristic for nlist; can be tuned externally later.
            nlist = int(os.getenv("FAISS_IVF_NLIST", "4096"))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            # IVF requires training before use; caller must ensure training
            return index
        if self.index_type == "hnsw":
            m = int(os.getenv("FAISS_HNSW_M", "32"))
            index = faiss.IndexHNSWFlat(dim, m)
            return index
        # Fallback: exact search
        return faiss.IndexFlatIP(dim)

    def _load_or_init_index(self, name: str, dim: int) -> None:
        path = self._index_paths[name]
        if path.exists():
            idx = faiss.read_index(str(path))
            if not isinstance(idx, faiss.IndexIDMap):
                idx = faiss.IndexIDMap(idx)
            self.indexes[name] = idx
            return
        base = self._build_base_index(dim)
        self.indexes[name] = faiss.IndexIDMap(base)

    def _load_metadata(self) -> None:
        if not self._meta_path.exists():
            self.metadata = {}
            self.id_to_chunk = {}
            self.chunk_to_id = {}
            self.next_id = 0
            return
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metadata = data.get("chunks", {}) or {}

            ids: list[int] = []
            self.chunk_to_id = {}
            self.id_to_chunk = {}
            for cid, meta in self.metadata.items():
                fid = int(meta.get("faiss_id", len(ids)))
                self.chunk_to_id[cid] = fid
                self.id_to_chunk[fid] = cid
                ids.append(fid)

            self.next_id = max(ids) + 1 if ids else 0
        except Exception:
            # Fall back to empty state on any parse error
            self.metadata = {}
            self.chunk_to_id = {}
            self.id_to_chunk = {}
            self.next_id = 0

    def _persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {"next_id": self.next_id, "chunks": self.metadata}
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        for name, idx in self.indexes.items():
            faiss.write_index(idx, str(self._index_paths[name]))

    def _reserve_id(self, chunk_id: str) -> int:
        if chunk_id in self.chunk_to_id:
            return self.chunk_to_id[chunk_id]
        fid = self.next_id
        self.next_id += 1
        self.chunk_to_id[chunk_id] = fid
        self.id_to_chunk[fid] = chunk_id
        return fid

    def insert_chunk(
        self,
        chunk_id: str,
        paper_id: str,
        chunk_text: str,
        vector_1: np.ndarray,
        vector_2: np.ndarray,
        section: Optional[str] = None,
        paragraph_index: Optional[int] = None,
        token_count: Optional[int] = None,
        overlap_tokens: Optional[int] = 0,
        overlap_percent: Optional[float] = 0.0,
        year: Optional[int] = None,
        citations_out: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        prev_chunk_id: Optional[str] = None,
        next_chunk_id: Optional[str] = None,
    ) -> bool:
        """Insert a single chunk into FAISS + metadata."""
        try:
            if chunk_id in self.metadata:
                # Avoid duplicating entries in the index; keep first write.
                return True
            with stage_timer("faiss_insert_single", extra={"paper_id": paper_id}):
                fid = self._reserve_id(chunk_id)
                v1 = _normalize(vector_1, self.vector_size_1)
                v2 = _normalize(vector_2, self.vector_size_2)

                self.indexes["vector_1"].add_with_ids(np.expand_dims(v1, axis=0), np.array([fid], dtype=np.int64))
                self.indexes["vector_2"].add_with_ids(np.expand_dims(v2, axis=0), np.array([fid], dtype=np.int64))

                self.metadata[chunk_id] = {
                    "faiss_id": int(fid),
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "section": section,
                    "paragraph_index": paragraph_index,
                    "chunk_text": chunk_text,
                    "year": year,
                    "prev_chunk_id": prev_chunk_id,
                    "next_chunk_id": next_chunk_id,
                    "token_count": token_count,
                    "overlap_tokens": overlap_tokens,
                    "overlap_percent": overlap_percent,
                    "citations_out": citations_out or [],
                    "keywords": keywords or [],
                }
            return True
        except Exception as e:
            print(f"✗ Failed to insert chunk {chunk_id}: {e}")
            return False

    def insert_chunks_batch(self, chunks_data: List[Dict[str, Any]]) -> bool:
        """Insert multiple chunks in batch into FAISS."""
        if not chunks_data:
            print("✓ No chunks to insert")
            return True
        try:
            with stage_timer("faiss_insert_batch", extra={"num_chunks": len(chunks_data)}):
                for chunk in chunks_data:
                    if not chunk:
                        continue
                    cid = chunk.get("chunk_id")
                    if not cid:
                        continue
                    if cid in self.metadata:
                        # Skip duplicates to avoid growing the index indefinitely
                        continue
                    self.insert_chunk(
                        chunk_id=cid,
                        paper_id=chunk.get("paper_id", ""),
                        chunk_text=chunk.get("chunk_text") or chunk.get("text", ""),
                        vector_1=np.asarray(chunk.get("vector_1"), dtype=np.float32),
                        vector_2=np.asarray(chunk.get("vector_2"), dtype=np.float32),
                        section=chunk.get("section"),
                        paragraph_index=chunk.get("paragraph_index"),
                        year=chunk.get("year"),
                        prev_chunk_id=chunk.get("prev_chunk_id"),
                        next_chunk_id=chunk.get("next_chunk_id"),
                    )

                self._persist()
            print(f"✓ Inserted {len(chunks_data)} chunks into FAISS")
            return True
        except Exception as e:
            print(f"✗ Failed to insert batch: {e}")
            return False

    def search_similar(
        self,
        query_vector: np.ndarray,
        vector_name: str = "vector_1",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        exclude_chunk_ids: Optional[set[str]] = None,
        exclude_paper_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using FAISS.

        For IVF indexes, the FAISS_NPROBE environment variable can be used to
        tune search recall vs. latency. For HNSW, FAISS_HNSW_EFSEARCH controls
        the breadth of search.
        """
        vn = str(vector_name or "vector_1").strip()
        if vn not in self.indexes:
            return []
        index = self.indexes[vn]

        # Tune index parameters when applicable
        base = index
        if isinstance(index, faiss.IndexIDMap):
            base = index.index
        if isinstance(base, faiss.IndexIVFFlat):
            try:
                nprobe = int(os.getenv("FAISS_NPROBE", "16"))
                base.nprobe = nprobe
            except Exception:
                pass
        if isinstance(base, faiss.IndexHNSWFlat):
            try:
                ef = int(os.getenv("FAISS_HNSW_EFSEARCH", "64"))
                base.hnsw.efSearch = ef
            except Exception:
                pass

        vec = _normalize(query_vector, self.vector_size_1 if vn == "vector_1" else self.vector_size_2)
        with stage_timer("faiss_search", extra={"vector_name": vn, "limit": int(limit)}):
            scores, ids = index.search(np.expand_dims(vec, axis=0), int(max(1, limit)))

        results: List[Dict[str, Any]] = []
        exclude_chunk_ids = exclude_chunk_ids or set()
        exclude_paper_ids = exclude_paper_ids or set()

        for score, fid in zip(scores[0], ids[0]):
            if int(fid) == -1:
                continue
            cid = self.id_to_chunk.get(int(fid))
            if not cid:
                continue
            meta = self.metadata.get(cid) or {}
            if cid in exclude_chunk_ids:
                continue
            if meta.get("paper_id") and meta.get("paper_id") in exclude_paper_ids:
                continue
            if score_threshold is not None and float(score) < float(score_threshold):
                continue
            results.append(
                {
                    "chunk_id": cid,
                    "score": float(score),
                    "chunk_text": meta.get("chunk_text", ""),
                    "paper_id": meta.get("paper_id", ""),
                    "section": meta.get("section", ""),
                    "year": meta.get("year"),
                }
            )
        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata by ID from the local store."""
        cid = str(chunk_id)
        meta = self.metadata.get(cid)
        if not meta:
            return None
        return dict(meta)

    def get_chunks_by_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific paper from the local store."""
        pid = str(paper_id)
        all_chunks = [dict(m) for m in self.metadata.values() if str(m.get("paper_id")) == pid]
        all_chunks.sort(key=lambda x: x.get("paragraph_index") or 0)
        return all_chunks

    def get_chunk_ids_by_paper(self, paper_id: str) -> List[str]:
        chunks = self.get_chunks_by_paper(paper_id)
        return [c.get("chunk_id") for c in chunks if c.get("chunk_id")]

    def close(self):
        """Persist indexes and metadata."""
        self._persist()
        print("✓ FAISS indexes saved")


# Example usage and testing
if __name__ == "__main__":
    db = DatabaseManager()
    db.connect_faiss()
    print("\n✓ FAISS connected and ready")
    db.close()
