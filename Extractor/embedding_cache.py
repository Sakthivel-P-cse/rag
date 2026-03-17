"""Embedding cache utilities.

Provides a simple SQLite-backed cache to avoid recomputing embeddings for
unchanged chunks. Keys are (chunk_id, text_hash, model_name).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class EmbeddingCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                chunk_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (chunk_id, text_hash, model_name)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def get(self, chunk_id: str, text_hash: str, model_name: str) -> Optional[np.ndarray]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT dim, vector FROM embedding_cache WHERE chunk_id=? AND text_hash=? AND model_name=?",
            (chunk_id, text_hash, model_name),
        )
        row = cur.fetchone()
        if not row:
            return None
        dim, blob = row
        arr = np.frombuffer(blob, dtype="float32")
        if arr.size != dim:
            return None
        return arr

    def put(self, chunk_id: str, text_hash: str, model_name: str, vector: np.ndarray) -> None:
        arr = np.asarray(vector, dtype="float32").reshape(-1)
        dim = int(arr.size)
        blob = arr.tobytes()
        self._conn.execute(
            "REPLACE INTO embedding_cache (chunk_id, text_hash, model_name, dim, vector) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, text_hash, model_name, dim, blob),
        )
        self._conn.commit()

    def get_many(self, keys: Tuple[Tuple[str, str, str], ...]) -> dict:
        """Batch get; returns a mapping from key tuple to vector.

        Keys are (chunk_id, text_hash, model_name).
        """
        if not keys:
            return {}
        cur = self._conn.cursor()
        placeholders = ",".join(["(?, ?, ?)"] * len(keys))
        flat: list[str] = []
        for cid, th, mn in keys:
            flat.extend([cid, th, mn])
        query = (
            "SELECT chunk_id, text_hash, model_name, dim, vector FROM embedding_cache WHERE (chunk_id, text_hash, model_name) IN ("
            + placeholders
            + ")"
        )
        cur.execute(query, flat)
        out = {}
        for cid, th, mn, dim, blob in cur.fetchall():
            arr = np.frombuffer(blob, dtype="float32")
            if arr.size != dim:
                continue
            out[(cid, th, mn)] = arr
        return out
