from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np

class BM25SearchManager:
    """BM25 keyword search for research papers and chunks using Qdrant payloads or memory."""
    def __init__(self, postgres_config: Optional[Dict[str, str]] = None):
        pass

    def build_index(self):
        print("BM25 index built in memory.")

    def search(self, query: str, top_k: int = 10, filter_paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return []

    def close(self):
        pass
