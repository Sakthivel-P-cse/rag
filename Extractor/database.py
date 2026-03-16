"""
Database Setup for Research Paper Chunking System
- PostgreSQL: Stores chunked texts and metadata
- Qdrant: Stores vector representations (2 vectors per chunk in FP32)
- Common key: chunk_id links both databases
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert chunk_id string to a valid UUID for Qdrant."""
    # Generate a deterministic UUID from the chunk_id string
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class DatabaseManager:
    """Manages PostgreSQL and Qdrant databases for chunked text storage."""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "research_papers"
    ):
        """
        Initialize database connections.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name for Qdrant collection
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        
        # Initialize connections
        self.qdrant_client = None
        
    def connect_postgres(self):
        """Deprecated."""
        pass
    
    def connect_qdrant(self):
        """Establish Qdrant connection."""
        try:
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            print(f"✓ Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        except Exception as e:
            print(f"✗ Failed to connect to Qdrant: {e}")
            raise
    
    def setup_postgres_schema(self):
        """Deprecated."""
        pass
    
    def setup_qdrant_collection(self, vector_size_1: int = 768, vector_size_2: int = 1024):
        """
        Create Qdrant collection with two vector fields.
        
        Args:
            vector_size_1: Dimension of first vector (e.g., sentence-transformers)
            vector_size_2: Dimension of second vector (e.g., custom embeddings)
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                print(f"⚠ Collection '{self.collection_name}' already exists")
                return
            
            # Create collection with named vectors (FP32 by default in Qdrant)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "vector_1": VectorParams(
                        size=vector_size_1,
                        distance=Distance.COSINE
                    ),
                    "vector_2": VectorParams(
                        size=vector_size_2,
                        distance=Distance.COSINE
                    )
                }
            )
            print(f"✓ Qdrant collection '{self.collection_name}' created")
            print(f"  - vector_1: {vector_size_1} dimensions (FP32, Cosine)")
            print(f"  - vector_2: {vector_size_2} dimensions (FP32, Cosine)")
        except Exception as e:
            print(f"✗ Failed to create Qdrant collection: {e}")
            raise
    
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
        next_chunk_id: Optional[str] = None
    ):
        """
        Insert chunk into both PostgreSQL and Qdrant.
        
        Args:
            chunk_id: Unique identifier (common key)
            paper_id: Paper identifier
            chunk_text: The actual text content
            vector_1: First vector representation (FP32 numpy array)
            vector_2: Second vector representation (FP32 numpy array)
            section: Section heading
            paragraph_index: Index of starting paragraph
            token_count: Number of tokens in chunk
            overlap_tokens: Number of overlapping tokens
            overlap_percent: Percentage of overlap
            year: Publication year
            citations_out: List of citation IDs
            keywords: List of keywords
            prev_chunk_id: ID of previous chunk
            next_chunk_id: ID of next chunk
        """
        try:
            # Insert into Qdrant
            point = PointStruct(
                id=chunk_id_to_uuid(chunk_id),  # Convert to UUID for Qdrant
                vector={
                    "vector_1": vector_1.astype(np.float32).tolist(),
                    "vector_2": vector_2.astype(np.float32).tolist()
                },
                payload={
                    "chunk_id": chunk_id,  # Store original chunk_id in payload
                    "paper_id": paper_id,
                    "section": section,
                    "paragraph_index": paragraph_index,
                    "chunk_text": chunk_text,
                    "year": year
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
        except Exception as e:
            print(f"✗ Failed to insert chunk {chunk_id}: {e}")
            return False
    
    def insert_chunks_batch(self, chunks_data: List[Dict[str, Any]]):
        """Insert multiple chunks in batch into Qdrant."""
        try:
            qdrant_points = []

            for chunk in chunks_data:
                qdrant_points.append(
                    PointStruct(
                        id=chunk_id_to_uuid(chunk["chunk_id"]),
                        vector={
                            "vector_1": chunk["vector_1"].astype(np.float32).tolist(),
                            "vector_2": chunk["vector_2"].astype(np.float32).tolist(),
                        },
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "paper_id": chunk["paper_id"],
                            "section": chunk.get("section", ""),
                            "paragraph_index": chunk.get("paragraph_index"),
                            "chunk_text": chunk["chunk_text"],
                            "year": chunk.get("year"),
                        },
                    )
                )

            self.qdrant_client.upsert(collection_name=self.collection_name, points=qdrant_points)

            print(f"✓ Inserted {len(chunks_data)} chunks successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to insert batch: {e}")
            return False
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        vector_name: str = "vector_1",
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_vector: Query vector (FP32 numpy array)
            vector_name: Which vector to search ("vector_1" or "vector_2")
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of dicts with chunk_id, score, and metadata
        """
        try:
            # Use query method for newer qdrant-client versions
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector.astype(np.float32).tolist(),
                using=vector_name,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for hit in search_result.points:
                # Get full metadata from Qdrant Payload
                chunk_id = hit.payload.get('chunk_id', str(hit.id))
                
                results.append({
                    "chunk_id": chunk_id,
                    "score": hit.score,
                    "chunk_text": hit.payload.get("chunk_text", ""),
                    "paper_id": hit.payload.get("paper_id", ""),
                    "section": hit.payload.get("section", ""),
                    "year": hit.payload.get("year")
                })
            
            return results
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata from Qdrant by ID."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            res = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchValue(value=chunk_id)
                        )
                    ]
                ),
                limit=1
            )
            records, _ = res
            if records:
                hit = records[0]
                return {
                    "chunk_id": hit.payload.get("chunk_id"),
                    "paper_id": hit.payload.get("paper_id"),
                    "section": hit.payload.get("section"),
                    "paragraph_index": hit.payload.get("paragraph_index"),
                    "chunk_text": hit.payload.get("chunk_text"),
                    "year": hit.payload.get("year"),
                    "created_at": None
                }
            return None
        except Exception as e:
            print(f"✗ Failed to get chunk: {e}")
            return None
    
    def get_chunks_by_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific paper using Qdrant."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            all_chunks = []
            offset = None
            while True:
                res = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="paper_id",
                                match=MatchValue(value=paper_id)
                            )
                        ]
                    ),
                    limit=100,
                    offset=offset
                )
                records, next_page_offset = res
                for hit in records:
                    all_chunks.append({
                        "chunk_id": hit.payload.get("chunk_id"),
                        "section": hit.payload.get("section"),
                        "paragraph_index": hit.payload.get("paragraph_index"),
                        "chunk_text": hit.payload.get("chunk_text"),
                        "year": hit.payload.get("year"),
                    })
                if next_page_offset is None:
                    break
                offset = next_page_offset
                
            all_chunks.sort(key=lambda x: x.get("paragraph_index") or 0)
            return all_chunks
        except Exception as e:
            print(f"✗ Failed to get chunks for paper: {e}")
            return []

    def get_chunk_ids_by_paper(self, paper_id: str) -> List[str]:
        """Retrieve all chunk_ids for a specific paper from Qdrant."""
        chunks = self.get_chunks_by_paper(paper_id)
        return [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
    
    def close(self):
        """Close database connections."""
        if self.qdrant_client:
            print("✓ Qdrant connection closed")


# Example usage and testing
if __name__ == "__main__":
    # Initialize database manager with Qdrant only
    db = DatabaseManager(
        qdrant_host='localhost',
        qdrant_port=6333,
        collection_name='research_papers'
    )
    
    try:
        db.connect_qdrant()
        db.setup_qdrant_collection(vector_size_1=384, vector_size_2=768)
        print("\n✓ Qdrant connected and collection ready")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        db.close()
