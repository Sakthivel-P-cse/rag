"""
Database Setup for Research Paper Chunking System
- PostgreSQL: Stores chunked texts and metadata
- Qdrant: Stores vector representations (2 vectors per chunk in FP32)
- Common key: chunk_id links both databases
"""

import psycopg2
from psycopg2.extras import execute_values
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
        postgres_config: Dict[str, str],
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "research_papers"
    ):
        """
        Initialize database connections.
        
        Args:
            postgres_config: Dict with keys: host, port, database, user, password
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name for Qdrant collection
        """
        self.postgres_config = postgres_config
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        
        # Initialize connections
        self.pg_conn = None
        self.qdrant_client = None
        
    def connect_postgres(self):
        """Establish PostgreSQL connection."""
        try:
            self.pg_conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config.get('port', 5432),
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config.get('password', '')
            )
            print(f"✓ Connected to PostgreSQL database: {self.postgres_config['database']}")
        except Exception as e:
            print(f"✗ Failed to connect to PostgreSQL: {e}")
            raise
    
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
        """Create PostgreSQL tables for storing chunk metadata and text."""
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            paper_id VARCHAR(255) NOT NULL,
            section VARCHAR(500),
            paragraph_index INTEGER,
            chunk_text TEXT NOT NULL,
            year INTEGER,

            -- Vector processing status
            vector_generated BOOLEAN DEFAULT FALSE,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
        CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);
        CREATE INDEX IF NOT EXISTS idx_chunks_vector_generated ON chunks(vector_generated);

        CREATE TABLE IF NOT EXISTS papers (
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
        """
        
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute(create_table_sql)
            self.pg_conn.commit()
            cursor.close()
            print("✓ PostgreSQL schema created successfully")
        except Exception as e:
            print(f"✗ Failed to create PostgreSQL schema: {e}")
            self.pg_conn.rollback()
            raise
    
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
            # Insert into PostgreSQL
            insert_sql = """
            INSERT INTO chunks (
                chunk_id, paper_id, section, paragraph_index, chunk_text, year, vector_generated
            ) VALUES (%s, %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT (chunk_id) DO UPDATE SET
                paper_id = EXCLUDED.paper_id,
                section = EXCLUDED.section,
                paragraph_index = EXCLUDED.paragraph_index,
                chunk_text = EXCLUDED.chunk_text,
                year = EXCLUDED.year,
                vector_generated = TRUE,
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor = self.pg_conn.cursor()
            cursor.execute(insert_sql, (chunk_id, paper_id, section, paragraph_index, chunk_text, year))
            self.pg_conn.commit()
            cursor.close()
            
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
            self.pg_conn.rollback()
            return False
    
    def insert_chunks_batch(self, chunks_data: List[Dict[str, Any]]):
        """Insert multiple chunks in batch into PostgreSQL + Qdrant."""
        try:
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
                vector_generated = TRUE,
                updated_at = CURRENT_TIMESTAMP
            """

            pg_values = []
            qdrant_points = []

            for chunk in chunks_data:
                pg_values.append(
                    (
                        chunk["chunk_id"],
                        chunk["paper_id"],
                        chunk.get("section"),
                        chunk.get("paragraph_index"),
                        chunk["chunk_text"],
                        chunk.get("year"),
                        True,
                    )
                )

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
                            "year": chunk.get("year"),
                        },
                    )
                )

            cursor = self.pg_conn.cursor()
            execute_values(cursor, insert_sql, pg_values)
            self.pg_conn.commit()
            cursor.close()

            self.qdrant_client.upsert(collection_name=self.collection_name, points=qdrant_points)

            print(f"✓ Inserted {len(chunks_data)} chunks successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to insert batch: {e}")
            self.pg_conn.rollback()
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
                # Get chunk_id from payload (original string ID)
                chunk_id = hit.payload.get('chunk_id', str(hit.id))
                
                # Get full metadata from PostgreSQL
                cursor = self.pg_conn.cursor()
                cursor.execute(
                    """SELECT chunk_id, paper_id, section, paragraph_index, chunk_text, year
                       FROM chunks WHERE chunk_id = %s""",
                    (chunk_id,)
                )
                row = cursor.fetchone()
                cursor.close()
                
                if row:
                    results.append({
                        "chunk_id": chunk_id,
                        "score": hit.score,
                        "chunk_text": row[4],
                        "paper_id": row[1],
                        "section": row[2],
                        "year": row[5]
                    })
            
            return results
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata from PostgreSQL by ID."""
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute(
                """SELECT chunk_id, paper_id, section, paragraph_index, chunk_text, year, created_at
                   FROM chunks WHERE chunk_id = %s""",
                (chunk_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return {
                    "chunk_id": row[0],
                    "paper_id": row[1],
                    "section": row[2],
                    "paragraph_index": row[3],
                    "chunk_text": row[4],
                    "year": row[5],
                    "created_at": row[6]
                }
            return None
        except Exception as e:
            print(f"✗ Failed to get chunk: {e}")
            return None
    
    def get_chunks_by_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific paper."""
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute(
                """SELECT chunk_id, section, paragraph_index, chunk_text, year
                   FROM chunks WHERE paper_id = %s
                   ORDER BY paragraph_index""",
                (paper_id,)
            )
            rows = cursor.fetchall()
            cursor.close()
            
            chunks = []
            for row in rows:
                chunks.append({
                    "chunk_id": row[0],
                    "section": row[1],
                    "paragraph_index": row[2],
                    "chunk_text": row[3],
                    "year": row[4]
                })
            return chunks
        except Exception as e:
            print(f"✗ Failed to get chunks for paper: {e}")
            return []

    def get_chunk_ids_by_paper(self, paper_id: str) -> List[str]:
        """Retrieve all chunk_ids for a specific paper (lightweight)."""
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute(
                """SELECT chunk_id
                   FROM chunks
                   WHERE paper_id = %s
                   ORDER BY paragraph_index""",
                (paper_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
            return [str(r[0]) for r in rows if r and r[0] is not None]
        except Exception as e:
            print(f"✗ Failed to get chunk ids for paper: {e}")
            return []
    
    def close(self):
        """Close database connections."""
        if self.pg_conn:
            self.pg_conn.close()
            print("✓ PostgreSQL connection closed")
        if self.qdrant_client:
            print("✓ Qdrant connection closed")


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    postgres_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'research_papers',
        'user': 'postgres',
        'password': 'password'
    }
    
    # Initialize database manager
    db = DatabaseManager(
        postgres_config=postgres_config,
        qdrant_host='localhost',
        qdrant_port=6333,
        collection_name='research_papers'
    )
    
    try:
        # Connect to databases
        db.connect_postgres()
        db.connect_qdrant()
        
        # Setup schemas
        db.setup_postgres_schema()
        db.setup_qdrant_collection(vector_size_1=768, vector_size_2=1024)
        
        # Example: Insert a single chunk
        example_chunk_id = "paper123_chunk-1"
        example_vector_1 = np.random.rand(768).astype(np.float32)
        example_vector_2 = np.random.rand(1024).astype(np.float32)
        
        success = db.insert_chunk(
            chunk_id=example_chunk_id,
            paper_id="paper123",
            chunk_text="This is an example chunk of text from a research paper.",
            vector_1=example_vector_1,
            vector_2=example_vector_2,
            section="Introduction",
            paragraph_index=0,
            token_count=15,
            year=2024,
            keywords=["example", "research", "paper"]
        )
        
        if success:
            print(f"\n✓ Example chunk inserted successfully")
            
            # Retrieve the chunk
            chunk = db.get_chunk_by_id(example_chunk_id)
            if chunk:
                print(f"\nRetrieved chunk:")
                print(f"  ID: {chunk['chunk_id']}")
                print(f"  Text: {chunk['chunk_text'][:50]}...")
                print(f"  Keywords: {chunk['keywords']}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        db.close()
