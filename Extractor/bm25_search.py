"""
BM25 Search Integration for Research Paper Database
====================================================
Adds keyword/sparse search capabilities using BM25 algorithm.

This complements the citation graph with traditional information retrieval:
- Find papers by keywords
- Search within paper content
- Fallback when citation matching fails
- Better retrieval accuracy

Two options:
1. Simple BM25 (using rank_bm25 library) - Easy to setup
2. ElasticSearch (production-grade) - More features
"""

from typing import List, Dict, Any, Optional
import psycopg2
from rank_bm25 import BM25Okapi
import numpy as np


class BM25SearchManager:
    """
    BM25 keyword search for research papers and chunks.
    
    Usage:
        search = BM25SearchManager(postgres_config)
        search.build_index()
        results = search.search("attention mechanism transformers", top_k=10)
    """
    
    def __init__(self, postgres_config: Dict[str, str]):
        self.postgres_config = postgres_config
        self.conn = None
        self.bm25_index = None
        self.chunk_ids = []
        self.chunk_metadata = {}
    
    def connect(self):
        """Connect to PostgreSQL database."""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config.get('port', 5433),
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config.get('password', '')
            )
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
    
    def build_index(self):
        """
        Build BM25 index from all chunks in database.
        This indexes all text content for keyword search.
        """
        print("🔨 Building BM25 index from database...")
        
        conn = self.connect()
        cursor = conn.cursor()
        
        # Get all chunks with their text
        cursor.execute("""
            SELECT c.chunk_id, c.paper_id, c.section, c.chunk_text,
                   p.title, p.year, p.authors
            FROM chunks c
            LEFT JOIN papers p ON c.paper_id = p.paper_id
            ORDER BY c.chunk_id
        """)
        
        corpus = []
        
        for row in cursor.fetchall():
            chunk_id, paper_id, section, text, title, year, authors = row
            
            # Tokenize text (simple word splitting)
            tokens = self._tokenize(text)
            corpus.append(tokens)
            
            # Store chunk_id and metadata
            self.chunk_ids.append(chunk_id)
            self.chunk_metadata[chunk_id] = {
                'paper_id': paper_id,
                'section': section,
                'text': text[:500],  # Store first 500 chars for preview
                'paper_title': title,
                'paper_year': year,
                'paper_authors': authors
            }
        
        cursor.close()
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(corpus)
        
        print(f"✅ BM25 index built with {len(self.chunk_ids)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split by whitespace.
        You can enhance this with stemming/lemmatization if needed.
        """
        return text.lower().split()
    
    def search(self, query: str, top_k: int = 10, 
               filter_paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using BM25.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filter_paper_ids: Optional list of paper_ids to search within
        
        Returns:
            List of dicts with chunk_id, score, and metadata
        """
        if not self.bm25_index:
            raise ValueError("Index not built! Call build_index() first")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Create results with scores
        results = []
        for idx, score in enumerate(scores):
            chunk_id = self.chunk_ids[idx]
            metadata = self.chunk_metadata[chunk_id]
            
            # Apply paper filter if specified
            if filter_paper_ids and metadata['paper_id'] not in filter_paper_ids:
                continue
            
            results.append({
                'chunk_id': chunk_id,
                'score': float(score),
                **metadata
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def search_in_cited_papers(self, source_chunk_id: str, query: str, 
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search within papers cited by a specific chunk.
        
        This combines citation graph + BM25 search:
        1. Find papers cited in source_chunk
        2. Search only within those papers using BM25
        
        Perfect for focused retrieval!
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Get cited paper_ids for this chunk
        cursor.execute("""
            SELECT DISTINCT r.cited_paper_id
            FROM chunks c
            CROSS JOIN UNNEST(c.citations_out) as citation_num
            JOIN references r ON r.citing_paper_id = c.paper_id 
                AND r.citation_number = citation_num
            WHERE c.chunk_id = %s
              AND r.cited_paper_id IS NOT NULL
        """, (source_chunk_id,))
        
        cited_paper_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        if not cited_paper_ids:
            print(f"⚠ No cited papers found for chunk {source_chunk_id}")
            return []
        
        # Search only within cited papers
        return self.search(query, top_k=top_k, filter_paper_ids=cited_paper_ids)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index."""
        if not self.bm25_index:
            return {'status': 'not_built'}
        
        return {
            'status': 'ready',
            'total_chunks': len(self.chunk_ids),
            'unique_papers': len(set(m['paper_id'] for m in self.chunk_metadata.values()))
        }


# Example usage
if __name__ == "__main__":
    postgres_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'research_papers',
        'user': 'postgres',
        'password': 'password'
    }
    
    print("🔍 BM25 Search System Demo\n")
    print("=" * 70)
    
    # Initialize
    search = BM25SearchManager(postgres_config)
    
    # Build index
    search.build_index()
    
    # Show stats
    stats = search.get_index_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Unique papers: {stats['unique_papers']}")
    
    # Test search
    print(f"\n🔍 Test Search: 'attention mechanism'")
    print("-" * 70)
    
    results = search.search("attention mechanism", top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Chunk: {result['chunk_id']}")
        print(f"   Paper: {result['paper_title']}")
        print(f"   Section: {result['section']}")
        print(f"   Preview: {result['text'][:100]}...")
    
    search.close()
    print("\n✅ BM25 search system ready for use!")
