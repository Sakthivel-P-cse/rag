"""
Citation Graph Database System
================================
Simple database system to manage research paper citations.

Key Features:
- Store citation metadata (which paper cites which)
- Match citations across multiple sources (arXiv, DOI, title)
- BM25/ElasticSearch integration for better retrieval

Database Tables:
1. papers - Paper metadata (already exists)
2. references - Citation number → cited paper mapping
3. citation_edges - Citation relationships between papers
"""

import json
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional, Tuple
import re
from Extractor.bm25_search import BM25SearchManager


class CitationGraphManager:
    """Manage citation database and retrieval with BM25 search integration."""
    
    def __init__(self, postgres_config: Dict[str, str], use_bm25: bool = True):
        self.postgres_config = postgres_config
        self.conn = None
        self.use_bm25 = use_bm25
        self.bm25_search = None
        
        # Initialize BM25 if enabled
        if self.use_bm25:
            try:
                self.bm25_search = BM25SearchManager(postgres_config)
                print("✓ BM25 search integration enabled")
            except Exception as e:
                print(f"⚠ BM25 search disabled: {e}")
                self.use_bm25 = False
    
    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config.get('port', 5433),
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config.get('password', '')
            )
            print(f"✓ Connected to PostgreSQL: {self.postgres_config['database']}")
        except Exception as e:
            print(f"✗ Failed to connect to PostgreSQL: {e}")
            raise
    
    def create_citation_schema(self):
        """Create tables for citation graph system."""
        
        schema_sql = """
        -- References table: Maps citation numbers to cited paper metadata
        -- This allows: citation number [1] → which paper it refers to
        CREATE TABLE IF NOT EXISTS "references" (
            reference_id SERIAL PRIMARY KEY,
            citing_paper_id VARCHAR(255) NOT NULL,  -- Paper that has the citation
            citation_number VARCHAR(10) NOT NULL,    -- The number in brackets [1], [2], etc.
            
            -- Cited paper metadata (from reference JSON)
            bibl_id VARCHAR(50),                     -- b0, b1, b2, etc from reference JSON
            cited_title TEXT,
            cited_authors TEXT[],
            cited_year INTEGER,
            cited_arxiv_id VARCHAR(100),
            cited_doi VARCHAR(100),
            cited_container TEXT,                    -- Journal/Conference name
            
            -- Resolved paper_id (if the cited paper exists in our database)
            cited_paper_id VARCHAR(255),             -- Foreign key to papers table
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Foreign keys
            CONSTRAINT fk_citing_paper FOREIGN KEY (citing_paper_id) 
                REFERENCES papers(paper_id) ON DELETE CASCADE,
            CONSTRAINT fk_cited_paper FOREIGN KEY (cited_paper_id) 
                REFERENCES papers(paper_id) ON DELETE SET NULL,
            
            -- Unique constraint: each citation number is unique per paper
            CONSTRAINT unique_citation_per_paper UNIQUE (citing_paper_id, citation_number)
        );
        
        -- Citation edges table: Graph representation for efficient traversal
        -- This enables multi-hop queries: Paper A → Paper B → Paper C
        CREATE TABLE IF NOT EXISTS citation_edges (
            edge_id SERIAL PRIMARY KEY,
            source_paper_id VARCHAR(255) NOT NULL,   -- Paper that cites
            target_paper_id VARCHAR(255) NOT NULL,   -- Paper being cited
            citation_numbers TEXT[],                 -- Which numbers cite this paper [1,5,12]
            citation_count INTEGER DEFAULT 1,        -- How many times cited
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Foreign keys
            CONSTRAINT fk_source_paper FOREIGN KEY (source_paper_id) 
                REFERENCES papers(paper_id) ON DELETE CASCADE,
            CONSTRAINT fk_target_paper FOREIGN KEY (target_paper_id) 
                REFERENCES papers(paper_id) ON DELETE CASCADE,
            
            -- Unique constraint: one edge per paper pair
            CONSTRAINT unique_citation_edge UNIQUE (source_paper_id, target_paper_id)
        );
        
        -- Create indexes for efficient citation lookups
        CREATE INDEX IF NOT EXISTS idx_citing_paper ON "references"(citing_paper_id);
        CREATE INDEX IF NOT EXISTS idx_citation_number ON "references"(citation_number);
        CREATE INDEX IF NOT EXISTS idx_cited_paper ON "references"(cited_paper_id);
        CREATE INDEX IF NOT EXISTS idx_cited_arxiv ON "references"(cited_arxiv_id);
        CREATE INDEX IF NOT EXISTS idx_cited_doi ON "references"(cited_doi);
        CREATE INDEX IF NOT EXISTS idx_cited_title ON "references"(cited_title);
        
        CREATE INDEX IF NOT EXISTS idx_edge_source ON citation_edges(source_paper_id);
        CREATE INDEX IF NOT EXISTS idx_edge_target ON citation_edges(target_paper_id);
        

        """
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(schema_sql)
            self.conn.commit()
            cursor.close()
            print("✓ Citation graph schema created successfully")
            print("  - references table: citation number → cited paper mapping")
            print("  - citation_edges table: graph representation for multi-hop")
        except Exception as e:
            print(f"✗ Failed to create citation graph schema: {e}")
            raise
    
    def extract_arxiv_id(self, idnos: List[Dict]) -> Optional[str]:
        """Extract arXiv ID from idnos array."""
        for idno in idnos:
            if idno.get('type') == 'arXiv':
                return idno.get('value')
        return None
    
    def extract_doi(self, idnos: List[Dict]) -> Optional[str]:
        """Extract DOI from idnos array."""
        for idno in idnos:
            if idno.get('type') in ['DOI', 'doi']:
                return idno.get('value')
        return None
    
    def normalize_arxiv_id(self, arxiv_id: str) -> str:
        """Normalize arXiv ID for matching (remove version, clean format)."""
        if not arxiv_id:
            return ""
        # Remove version number (v1, v2, etc.)
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        # Remove arXiv: prefix if present
        arxiv_id = arxiv_id.replace('arXiv:', '').strip()
        # Remove [cs.CL] or similar category tags
        arxiv_id = re.sub(r'\[.*?\]', '', arxiv_id).strip()
        return arxiv_id
    
    def load_references_from_json(self, reference_json_path: Path):
        """
        Load references from a single reference JSON file.
        Maps citation numbers to cited paper metadata.
        """
        try:
            with open(reference_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            citing_paper_id = data.get('paper_id')
            if not citing_paper_id:
                print(f"⚠ No paper_id in {reference_json_path.name}")
                return
            
            references = data.get('references', [])
            if not references:
                print(f"⚠ No references in {reference_json_path.name}")
                return
            
            # Prepare reference records
            reference_records = []
            
            for idx, ref in enumerate(references):
                # Citation number is the index + 1 (since citations are 1-indexed)
                citation_number = str(idx + 1)
                
                # Extract metadata
                bibl_id = ref.get('bibl_id', '')
                cited_title = ref.get('title', '').strip()
                cited_year = ref.get('year')
                cited_container = ref.get('container_title', '')
                
                # Extract authors
                authors = ref.get('authors', [])
                cited_authors = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip() 
                    for a in authors if a
                ]
                
                # Extract identifiers
                idnos = ref.get('idnos', [])
                cited_arxiv_id = self.extract_arxiv_id(idnos)
                cited_doi = self.extract_doi(idnos)
                
                reference_records.append((
                    citing_paper_id,
                    citation_number,
                    bibl_id,
                    cited_title,
                    cited_authors,
                    cited_year,
                    cited_arxiv_id,
                    cited_doi,
                    cited_container
                ))
            
            # Insert into database
            cursor = self.conn.cursor()
            insert_sql = """
                INSERT INTO "references" (
                    citing_paper_id, citation_number, bibl_id, 
                    cited_title, cited_authors, cited_year,
                    cited_arxiv_id, cited_doi, cited_container
                ) VALUES %s
                ON CONFLICT (citing_paper_id, citation_number) 
                DO UPDATE SET
                    bibl_id = EXCLUDED.bibl_id,
                    cited_title = EXCLUDED.cited_title,
                    cited_authors = EXCLUDED.cited_authors,
                    cited_year = EXCLUDED.cited_year,
                    cited_arxiv_id = EXCLUDED.cited_arxiv_id,
                    cited_doi = EXCLUDED.cited_doi,
                    cited_container = EXCLUDED.cited_container
            """
            
            execute_values(cursor, insert_sql, reference_records)
            self.conn.commit()
            cursor.close()
            
            print(f"✓ Loaded {len(reference_records)} references from {reference_json_path.name}")
            
        except Exception as e:
            print(f"✗ Failed to load references from {reference_json_path.name}: {e}")
            raise
    
    def load_all_references(self, reference_dir: Path):
        """Load references from all JSON files in the directory."""
        json_files = list(reference_dir.glob("*.json"))
        print(f"\n📚 Loading references from {len(json_files)} files...")
        
        for json_file in json_files:
            self.load_references_from_json(json_file)
        
        print(f"\n✓ All references loaded successfully")
    
    def resolve_citations(self):
        """
        Resolve citations by matching cited papers to papers in database.
        Updates references.cited_paper_id when a match is found.
        
        Matching strategy:
        1. Match by arXiv ID (most reliable)
        2. Match by DOI
        3. Match by title similarity (fuzzy matching)
        """
        cursor = self.conn.cursor()
        
        print("\n🔍 Resolving citations to papers in database...")
        
        # Strategy 1: Match by arXiv ID
        resolve_arxiv_sql = """
            UPDATE "references" r
            SET cited_paper_id = p.paper_id
            FROM papers p
            WHERE r.cited_paper_id IS NULL
              AND r.cited_arxiv_id IS NOT NULL
              AND p.arxiv_id IS NOT NULL
              AND REGEXP_REPLACE(r.cited_arxiv_id, 'v\\d+|arXiv:|\\[.*?\\]', '', 'g') = 
                  REGEXP_REPLACE(p.arxiv_id, 'v\\d+|arXiv:|\\[.*?\\]', '', 'g')
        """
        
        cursor.execute(resolve_arxiv_sql)
        arxiv_matches = cursor.rowcount
        self.conn.commit()
        print(f"  ✓ Matched {arxiv_matches} citations by arXiv ID")
        
        # Strategy 2: Match by exact title (case-insensitive)
        resolve_title_sql = """
            UPDATE "references" r
            SET cited_paper_id = p.paper_id
            FROM papers p
            WHERE r.cited_paper_id IS NULL
              AND r.cited_title IS NOT NULL
              AND p.title IS NOT NULL
              AND LOWER(TRIM(r.cited_title)) = LOWER(TRIM(p.title))
        """
        
        cursor.execute(resolve_title_sql)
        title_matches = cursor.rowcount
        self.conn.commit()
        print(f"  ✓ Matched {title_matches} citations by exact title")
        
        # Strategy 3: Fuzzy title matching for papers from non-arXiv sources
        # This handles IEEE, ACM, ACL, PubMed papers with slight title variations
        print(f"  🔍 Fuzzy matching for remaining citations (IEEE, ACM, ACL, etc.)...")
        fuzzy_matches = self._fuzzy_title_matching(cursor)
        print(f"  ✓ Matched {fuzzy_matches} citations by fuzzy title matching")
        
        # Strategy 4: Match by title + year (even more flexible)
        resolve_title_year_sql = """
            UPDATE "references" r
            SET cited_paper_id = p.paper_id
            FROM papers p
            WHERE r.cited_paper_id IS NULL
              AND r.cited_title IS NOT NULL
              AND r.cited_year IS NOT NULL
              AND p.title IS NOT NULL
              AND p.year IS NOT NULL
              AND r.cited_year = p.year
              AND similarity(LOWER(r.cited_title), LOWER(p.title)) > 0.75
        """
        
        # Check if pg_trgm extension is available for similarity
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
        has_trgm = cursor.fetchone()[0]
        
        if has_trgm:
            cursor.execute(resolve_title_year_sql)
            title_year_matches = cursor.rowcount
            self.conn.commit()
            print(f"  ✓ Matched {title_year_matches} citations by title + year similarity")
        else:
            title_year_matches = 0
            print(f"  ⚠ pg_trgm extension not available, skipping title+year fuzzy matching")
        
        # Strategy 4: BM25 title search for remaining unresolved citations
        bm25_matches = 0
        if self.use_bm25 and self.bm25_search:
            print(f"  🔍 Using BM25 search for remaining citations...")
            bm25_matches = self._resolve_with_bm25(cursor)
            print(f"  ✓ Matched {bm25_matches} citations by BM25 title search")
        
        cursor.close()
        
        total_resolved = arxiv_matches + title_matches + fuzzy_matches + title_year_matches + bm25_matches
        print(f"\n✓ Total citations resolved: {total_resolved}")
        print(f"  📊 By arXiv ID: {arxiv_matches}")
        print(f"  📊 By exact title: {title_matches}")
        print(f"  📊 By fuzzy title: {fuzzy_matches}")
        print(f"  📊 By title+year: {title_year_matches}")
        print(f"  📊 By BM25 search: {bm25_matches}")
        
        return total_resolved
    
    def _fuzzy_title_matching(self, cursor) -> int:
        """
        Advanced fuzzy matching for papers from ANY source (IEEE, ACM, ACL, PubMed, etc.)
        
        This handles:
        - Title variations (punctuation, capitalization)
        - Papers without arXiv ID or DOI
        - Conference papers, journal papers from any publisher
        - Papers with incomplete metadata
        
        Uses Levenshtein distance for similarity scoring.
        """
        # Get all unresolved references with titles
        cursor.execute("""
            SELECT reference_id, cited_title, cited_year, cited_authors
            FROM "references"
            WHERE cited_paper_id IS NULL
              AND cited_title IS NOT NULL
              AND LENGTH(TRIM(cited_title)) > 10
        """)
        
        unresolved = cursor.fetchall()
        
        if not unresolved:
            return 0
        
        # Get all papers from database for matching
        cursor.execute("""
            SELECT paper_id, title, year, authors
            FROM papers
            WHERE title IS NOT NULL
        """)
        
        all_papers = cursor.fetchall()
        
        matches = []
        
        for ref_id, ref_title, ref_year, ref_authors in unresolved:
            best_match = None
            best_score = 0.0
            
            # Normalize reference title
            ref_title_normalized = self._normalize_title_for_matching(ref_title)
            
            for paper_id, paper_title, paper_year, paper_authors in all_papers:
                # Normalize paper title
                paper_title_normalized = self._normalize_title_for_matching(paper_title)
                
                # Calculate similarity score
                score = self._calculate_similarity(ref_title_normalized, paper_title_normalized)
                
                # Boost score if years match
                if ref_year and paper_year and ref_year == paper_year:
                    score += 0.1
                
                # Boost score if first author matches
                if ref_authors and paper_authors and len(ref_authors) > 0:
                    ref_first_author = ref_authors[0].lower() if ref_authors[0] else ""
                    paper_author_str = str(paper_authors).lower() if paper_authors else ""
                    if ref_first_author and ref_first_author in paper_author_str:
                        score += 0.05
                
                # Consider it a match if score is high enough
                if score > best_score and score >= 0.80:  # 80% similarity threshold
                    best_score = score
                    best_match = paper_id
            
            # Store match if found
            if best_match:
                matches.append((best_match, ref_id))
        
        # Batch update matched references (already validated paper_ids exist)
        if matches:
            for paper_id, ref_id in matches:
                cursor.execute("""
                    UPDATE "references"
                    SET cited_paper_id = %s
                    WHERE reference_id = %s
                """, (paper_id, ref_id))
            
            self.conn.commit()
        
        return len(matches)
    
    def _resolve_with_bm25(self, cursor) -> int:
        """
        Use BM25 search to find papers by cited title.
        This is a hybrid approach: database matching + BM25 text search.
        
        Works great for:
        - Papers with slightly different titles
        - Papers from non-arXiv sources
        - Abbreviated titles in references
        """
        if not self.bm25_search:
            return 0
        
        # Build BM25 index if not built
        if not self.bm25_search.bm25_index:
            self.bm25_search.build_index()
        
        # Get all unresolved references with titles
        cursor.execute("""
            SELECT reference_id, citing_paper_id, cited_title, cited_year
            FROM "references"
            WHERE cited_paper_id IS NULL
              AND cited_title IS NOT NULL
              AND LENGTH(TRIM(cited_title)) > 15
        """)
        
        unresolved = cursor.fetchall()
        
        if not unresolved:
            return 0
        
        matches = []
        
        for ref_id, citing_paper_id, cited_title, cited_year in unresolved:
            # Search for papers using BM25
            search_results = self.bm25_search.search(
                query=cited_title,
                top_k=5  # Get top 5 candidates
            )
            
            if not search_results:
                continue
            
            # Get unique paper_ids from results (exclude citing paper itself)
            candidate_papers = {}
            for result in search_results:
                paper_id = result['paper_id']
                # Skip if this is the same paper that's doing the citing
                if paper_id == citing_paper_id:
                    continue
                if paper_id not in candidate_papers:
                    candidate_papers[paper_id] = {
                        'score': result['score'],
                        'title': result['paper_title'],
                        'year': result['paper_year']
                    }
            
            # Find best match
            best_match = None
            best_score = 0.0
            
            for paper_id, info in candidate_papers.items():
                score = info['score']
                
                # Boost score if year matches
                if cited_year and info['year'] and cited_year == info['year']:
                    score *= 1.2
                
                # Only accept if BM25 score is high enough
                if score > best_score and score >= 2.5:  # BM25 threshold
                    best_score = score
                    best_match = paper_id
            
            if best_match:
                matches.append((best_match, ref_id))
        
        # Batch update matched references
        # First verify all paper_ids exist in papers table
        valid_matches = []
        if matches:
            paper_ids_to_check = list(set([paper_id for paper_id, _ in matches]))
            cursor.execute("""
                SELECT paper_id FROM papers
                WHERE paper_id = ANY(%s)
            """, (paper_ids_to_check,))
            valid_paper_ids = set([row[0] for row in cursor.fetchall()])
            
            # Filter to only valid matches
            for paper_id, ref_id in matches:
                if paper_id in valid_paper_ids:
                    valid_matches.append((paper_id, ref_id))
            
            # Update only valid matches
            for paper_id, ref_id in valid_matches:
                cursor.execute("""
                    UPDATE "references"
                    SET cited_paper_id = %s
                    WHERE reference_id = %s
                """, (paper_id, ref_id))
            
            self.conn.commit()
        
        return len(valid_matches)
    
    def _normalize_title_for_matching(self, title: str) -> str:
        """
        Normalize title for fuzzy matching.
        Removes punctuation, extra spaces, converts to lowercase.
        """
        if not title:
            return ""
        
        # Lowercase
        title = title.lower()
        
        # Remove special characters but keep spaces
        title = re.sub(r'[^\w\s]', ' ', title)
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using simple ratio.
        Returns value between 0.0 and 1.0.
        """
        if not str1 or not str2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def build_citation_edges(self):
        """
        Build citation_edges table from resolved references.
        Creates graph edges only for citations that were successfully resolved.
        """
        cursor = self.conn.cursor()
        
        print("\n🔗 Building citation graph edges...")
        
        # Create edges by aggregating resolved citations
        build_edges_sql = """
            INSERT INTO citation_edges (
                source_paper_id, 
                target_paper_id, 
                citation_numbers,
                citation_count
            )
            SELECT 
                citing_paper_id as source_paper_id,
                cited_paper_id as target_paper_id,
                ARRAY_AGG(citation_number ORDER BY citation_number::integer) as citation_numbers,
                COUNT(*) as citation_count
            FROM "references"
            WHERE cited_paper_id IS NOT NULL
            GROUP BY citing_paper_id, cited_paper_id
            ON CONFLICT (source_paper_id, target_paper_id)
            DO UPDATE SET
                citation_numbers = EXCLUDED.citation_numbers,
                citation_count = EXCLUDED.citation_count
        """
        
        cursor.execute(build_edges_sql)
        edge_count = cursor.rowcount
        self.conn.commit()
        cursor.close()
        
        print(f"✓ Created {edge_count} citation edges")
        return edge_count
    
    def get_citations_for_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Given a chunk_id, return all citation metadata for that chunk.
        
        Returns: List of dicts with:
        - citation_number: The number in brackets [1], [2]
        - cited_title: Title of cited paper
        - cited_authors: Authors of cited paper
        - cited_year: Publication year
        - cited_paper_id: paper_id if paper exists in database (None otherwise)
        """
        cursor = self.conn.cursor()
        
        # Get the paper_id and citations_out for the chunk first
        query_paper = """
            SELECT paper_id, citations_out FROM chunks WHERE chunk_id = %s
        """
        
        try:
            cursor.execute(query_paper, (chunk_id,))
            result = cursor.fetchone()
            if not result:
                cursor.close()
                return []
            paper_id, citations_out = result
            
            if not citations_out or len(citations_out) == 0:
                cursor.close()
                return []
        except Exception as e:
            cursor.close()
            print(f"  [ERROR] Failed to get chunk info: {e}")
            return []
        
        query = """
            SELECT 
                r.citation_number,
                r.cited_title,
                r.cited_authors,
                r.cited_year,
                r.cited_arxiv_id,
                r.cited_doi,
                r.cited_paper_id,
                p.title as resolved_title,
                p.total_chunks as resolved_chunks
            FROM "references" r
            LEFT JOIN papers p ON p.paper_id = r.cited_paper_id
            WHERE r.citing_paper_id = %s
              AND r.citation_number = ANY(%s)
            ORDER BY CAST(r.citation_number AS INTEGER)
        """
        
        try:
            cursor.execute(query, (paper_id, citations_out))
            rows = cursor.fetchall()
        except Exception as e:
            cursor.close()
            print(f"  [ERROR] Failed to get citations for chunk {chunk_id}: {e}")
            return []
        
        cursor.close()
        
        results = []
        for row in rows:
            results.append({
                'citation_number': row[0],
                'cited_title': row[1],
                'cited_authors': row[2],
                'cited_year': row[3],
                'cited_arxiv_id': row[4],
                'cited_doi': row[5],
                'cited_paper_id': row[6],  # None if not in database
                'resolved_title': row[7],
                'resolved_chunks': row[8]
            })
        
        return results
    
    def get_cited_papers_for_chunk(self, chunk_id: str) -> List[str]:
        """
        Get list of paper_ids that are cited by a given chunk.
        Only returns papers that exist in the database (resolved citations).
        
        This is the key function for multi-hop retrieval:
        chunk → citation numbers → cited paper_ids → retrieve from those papers
        """
        cursor = self.conn.cursor()
        
        # Get the paper_id for the chunk first
        query_paper = """
            SELECT paper_id, citations_out FROM chunks WHERE chunk_id = %s
        """
        
        try:
            cursor.execute(query_paper, (chunk_id,))
            result = cursor.fetchone()
            if not result:
                cursor.close()
                return []
            paper_id, citations_out = result
            
            if not citations_out or len(citations_out) == 0:
                cursor.close()
                return []
        except Exception as e:
            cursor.close()
            print(f"  [ERROR] Failed to get chunk info: {e}")
            return []
        
        # Get cited paper IDs from references table
        query = """
            SELECT DISTINCT cited_paper_id
            FROM "references"
            WHERE citing_paper_id = %s
              AND citation_number = ANY(%s)
              AND cited_paper_id IS NOT NULL
        """
        
        try:
            cursor.execute(query, (paper_id, citations_out))
            cited_paper_ids = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            cited_paper_ids = []
            print(f"  [ERROR] Failed to get cited papers: {e}")
        
        cursor.close()
        
        return cited_paper_ids
    
    def get_chunks_by_citation_numbers(self, source_chunk_id: str, 
                                       citation_numbers: List[str]) -> List[Dict[str, Any]]:
        """
        AI-guided citation retrieval: Get chunks from ONLY selected citations.
        
        This is the key method for your workflow:
        1. AI decides which citations to follow (e.g., ["13", "21"])
        2. This method returns chunks ONLY from those cited papers
        3. Returns chunk_ids that can be used for Qdrant vector search
        
        Args:
            source_chunk_id: The chunk that has citations
            citation_numbers: List of citation numbers AI wants to follow (e.g., ["1", "13", "21"])
        
        Returns:
            List of dicts with:
            - chunk_id: For database queries
            - paper_id: Which paper this chunk is from
            - text: Chunk content
            - section: Section in paper
            - token_count: Chunk size
            - cited_as: Which citation number led to this chunk (e.g., "13")
            - cited_paper_title: Title of the cited paper
        """
        cursor = self.conn.cursor()
        
        # Convert citation_numbers to proper format
        citation_nums = [str(num) for num in citation_numbers]
        
        # Get the paper_id for the source chunk first
        query_paper = """
            SELECT paper_id FROM chunks WHERE chunk_id = %s
        """
        
        try:
            cursor.execute(query_paper, (source_chunk_id,))
            result = cursor.fetchone()
            if not result:
                cursor.close()
                print(f"  [ERROR] Source chunk {source_chunk_id} not found")
                return []
            citing_paper_id = result[0]
        except Exception as e:
            cursor.close()
            print(f"  [ERROR] Failed to get source paper: {e}")
            return []
        
        # Get cited paper_ids for the selected citation numbers
        query = """
            SELECT DISTINCT r.cited_paper_id, r.citation_number, p.title
            FROM "references" r
            LEFT JOIN papers p ON p.paper_id = r.cited_paper_id
            WHERE r.citing_paper_id = %s
              AND r.citation_number = ANY(%s)
              AND r.cited_paper_id IS NOT NULL
        """
        
        try:
            cursor.execute(query, (citing_paper_id, citation_nums))
            citation_mapping = cursor.fetchall()  # [(paper_id, citation_num, title), ...]
        except Exception as e:
            cursor.close()
            print(f"  [ERROR] Failed to retrieve citation mapping: {e}")
            return []
        
        if not citation_mapping:
            cursor.close()
            return []
        
        # Get all chunks from these cited papers
        cited_paper_ids = [row[0] for row in citation_mapping]
        
        # Create mapping of paper_id -> citation_number
        paper_to_citation = {row[0]: (row[1], row[2]) for row in citation_mapping}
        
        # Retrieve chunks from cited papers
        query = """
            SELECT chunk_id, paper_id, section, chunk_text, token_count, paragraph_index
            FROM chunks
            WHERE paper_id = ANY(%s)
            ORDER BY paper_id, paragraph_index
        """
        
        try:
            cursor.execute(query, (cited_paper_ids,))
            rows = cursor.fetchall()
        except Exception as e:
            cursor.close()
            print(f"  [ERROR] Failed to retrieve chunks from cited papers: {e}")
            return []
        
        cursor.close()
        
        results = []
        for row in rows:
            chunk_id, paper_id, section, text, token_count, para_idx = row
            citation_num, paper_title = paper_to_citation.get(paper_id, (None, None))
            
            results.append({
                'chunk_id': chunk_id,
                'paper_id': paper_id,
                'section': section,
                'text': text,
                'token_count': token_count,
                'paragraph_index': para_idx,
                'cited_as': citation_num,  # Which citation number this came from
                'cited_paper_title': paper_title
            })
        
        return results
    
    def get_vector_ids_for_citations(self, source_chunk_id: str, 
                                     citation_numbers: List[str],
                                     use_bm25_fallback: bool = True) -> Dict[str, Any]:
        """
        Get Qdrant vector IDs for chunks from selected citations.
        
        THIS IS THE METHOD YOU NEED FOR VECTOR SEARCH PRIORITY!
        
        Workflow:
        1. AI analyzes citations and returns: ["13", "21"]
        2. Call this method: get_vector_ids_for_citations(chunk_id, ["13", "21"])
        3. Returns: dict with vector_ids, resolved_citations, etc.
        4. Use these IDs for Qdrant priority search
        
        Args:
            source_chunk_id: The chunk that has citations
            citation_numbers: Citation numbers AI selected (e.g., ["1", "13", "21"])
            use_bm25_fallback: Whether to use BM25 for unresolved citations
        
        Returns:
            Dict with:
            - vector_ids: List of chunk_ids for Qdrant search
            - resolved_citations: List of successfully resolved citations
            - unresolved_citations: List of citations not in database
            - bm25_matches: List of citations matched via BM25 (if enabled)
        """
        cursor = self.conn.cursor()
        
        # Convert citation_numbers to proper format
        citation_nums = [str(num) for num in citation_numbers]
        
        # Get all citations for this chunk
        query = """
            SELECT DISTINCT 
                r.citation_number,
                r.cited_paper_id,
                r.cited_title,
                r.cited_year,
                p.title as resolved_title
            FROM chunks c
            CROSS JOIN UNNEST(c.citations_out) as citation_num
            JOIN "references" r ON r.citing_paper_id = c.paper_id 
                AND r.citation_number = citation_num
            LEFT JOIN papers p ON p.paper_id = r.cited_paper_id
            WHERE c.chunk_id = %s
              AND r.citation_number = ANY(%s)
        """
        
        try:
            cursor.execute(query, (source_chunk_id, citation_nums))
            rows = cursor.fetchall()
        except Exception as e:
            cursor.close()
            print(f"  [ERROR] Failed to query citations: {e}")
            return {
                'vector_ids': [],
                'resolved_citations': [],
                'unresolved_citations': citation_numbers,
                'bm25_matches': []
            }
        
        resolved_citations = []
        unresolved_citations = []
        cited_paper_ids = []
        
        for row in rows:
            citation_num, cited_paper_id, cited_title, cited_year, resolved_title = row
            
            if cited_paper_id:
                # Citation resolved to a paper in our database
                resolved_citations.append({
                    'citation_number': citation_num,
                    'cited_paper_id': cited_paper_id,
                    'cited_title': resolved_title or cited_title,
                    'cited_year': cited_year
                })
                cited_paper_ids.append(cited_paper_id)
            else:
                # Citation not resolved
                unresolved_citations.append({
                    'citation_number': citation_num,
                    'cited_title': cited_title,
                    'cited_year': cited_year
                })
        
        # Get all chunk_ids from resolved cited papers
        vector_ids = []
        bm25_matches = []
        
        if cited_paper_ids:
            query = """
                SELECT chunk_id, paper_id
                FROM chunks
                WHERE paper_id = ANY(%s)
                ORDER BY paper_id, paragraph_index
            """
            
            try:
                cursor.execute(query, (cited_paper_ids,))
                vector_ids = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                print(f"  [ERROR] Failed to get chunks from cited papers: {e}")
        
        # BM25 fallback for unresolved citations
        if use_bm25_fallback and unresolved_citations and self.use_bm25 and self.bm25_search:
            try:
                for unresolved in unresolved_citations:
                    cited_title = unresolved.get('cited_title', '')
                    if not cited_title or len(cited_title) < 10:
                        continue
                    
                    # Search using BM25
                    bm25_results = self.bm25_search.search(cited_title, top_k=3)
                    
                    if bm25_results:
                        # Add chunks from BM25 matches
                        for result in bm25_results:
                            chunk_id = result.get('chunk_id')
                            if chunk_id and chunk_id not in vector_ids:
                                vector_ids.append(chunk_id)
                                bm25_matches.append({
                                    'citation_number': unresolved['citation_number'],
                                    'matched_chunk': chunk_id,
                                    'matched_paper': result.get('paper_id'),
                                    'bm25_score': result.get('score')
                                })
            except Exception as e:
                print(f"  [WARNING] BM25 fallback failed: {e}")
        
        cursor.close()
        
        return {
            'vector_ids': vector_ids,
            'resolved_citations': resolved_citations,
            'unresolved_citations': [u['citation_number'] for u in unresolved_citations],
            'bm25_matches': bm25_matches
        }
    
    def get_citation_context_for_ai(self, chunk_id: str) -> str:
        """
        Helper method: Format citations for AI to decide which to follow.
        
        Returns a formatted string you can include in your AI prompt.
        
        Example output:
        '''
        Available citations:
        [1] Layer normalization (2016) - Jimmy Lei Ba et al.
        [13] Long short-term memory (1997) - Hochreiter et al.
        [21] Neural machine translation (2014) - Bahdanau et al.
        '''
        """
        citations = self.get_citations_for_chunk(chunk_id)
        
        if not citations:
            return "No citations available."
        
        lines = ["Available citations:"]
        for cite in citations:
            title = cite['cited_title'][:60] + "..." if cite['cited_title'] and len(cite['cited_title']) > 60 else cite['cited_title']
            year = f"({cite['cited_year']})" if cite['cited_year'] else ""
            authors = cite['cited_authors'][0] if cite['cited_authors'] and len(cite['cited_authors']) > 0 else ""
            
            if cite['cited_paper_id']:
                status = f"✓ In database ({cite['resolved_chunks']} chunks)"
            else:
                status = "✗ Not in database"
            
            lines.append(f"[{cite['citation_number']}] {title} {year} - {authors}")
            lines.append(f"     {status}")
        
        return "\n".join(lines)
    
    def get_stats(self):
        """Get statistics about the citation graph."""
        cursor = self.conn.cursor()
        
        # Total references
        cursor.execute('SELECT COUNT(*) FROM "references"')
        total_refs = cursor.fetchone()[0]
        
        # Resolved references
        cursor.execute('SELECT COUNT(*) FROM "references" WHERE cited_paper_id IS NOT NULL')
        resolved_refs = cursor.fetchone()[0]
        
        # Citation edges
        cursor.execute("SELECT COUNT(*) FROM citation_edges")
        total_edges = cursor.fetchone()[0]
        
        # Papers with citations
        cursor.execute('SELECT COUNT(DISTINCT citing_paper_id) FROM "references"')
        papers_with_citations = cursor.fetchone()[0]
        
        # Papers being cited
        cursor.execute('SELECT COUNT(DISTINCT cited_paper_id) FROM "references" WHERE cited_paper_id IS NOT NULL')
        papers_being_cited = cursor.fetchone()[0]
        
        cursor.close()
        
        print("\n" + "="*60)
        print("📊 CITATION GRAPH STATISTICS")
        print("="*60)
        print(f"Total references loaded:        {total_refs}")
        print(f"Resolved references:            {resolved_refs} ({resolved_refs/total_refs*100:.1f}%)")
        print(f"Unresolved references:          {total_refs - resolved_refs}")
        print(f"Citation edges (paper→paper):   {total_edges}")
        print(f"Papers with outgoing citations: {papers_with_citations}")
        print(f"Papers with incoming citations: {papers_being_cited}")
        print("="*60)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


if __name__ == "__main__":
    """
    CITATION GRAPH RETRIEVAL EXAMPLES
    
    NOTE: Before using this module, run load_to_postgres.py to setup the database.
    That script will create all tables, load papers/chunks/citations, and build the citation graph.
    """
    
    # Configuration
    postgres_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'research_papers',
        'user': 'postgres',
        'password': 'password'
    }
    
    # Initialize citation manager for retrieval
    manager = CitationGraphManager(postgres_config, use_bm25=True)
    
    try:
        manager.connect()
        
        print("\n" + "="*60)
        print("CITATION GRAPH - RETRIEVAL EXAMPLES")
        print("="*60)
        
        # Example 1: Get citation context for AI prompt
        print("\n📝 Example 1: Get citation context for AI")
        chunk_id = "Attention.tei_chunk_1"  # Replace with actual chunk ID
        citation_context = manager.get_citation_context_for_ai(chunk_id)
        if citation_context:
            print(citation_context)
        else:
            print(f"No citations found for {chunk_id}")
        
        # Example 2: Get vector IDs for priority search
        print("\n🎯 Example 2: Get vector IDs for Qdrant priority search")
        # AI selected these citation numbers as relevant
        selected_citations = [1, 3, 5]
        result = manager.get_vector_ids_for_citations(
            chunk_id, 
            selected_citations, 
            use_bm25_fallback=True
        )
        print(f"Vector IDs for priority search: {len(result['vector_ids'])} vectors")
        print(f"Resolved citations: {len(result['resolved_citations'])}")
        print(f"Unresolved citations: {len(result['unresolved_citations'])}")
        if result['bm25_matches']:
            print(f"BM25 fallback matches: {len(result['bm25_matches'])}")
        
        # Example 3: Get full chunk data for selected citations
        print("\n📚 Example 3: Get chunks from selected citations")
        chunks = manager.get_chunks_by_citation_numbers(chunk_id, selected_citations)
        print(f"Retrieved {len(chunks)} chunks from cited papers")
        for chunk in chunks[:2]:  # Show first 2
            print(f"  - {chunk['chunk_id'][:30]}... (from paper: {chunk['paper_id']})")
        
        print("\n" + "="*60)
        print("✅ RETRIEVAL EXAMPLES COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you've run load_to_postgres.py first!")
        
    finally:
        manager.close()
