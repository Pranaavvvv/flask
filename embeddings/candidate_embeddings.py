import json
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import os
import re
import hashlib
from flask import Blueprint, Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
candidates_bp = Blueprint('candidates_embeddings', __name__)


class EnhancedPostgresVectorSearch:
    def __init__(self, 
                 database_url: str,
                 model_name: str = 'paraphrase-MiniLM-L6-v2'):  # Better model
        
        # Use better models for higher quality embeddings
        # Options: 'all-mpnet-base-v2', 'all-MiniLM-L12-v2', 'multi-qa-mpnet-base-dot-v1'
        self.model = SentenceTransformer(model_name)
        self.database_url = database_url
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize skill standardization
        self.skill_synonyms = self._load_skill_synonyms()
        
        self._setup_database()

    def _get_connection(self):
        """Get database connection using connection string"""
        return psycopg2.connect(self.database_url)

    def _load_skill_synonyms(self) -> Dict[str, str]:
        """Load skill synonyms for standardization"""
        return {
            # Programming languages
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'golang': 'go',
            'c#': 'csharp',
            '.net': 'dotnet',
            
            # Frameworks
            'reactjs': 'react',
            'vuejs': 'vue',
            'angularjs': 'angular',
            'nodejs': 'node.js',
            'nextjs': 'next.js',
            
            # Databases
            'postgresql': 'postgres',
            'mysql': 'sql',
            'mongodb': 'mongo',
            
            # Cloud
            'amazon web services': 'aws',
            'google cloud platform': 'gcp',
            'microsoft azure': 'azure',
            
            # Add more based on your domain
        }

    def _clean_and_standardize_text(self, text: str) -> str:
        """Clean and standardize text for better embeddings"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Standardize skills using synonyms
        for synonym, standard in self.skill_synonyms.items():
            text = re.sub(rf'\b{re.escape(synonym)}\b', standard, text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\+\#\.\-]', ' ', text)
        
        return text.strip()

    def _create_enhanced_candidate_text(self, candidate: Dict) -> str:
        """Create enhanced text representation with weighted importance"""
        
        # Define field weights (higher = more important)
        weighted_fields = {
            'title': 3,
            'skills': 3,
            'summary': 2,
            'past_companies': 2,
            'education': 1,
            'work_preference': 1,
            'location': 1
        }
        
        text_parts = []
        
        # Add weighted fields
        for field, weight in weighted_fields.items():
            value = candidate.get(field, '')
            
            if field == 'skills' and isinstance(value, list):
                value = ' '.join(value)
            elif field == 'past_companies' and isinstance(value, list):
                value = ' '.join(value)
            
            if value:
                cleaned_value = self._clean_and_standardize_text(str(value))
                # Repeat important fields based on weight
                text_parts.extend([cleaned_value] * weight)
        
        # Add experience context
        years_exp = candidate.get('years_of_experience', 0)
        if years_exp:
            exp_context = self._get_experience_context(years_exp)
            text_parts.append(exp_context)
        
        # Add name (but with lower weight)
        name = candidate.get('name', '')
        if name:
            text_parts.append(self._clean_and_standardize_text(name))
        
        return ' '.join(text_parts)

    def _get_experience_context(self, years: int) -> str:
        """Add experience level context for better matching"""
        if years < 2:
            return "junior entry level beginner"
        elif years < 5:
            return "mid level intermediate"
        elif years < 10:
            return "senior experienced"
        else:
            return "senior lead expert principal architect"

    def _setup_database(self):
        """Setup PostgreSQL database with pgvector extension and embedding column"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Enable required extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
            
            # Add embedding column to existing candidates table if it doesn't exist
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'candidates' AND column_name = 'embedding';
            """)
            
            if not cur.fetchone():
                logger.info("Adding embedding column to candidates table...")
                cur.execute(f"ALTER TABLE candidates ADD COLUMN embedding vector({self.dimension});")
            
            # Add metadata columns for embedding quality tracking
            metadata_columns = [
                ('embedding_model', 'VARCHAR(100)'),
                ('embedding_version', 'INTEGER DEFAULT 1'),
                ('text_hash', 'VARCHAR(64)')  # To track if text changed
            ]
            
            for col_name, col_type in metadata_columns:
                cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'candidates' AND column_name = '{col_name}';
                """)
                
                if not cur.fetchone():
                    cur.execute(f"ALTER TABLE candidates ADD COLUMN {col_name} {col_type};")
            
            # Create indexes for vector search and filters
            cur.execute("CREATE INDEX IF NOT EXISTS idx_years_exp ON candidates(years_of_experience);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_work_pref ON candidates(work_preference);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_status ON candidates(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_available_from ON candidates(available_from);")
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info("Database setup completed successfully")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise

    def _create_vector_index(self):
        """Create optimized vector index"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Drop existing index if it exists
            cur.execute("DROP INDEX IF EXISTS candidates_embedding_idx;")
            
            # Get count of candidates to optimize index parameters
            cur.execute("SELECT COUNT(*) FROM candidates WHERE embedding IS NOT NULL;")
            count = cur.fetchone()[0]
            
            # Calculate optimal lists parameter (rule of thumb: sqrt(rows))
            lists = max(1, min(1000, int(count ** 0.5)))
            
            logger.info(f"Creating vector index with {lists} lists for {count} candidates...")
            cur.execute(f"""
                CREATE INDEX candidates_embedding_idx 
                ON candidates USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """)
            conn.commit()
            logger.info("Vector index created successfully")
            
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Vector index creation failed: {e}")

    def generate_embeddings_for_existing_candidates(self, batch_size: int = 32, force_regenerate: bool = False):
        """Generate embeddings with improved text processing"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get candidates that need embeddings
            if force_regenerate:
                cur.execute("""
                    SELECT id, name, email, title, location, years_of_experience,
                           skills, work_preference, education, past_companies, summary
                    FROM candidates 
                    ORDER BY created_at;
                """)
            else:
                cur.execute("""
                    SELECT id, name, email, title, location, years_of_experience,
                           skills, work_preference, education, past_companies, summary
                    FROM candidates 
                    WHERE embedding IS NULL OR embedding_model IS NULL
                    ORDER BY created_at;
                """)
            
            candidates = cur.fetchall()
            
            if not candidates:
                logger.info("All candidates already have current embeddings")
                cur.close()
                conn.close()
                return
            
            logger.info(f"Generating enhanced embeddings for {len(candidates)} candidates...")
            
            # Process in batches for better memory management
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i+batch_size]
                
                # Generate enhanced text representations
                texts = []
                candidate_ids = []
                text_hashes = []
                
                for candidate in batch:
                    enhanced_text = self._create_enhanced_candidate_text(dict(candidate))
                    texts.append(enhanced_text)
                    candidate_ids.append(candidate['id'])
                    
                    # Create hash of text for change tracking
                    text_hash = hashlib.md5(enhanced_text.encode()).hexdigest()
                    text_hashes.append(text_hash)
                
                # Generate embeddings with better normalization
                embeddings = self.model.encode(
                    texts, 
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=True
                )
                
                # Update database with metadata
                update_cur = conn.cursor()
                for candidate_id, embedding, text_hash in zip(candidate_ids, embeddings, text_hashes):
                    update_cur.execute("""
                        UPDATE candidates 
                        SET embedding = %s, 
                            embedding_model = %s,
                            embedding_version = 2,
                            text_hash = %s,
                            updated_at = NOW()
                        WHERE id = %s;
                    """, (embedding.tolist(), self.model.get_sentence_embedding_dimension(), text_hash, candidate_id))
                
                conn.commit()
                update_cur.close()
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(candidates)-1)//batch_size + 1}")
            
            cur.close()
            conn.close()
            
            # Create optimized vector index
            self._create_vector_index()
            
            logger.info("Enhanced embedding generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def search_with_hybrid_scoring(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Enhanced search with hybrid scoring (semantic + keyword)"""
        try:
            # Clean and enhance the query
            enhanced_query = self._clean_and_standardize_text(query)
            
            # Generate query embedding
            query_embedding = self.model.encode([enhanced_query], normalize_embeddings=True)[0]
            
            # Build hybrid search query
            sql = """
                SELECT 
                    id, name, email, phone, photo, title, location, 
                    years_of_experience, skills, work_preference, education, 
                    past_companies, summary, available_from, linkedin_url, 
                    portfolio_url, status,
                    -- Semantic similarity score (0-100)
                    ROUND((1 - (embedding <=> %s::vector)) * 100) as semantic_score,
                    -- Keyword matching score
                    CASE 
                        WHEN LOWER(title) LIKE LOWER(%s) THEN 20
                        WHEN EXISTS(SELECT 1 FROM unnest(skills) s WHERE LOWER(s) LIKE LOWER(%s)) THEN 15
                        WHEN LOWER(summary) LIKE LOWER(%s) THEN 10
                        ELSE 0
                    END as keyword_score,
                    -- Combined hybrid score
                    ROUND(
                        (1 - (embedding <=> %s::vector)) * 70 + 
                        CASE 
                            WHEN LOWER(title) LIKE LOWER(%s) THEN 20
                            WHEN EXISTS(SELECT 1 FROM unnest(skills) s WHERE LOWER(s) LIKE LOWER(%s)) THEN 15
                            WHEN LOWER(summary) LIKE LOWER(%s) THEN 10
                            ELSE 0
                        END * 0.3
                    ) as hybrid_score
                FROM candidates
                WHERE embedding IS NOT NULL
            """
            
            # Prepare query parameters for keyword matching
            keyword_pattern = f"%{enhanced_query}%"
            params = [
                query_embedding.tolist(),
                keyword_pattern, keyword_pattern, keyword_pattern,
                query_embedding.tolist(),
                keyword_pattern, keyword_pattern, keyword_pattern
            ]
            
            where_conditions = []
            
            # Add filters
            if filters:
                if filters.get('min_experience'):
                    where_conditions.append("years_of_experience >= %s")
                    params.append(filters['min_experience'])
                
                if filters.get('max_experience'):
                    where_conditions.append("years_of_experience <= %s")
                    params.append(filters['max_experience'])
                
                if filters.get('work_preference'):
                    where_conditions.append("LOWER(work_preference) = LOWER(%s)")
                    params.append(filters['work_preference'])
                
                if filters.get('location'):
                    where_conditions.append("LOWER(location) ILIKE LOWER(%s)")
                    params.append(f"%{filters['location']}%")
                
                if filters.get('status'):
                    where_conditions.append("LOWER(status) = LOWER(%s)")
                    params.append(filters['status'])
                
                if filters.get('available_from'):
                    where_conditions.append("available_from <= %s")
                    params.append(filters['available_from'])
                
                if filters.get('skills'):
                    skill_conditions = []
                    for skill in filters['skills']:
                        skill_conditions.append("LOWER(%s) = ANY(SELECT LOWER(unnest(skills)))")
                        params.append(skill.lower())
                    if skill_conditions:
                        where_conditions.append(f"({' OR '.join(skill_conditions)})")
            
            if where_conditions:
                sql += " AND " + " AND ".join(where_conditions)
            
            sql += " ORDER BY hybrid_score DESC LIMIT %s;"
            params.append(k)
            
            # Execute query
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql, params)
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            # Process results
            candidates = []
            for row in results:
                candidate = dict(row)
                if candidate.get('id'):
                    candidate['id'] = str(candidate['id'])
                if candidate.get('available_from'):
                    candidate['available_from'] = candidate['available_from'].isoformat()
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    def search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Main search method using hybrid scoring"""
        return self.search_with_hybrid_scoring(query, k, filters)

    def get_embedding_quality_stats(self) -> Dict:
        """Get statistics about embedding quality"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Basic stats
            cur.execute("SELECT COUNT(*) FROM candidates;")
            total_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM candidates WHERE embedding IS NOT NULL;")
            embedded_count = cur.fetchone()[0]
            
            # Model distribution
            cur.execute("""
                SELECT embedding_model, COUNT(*) 
                FROM candidates 
                WHERE embedding_model IS NOT NULL
                GROUP BY embedding_model;
            """)
            model_stats = cur.fetchall()
            
            # Version distribution
            cur.execute("""
                SELECT embedding_version, COUNT(*) 
                FROM candidates 
                WHERE embedding_version IS NOT NULL
                GROUP BY embedding_version;
            """)
            version_stats = cur.fetchall()
            
            cur.close()
            conn.close()
            
            return {
                'total_candidates': total_count,
                'candidates_with_embeddings': embedded_count,
                'embedding_coverage': f"{(embedded_count/total_count*100):.1f}%" if total_count > 0 else "0%",
                'model_distribution': dict(model_stats) if model_stats else {},
                'version_distribution': dict(version_stats) if version_stats else {},
                'current_model_dimension': self.dimension
            }
            
        except Exception as e:
            logger.error(f"Error getting embedding quality stats: {e}")
            return {'error': str(e)}


# Initialize with connection string
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/candidates_db')
postgres_search = EnhancedPostgresVectorSearch(DATABASE_URL)

# Routes
@candidates_bp.route('/search', methods=['POST'])
def search_candidates():
    data = request.get_json()
    if not data or not data.get('query'):
        return jsonify({'error': 'Missing query'}), 400
    
    try:
        filters = {}
        if data.get('min_experience'):
            filters['min_experience'] = data['min_experience']
        if data.get('max_experience'):
            filters['max_experience'] = data['max_experience']
        if data.get('work_preference'):
            filters['work_preference'] = data['work_preference']
        if data.get('location'):
            filters['location'] = data['location']
        if data.get('status'):
            filters['status'] = data['status']
        if data.get('available_from'):
            filters['available_from'] = data['available_from']
        if data.get('skills'):
            filters['skills'] = data['skills']
        
        results = postgres_search.search(
            query=data['query'],
            k=data.get('top_k', 5),
            filters=filters if filters else None
        )
        
        return jsonify({
            'status': 'success',
            'query': data['query'],
            'filters': filters,
            'candidates': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({'error': str(e)}), 500

@candidates_bp.route('/regenerate-embeddings', methods=['POST'])
def regenerate_embeddings():
    """Force regenerate all embeddings with enhanced quality"""
    try:
        postgres_search.generate_embeddings_for_existing_candidates(force_regenerate=True)
        return jsonify({
            'status': 'success',
            'message': 'All embeddings regenerated with enhanced quality'
        })
    except Exception as e:
        logger.error(f"Embedding regeneration failed: {e}")
        return jsonify({'error': str(e)}), 500

@candidates_bp.route('/embedding-stats', methods=['GET'])
def get_embedding_stats():
    """Get embedding quality statistics"""
    return jsonify(postgres_search.get_embedding_quality_stats())

@candidates_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Enhanced Candidate Search API',
        'model_dimension': postgres_search.dimension
    })

# Flask app setup
def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register blueprint
    app.register_blueprint(candidates_bp, url_prefix='/api/candidates')
    
    @app.route('/')
    def index():
        return jsonify({
            'service': 'Enhanced Candidate Search API',
            'version': '2.0',
            'endpoints': {
                'search': '/api/candidates/search',
                'regenerate_embeddings': '/api/candidates/regenerate-embeddings',
                'embedding_stats': '/api/candidates/embedding-stats',
                'health': '/api/candidates/health'
            }
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_ENV') == 'development')