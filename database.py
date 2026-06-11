import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from config import Config, ALLOWED_TABLES
from typing import List, Dict, Any, Optional, Tuple
import os
import re

logger = logging.getLogger(__name__)


def validate_table_name(table_name: str) -> str:
    """Validate table name against whitelist to prevent SQL injection"""
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}. Allowed: {ALLOWED_TABLES}")
    return table_name


class DatabaseManager:
    def __init__(self):
        self.connection = None
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        self._initialized = False
        # Don't connect immediately - lazy initialization
        logger.info("DatabaseManager created (lazy initialization)")

    def ensure_connection(self):
        """Ensure database connection is established (lazy initialization)"""
        if not self.connection or self.connection.closed:
            self.connect()
        if self.connection and not self._initialized:
            self.initialize_dummy_data()
            self._initialized = True

    def connect(self):
        "Establish database connection using db_config property"
        try:
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                    
            config = Config()
            db_config = config.db_config
            
            self.connection = psycopg2.connect(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"],
                sslmode=db_config.get("sslmode", "prefer"),
                cursor_factory=RealDictCursor
            )
            logger.info(f"✅ Database connection established to {db_config['host']}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to database: {e}")
            logger.info("📄 App will continue without database support (PDF/Chat only)")
            self.connection = None

    def initialize_dummy_data(self):
        "Initialize database with dummy data for demonstration"
        # DON'T call ensure_connection here - it causes infinite recursion
        if not self.connection:
            logger.warning("Cannot initialize dummy data: database not connected")
            return
        try:
            with self.connection.cursor() as cursor:
                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        email VARCHAR(150) UNIQUE NOT NULL,
                        department VARCHAR(100),
                        position VARCHAR(100),
                        phone VARCHAR(20),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS products (
                        id SERIAL PRIMARY KEY, 
                        name VARCHAR(200) NOT NULL,
                        category VARCHAR(100),
                        price DECIMAL(10, 2),
                        description TEXT,
                        stock_quantity INTEGER, 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS orders (
                        id SERIAL PRIMARY KEY, 
                        user_id INTEGER REFERENCES user_profiles(id),
                        product_id INTEGER REFERENCES products(id),
                        quantity INTEGER,
                        total_amount DECIMAL(10, 2),
                        status VARCHAR(50),
                        order_date DATE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

              
                """)

                cursor.execute("SELECT COUNT(*) FROM user_profiles;")

                if cursor.fetchone()['count'] == 0:
                    # Insert dummy user profiles - updated untuk match dengan test questions
                    cursor.execute("""
                        INSERT INTO user_profiles (name, email, department, position, phone) VALUES
                        ('Ahmad Wijaya', 'ahmad.wijaya@tmb.co.id', 'IT', 'IT Manager', '+62-812-3456-7890'),
                        ('Sari Dewi', 'sari.dewi@tmb.co.id', 'IT', 'Business Analyst', '+62-813-4567-8901'),
                        ('Budi Santoso', 'budi.santoso@tmb.co.id', 'IT', 'Senior Developer', '+62-814-5678-9012'),
                        ('Maya Sari', 'maya.sari@tmb.co.id', 'IT', 'QA Lead', '+62-815-6789-0123'),
                        ('Rizki Pratama', 'rizki.pratama@tmb.co.id', 'IT', 'DevOps Engineer', '+62-816-7890-1234'),
                        ('Andi Firmansyah', 'andi.firmansyah@vendor.com', 'External', 'Vendor PM', '+62-817-8901-2345');
                    """)

                cursor.execute("SELECT COUNT(*) as count FROM products")
                if cursor.fetchone()['count'] == 0:
                    cursor.execute("""
                        INSERT INTO products (name, category, price, description, stock_quantity) VALUES
                        ('JetBrains All Products Pack', 'Software Tools', 1500000, 'IDE development tools - IntelliJ IDEA, PyCharm, WebStorm, DataGrip, etc', 100),
                        ('SAP S/4HANA License', 'Enterprise Software', 35000000, 'SAP ERP system license untuk enterprise resource planning', 10),
                        ('Jira Software Cloud', 'Project Management', 2500000, 'Agile project management dan issue tracking untuk tim software', 50),
                        ('Confluence Cloud', 'Collaboration', 1800000, 'Team collaboration dan documentation platform', 50),
                        ('AWS Enterprise Support', 'Cloud Services', 8000000, 'AWS cloud support dengan TAM dan 24/7 assistance', 5),
                        ('Datadog Pro', 'Monitoring Tools', 2000000, 'Infrastructure dan application monitoring platform', 20),
                        ('SonarQube Enterprise', 'Security Tools', 3500000, 'Code quality dan security scanning platform', 15),
                        ('GitLab Ultimate', 'DevOps Platform', 4500000, 'Complete DevOps platform dengan CI/CD dan security', 25),
                        ('Laptop ThinkPad X1', 'Hardware', 15000000, 'Business laptop dengan processor Intel i7 dan RAM 16GB', 25),
                        ('Wireless Mouse', 'Hardware', 350000, 'Mouse nirkabel dengan precision sensor', 75);
                    """)

                cursor.execute("SELECT COUNT(*) as count FROM orders")
                if cursor.fetchone()['count'] == 0:
                    cursor.execute("""
                        INSERT INTO orders (user_id, product_id, quantity, total_amount, status, order_date) VALUES
                        (1, 2, 1, 35000000, 'completed', '2024-11-15'),
                        (1, 1, 10, 15000000, 'completed', '2024-12-01'),
                        (4, 3, 5, 12500000, 'completed', '2024-12-10'),
                        (5, 5, 1, 8000000, 'processing', '2025-01-02'),
                        (3, 6, 3, 6000000, 'processing', '2025-01-03');
                    """)

                self.connection.commit()
                logger.info("Dummy data initialized in database")
                
                # Initialize Full-Text Search
                self.initialize_fts()
                
        except Exception as e:
            logger.error(f"Failed to initialize dummy data: {e}")
            self.connection.rollback()

    def initialize_fts(self):
        """Initialize Full-Text Search with tsvector columns and GIN indexes"""
        try:
            with self.connection.cursor() as cursor:
                # Create pg_trgm extension for fuzzy matching (if available)
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                    logger.info("pg_trgm extension enabled")
                except Exception as e:
                    logger.warning(f"pg_trgm extension not available: {e}")

                # Add tsvector columns and triggers for each table
                fts_configs = [
                    ('user_profiles', ['name', 'email', 'department', 'position']),
                    ('products', ['name', 'category', 'description']),
                    ('orders', ['status'])
                ]

                for table_name, text_columns in fts_configs:
                    # Check if search_vector column exists
                    cursor.execute("""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = %s AND column_name = 'search_vector'
                    """, (table_name,))
                    
                    if not cursor.fetchone():
                        # Add tsvector column
                        cursor.execute(f"""
                            ALTER TABLE {table_name} 
                            ADD COLUMN IF NOT EXISTS search_vector tsvector;
                        """)
                        
                        # Build tsvector from text columns
                        coalesce_parts = " || ' ' || ".join(
                            [f"COALESCE({col}::text, '')" for col in text_columns]
                        )
                        
                        # Update existing rows
                        cursor.execute(f"""
                            UPDATE {table_name} 
                            SET search_vector = to_tsvector('indonesian', {coalesce_parts})
                            WHERE search_vector IS NULL;
                        """)
                        
                        # Create GIN index for fast search
                        cursor.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_search 
                            ON {table_name} USING GIN(search_vector);
                        """)
                        
                        logger.info(f"FTS initialized for table {table_name}")

                self.connection.commit()
                logger.info("Full-Text Search initialization completed")
                
        except Exception as e:
            logger.warning(f"FTS initialization failed (will use fallback ILIKE): {e}")
            self.connection.rollback()

    def check_connection_health(self) -> bool:
        """Check if database connection is alive, reconnect if needed"""
        import time
        
        # Return False immediately if no connection
        if not self.connection:
            return False
        
        try:
            # Test current connection
            if self.connection and not self.connection.closed:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    self._last_health_check = current_time
                    return True
        except Exception as e:
            logger.warning(f"Database connection test failed: {e}")
        
        # Connection is dead, try to reconnect
        try:
            logger.info("Attempting to reconnect to database...")
            self.connect()
            self._last_health_check = current_time
            return True
        except Exception as e:
            logger.error(f"Failed to reconnect to database: {e}")
            return False
    
    def is_healthy(self) -> Dict[str, Any]:
        """Check database health status"""
        try:
            if not self.check_connection_health():
                return {
                    "status": "disconnected",
                    "message": "Database connection failed",
                    "can_query": False
                }
            
            # Test basic query
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as total FROM information_schema.tables WHERE table_schema = 'public'")
                result = cursor.fetchone()
                table_count = result['total'] if result else 0
                
                return {
                    "status": "connected",
                    "message": f"Database healthy with {table_count} tables",
                    "can_query": True,
                    "table_count": table_count
                }
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Database health check failed: {str(e)}",
                "can_query": False
            }
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a table"""
        self.ensure_connection()
        
        if not self.connection:
            logger.warning("No database connection available")
            return []
            
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                
                """, (table_name,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            return []

    def execute_query(self, query: str, params: Optional[tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        self.ensure_connection()
        if not self.connection:
            return []
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def search_accross_tables(self, search_terms: List[str], limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all configured tables"""
        self.ensure_connection()
        if not self.connection:
            return {}
        results = {}
       
        for table_name in Config().db_tables:
            try:
                table_results = self.search_in_table(table_name, search_terms, limit)
                if table_results:
                    results[table_name] = table_results
            except Exception as e:
                logger.error(f"Search in table {table_name} failed: {e}")
                continue
            
        return results

    def search_in_specific_tables(self, search_terms: List[str], tables: List[str], limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Search across specific tables only (smart routing)"""
        logger.info(f"🔎 search_in_specific_tables called with terms: {search_terms}, tables: {tables}")
        self.ensure_connection()
        if not self.connection:
            return {}
        results = {}
        
        configured_tables = Config().db_tables
        logger.info(f"🔎 Configured tables: {configured_tables}")
        
        for table_name in tables:
            if table_name not in configured_tables:
                logger.warning(f"Table {table_name} not in configured tables, skipping")
                continue
            
            logger.info(f"🔎 Searching in table: {table_name}")
            try:
                table_results = self.search_in_table(table_name, search_terms, limit)
                logger.info(f"🔎 Got {len(table_results) if table_results else 0} results from {table_name}")
                if table_results:
                    results[table_name] = table_results
                    logger.info(f"Found {len(table_results)} results in {table_name}")
            except Exception as e:
                logger.error(f"Search in table {table_name} failed: {e}")
                continue
        
        logger.info(f"🔎 Total results: {results}")
        return results
    
    # Common phrases that should be searched together
    PHRASE_PATTERNS = {
        ('project', 'manager'): 'project manager',
        ('tech', 'lead'): 'tech lead',
        ('qa', 'lead'): 'qa lead',
        ('backend', 'developer'): 'backend developer',
        ('frontend', 'developer'): 'frontend developer',
        ('devops', 'engineer'): 'devops engineer',
        ('finance', 'manager'): 'finance manager',
        ('hr', 'manager'): 'hr manager',
        ('it', 'director'): 'it director',
        ('business', 'analyst'): 'business analyst',
    }
    
    def detect_phrases(self, search_terms: List[str]) -> tuple[List[str], List[str]]:
        """Detect common phrases in search terms and return (phrases, remaining_terms)"""
        terms_lower = [t.lower() for t in search_terms]
        detected_phrases = []
        used_terms = set()
        
        for (word1, word2), phrase in self.PHRASE_PATTERNS.items():
            if word1 in terms_lower and word2 in terms_lower:
                detected_phrases.append(phrase)
                used_terms.add(word1)
                used_terms.add(word2)
        
        remaining_terms = [t for t in search_terms if t.lower() not in used_terms]
        return detected_phrases, remaining_terms
    
    # database.py - Add this method if not exists
    def search_in_table(self, table_name: str, search_terms: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for terms across all columns in a table with FTS and scoring"""
        self.ensure_connection()
        if not self.connection:
            return []
        try:
            logger.info(f"🔎 Searching table '{table_name}' with terms: {search_terms}")
            
            # Detect phrases first
            phrases, remaining_terms = self.detect_phrases(search_terms)
            logger.info(f"🔎 Detected phrases: {phrases}, remaining terms: {remaining_terms}")
            
            # Try Full-Text Search first
            results = self.search_with_fts(table_name, search_terms, limit, phrases)
            if results:
                logger.info(f"✅ FTS returned {len(results)} results from {table_name}")
                return results
            
            logger.info(f"⚠️ FTS returned no results, trying ILIKE fallback...")
            # Fallback to ILIKE if FTS fails or returns no results
            ilike_results = self.search_with_ilike(table_name, search_terms, limit, phrases)
            logger.info(f"{'✅' if ilike_results else '❌'} ILIKE returned {len(ilike_results)} results from {table_name}")
            return ilike_results
            
        except Exception as e:
            logger.error(f"Search in table {table_name} failed: {str(e)}")
            return []

    def search_with_fts(self, table_name: str, search_terms: List[str], limit: int = 10, phrases: List[str] = None) -> List[Dict[str, Any]]:
        """Full-Text Search with ts_rank scoring - prioritizes exact matches and phrases"""
        self.ensure_connection()
        if not self.connection:
            logger.error("Database connection not available for FTS search")
            return []
            
        try:
            # Validate table name against whitelist
            table_name = validate_table_name(table_name)
            
            # Check if table has search_vector column
            schema = self.get_table_schema(table_name)
            has_fts = any(col['column_name'] == 'search_vector' for col in schema)
            
            if not has_fts:
                return []
            
            # Build tsquery from search terms with OR logic
            # Each term is searched separately and combined with OR (|)
            stemmed_terms = [self.indonesian_stem(term.lower()) for term in search_terms]
            # Filter out empty terms and create tsquery format
            valid_terms = [t for t in stemmed_terms if t and len(t) >= 2]
            
            if not valid_terms:
                logger.info(f"No valid terms for FTS in {table_name}")
                return []
            
            # Use to_tsquery with OR logic: term1:* | term2:* (prefix matching)
            query_parts = [f"{term}:*" for term in valid_terms]
            tsquery_string = ' | '.join(query_parts)
            
            logger.info(f"🔍 FTS query for {table_name}: {tsquery_string}")
            logger.info(f"🔍 Original search terms: {search_terms}")
            
            # Get text columns for exact match and phrase boost
            text_columns = [col['column_name'] for col in schema 
                           if col['data_type'] in ['text', 'character varying', 'varchar']
                           and col['column_name'] != 'search_vector']
            
            # Build boost expressions
            boost_parts = []
            boost_params = []
            
            # 1. Exact term match boost (50 points for exact match in name column)
            for term in search_terms:
                if term:  # Original term, not stemmed
                    # Check if term appears as whole word in name
                    boost_parts.append(f"CASE WHEN LOWER(name) LIKE LOWER(%s) THEN 50.0 ELSE 0.0 END")
                    boost_params.append(f"%{term}%")
            
            # 2. Phrase match boost (20 points)
            if phrases:
                for phrase in phrases:
                    for col in text_columns:
                        boost_parts.append(f"CASE WHEN LOWER({col}) LIKE LOWER(%s) THEN 20.0 ELSE 0.0 END")
                        boost_params.append(f"%{phrase}%")
            
            if boost_parts:
                boost_expr = " + ".join(boost_parts)
                query = f"""
                    SELECT *, 
                           ts_rank(search_vector, to_tsquery('simple', %s)) + ({boost_expr}) as relevance_score
                    FROM {table_name}
                    WHERE search_vector @@ to_tsquery('simple', %s)
                    ORDER BY relevance_score DESC
                    LIMIT %s
                """
                params = tuple([tsquery_string] + boost_params + [tsquery_string, limit])
            else:
                query = f"""
                    SELECT *, 
                           ts_rank(search_vector, to_tsquery('simple', %s)) as relevance_score
                    FROM {table_name}
                    WHERE search_vector @@ to_tsquery('simple', %s)
                    ORDER BY relevance_score DESC
                    LIMIT %s
                """
                params = (tsquery_string, tsquery_string, limit)
            
            results = self.execute_query(query, params)
            
            logger.info(f"✅ FTS found {len(results)} results in {table_name}")
            if results:
                # Log top 3 results for debugging
                for i, r in enumerate(results[:3], 1):
                    logger.info(f"  {i}. {r.get('name', 'N/A')} - score: {r.get('relevance_score', 0):.4f}")
            
            return results
            
        except Exception as e:
            logger.warning(f"FTS search failed for {table_name}: {e}")
            return []

    def search_with_ilike(self, table_name: str, search_terms: List[str], limit: int = 10, phrases: List[str] = None) -> List[Dict[str, Any]]:
        """Fallback ILIKE search with basic scoring - case insensitive, prioritizes phrase matches"""
        self.ensure_connection()
        if not self.connection:
            logger.error("Database connection not available for ILIKE search")
            return []
            
        try:
            # Validate table name against whitelist
            table_name = validate_table_name(table_name)
            
            schema = self.get_table_schema(table_name)
            text_columns = [col['column_name'] for col in schema 
                        if col['data_type'] in ['text', 'character varying', 'varchar']
                        and col['column_name'] != 'search_vector']
            
            if not text_columns:
                return []
            
            # Build conditions for WHERE clause
            conditions = []
            where_params = []
            
            # Build score expression separately
            score_parts = []
            score_params = []
            
            # Add phrase matching with higher score (10 points per phrase match)
            if phrases:
                for column in text_columns:
                    for phrase in phrases:
                        conditions.append(f"LOWER({column}) LIKE LOWER(%s)")
                        where_params.append(f"%{phrase}%")
                        score_parts.append(f"CASE WHEN LOWER({column}) LIKE LOWER(%s) THEN 10 ELSE 0 END")
                        score_params.append(f"%{phrase}%")
            
            # Add individual term matching (1 point per term match)
            for column in text_columns:
                for term in search_terms:
                    # For WHERE clause
                    conditions.append(f"LOWER({column}) LIKE LOWER(%s)")
                    where_params.append(f"%{term}%")
                    
                    # For score calculation
                    score_parts.append(f"CASE WHEN LOWER({column}) LIKE LOWER(%s) THEN 1 ELSE 0 END")
                    score_params.append(f"%{term}%")
            
            score_expr = " + ".join(score_parts) if score_parts else "0"
            where_clause = " OR ".join(conditions)
            
            # Parameters order: score_params first, then where_params, then limit
            all_params = score_params + where_params + [limit]
            
            query = f"""
                SELECT *, ({score_expr}) as relevance_score 
                FROM {table_name} 
                WHERE {where_clause} 
                ORDER BY relevance_score DESC
                LIMIT %s
            """
            
            logger.info(f"🔍 ILIKE search in {table_name} for terms: {search_terms}, phrases: {phrases}")
            results = self.execute_query(query, tuple(all_params))
            logger.info(f"ILIKE found {len(results)} results in {table_name}")
            return results
            
        except Exception as e:
            logger.error(f"ILIKE search in table {table_name} failed: {str(e)}")
            return []

    def indonesian_stem(self, word: str) -> str:
        """Simple Indonesian stemming - remove common suffixes"""
        word = word.lower().strip()
        
        # Common Indonesian suffixes
        suffixes = ['kan', 'an', 'i', 'nya', 'lah', 'kah', 'pun']
        prefixes = ['me', 'di', 'ke', 'se', 'ber', 'ter', 'pe']
        
        # Remove suffixes
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        # Remove prefixes
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
        
        return word  
      
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows from a table"""
        self.ensure_connection()
        if not self.connection:
            return []
        try:
            # Validate table name against whitelist
            table_name = validate_table_name(table_name)
            query = f"SELECT * FROM {table_name} LIMIT %s"
            return self.execute_query(query, (limit,))
        except Exception as e:
            logger.error(f"Failed to get sample from table {table_name}: {e}")
            return []

    def close(self):
        "Close database connection"
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database"""
        self.ensure_connection()
        if not self.connection:
            logger.error("Database connection not available for listing tables")
            return []
            
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public';
                """)
                tables = cursor.fetchall()
                return [table['table_name'] for table in tables]
        except Exception as e:
            logger.error(f"Failed to get tables: {e}")
            return []
db_manager = DatabaseManager()