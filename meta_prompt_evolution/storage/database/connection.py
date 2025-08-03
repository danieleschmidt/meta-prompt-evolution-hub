"""Database connection management."""

import os
import json
import sqlite3
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass
import logging


@dataclass
class DatabaseConfig:
    """Database configuration."""
    database_path: str = "meta_prompt_hub.db"
    connection_pool_size: int = 10
    timeout: float = 30.0


class DatabaseConnection:
    """Simplified database connection using SQLite for development."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database connection."""
        self.config = config or DatabaseConfig()
        self.db_path = self.config.database_path
        self.logger = logging.getLogger(__name__)
        
        # Create database and tables
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Prompts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    generation INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    fitness_scores TEXT,
                    parent_ids TEXT
                )
            """)
            
            # Populations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS populations (
                    id TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    algorithm_used TEXT,
                    config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    diversity_score REAL,
                    best_fitness REAL
                )
            """)
            
            # Population members junction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS population_members (
                    population_id TEXT,
                    prompt_id TEXT,
                    rank INTEGER,
                    PRIMARY KEY (population_id, prompt_id),
                    FOREIGN KEY (population_id) REFERENCES populations(id),
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
                )
            """)
            
            # Evolution runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_runs (
                    id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_generations INTEGER,
                    final_best_fitness REAL,
                    config TEXT,
                    status TEXT DEFAULT 'running'
                )
            """)
            
            # Evaluation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    test_case_id TEXT,
                    metrics TEXT,
                    execution_time REAL,
                    model_used TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error TEXT,
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompts_generation ON prompts(generation)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompts_created_at ON prompts(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_populations_generation ON populations(generation)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_results_prompt_id ON evaluation_results(prompt_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_results_timestamp ON evaluation_results(timestamp)")
            
            conn.commit()
            self.logger.info("Database schema initialized successfully")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path, timeout=self.config.timeout)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Convert rows to dictionaries
            results = []
            for row in cursor.fetchall():
                results.append(dict(row))
            
            return results
    
    def execute_insert(self, query: str, params: Optional[tuple] = None) -> str:
        """Execute an INSERT query and return the lastrowid."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.lastrowid
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an UPDATE/DELETE query and return affected rows."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute multiple queries with different parameters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            self.execute_query("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get statistics about table sizes."""
        stats = {}
        tables = ["prompts", "populations", "population_members", "evolution_runs", "evaluation_results"]
        
        for table in tables:
            try:
                result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                stats[table] = result[0]["count"]
            except Exception as e:
                self.logger.warning(f"Could not get stats for table {table}: {e}")
                stats[table] = -1
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old evaluation results and temporary data."""
        query = """
            DELETE FROM evaluation_results 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days_to_keep)
        
        return self.execute_update(query)
    
    @staticmethod
    def serialize_json(data: Any) -> str:
        """Serialize data to JSON string for database storage."""
        if data is None:
            return None
        return json.dumps(data, default=str)
    
    @staticmethod
    def deserialize_json(json_str: Optional[str]) -> Any:
        """Deserialize JSON string from database."""
        if json_str is None:
            return None
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return None