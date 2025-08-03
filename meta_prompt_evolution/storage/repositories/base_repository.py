"""Base repository with common CRUD operations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generic, TypeVar
import uuid
import time
from datetime import datetime

from ..database.connection import DatabaseConnection

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """Base repository class with common CRUD operations."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize repository with database connection."""
        self.db = db
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Get the table name for this repository."""
        pass
    
    @abstractmethod
    def _from_db_row(self, row: Dict[str, Any]) -> T:
        """Convert database row to domain object."""
        pass
    
    @abstractmethod
    def _to_db_params(self, obj: T) -> Dict[str, Any]:
        """Convert domain object to database parameters."""
        pass
    
    def generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())
    
    def find_by_id(self, obj_id: str) -> Optional[T]:
        """Find object by ID."""
        query = f"SELECT * FROM {self.table_name} WHERE id = ?"
        results = self.db.execute_query(query, (obj_id,))
        
        if results:
            return self._from_db_row(results[0])
        return None
    
    def find_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """Find all objects with optional pagination."""
        query = f"SELECT * FROM {self.table_name} ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        results = self.db.execute_query(query)
        return [self._from_db_row(row) for row in results]
    
    def save(self, obj: T) -> T:
        """Save object to database (insert or update)."""
        params = self._to_db_params(obj)
        
        # Check if object exists
        if hasattr(obj, 'id') and obj.id and self.find_by_id(obj.id):
            return self._update(obj)
        else:
            return self._insert(obj)
    
    def _insert(self, obj: T) -> T:
        """Insert new object."""
        params = self._to_db_params(obj)
        
        # Generate ID if not present
        if 'id' not in params or not params['id']:
            params['id'] = self.generate_id()
            if hasattr(obj, 'id'):
                obj.id = params['id']
        
        # Add timestamps
        current_time = datetime.now().isoformat()
        params['created_at'] = current_time
        params['updated_at'] = current_time
        
        # Build INSERT query
        columns = list(params.keys())
        placeholders = ', '.join(['?' for _ in columns])
        query = f"INSERT INTO {self.table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        self.db.execute_insert(query, tuple(params.values()))
        return obj
    
    def _update(self, obj: T) -> T:
        """Update existing object."""
        params = self._to_db_params(obj)
        
        # Add updated timestamp
        params['updated_at'] = datetime.now().isoformat()
        
        # Remove id from params for SET clause
        obj_id = params.pop('id')
        
        # Build UPDATE query
        set_clause = ', '.join([f"{col} = ?" for col in params.keys()])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = ?"
        
        values = list(params.values()) + [obj_id]
        self.db.execute_update(query, tuple(values))
        return obj
    
    def delete_by_id(self, obj_id: str) -> bool:
        """Delete object by ID."""
        query = f"DELETE FROM {self.table_name} WHERE id = ?"
        affected_rows = self.db.execute_update(query, (obj_id,))
        return affected_rows > 0
    
    def count(self) -> int:
        """Count total objects."""
        query = f"SELECT COUNT(*) as count FROM {self.table_name}"
        result = self.db.execute_query(query)
        return result[0]['count']
    
    def exists(self, obj_id: str) -> bool:
        """Check if object exists."""
        query = f"SELECT 1 FROM {self.table_name} WHERE id = ? LIMIT 1"
        results = self.db.execute_query(query, (obj_id,))
        return len(results) > 0
    
    def find_by_criteria(self, criteria: Dict[str, Any], limit: Optional[int] = None) -> List[T]:
        """Find objects by criteria."""
        if not criteria:
            return self.find_all(limit=limit)
        
        where_clauses = []
        params = []
        
        for key, value in criteria.items():
            if value is not None:
                where_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not where_clauses:
            return self.find_all(limit=limit)
        
        where_clause = " AND ".join(where_clauses)
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause} ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.db.execute_query(query, tuple(params))
        return [self._from_db_row(row) for row in results]
    
    def batch_save(self, objects: List[T]) -> List[T]:
        """Save multiple objects efficiently."""
        if not objects:
            return []
        
        saved_objects = []
        for obj in objects:
            saved_obj = self.save(obj)
            saved_objects.append(saved_obj)
        
        return saved_objects
    
    def delete_by_criteria(self, criteria: Dict[str, Any]) -> int:
        """Delete objects by criteria."""
        if not criteria:
            return 0
        
        where_clauses = []
        params = []
        
        for key, value in criteria.items():
            if value is not None:
                where_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not where_clauses:
            return 0
        
        where_clause = " AND ".join(where_clauses)
        query = f"DELETE FROM {self.table_name} WHERE {where_clause}"
        
        return self.db.execute_update(query, tuple(params))