"""Repository for prompt data access."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .base_repository import BaseRepository
from ..database.connection import DatabaseConnection
from ...evolution.population import Prompt


class PromptRepository(BaseRepository[Prompt]):
    """Repository for managing prompt persistence."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize prompt repository."""
        super().__init__(db)
    
    @property
    def table_name(self) -> str:
        """Get table name."""
        return "prompts"
    
    def _from_db_row(self, row: Dict[str, Any]) -> Prompt:
        """Convert database row to Prompt object."""
        prompt = Prompt(
            text=row['text'],
            fitness_scores=self.db.deserialize_json(row['fitness_scores']),
            generation=row['generation'],
            parent_ids=self.db.deserialize_json(row['parent_ids']),
            id=row['id']
        )
        return prompt
    
    def _to_db_params(self, prompt: Prompt) -> Dict[str, Any]:
        """Convert Prompt object to database parameters."""
        return {
            'id': prompt.id,
            'text': prompt.text,
            'generation': prompt.generation,
            'metadata': self.db.serialize_json({}),  # Future extension
            'fitness_scores': self.db.serialize_json(prompt.fitness_scores),
            'parent_ids': self.db.serialize_json(prompt.parent_ids)
        }
    
    def find_by_generation(self, generation: int) -> List[Prompt]:
        """Find all prompts from a specific generation."""
        return self.find_by_criteria({'generation': generation})
    
    def find_top_by_fitness(self, metric: str = 'fitness', limit: int = 10) -> List[Prompt]:
        """Find top prompts by fitness metric."""
        # For SQLite, we need to handle JSON queries differently
        # This is a simplified approach
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE fitness_scores IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """
        
        results = self.db.execute_query(query, (limit * 2,))  # Get more to filter
        prompts = [self._from_db_row(row) for row in results]
        
        # Filter and sort by the specified metric
        scored_prompts = []
        for prompt in prompts:
            if prompt.fitness_scores and metric in prompt.fitness_scores:
                scored_prompts.append((prompt, prompt.fitness_scores[metric]))
        
        # Sort by fitness score (descending)
        scored_prompts.sort(key=lambda x: x[1], reverse=True)
        
        return [prompt for prompt, _ in scored_prompts[:limit]]
    
    def find_by_text_similarity(self, text: str, limit: int = 10) -> List[Prompt]:
        """Find prompts with similar text (simple implementation)."""
        # Simple similarity based on common words
        words = set(text.lower().split())
        
        query = f"SELECT * FROM {self.table_name} ORDER BY created_at DESC LIMIT ?"
        results = self.db.execute_query(query, (limit * 3,))  # Get more to filter
        
        prompts = [self._from_db_row(row) for row in results]
        
        # Calculate similarity scores
        scored_prompts = []
        for prompt in prompts:
            prompt_words = set(prompt.text.lower().split())
            if prompt_words:
                similarity = len(words.intersection(prompt_words)) / len(words.union(prompt_words))
                scored_prompts.append((prompt, similarity))
        
        # Sort by similarity (descending)
        scored_prompts.sort(key=lambda x: x[1], reverse=True)
        
        return [prompt for prompt, _ in scored_prompts[:limit]]
    
    def find_by_parent_id(self, parent_id: str) -> List[Prompt]:
        """Find all prompts that have the given parent ID."""
        # SQLite doesn't have good JSON query support, so we'll do a simple text search
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE parent_ids LIKE ?
            ORDER BY created_at DESC
        """
        
        results = self.db.execute_query(query, (f'%"{parent_id}"%',))
        prompts = [self._from_db_row(row) for row in results]
        
        # Filter more precisely
        filtered_prompts = []
        for prompt in prompts:
            if prompt.parent_ids and parent_id in prompt.parent_ids:
                filtered_prompts.append(prompt)
        
        return filtered_prompts
    
    def get_genealogy(self, prompt_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get the genealogy tree for a prompt."""
        prompt = self.find_by_id(prompt_id)
        if not prompt:
            return {}
        
        def build_tree(current_prompt: Prompt, depth: int) -> Dict[str, Any]:
            if depth >= max_depth or not current_prompt.parent_ids:
                return {
                    'prompt': current_prompt,
                    'parents': []
                }
            
            parents = []
            for parent_id in current_prompt.parent_ids:
                parent_prompt = self.find_by_id(parent_id)
                if parent_prompt:
                    parent_tree = build_tree(parent_prompt, depth + 1)
                    parents.append(parent_tree)
            
            return {
                'prompt': current_prompt,
                'parents': parents
            }
        
        return build_tree(prompt, 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about prompts in the database."""
        stats = {}
        
        # Total count
        stats['total_prompts'] = self.count()
        
        # Generation distribution
        query = """
            SELECT generation, COUNT(*) as count
            FROM prompts
            GROUP BY generation
            ORDER BY generation
        """
        generation_stats = self.db.execute_query(query)
        stats['generation_distribution'] = {row['generation']: row['count'] for row in generation_stats}
        
        # Recent activity (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        recent_query = f"SELECT COUNT(*) as count FROM {self.table_name} WHERE created_at > ?"
        recent_result = self.db.execute_query(recent_query, (yesterday,))
        stats['recent_prompts_24h'] = recent_result[0]['count']
        
        # Fitness score statistics
        query = f"""
            SELECT fitness_scores
            FROM {self.table_name}
            WHERE fitness_scores IS NOT NULL
        """
        fitness_results = self.db.execute_query(query)
        
        fitness_values = []
        for row in fitness_results:
            scores = self.db.deserialize_json(row['fitness_scores'])
            if scores and 'fitness' in scores:
                fitness_values.append(scores['fitness'])
        
        if fitness_values:
            stats['fitness_stats'] = {
                'count': len(fitness_values),
                'avg': sum(fitness_values) / len(fitness_values),
                'min': min(fitness_values),
                'max': max(fitness_values)
            }
        else:
            stats['fitness_stats'] = {'count': 0}
        
        return stats
    
    def cleanup_low_fitness_prompts(self, fitness_threshold: float = 0.1, keep_count: int = 1000) -> int:
        """Clean up prompts with low fitness scores, keeping the best ones."""
        # Get prompts with fitness scores
        query = f"""
            SELECT id, fitness_scores
            FROM {self.table_name}
            WHERE fitness_scores IS NOT NULL
            ORDER BY created_at DESC
        """
        
        results = self.db.execute_query(query)
        
        # Parse fitness scores and identify low-performing prompts
        low_fitness_ids = []
        high_fitness_count = 0
        
        for row in results:
            scores = self.db.deserialize_json(row['fitness_scores'])
            if scores and 'fitness' in scores:
                fitness = scores['fitness']
                if fitness < fitness_threshold and high_fitness_count >= keep_count:
                    low_fitness_ids.append(row['id'])
                elif fitness >= fitness_threshold:
                    high_fitness_count += 1
        
        # Delete low fitness prompts
        deleted_count = 0
        for prompt_id in low_fitness_ids:
            if self.delete_by_id(prompt_id):
                deleted_count += 1
        
        return deleted_count