"""Repository for population data access."""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_repository import BaseRepository
from .prompt_repository import PromptRepository
from ..database.connection import DatabaseConnection
from ...evolution.population import PromptPopulation, Prompt


class PopulationRepository(BaseRepository[PromptPopulation]):
    """Repository for managing population persistence."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize population repository."""
        super().__init__(db)
        self.prompt_repo = PromptRepository(db)
    
    @property
    def table_name(self) -> str:
        """Get table name."""
        return "populations"
    
    def _from_db_row(self, row: Dict[str, Any]) -> PromptPopulation:
        """Convert database row to PromptPopulation object."""
        # Get associated prompts
        prompts = self._get_population_prompts(row['id'])
        
        population = PromptPopulation(prompts)
        population.id = row['id']
        population.generation = row['generation']
        population.algorithm_used = row['algorithm_used']
        population.config = self.db.deserialize_json(row['config'])
        population.diversity_score = row['diversity_score']
        population.best_fitness = row['best_fitness']
        
        return population
    
    def _to_db_params(self, population: PromptPopulation) -> Dict[str, Any]:
        """Convert PromptPopulation object to database parameters."""
        # Calculate best fitness from prompts
        best_fitness = 0.0
        if population.prompts:
            fitness_scores = []
            for prompt in population.prompts:
                if prompt.fitness_scores and 'fitness' in prompt.fitness_scores:
                    fitness_scores.append(prompt.fitness_scores['fitness'])
            
            if fitness_scores:
                best_fitness = max(fitness_scores)
        
        return {
            'id': getattr(population, 'id', None) or self.generate_id(),
            'generation': population.generation,
            'algorithm_used': getattr(population, 'algorithm_used', 'unknown'),
            'config': self.db.serialize_json(getattr(population, 'config', {})),
            'diversity_score': getattr(population, 'diversity_score', 0.0),
            'best_fitness': best_fitness
        }
    
    def save(self, population: PromptPopulation) -> PromptPopulation:
        """Save population and its prompts."""
        # First save all prompts
        for prompt in population.prompts:
            self.prompt_repo.save(prompt)
        
        # Ensure population has an ID
        if not hasattr(population, 'id') or not population.id:
            population.id = self.generate_id()
        
        # Save population
        saved_population = super().save(population)
        
        # Update population members
        self._update_population_members(saved_population)
        
        return saved_population
    
    def _update_population_members(self, population: PromptPopulation):
        """Update the population_members junction table."""
        population_id = population.id
        
        # Clear existing members
        delete_query = "DELETE FROM population_members WHERE population_id = ?"
        self.db.execute_update(delete_query, (population_id,))
        
        # Add current members
        if population.prompts:
            member_data = []
            for rank, prompt in enumerate(population.prompts):
                member_data.append((population_id, prompt.id, rank))
            
            insert_query = "INSERT INTO population_members (population_id, prompt_id, rank) VALUES (?, ?, ?)"
            self.db.execute_many(insert_query, member_data)
    
    def _get_population_prompts(self, population_id: str) -> List[Prompt]:
        """Get all prompts for a population, ordered by rank."""
        query = """
            SELECT p.* FROM prompts p
            JOIN population_members pm ON p.id = pm.prompt_id
            WHERE pm.population_id = ?
            ORDER BY pm.rank
        """
        
        results = self.db.execute_query(query, (population_id,))
        return [self.prompt_repo._from_db_row(row) for row in results]
    
    def find_by_generation(self, generation: int) -> List[PromptPopulation]:
        """Find all populations from a specific generation."""
        return self.find_by_criteria({'generation': generation})
    
    def find_by_algorithm(self, algorithm: str) -> List[PromptPopulation]:
        """Find all populations created with a specific algorithm."""
        return self.find_by_criteria({'algorithm_used': algorithm})
    
    def get_evolution_progress(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get evolution progress across generations."""
        query = f"""
            SELECT generation, algorithm_used, best_fitness, diversity_score, created_at
            FROM {self.table_name}
            ORDER BY generation ASC, created_at ASC
            LIMIT ?
        """
        
        results = self.db.execute_query(query, (limit,))
        
        # Group by generation and calculate statistics
        progress = []
        current_generation = None
        generation_data = []
        
        for row in results:
            if current_generation != row['generation']:
                if generation_data:
                    # Process previous generation
                    progress.append(self._summarize_generation_data(generation_data))
                
                current_generation = row['generation']
                generation_data = []
            
            generation_data.append(row)
        
        # Process last generation
        if generation_data:
            progress.append(self._summarize_generation_data(generation_data))
        
        return progress
    
    def _summarize_generation_data(self, generation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize data for a single generation."""
        if not generation_data:
            return {}
        
        generation = generation_data[0]['generation']
        algorithms_used = list(set(row['algorithm_used'] for row in generation_data))
        
        fitness_scores = [row['best_fitness'] for row in generation_data if row['best_fitness'] is not None]
        diversity_scores = [row['diversity_score'] for row in generation_data if row['diversity_score'] is not None]
        
        summary = {
            'generation': generation,
            'population_count': len(generation_data),
            'algorithms_used': algorithms_used,
            'created_at': generation_data[0]['created_at']
        }
        
        if fitness_scores:
            summary['fitness_stats'] = {
                'best': max(fitness_scores),
                'avg': sum(fitness_scores) / len(fitness_scores),
                'worst': min(fitness_scores)
            }
        
        if diversity_scores:
            summary['diversity_stats'] = {
                'avg': sum(diversity_scores) / len(diversity_scores),
                'max': max(diversity_scores),
                'min': min(diversity_scores)
            }
        
        return summary
    
    def get_best_populations(self, limit: int = 10) -> List[PromptPopulation]:
        """Get populations with highest fitness scores."""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE best_fitness IS NOT NULL
            ORDER BY best_fitness DESC, created_at DESC
            LIMIT ?
        """
        
        results = self.db.execute_query(query, (limit,))
        return [self._from_db_row(row) for row in results]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about populations in the database."""
        stats = {}
        
        # Total count
        stats['total_populations'] = self.count()
        
        # Algorithm distribution
        query = """
            SELECT algorithm_used, COUNT(*) as count
            FROM populations
            GROUP BY algorithm_used
            ORDER BY count DESC
        """
        algorithm_stats = self.db.execute_query(query)
        stats['algorithm_distribution'] = {row['algorithm_used']: row['count'] for row in algorithm_stats}
        
        # Generation range
        query = "SELECT MIN(generation) as min_gen, MAX(generation) as max_gen FROM populations"
        gen_result = self.db.execute_query(query)
        if gen_result[0]['min_gen'] is not None:
            stats['generation_range'] = {
                'min': gen_result[0]['min_gen'],
                'max': gen_result[0]['max_gen']
            }
        
        # Fitness statistics
        query = """
            SELECT best_fitness
            FROM populations
            WHERE best_fitness IS NOT NULL
        """
        fitness_results = self.db.execute_query(query)
        
        if fitness_results:
            fitness_values = [row['best_fitness'] for row in fitness_results]
            stats['fitness_stats'] = {
                'count': len(fitness_values),
                'avg': sum(fitness_values) / len(fitness_values),
                'min': min(fitness_values),
                'max': max(fitness_values)
            }
        
        # Diversity statistics
        query = """
            SELECT diversity_score
            FROM populations
            WHERE diversity_score IS NOT NULL
        """
        diversity_results = self.db.execute_query(query)
        
        if diversity_results:
            diversity_values = [row['diversity_score'] for row in diversity_results]
            stats['diversity_stats'] = {
                'count': len(diversity_values),
                'avg': sum(diversity_values) / len(diversity_values),
                'min': min(diversity_values),
                'max': max(diversity_values)
            }
        
        return stats
    
    def cleanup_old_populations(self, generations_to_keep: int = 50) -> int:
        """Clean up old populations, keeping the most recent generations."""
        # Get the maximum generation
        query = "SELECT MAX(generation) as max_gen FROM populations"
        result = self.db.execute_query(query)
        
        if not result[0]['max_gen']:
            return 0
        
        max_generation = result[0]['max_gen']
        cutoff_generation = max_generation - generations_to_keep
        
        if cutoff_generation <= 0:
            return 0
        
        # Delete old populations and their members
        delete_members_query = """
            DELETE FROM population_members
            WHERE population_id IN (
                SELECT id FROM populations WHERE generation < ?
            )
        """
        self.db.execute_update(delete_members_query, (cutoff_generation,))
        
        delete_populations_query = "DELETE FROM populations WHERE generation < ?"
        return self.db.execute_update(delete_populations_query, (cutoff_generation,))