#!/usr/bin/env python3
"""
Generation 2: Robust Evolution System
Enhanced with comprehensive error handling, validation, logging, and monitoring.
"""

import json
import time
import logging
import traceback
import hashlib
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import os
import sys


# Enhanced logging configuration
def setup_robust_logging():
    """Setup comprehensive logging system."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f"logs/evolution_{int(time.time())}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


@dataclass
class RobustPrompt:
    """Enhanced prompt with validation and error handling."""
    id: str
    text: str
    fitness: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = None
    validation_errors: List[str] = None
    checksum: str = ""
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
        if self.validation_errors is None:
            self.validation_errors = []
        if self.created_at == 0.0:
            self.created_at = time.time()
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        content = f"{self.text}|{self.generation}|{self.created_at}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def validate(self) -> bool:
        """Validate prompt integrity and content."""
        self.validation_errors.clear()
        
        # Check text content
        if not self.text or not self.text.strip():
            self.validation_errors.append("Empty or whitespace-only text")
        
        if len(self.text) > 1000:
            self.validation_errors.append("Text exceeds maximum length (1000 chars)")
        
        if len(self.text.split()) < 2:
            self.validation_errors.append("Text has fewer than 2 words")
        
        # Check fitness range
        if not (0.0 <= self.fitness <= 1.0):
            self.validation_errors.append(f"Fitness {self.fitness} outside valid range [0.0, 1.0]")
        
        # Check generation
        if self.generation < 0:
            self.validation_errors.append(f"Negative generation number: {self.generation}")
        
        # Verify checksum
        expected_checksum = self._calculate_checksum()
        if self.checksum != expected_checksum:
            self.validation_errors.append("Checksum mismatch - data corruption detected")
        
        return len(self.validation_errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with validation."""
        if not self.validate():
            raise ValueError(f"Prompt validation failed: {self.validation_errors}")
        return asdict(self)


class RobustEvolutionEngine:
    """Production-ready evolution engine with comprehensive error handling."""
    
    def __init__(
        self, 
        population_size: int = 20, 
        mutation_rate: float = 0.1,
        max_retries: int = 3,
        backup_frequency: int = 5
    ):
        self.logger = setup_robust_logging()
        self.population_size = self._validate_population_size(population_size)
        self.mutation_rate = self._validate_mutation_rate(mutation_rate)
        self.max_retries = max_retries
        self.backup_frequency = backup_frequency
        
        # Evolution state
        self.generation = 0
        self.history = []
        self.error_log = []
        self.performance_metrics = {}
        
        # Backup and recovery
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized RobustEvolutionEngine: pop_size={self.population_size}, mutation_rate={self.mutation_rate}")
    
    def _validate_population_size(self, size: int) -> int:
        """Validate and sanitize population size."""
        if not isinstance(size, int):
            raise TypeError(f"Population size must be integer, got {type(size)}")
        if size < 5:
            self.logger.warning(f"Population size {size} too small, using minimum 5")
            return 5
        if size > 1000:
            self.logger.warning(f"Population size {size} too large, using maximum 1000")
            return 1000
        return size
    
    def _validate_mutation_rate(self, rate: float) -> float:
        """Validate and sanitize mutation rate."""
        if not isinstance(rate, (int, float)):
            raise TypeError(f"Mutation rate must be numeric, got {type(rate)}")
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"Mutation rate must be in [0.0, 1.0], got {rate}")
        return float(rate)
    
    def evolve_prompts(
        self, 
        seed_prompts: List[str], 
        fitness_evaluator: Callable[[str], float],
        generations: int = 10,
        termination_criteria: Optional[Callable[[List[RobustPrompt]], bool]] = None
    ) -> List[RobustPrompt]:
        """
        Robust evolution with comprehensive error handling and recovery.
        """
        try:
            self.logger.info(f"Starting robust evolution: {len(seed_prompts)} seeds → {generations} generations")
            
            # Validate inputs
            self._validate_inputs(seed_prompts, fitness_evaluator, generations)
            
            # Initialize population with error handling
            population = self._initialize_population(seed_prompts, fitness_evaluator)
            
            # Evolution loop with recovery mechanisms
            for gen in range(generations):
                try:
                    start_time = time.time()
                    self.generation = gen + 1
                    
                    self.logger.info(f"Starting generation {self.generation}")
                    
                    # Backup current state
                    if gen % self.backup_frequency == 0:
                        self._backup_population(population, gen)
                    
                    # Evolve generation with retry logic
                    population = self._evolve_generation_with_retry(population, fitness_evaluator)
                    
                    # Validate population integrity
                    population = self._validate_and_repair_population(population)
                    
                    # Track performance metrics
                    gen_metrics = self._calculate_generation_metrics(population, start_time)
                    self.history.append(gen_metrics)
                    
                    self.logger.info(
                        f"Generation {self.generation} completed: "
                        f"Best={gen_metrics['best_fitness']:.3f}, "
                        f"Avg={gen_metrics['avg_fitness']:.3f}, "
                        f"Errors={gen_metrics['validation_errors']}"
                    )
                    
                    # Check termination criteria
                    if termination_criteria and termination_criteria(population):
                        self.logger.info("Termination criteria met, stopping evolution")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error in generation {gen}: {e}")
                    self.error_log.append({
                        "generation": gen,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": time.time()
                    })
                    
                    # Attempt recovery
                    population = self._recover_from_error(population, gen)
                    
                    if not population:
                        self.logger.critical("Recovery failed, terminating evolution")
                        break
            
            # Final validation and cleanup
            population = self._finalize_population(population)
            self._save_comprehensive_results(population)
            
            return population
            
        except Exception as e:
            self.logger.critical(f"Critical error in evolution: {e}")
            self.logger.critical(traceback.format_exc())
            raise
    
    def _validate_inputs(self, seed_prompts: List[str], fitness_evaluator: Callable, generations: int):
        """Comprehensive input validation."""
        if not seed_prompts:
            raise ValueError("Seed prompts cannot be empty")
        
        if not all(isinstance(prompt, str) and prompt.strip() for prompt in seed_prompts):
            raise ValueError("All seed prompts must be non-empty strings")
        
        if not callable(fitness_evaluator):
            raise TypeError("Fitness evaluator must be callable")
        
        if not isinstance(generations, int) or generations < 1:
            raise ValueError("Generations must be positive integer")
        
        # Test fitness evaluator
        try:
            test_score = fitness_evaluator(seed_prompts[0])
            if not isinstance(test_score, (int, float)) or not (0.0 <= test_score <= 1.0):
                raise ValueError("Fitness evaluator must return float in [0.0, 1.0]")
        except Exception as e:
            raise ValueError(f"Fitness evaluator test failed: {e}")
    
    def _initialize_population(self, seed_prompts: List[str], fitness_evaluator: Callable) -> List[RobustPrompt]:
        """Initialize population with comprehensive error handling."""
        population = []
        
        for i, seed in enumerate(seed_prompts):
            try:
                prompt = RobustPrompt(
                    id=f"seed_{i}",
                    text=seed.strip(),
                    generation=0
                )
                
                # Evaluate fitness with retry logic
                prompt.fitness = self._evaluate_fitness_with_retry(prompt.text, fitness_evaluator)
                
                if prompt.validate():
                    population.append(prompt)
                else:
                    self.logger.warning(f"Seed prompt {i} failed validation: {prompt.validation_errors}")
                    
            except Exception as e:
                self.logger.error(f"Error initializing seed {i}: {e}")
                self.error_log.append({
                    "phase": "initialization",
                    "seed_index": i,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        # Fill population to target size
        while len(population) < self.population_size and population:
            try:
                parent = self._safe_random_choice(population)
                if parent:
                    mutant = self._safe_mutate_prompt(parent)
                    if mutant:
                        mutant.fitness = self._evaluate_fitness_with_retry(mutant.text, fitness_evaluator)
                        if mutant.validate():
                            population.append(mutant)
            except Exception as e:
                self.logger.error(f"Error filling population: {e}")
                break
        
        if len(population) < 5:
            raise RuntimeError("Failed to initialize minimum viable population")
        
        self.logger.info(f"Initialized population with {len(population)} valid prompts")
        return population
    
    def _evaluate_fitness_with_retry(self, text: str, fitness_evaluator: Callable) -> float:
        """Evaluate fitness with retry logic and error handling."""
        for attempt in range(self.max_retries):
            try:
                score = fitness_evaluator(text)
                if isinstance(score, (int, float)) and 0.0 <= score <= 1.0:
                    return float(score)
                else:
                    self.logger.warning(f"Invalid fitness score {score}, attempt {attempt + 1}")
            except Exception as e:
                self.logger.warning(f"Fitness evaluation failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All fitness evaluation attempts failed for text: {text[:50]}...")
                    return 0.0
        return 0.0
    
    def _evolve_generation_with_retry(self, population: List[RobustPrompt], fitness_evaluator: Callable) -> List[RobustPrompt]:
        """Evolve generation with retry mechanisms."""
        for attempt in range(self.max_retries):
            try:
                return self._evolve_generation(population, fitness_evaluator)
            except Exception as e:
                self.logger.warning(f"Generation evolution failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error("All generation evolution attempts failed")
                    return population  # Return unchanged population
        return population
    
    def _evolve_generation(self, population: List[RobustPrompt], fitness_evaluator: Callable) -> List[RobustPrompt]:
        """Core generation evolution logic."""
        new_population = []
        
        # Elitism with validation
        population.sort(key=lambda p: p.fitness, reverse=True)
        elite_count = max(1, int(self.population_size * 0.2))
        
        for prompt in population[:elite_count]:
            if prompt.validate():
                new_population.append(prompt)
        
        # Generate offspring
        while len(new_population) < self.population_size:
            try:
                # Tournament selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                if not parent1 or not parent2:
                    break
                
                # Crossover
                if len(new_population) % 3 == 0:  # 33% crossover rate
                    child = self._safe_crossover(parent1, parent2)
                else:
                    child = self._safe_random_choice([parent1, parent2])
                
                if not child:
                    continue
                
                # Mutation
                if len(new_population) % int(1/self.mutation_rate) == 0:
                    child = self._safe_mutate_prompt(child)
                
                if child:
                    child.generation = self.generation
                    child.fitness = self._evaluate_fitness_with_retry(child.text, fitness_evaluator)
                    
                    if child.validate():
                        new_population.append(child)
                    
            except Exception as e:
                self.logger.warning(f"Error generating offspring: {e}")
                continue
        
        return new_population[:self.population_size]
    
    def _safe_random_choice(self, items: List[Any]) -> Optional[Any]:
        """Safe random choice with error handling."""
        try:
            if not items:
                return None
            import random
            return random.choice(items)
        except Exception as e:
            self.logger.warning(f"Safe random choice failed: {e}")
            return items[0] if items else None
    
    def _safe_mutate_prompt(self, prompt: RobustPrompt) -> Optional[RobustPrompt]:
        """Safe mutation with comprehensive error handling."""
        try:
            words = prompt.text.split()
            if not words:
                return None
            
            import random
            mutation_type = random.choice([
                "word_substitute", "word_insert", "word_delete", 
                "word_swap", "phrase_modify"
            ])
            
            new_words = words.copy()
            
            if mutation_type == "word_substitute" and words:
                idx = random.randint(0, len(words) - 1)
                new_words[idx] = self._generate_safe_word_variant(words[idx])
                
            elif mutation_type == "word_insert":
                insert_words = ["please", "help", "assist", "explain", "describe", "analyze"]
                idx = random.randint(0, len(words))
                new_words.insert(idx, random.choice(insert_words))
                
            elif mutation_type == "word_delete" and len(words) > 2:
                idx = random.randint(0, len(words) - 1)
                new_words.pop(idx)
                
            elif mutation_type == "word_swap" and len(words) > 1:
                idx1, idx2 = random.sample(range(len(words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
                
            elif mutation_type == "phrase_modify":
                prefixes = ["Please", "Could you", "Can you", "I need you to"]
                suffixes = ["clearly", "in detail", "step by step", "concisely"]
                
                if random.random() < 0.5:
                    new_words.insert(0, random.choice(prefixes))
                if random.random() < 0.5:
                    new_words.append(random.choice(suffixes))
            
            mutated_text = " ".join(new_words)
            
            # Validate mutated text
            if not mutated_text.strip() or len(mutated_text) > 1000:
                return None
            
            return RobustPrompt(
                id=str(uuid.uuid4()),
                text=mutated_text,
                generation=prompt.generation,
                metadata={"parent_id": prompt.id, "mutation": mutation_type}
            )
            
        except Exception as e:
            self.logger.warning(f"Mutation failed: {e}")
            return None
    
    def _safe_crossover(self, parent1: RobustPrompt, parent2: RobustPrompt) -> Optional[RobustPrompt]:
        """Safe crossover with error handling."""
        try:
            words1 = parent1.text.split()
            words2 = parent2.text.split()
            
            if not words1 and not words2:
                return None
            
            import random
            if words1 and words2:
                split1 = random.randint(0, len(words1))
                split2 = random.randint(0, len(words2))
                new_words = words1[:split1] + words2[split2:]
            else:
                new_words = words1 if words1 else words2
            
            crossover_text = " ".join(new_words)
            
            if not crossover_text.strip():
                return None
            
            return RobustPrompt(
                id=str(uuid.uuid4()),
                text=crossover_text,
                generation=max(parent1.generation, parent2.generation),
                metadata={"parent1_id": parent1.id, "parent2_id": parent2.id, "operation": "crossover"}
            )
            
        except Exception as e:
            self.logger.warning(f"Crossover failed: {e}")
            return None
    
    def _generate_safe_word_variant(self, word: str) -> str:
        """Generate safe word variants with fallback."""
        try:
            variants = {
                "help": ["assist", "aid", "support"],
                "explain": ["describe", "clarify", "detail"],
                "analyze": ["examine", "evaluate", "assess"],
                "create": ["generate", "build", "make"],
                "understand": ["grasp", "comprehend", "learn"]
            }
            
            import random
            return random.choice(variants.get(word.lower(), [word]))
        except Exception:
            return word
    
    def _tournament_select(self, population: List[RobustPrompt]) -> Optional[RobustPrompt]:
        """Tournament selection with error handling."""
        try:
            import random
            tournament = random.sample(population, min(3, len(population)))
            return max(tournament, key=lambda p: p.fitness)
        except Exception as e:
            self.logger.warning(f"Tournament selection failed: {e}")
            return population[0] if population else None
    
    def _validate_and_repair_population(self, population: List[RobustPrompt]) -> List[RobustPrompt]:
        """Validate and repair population integrity."""
        valid_population = []
        repair_count = 0
        
        for prompt in population:
            if prompt.validate():
                valid_population.append(prompt)
            else:
                self.logger.warning(f"Prompt {prompt.id} validation failed: {prompt.validation_errors}")
                repair_count += 1
                
                # Attempt basic repair
                if prompt.text and prompt.text.strip():
                    repaired = RobustPrompt(
                        id=str(uuid.uuid4()),
                        text=prompt.text.strip()[:500],  # Truncate if too long
                        fitness=max(0.0, min(1.0, prompt.fitness)),  # Clamp fitness
                        generation=prompt.generation,
                        metadata={"repaired_from": prompt.id}
                    )
                    
                    if repaired.validate():
                        valid_population.append(repaired)
                        self.logger.info(f"Successfully repaired prompt {prompt.id}")
        
        if repair_count > 0:
            self.logger.info(f"Repaired {repair_count} prompts in population")
        
        return valid_population
    
    def _calculate_generation_metrics(self, population: List[RobustPrompt], start_time: float) -> Dict[str, Any]:
        """Calculate comprehensive generation metrics."""
        if not population:
            return {"error": "Empty population"}
        
        fitnesses = [p.fitness for p in population]
        valid_prompts = [p for p in population if p.validate()]
        
        return {
            "generation": self.generation,
            "population_size": len(population),
            "valid_prompts": len(valid_prompts),
            "validation_errors": len(population) - len(valid_prompts),
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            "fitness_std": self._calculate_std(fitnesses),
            "diversity": self._calculate_robust_diversity(population),
            "execution_time": time.time() - start_time,
            "error_rate": len(self.error_log) / max(1, self.generation),
            "timestamp": time.time()
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation safely."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_robust_diversity(self, population: List[RobustPrompt]) -> float:
        """Calculate diversity with error handling."""
        try:
            if len(population) < 2:
                return 0.0
            
            total_distance = 0.0
            comparisons = 0
            
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    try:
                        similarity = self._text_similarity(population[i].text, population[j].text)
                        total_distance += (1.0 - similarity)
                        comparisons += 1
                    except Exception:
                        continue
            
            return total_distance / comparisons if comparisons > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Diversity calculation failed: {e}")
            return 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity with error handling."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def _backup_population(self, population: List[RobustPrompt], generation: int):
        """Backup population state for recovery."""
        try:
            backup_data = {
                "generation": generation,
                "timestamp": time.time(),
                "population": [prompt.to_dict() for prompt in population if prompt.validate()],
                "config": {
                    "population_size": self.population_size,
                    "mutation_rate": self.mutation_rate
                }
            }
            
            backup_file = self.backup_dir / f"backup_gen_{generation}.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            self.logger.info(f"Backup saved: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
    
    def _recover_from_error(self, population: List[RobustPrompt], generation: int) -> Optional[List[RobustPrompt]]:
        """Attempt recovery from errors."""
        try:
            # Try to recover from last backup
            for gen in range(generation, max(-1, generation - 5), -1):
                backup_file = self.backup_dir / f"backup_gen_{gen}.json"
                if backup_file.exists():
                    self.logger.info(f"Attempting recovery from {backup_file}")
                    
                    with open(backup_file, 'r') as f:
                        backup_data = json.load(f)
                    
                    recovered_population = []
                    for prompt_data in backup_data["population"]:
                        prompt = RobustPrompt(**prompt_data)
                        if prompt.validate():
                            recovered_population.append(prompt)
                    
                    if len(recovered_population) >= 5:
                        self.logger.info(f"Successfully recovered {len(recovered_population)} prompts")
                        return recovered_population
            
            # If no backup available, create minimal population
            if population and len([p for p in population if p.validate()]) >= 2:
                valid_prompts = [p for p in population if p.validate()]
                self.logger.info(f"Using {len(valid_prompts)} valid prompts from current population")
                return valid_prompts
            
            self.logger.error("Recovery failed - no valid prompts available")
            return None
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return None
    
    def _finalize_population(self, population: List[RobustPrompt]) -> List[RobustPrompt]:
        """Final validation and sorting of population."""
        try:
            # Filter and validate
            valid_population = [p for p in population if p.validate()]
            
            # Sort by fitness
            valid_population.sort(key=lambda p: p.fitness, reverse=True)
            
            self.logger.info(f"Finalized population: {len(valid_population)} valid prompts")
            return valid_population
            
        except Exception as e:
            self.logger.error(f"Population finalization failed: {e}")
            return population
    
    def _save_comprehensive_results(self, population: List[RobustPrompt]):
        """Save comprehensive results with error handling."""
        try:
            results = {
                "generation_2_robust": {
                    "config": {
                        "population_size": self.population_size,
                        "mutation_rate": self.mutation_rate,
                        "max_retries": self.max_retries,
                        "backup_frequency": self.backup_frequency
                    },
                    "final_population": [p.to_dict() for p in population if p.validate()],
                    "evolution_history": self.history,
                    "error_log": self.error_log,
                    "performance_summary": self._generate_performance_summary(),
                    "timestamp": time.time()
                }
            }
            
            results_file = f"generation_2_robust_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Comprehensive results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        if not self.history:
            return {}
        
        try:
            total_time = sum(gen.get("execution_time", 0) for gen in self.history)
            final_gen = self.history[-1]
            
            return {
                "total_generations": len(self.history),
                "total_execution_time": total_time,
                "final_best_fitness": final_gen.get("best_fitness", 0.0),
                "final_diversity": final_gen.get("diversity", 0.0),
                "total_errors": len(self.error_log),
                "average_generation_time": total_time / len(self.history),
                "fitness_improvement": final_gen.get("best_fitness", 0.0) - self.history[0].get("best_fitness", 0.0)
            }
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {e}")
            return {"error": str(e)}


def enhanced_robust_fitness_evaluator(prompt_text: str) -> float:
    """Enhanced fitness evaluator with robust error handling."""
    try:
        if not prompt_text or not prompt_text.strip():
            return 0.0
        
        score = 0.0
        words = prompt_text.split()
        text_lower = prompt_text.lower()
        
        # Length optimization (8-15 words optimal)
        if 8 <= len(words) <= 15:
            score += 0.3
        elif 5 <= len(words) <= 20:
            score += 0.2
        else:
            score += 0.1
        
        # Quality indicators
        quality_terms = [
            "please", "help", "explain", "describe", "analyze", 
            "step", "detail", "clearly", "concisely", "specifically",
            "could you", "would you", "can you"
        ]
        quality_score = sum(0.05 for term in quality_terms if term in text_lower)
        score += min(quality_score, 0.35)
        
        # Structure and clarity
        structure_indicators = [":", "?", "step by step", "how to", "first", "then"]
        structure_score = sum(0.08 for indicator in structure_indicators if indicator in text_lower)
        score += min(structure_score, 0.2)
        
        # Avoid excessive repetition
        unique_words = len(set(word.lower() for word in words))
        repetition_ratio = unique_words / len(words) if words else 0
        score += repetition_ratio * 0.15
        
        return max(0.0, min(1.0, score))
        
    except Exception:
        return 0.0


def run_generation_2_robust():
    """Run Generation 2 robust evolution system."""
    logger = setup_robust_logging()
    
    print("=" * 70)
    print("🛡️  GENERATION 2: ROBUST EVOLUTION SYSTEM")
    print("=" * 70)
    
    try:
        # Initialize robust engine
        engine = RobustEvolutionEngine(
            population_size=30,
            mutation_rate=0.15,
            max_retries=3,
            backup_frequency=3
        )
        
        # Enhanced seed prompts
        seed_prompts = [
            "Help me understand this topic clearly and thoroughly",
            "Please explain the concept step by step with examples",
            "I need detailed assistance with comprehensive analysis",
            "Could you describe the process in a structured way",
            "Can you help me analyze this information systematically",
            "Please provide a clear and detailed explanation",
            "I would like you to explain this concept clearly",
            "Help me comprehend this subject with specific details",
            "Could you assist me in understanding this thoroughly",
            "Please break down this topic into manageable parts"
        ]
        
        print(f"🌱 Starting with {len(seed_prompts)} enhanced seed prompts")
        print(f"🛡️  Robust population size: {engine.population_size}")
        print(f"🔀 Mutation rate: {engine.mutation_rate}")
        print(f"🔄 Max retries: {engine.max_retries}")
        print(f"💾 Backup frequency: {engine.backup_frequency}")
        print()
        
        start_time = time.time()
        
        # Run robust evolution
        evolved_prompts = engine.evolve_prompts(
            seed_prompts=seed_prompts,
            fitness_evaluator=enhanced_robust_fitness_evaluator,
            generations=15
        )
        
        total_time = time.time() - start_time
        
        # Results analysis
        print("\n" + "=" * 70)
        print("📊 GENERATION 2 ROBUST RESULTS")
        print("=" * 70)
        
        if evolved_prompts:
            print(f"⏱️  Total execution time: {total_time:.2f} seconds")
            print(f"🏆 Best fitness achieved: {evolved_prompts[0].fitness:.3f}")
            print(f"📈 Population size: {len(evolved_prompts)}")
            print(f"🔧 Total errors encountered: {len(engine.error_log)}")
            print(f"💾 Backups created: {len(list(engine.backup_dir.glob('*.json')))}")
            
            print(f"\n🏆 TOP 10 ROBUST EVOLVED PROMPTS:")
            for i, prompt in enumerate(evolved_prompts[:10], 1):
                validation_status = "✅" if prompt.validate() else "❌"
                print(f"{i:2d}. {validation_status} [{prompt.fitness:.3f}] {prompt.text}")
            
            # Quality gates for Generation 2
            print(f"\n🔍 GENERATION 2 ROBUST QUALITY GATES:")
            gates_passed = 0
            total_gates = 6
            
            # Gate 1: Best fitness threshold
            if evolved_prompts[0].fitness >= 0.7:
                print("✅ Gate 1: Best fitness >= 0.7 PASSED")
                gates_passed += 1
            else:
                print(f"❌ Gate 1: Best fitness >= 0.7 FAILED ({evolved_prompts[0].fitness:.3f})")
            
            # Gate 2: Population validation
            valid_prompts = [p for p in evolved_prompts if p.validate()]
            if len(valid_prompts) >= len(evolved_prompts) * 0.9:
                print("✅ Gate 2: 90%+ population valid PASSED")
                gates_passed += 1
            else:
                print(f"❌ Gate 2: 90%+ population valid FAILED ({len(valid_prompts)}/{len(evolved_prompts)})")
            
            # Gate 3: Error resilience
            if len(engine.error_log) <= engine.generation * 0.1:
                print("✅ Gate 3: Error rate < 10% PASSED")
                gates_passed += 1
            else:
                print(f"❌ Gate 3: Error rate < 10% FAILED ({len(engine.error_log)}/{engine.generation})")
            
            # Gate 4: Execution time
            if total_time < 30.0:
                print("✅ Gate 4: Execution time < 30s PASSED")
                gates_passed += 1
            else:
                print(f"❌ Gate 4: Execution time < 30s FAILED ({total_time:.2f}s)")
            
            # Gate 5: Diversity maintenance
            final_diversity = engine.history[-1]["diversity"] if engine.history else 0
            if final_diversity > 0.25:
                print("✅ Gate 5: Diversity > 0.25 PASSED")
                gates_passed += 1
            else:
                print(f"❌ Gate 5: Diversity > 0.25 FAILED ({final_diversity:.3f})")
            
            # Gate 6: Recovery capability
            if len(list(engine.backup_dir.glob('*.json'))) > 0:
                print("✅ Gate 6: Recovery backups created PASSED")
                gates_passed += 1
            else:
                print("❌ Gate 6: Recovery backups created FAILED")
            
            print(f"\n🎯 Robust Quality Gates: {gates_passed}/{total_gates} passed")
            
            success = gates_passed >= total_gates * 0.8  # 80% pass rate
            
            return success
        else:
            print("❌ No evolved prompts generated")
            return False
            
    except Exception as e:
        logger.critical(f"Generation 2 robust evolution failed: {e}")
        logger.critical(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_generation_2_robust()
    
    if success:
        print("\n" + "="*70)
        print("✨ GENERATION 2 ROBUST EVOLUTION COMPLETE")
        print("Ready for Generation 3: Scalable Optimization")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("🔧 GENERATION 2 NEEDS FURTHER OPTIMIZATION")
        print("Reviewing robust implementation before proceeding")
        print("="*70)