"""
Quantum-Inspired Evolutionary Algorithm (QIEA)
Next-generation algorithm incorporating quantum computing principles for prompt evolution.
"""

import math
import random
import cmath
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json

from .base import EvolutionAlgorithm, AlgorithmConfig
from ..population import PromptPopulation, Prompt


@dataclass
class QuantumConfig(AlgorithmConfig):
    """Configuration for Quantum-Inspired Evolutionary Algorithm."""
    population_size: int = 50
    max_generations: int = 100
    quantum_population_size: int = 10  # Quantum individuals
    rotation_angle: float = 0.01 * math.pi  # Rotation gate angle
    quantum_crossover_rate: float = 0.8
    quantum_mutation_rate: float = 0.1
    superposition_collapse_probability: float = 0.7
    entanglement_strength: float = 0.3
    
    
class QuantumIndividual:
    """
    Quantum individual representing superposition of multiple prompt states.
    """
    
    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions
        # Quantum bits (qubits) represented as complex probability amplitudes
        self.qubits = [complex(random.gauss(0.707, 0.1), random.gauss(0.707, 0.1)) 
                       for _ in range(dimensions)]
        self._normalize_qubits()
        
        # Classical observations (collapsed states)
        self.observed_prompts = []
        self.fitness_history = []
        self.best_fitness = 0.0
        
    def _normalize_qubits(self):
        """Normalize quantum state to maintain probability constraint."""
        for i in range(len(self.qubits)):
            magnitude = abs(self.qubits[i])
            if magnitude > 0:
                self.qubits[i] = self.qubits[i] / magnitude * math.sqrt(0.5)
    
    def observe(self, prompt_generator) -> Prompt:
        """Collapse quantum superposition to classical prompt."""
        # Probabilistic observation based on quantum amplitudes
        binary_string = ""
        for qubit in self.qubits:
            # Probability of measuring |0⟩ vs |1⟩
            prob_0 = abs(qubit.real) ** 2
            bit = "0" if random.random() < prob_0 else "1"
            binary_string += bit
        
        # Convert binary pattern to prompt characteristics
        prompt = prompt_generator(binary_string)
        self.observed_prompts.append(prompt)
        return prompt
    
    def update_rotation(self, best_individual, rotation_angle: float):
        """Apply quantum rotation gates based on best individual."""
        for i, (my_qubit, best_qubit) in enumerate(zip(self.qubits, best_individual.qubits)):
            # Rotation direction based on fitness comparison
            if self.best_fitness < best_individual.best_fitness:
                # Rotate towards the better state
                rotation = cmath.exp(1j * rotation_angle)
                self.qubits[i] = my_qubit * rotation
        
        self._normalize_qubits()


class QuantumInspiredEvolution(EvolutionAlgorithm):
    """
    Quantum-Inspired Evolutionary Algorithm for prompt evolution.
    
    Combines quantum computing principles with evolutionary computation:
    - Superposition: Multiple prompt possibilities in quantum individuals  
    - Entanglement: Correlated evolution of related prompts
    - Quantum gates: Rotation and interference operations
    - Measurement: Collapse to classical prompts for evaluation
    """
    
    def __init__(self, config: QuantumConfig):
        super().__init__(config)
        self.config = config
        self.quantum_population = []
        self.classical_archive = []
        self.generation = 0
        
        # Initialize quantum population
        for _ in range(config.quantum_population_size):
            self.quantum_population.append(QuantumIndividual())
    
    def evolve_generation(self, population: PromptPopulation, fitness_fn) -> PromptPopulation:
        """Evolve one generation using quantum-inspired operations."""
        self.generation += 1
        
        # Step 1: Quantum observation (collapse to classical prompts)
        observed_prompts = []
        for q_individual in self.quantum_population:
            for _ in range(self.config.population_size // self.config.quantum_population_size):
                prompt = q_individual.observe(self._binary_to_prompt_generator)
                prompt.fitness_scores = {"fitness": fitness_fn(prompt)}
                observed_prompts.append(prompt)
        
        # Fill remaining slots if needed
        while len(observed_prompts) < self.config.population_size:
            q_ind = random.choice(self.quantum_population)
            prompt = q_ind.observe(self._binary_to_prompt_generator)
            prompt.fitness_scores = {"fitness": fitness_fn(prompt)}
            observed_prompts.append(prompt)
        
        # Step 2: Update quantum individuals based on fitness
        self._update_quantum_population(observed_prompts)
        
        # Step 3: Quantum operations (rotation, entanglement)
        self._apply_quantum_operations()
        
        # Step 4: Maintain classical archive of best solutions
        self._update_classical_archive(observed_prompts)
        
        # Create new population
        new_population = PromptPopulation(observed_prompts)
        new_population.generation = self.generation
        
        return new_population
    
    def _binary_to_prompt_generator(self, binary_string: str) -> Prompt:
        """Convert binary pattern to prompt characteristics."""
        # Decode binary string into prompt features
        length_bits = binary_string[:4]
        structure_bits = binary_string[4:8] 
        tone_bits = binary_string[8:12]
        content_bits = binary_string[12:]
        
        # Map to prompt components
        length = int(length_bits, 2) + 3  # 3-18 words
        
        structures = [
            "Please help me",
            "Can you explain",
            "I need assistance with", 
            "Could you describe",
            "Help me understand",
            "Please analyze",
            "Can you break down",
            "I would like to know"
        ]
        structure = structures[int(structure_bits, 2) % len(structures)]
        
        tones = ["", "clearly", "in detail", "step by step", "concisely", 
                "thoroughly", "simply", "precisely"]
        tone = tones[int(tone_bits, 2) % len(tones)]
        
        # Generate content based on remaining bits
        content_words = ["topic", "concept", "process", "method", "approach", 
                        "technique", "strategy", "solution", "answer", "explanation"]
        content = content_words[int(content_bits[:4], 2) % len(content_words)]
        
        # Assemble prompt
        prompt_parts = [structure, content]
        if tone:
            prompt_parts.append(tone)
        
        prompt_text = " ".join(prompt_parts)
        
        return Prompt(
            text=prompt_text,
            metadata={"quantum_generated": True, "binary_pattern": binary_string}
        )
    
    def _update_quantum_population(self, observed_prompts: List[Prompt]):
        """Update quantum individuals based on observed fitness."""
        # Group observations by quantum individual
        prompts_per_individual = len(observed_prompts) // len(self.quantum_population)
        
        for i, q_individual in enumerate(self.quantum_population):
            start_idx = i * prompts_per_individual
            end_idx = start_idx + prompts_per_individual
            individual_prompts = observed_prompts[start_idx:end_idx]
            
            if individual_prompts:
                best_prompt = max(individual_prompts, 
                                key=lambda p: p.fitness_scores.get("fitness", 0))
                q_individual.best_fitness = best_prompt.fitness_scores.get("fitness", 0)
    
    def _apply_quantum_operations(self):
        """Apply quantum gates and operations."""
        # Find best quantum individual
        best_q_individual = max(self.quantum_population, key=lambda q: q.best_fitness)
        
        # Apply rotation gates
        for q_individual in self.quantum_population:
            if q_individual != best_q_individual:
                q_individual.update_rotation(best_q_individual, self.config.rotation_angle)
        
        # Quantum crossover (entanglement)
        if random.random() < self.config.quantum_crossover_rate:
            self._quantum_crossover()
        
        # Quantum mutation (random rotation)
        if random.random() < self.config.quantum_mutation_rate:
            self._quantum_mutation()
    
    def _quantum_crossover(self):
        """Quantum crossover through entanglement."""
        if len(self.quantum_population) >= 2:
            parent1, parent2 = random.sample(self.quantum_population, 2)
            
            # Entangle qubits with controlled probability
            for i in range(min(len(parent1.qubits), len(parent2.qubits))):
                if random.random() < self.config.entanglement_strength:
                    # Quantum entanglement operation
                    alpha1, alpha2 = parent1.qubits[i], parent2.qubits[i]
                    
                    # Bell state creation (simplified)
                    new_alpha1 = (alpha1 + alpha2) / math.sqrt(2)
                    new_alpha2 = (alpha1 - alpha2) / math.sqrt(2)
                    
                    parent1.qubits[i] = new_alpha1
                    parent2.qubits[i] = new_alpha2
            
            parent1._normalize_qubits()
            parent2._normalize_qubits()
    
    def _quantum_mutation(self):
        """Apply quantum mutation through random rotations."""
        mutant = random.choice(self.quantum_population)
        
        for i in range(len(mutant.qubits)):
            if random.random() < 0.1:  # Mutation probability per qubit
                # Random rotation
                angle = random.uniform(-math.pi/4, math.pi/4)
                rotation = cmath.exp(1j * angle)
                mutant.qubits[i] *= rotation
        
        mutant._normalize_qubits()
    
    def _update_classical_archive(self, prompts: List[Prompt]):
        """Maintain archive of best classical solutions."""
        # Add new prompts to archive
        self.classical_archive.extend(prompts)
        
        # Sort by fitness and keep top solutions
        self.classical_archive.sort(key=lambda p: p.fitness_scores.get("fitness", 0), 
                                  reverse=True)
        
        # Limit archive size
        max_archive_size = 100
        if len(self.classical_archive) > max_archive_size:
            self.classical_archive = self.classical_archive[:max_archive_size]
    
    def get_quantum_state_analysis(self) -> Dict[str, Any]:
        """Analyze quantum population state for insights."""
        analysis = {
            "generation": self.generation,
            "quantum_population_size": len(self.quantum_population),
            "quantum_coherence": [],
            "entanglement_measures": [],
            "classical_archive_size": len(self.classical_archive)
        }
        
        for i, q_individual in enumerate(self.quantum_population):
            # Quantum coherence (measure of superposition)
            coherence = sum(abs(qubit.imag) for qubit in q_individual.qubits) / len(q_individual.qubits)
            analysis["quantum_coherence"].append(coherence)
            
            # Fitness statistics
            analysis[f"quantum_individual_{i}"] = {
                "best_fitness": q_individual.best_fitness,
                "observations": len(q_individual.observed_prompts)
            }
        
        return analysis
    
    def export_quantum_state(self, filename: str):
        """Export quantum state for analysis and visualization."""
        state_data = {
            "algorithm": "quantum_inspired_evolution",
            "generation": self.generation,
            "config": {
                "population_size": self.config.population_size,
                "quantum_population_size": self.config.quantum_population_size,
                "rotation_angle": self.config.rotation_angle,
                "entanglement_strength": self.config.entanglement_strength
            },
            "quantum_analysis": self.get_quantum_state_analysis(),
            "best_classical_solutions": [
                {
                    "text": prompt.text,
                    "fitness": prompt.fitness_scores.get("fitness", 0),
                    "metadata": prompt.metadata
                }
                for prompt in self.classical_archive[:10]
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        print(f"🔬 Quantum state exported to {filename}")


# Demonstration usage
if __name__ == "__main__":
    from ..population import PromptPopulation
    from ...evaluation.base import TestCase
    
    # Initialize quantum evolution
    config = QuantumConfig(
        population_size=20,
        quantum_population_size=4,
        max_generations=15,
        rotation_angle=0.02 * math.pi
    )
    
    qiea = QuantumInspiredEvolution(config)
    
    # Create initial population
    seeds = ["Help me understand", "Please explain", "Can you describe"]
    initial_population = PromptPopulation.from_seeds(seeds)
    
    # Simple fitness function
    def quantum_fitness(prompt):
        words = prompt.text.lower().split()
        score = len(words) * 0.1  # Length component
        score += sum(0.2 for word in ["please", "help", "explain"] if word in words)
        score += random.uniform(-0.1, 0.1)  # Exploration noise
        return max(0, min(1, score))
    
    # Run evolution
    print("🚀 Starting Quantum-Inspired Evolution")
    evolved_population = qiea.evolve_generation(initial_population, quantum_fitness)
    
    # Show results
    best_prompts = evolved_population.get_top_k(5)
    print("\n🏆 Best Quantum-Evolved Prompts:")
    for i, prompt in enumerate(best_prompts, 1):
        fitness = prompt.fitness_scores.get("fitness", 0)
        print(f"{i}. [{fitness:.3f}] {prompt.text}")
    
    # Export quantum analysis
    qiea.export_quantum_state("quantum_evolution_state.json")
    
    print(f"\n📊 Quantum Analysis:")
    analysis = qiea.get_quantum_state_analysis()
    print(f"Average Quantum Coherence: {sum(analysis['quantum_coherence'])/len(analysis['quantum_coherence']):.3f}")
    print(f"Classical Archive: {analysis['classical_archive_size']} solutions")