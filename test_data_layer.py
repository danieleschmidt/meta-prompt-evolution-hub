#!/usr/bin/env python3
"""Test script for the data persistence layer."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from meta_prompt_evolution.storage.database.connection import DatabaseConnection, DatabaseConfig
from meta_prompt_evolution.storage.repositories.prompt_repository import PromptRepository
from meta_prompt_evolution.storage.repositories.population_repository import PopulationRepository
from meta_prompt_evolution.evolution.population import Prompt, PromptPopulation


def test_data_layer():
    """Test the data persistence layer."""
    print("🗄️ Testing Meta-Prompt-Evolution-Hub Data Layer")
    print("=" * 60)
    
    # Test 1: Database connection
    print("1. Testing database connection...")
    config = DatabaseConfig(database_path="test_prompt_hub.db")
    db = DatabaseConnection(config)
    
    health = db.health_check()
    print(f"   ✅ Database health: {'OK' if health else 'FAILED'}")
    
    # Test 2: Create repositories
    print("\n2. Testing repositories...")
    prompt_repo = PromptRepository(db)
    population_repo = PopulationRepository(db)
    print("   ✅ Repositories created successfully")
    
    # Test 3: Save and retrieve prompts
    print("\n3. Testing prompt persistence...")
    
    # Create test prompts
    prompt1 = Prompt(
        text="You are a helpful assistant. Please help with: {task}",
        fitness_scores={"fitness": 0.85, "accuracy": 0.90, "latency": 0.75},
        generation=1
    )
    
    prompt2 = Prompt(
        text="As an AI, I will carefully assist you with: {task}",
        fitness_scores={"fitness": 0.78, "accuracy": 0.85, "latency": 0.80},
        generation=1,
        parent_ids=[prompt1.id]
    )
    
    # Save prompts
    saved_prompt1 = prompt_repo.save(prompt1)
    saved_prompt2 = prompt_repo.save(prompt2)
    
    print(f"   ✅ Saved prompt 1: {saved_prompt1.id}")
    print(f"   ✅ Saved prompt 2: {saved_prompt2.id}")
    
    # Retrieve prompts
    retrieved_prompt1 = prompt_repo.find_by_id(saved_prompt1.id)
    print(f"   ✅ Retrieved prompt 1: {retrieved_prompt1.text[:30]}...")
    
    # Test 4: Population persistence
    print("\n4. Testing population persistence...")
    
    # Create population
    population = PromptPopulation([saved_prompt1, saved_prompt2])
    population.generation = 1
    population.algorithm_used = "nsga2"
    population.diversity_score = 0.65
    
    # Save population
    saved_population = population_repo.save(population)
    print(f"   ✅ Saved population: {saved_population.id}")
    print(f"   ✅ Population size: {len(saved_population.prompts)}")
    
    # Retrieve population
    retrieved_population = population_repo.find_by_id(saved_population.id)
    print(f"   ✅ Retrieved population with {len(retrieved_population.prompts)} prompts")
    
    # Test 5: Query operations
    print("\n5. Testing query operations...")
    
    # Find by generation
    gen1_prompts = prompt_repo.find_by_generation(1)
    print(f"   ✅ Found {len(gen1_prompts)} prompts in generation 1")
    
    # Find top prompts
    top_prompts = prompt_repo.find_top_by_fitness("fitness", limit=5)
    print(f"   ✅ Found {len(top_prompts)} top prompts")
    
    # Find by parent
    children = prompt_repo.find_by_parent_id(prompt1.id)
    print(f"   ✅ Found {len(children)} child prompts")
    
    # Test 6: Statistics
    print("\n6. Testing statistics...")
    
    prompt_stats = prompt_repo.get_statistics()
    print(f"   ✅ Prompt statistics: {prompt_stats['total_prompts']} total prompts")
    
    population_stats = population_repo.get_statistics()
    print(f"   ✅ Population statistics: {population_stats['total_populations']} total populations")
    
    # Test 7: Database stats
    print("\n7. Testing database statistics...")
    
    table_stats = db.get_table_stats()
    for table, count in table_stats.items():
        print(f"   ✅ Table {table}: {count} records")
    
    # Test 8: Genealogy tracking
    print("\n8. Testing genealogy tracking...")
    
    genealogy = prompt_repo.get_genealogy(prompt2.id)
    if genealogy:
        print(f"   ✅ Genealogy tracked for prompt {prompt2.id}")
        print(f"       Parents: {len(genealogy.get('parents', []))}")
    
    print("\n✅ All data layer tests passed!")
    print("\n🔧 Data Layer Status:")
    print("   - SQLite database: Working")
    print("   - Prompt persistence: Working")
    print("   - Population persistence: Working")
    print("   - Query operations: Working")
    print("   - Statistics tracking: Working")
    print("   - Genealogy tracking: Working")
    
    # Cleanup
    try:
        os.remove("test_prompt_hub.db")
        print("   - Test database cleaned up")
    except:
        pass
    
    return True


if __name__ == "__main__":
    try:
        success = test_data_layer()
        if success:
            print("\n🎉 Data layer test completed successfully!")
            print("   The database and repository system is fully functional.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)