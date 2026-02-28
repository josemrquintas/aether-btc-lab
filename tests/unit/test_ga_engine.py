"""Tests for GA engine — Phase 3 GA Optimization Gate."""

import pytest

from aether_btc.ga.chromosome import Chromosome
from aether_btc.ga.engine import GAConfig, GAEngine
from aether_btc.ga.fitness import FitnessResult


def simple_fitness(chromosome: Chromosome) -> FitnessResult:
    """Simple fitness function for testing: sum of all gene values (normalized)."""
    total = sum(float(v) for v in chromosome.genes.values())
    return FitnessResult(fitness=total / 100, total_trades=10)


class TestGAEvolution:
    """Test GA evolution mechanics."""

    def test_fitness_improves_over_generations(self):
        """After 10 generations, best fitness > initial best fitness."""
        config = GAConfig(
            population_size=20, generations=10,
            elite_size=2, random_seed=42,
        )
        engine = GAEngine(config, simple_fitness)
        result = engine.run()

        # First generation should be worse than final
        assert result.history[-1].best_fitness >= result.history[0].best_fitness

    def test_elite_preserved(self):
        """Top N chromosomes survive to next generation unchanged."""
        config = GAConfig(
            population_size=10, generations=2,
            elite_size=2, random_seed=42,
        )
        engine = GAEngine(config, simple_fitness)

        engine._initialize_population()
        engine._evaluate_population()

        # Get top 2 chromosomes
        top_indices = sorted(
            range(len(engine.fitness_scores)),
            key=lambda i: engine.fitness_scores[i],
            reverse=True,
        )[:2]
        elite_genes = [engine.population[i].genes.copy() for i in top_indices]

        engine._evolve()

        # Elites should be in new population
        for elite in elite_genes:
            found = any(c.genes == elite for c in engine.population)
            assert found, "Elite chromosome not found in next generation"

    def test_early_stopping(self):
        """If fitness stalls for N generations, GA stops early."""
        def constant_fitness(c: Chromosome) -> FitnessResult:
            return FitnessResult(fitness=1.0)

        config = GAConfig(
            population_size=10, generations=100,
            early_stop_generations=5, random_seed=42,
        )
        engine = GAEngine(config, constant_fitness)
        result = engine.run()

        # Should stop well before 100 generations
        assert result.generations_run < 100
        assert result.generations_run <= 10  # Should stop around 5-6

    def test_population_size_maintained(self):
        """Population stays at configured size each generation."""
        config = GAConfig(
            population_size=20, generations=5,
            elite_size=3, random_seed=42,
        )
        engine = GAEngine(config, simple_fitness)

        engine._initialize_population()
        assert len(engine.population) == 20

        engine._evaluate_population()
        engine._evolve()
        assert len(engine.population) == 20
