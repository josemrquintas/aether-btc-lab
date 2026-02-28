"""GA Evolution Engine.

Reused from Aether v3 with minimal changes.
Multi-objective fitness (Sharpe + Calmar), parallel evaluation.
"""

import json
import multiprocessing
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from aether_btc.ga.chromosome import Chromosome
from aether_btc.ga.fitness import FitnessResult

log = structlog.get_logger()

_PARALLEL_FITNESS_FN = None


def _parallel_eval(chromosome: Chromosome) -> FitnessResult:
    """Module-level worker for fork-based parallel evaluation."""
    return _PARALLEL_FITNESS_FN(chromosome)


@dataclass
class GAConfig:
    """GA configuration."""

    population_size: int = 80
    generations: int = 50
    elite_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    early_stop_generations: int = 10
    parallel_workers: int = 1
    random_seed: int | None = None
    sharpe_weight: float = 0.6
    calmar_weight: float = 0.4


@dataclass
class GenerationStats:
    """Statistics for a single generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    median_fitness: float
    worst_fitness: float
    fitness_std: float
    best_chromosome: Chromosome
    best_result: FitnessResult | None = None

    def to_dict(self) -> dict:
        d = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "median_fitness": self.median_fitness,
            "worst_fitness": self.worst_fitness,
            "fitness_std": self.fitness_std,
            "best_genes": self.best_chromosome.genes,
        }
        if self.best_result:
            d["best_result"] = self.best_result.to_dict()
        return d


@dataclass
class EvolutionResult:
    """Results from GA evolution."""

    best_chromosome: Chromosome
    best_fitness: float
    best_result: FitnessResult
    generations_run: int
    history: list[GenerationStats]

    def to_dict(self) -> dict:
        return {
            "best_genes": self.best_chromosome.genes,
            "best_fitness": self.best_fitness,
            "best_result": self.best_result.to_dict(),
            "generations_run": self.generations_run,
            "history": [h.to_dict() for h in self.history],
        }


class GAEngine:
    """Genetic Algorithm Evolution Engine."""

    def __init__(
        self,
        config: GAConfig,
        fitness_fn: Callable[[Chromosome], FitnessResult],
        progress_callback: Callable[[int, GenerationStats], None] | None = None,
    ) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self.progress_callback = progress_callback

        self.population: list[Chromosome] = []
        self.fitness_scores: list[float] = []
        self.fitness_results: list[FitnessResult] = []
        self.best_chromosome: Chromosome | None = None
        self.best_fitness: float = float("-inf")
        self.best_result: FitnessResult | None = None
        self.history: list[GenerationStats] = []

        if config.random_seed is not None:
            random.seed(config.random_seed)

    def run(self) -> EvolutionResult:
        """Run the GA evolution."""
        log.info(
            "ga_starting",
            population=self.config.population_size,
            generations=self.config.generations,
        )

        self._initialize_population()
        stall = 0

        for gen in range(self.config.generations):
            self._evaluate_population()
            stats = self._compute_stats(gen)

            if stats.best_fitness > self.best_fitness:
                best_idx = self.fitness_scores.index(max(self.fitness_scores))
                self.best_chromosome = self.population[best_idx].copy()
                self.best_fitness = stats.best_fitness
                self.best_result = self.fitness_results[best_idx]
                stall = 0
            else:
                stall += 1

            self.history.append(stats)

            if self.progress_callback:
                self.progress_callback(gen, stats)

            log.info(
                "generation_complete",
                gen=gen,
                best=f"{stats.best_fitness:.3f}",
                avg=f"{stats.avg_fitness:.3f}",
                stall=stall,
            )

            if stall >= self.config.early_stop_generations:
                log.info("early_stopping", generation=gen)
                break

            self._evolve()

        return EvolutionResult(
            best_chromosome=self.best_chromosome or Chromosome.from_defaults(),
            best_fitness=self.best_fitness,
            best_result=self.best_result or FitnessResult(fitness=0),
            generations_run=len(self.history),
            history=self.history,
        )

    def _initialize_population(self) -> None:
        self.population = [Chromosome.random() for _ in range(self.config.population_size)]
        self.population[0] = Chromosome.from_defaults()

    def _evaluate_population(self) -> None:
        workers = self.config.parallel_workers
        if workers > 1:
            global _PARALLEL_FITNESS_FN
            _PARALLEL_FITNESS_FN = self.fitness_fn
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=workers) as pool:
                self.fitness_results = pool.map(_parallel_eval, self.population)
        else:
            self.fitness_results = [self.fitness_fn(c) for c in self.population]
        self.fitness_scores = [r.fitness for r in self.fitness_results]

    def _compute_stats(self, gen: int) -> GenerationStats:
        scores = self.fitness_scores
        best_idx = scores.index(max(scores))
        return GenerationStats(
            generation=gen,
            best_fitness=max(scores),
            avg_fitness=float(np.mean(scores)),
            median_fitness=float(np.median(scores)),
            worst_fitness=min(scores),
            fitness_std=float(np.std(scores)),
            best_chromosome=self.population[best_idx].copy(),
            best_result=self.fitness_results[best_idx],
        )

    def _evolve(self) -> None:
        """Create next generation via elitism + tournament selection + crossover + mutation."""
        elite_indices = sorted(
            range(len(self.fitness_scores)),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )[: self.config.elite_size]
        elite = [self.population[i].copy() for i in elite_indices]

        children: list[Chromosome] = []
        while len(children) < self.config.population_size - len(elite):
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            if random.random() < self.config.crossover_rate:
                c1, c2 = p1.crossover(p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            children.append(c1.mutate(self.config.mutation_rate))
            if len(children) < self.config.population_size - len(elite):
                children.append(c2.mutate(self.config.mutation_rate))

        self.population = elite + children

    def _tournament_select(self, size: int = 3) -> Chromosome:
        indices = random.sample(range(len(self.population)), min(size, len(self.population)))
        best = max(indices, key=lambda i: self.fitness_scores[i])
        return self.population[best]

    def save_results(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        result = EvolutionResult(
            best_chromosome=self.best_chromosome or Chromosome.from_defaults(),
            best_fitness=self.best_fitness,
            best_result=self.best_result or FitnessResult(fitness=0),
            generations_run=len(self.history),
            history=self.history,
        )
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
