"""Genetic Algorithm for selecting the hybrid confidence threshold."""

import random
from typing import Sequence

import config


def _fitness(threshold: float, confidences: Sequence[float]) -> float:
    """Fitness = 0.7 * average retained confidence + 0.3 * retention proportion."""
    if not confidences:
        return 0.0

    retained = [c for c in confidences if c >= threshold]
    if not retained:
        return 0.0

    avg_confidence = sum(retained) / len(retained)
    retention = len(retained) / len(confidences)
    return (
        config.CONFIDENCE_WEIGHT * avg_confidence
        + config.RETENTION_WEIGHT * retention
    )


def optimize_threshold(confidences: Sequence[float]) -> tuple[float, float, list[float]]:
    """Search for a confidence threshold using a small real-valued GA.

    Returns:
        best_threshold, best_fitness, fitness_history
    """
    values = [float(c) for c in confidences if c is not None]
    if not values:
        return config.LOW_CONF, 0.0, []

    rng = random.Random(config.RANDOM_SEED)

    def random_threshold() -> float:
        return rng.uniform(config.THRESHOLD_MIN, config.THRESHOLD_MAX)

    population = [random_threshold() for _ in range(config.POP_SIZE)]
    history: list[float] = []
    best_threshold = population[0]
    best_fitness = _fitness(best_threshold, values)

    for _ in range(config.GENERATIONS):
        scored = sorted(
            ((candidate, _fitness(candidate, values)) for candidate in population),
            key=lambda item: item[1],
            reverse=True,
        )

        if scored[0][1] > best_fitness:
            best_threshold, best_fitness = scored[0]
        history.append(best_fitness)

        elites = [candidate for candidate, _ in scored[: config.ELITISM]]
        next_population = elites.copy()

        while len(next_population) < config.POP_SIZE:
            parent_a = _tournament_select(scored, rng)
            parent_b = _tournament_select(scored, rng)
            child = (parent_a + parent_b) / 2.0

            if rng.random() < config.MUTATION:
                # Small mutation relative to the search interval.
                span = config.THRESHOLD_MAX - config.THRESHOLD_MIN
                child += rng.gauss(0.0, span * 0.10)

            child = max(config.THRESHOLD_MIN, min(config.THRESHOLD_MAX, child))
            next_population.append(child)

        population = next_population

    return float(best_threshold), float(best_fitness), history


def _tournament_select(scored_population, rng: random.Random) -> float:
    candidates = rng.sample(
        scored_population,
        k=min(config.TOURNAMENT_SIZE, len(scored_population)),
    )
    return max(candidates, key=lambda item: item[1])[0]
