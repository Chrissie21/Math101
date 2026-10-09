"""Descriptive statistics for finite numeric observations."""

import math
from collections.abc import Iterable

from math101._validation import finite_observations, finite_sum


def mean(observations: Iterable[int | float]) -> float:
    """Return the arithmetic mean of one or more finite observations.

    Raise ValueError for empty input or a non-finite observation, and TypeError
    for a non-numeric observation. A one-pass iterable is accepted.
    """
    numbers = finite_observations(observations)
    if not numbers:
        raise ValueError("mean requires at least one observation")
    return finite_sum(numbers, name="observation total") / len(numbers)


def population_variance(observations: Iterable[int | float]) -> float:
    """Return variance divided by the population size.

    Require at least one finite observation. Use a running mean to avoid
    subtracting two large, nearly equal squared totals.
    """
    numbers = finite_observations(observations)
    if not numbers:
        raise ValueError("population variance requires at least one observation")
    return _variance(numbers, sample=False)


def sample_variance(observations: Iterable[int | float]) -> float:
    """Return variance divided by sample size minus one.

    Require at least two finite observations.
    """
    numbers = finite_observations(observations)
    if len(numbers) < 2:
        raise ValueError("sample variance requires at least two observations")
    return _variance(numbers, sample=True)


def population_standard_deviation(observations: Iterable[int | float]) -> float:
    """Return the square root of population variance."""
    return math.sqrt(population_variance(observations))


def sample_standard_deviation(observations: Iterable[int | float]) -> float:
    """Return the square root of sample variance."""
    return math.sqrt(sample_variance(observations))


def _variance(numbers: tuple[float, ...], *, sample: bool) -> float:
    running_mean = 0.0
    squared_deviation_total = 0.0
    for count, number in enumerate(numbers, start=1):
        deviation = number - running_mean
        running_mean += deviation / count
        squared_deviation_total += deviation * (number - running_mean)
    if not math.isfinite(squared_deviation_total):
        raise ValueError("variance exceeds the finite float range")
    divisor = len(numbers) - 1 if sample else len(numbers)
    return max(0.0, squared_deviation_total / divisor)
