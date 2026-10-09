"""Probabilities for finite, equally likely events and Bernoulli trials."""

from collections.abc import Hashable, Iterable
import math
import random

from math101._validation import finite_number


def event_probability(
    sample_space: Iterable[Hashable], favorable_outcomes: Iterable[Hashable]
) -> float:
    """Return the probability of an event in a uniform finite sample space.

    Outcomes must be unique and hashable. The sample space must be nonempty,
    and every favorable outcome must belong to it. An empty event has
    probability zero. This function treats each outcome as equally likely.
    """
    possible = _unique_outcomes(sample_space, name="sample space")
    if not possible:
        raise ValueError("sample space must contain at least one outcome")

    favorable = _unique_outcomes(favorable_outcomes, name="favorable outcomes")
    outside = favorable - possible
    if outside:
        raise ValueError(f"favorable outcomes must belong to the sample space: {outside!r}")
    return len(favorable) / len(possible)


def bernoulli_pmf(success: bool, success_probability: int | float) -> float:
    """Return the probability of one success or failure.

    ``success`` must be a bool and ``success_probability`` must be in [0, 1].
    """
    if not isinstance(success, bool):
        raise TypeError(f"success must be a bool, got {type(success).__name__}")
    probability = _probability(success_probability)
    return probability if success else 1.0 - probability


def binomial_pmf(
    successes: int, trials: int, success_probability: int | float
) -> float:
    """Return the probability of exactly ``successes`` in ``trials`` trials.

    Trials must be a nonnegative integer. An integer success count outside
    [0, trials] is impossible and returns zero. The trials are assumed
    independent with the same success probability.
    """
    _integer(trials, name="trials")
    _integer(successes, name="successes")
    if trials < 0:
        raise ValueError(f"trials must be nonnegative, got {trials}")
    probability = _probability(success_probability)
    if successes < 0 or successes > trials:
        return 0.0
    if probability == 0.0:
        return 1.0 if successes == 0 else 0.0
    if probability == 1.0:
        return 1.0 if successes == trials else 0.0

    # Log space avoids converting a huge binomial coefficient to float.
    try:
        log_probability = (
            math.lgamma(trials + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(trials - successes + 1)
            + successes * math.log(probability)
            + (trials - successes) * math.log1p(-probability)
        )
        probability_mass = math.exp(log_probability)
    except (OverflowError, ValueError) as error:
        raise ValueError("binomial parameters exceed the supported float range") from error
    if not 0.0 <= probability_mass <= 1.0:
        raise ValueError("binomial probability could not be represented accurately")
    return probability_mass


def sample_bernoulli(success_probability: int | float, *, rng: random.Random) -> bool:
    """Draw one Bernoulli outcome using the caller's random generator.

    Passing an explicit generator makes a sequence reproducible with a seed
    and leaves Python's module-level random state untouched.
    """
    probability = _probability(success_probability)
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be a random.Random instance")
    return rng.random() < probability


def _probability(number: int | float) -> float:
    probability = finite_number(number, name="success probability")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"success probability must be between 0 and 1, got {probability}")
    return probability


def _integer(number: int, *, name: str) -> None:
    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError(f"{name} must be an integer, got {type(number).__name__}")


def _unique_outcomes(outcomes: Iterable[Hashable], *, name: str) -> set[Hashable]:
    if isinstance(outcomes, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of outcomes, not text")
    try:
        outcome_iter = iter(outcomes)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of outcomes") from error

    unique: set[Hashable] = set()
    for index, outcome in enumerate(outcome_iter):
        try:
            if outcome in unique:
                raise ValueError(f"{name} contains duplicate outcome at index {index}: {outcome!r}")
            unique.add(outcome)
        except TypeError as error:
            raise TypeError(f"{name} outcome at index {index} must be hashable") from error
    return unique
