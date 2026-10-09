"""Contracts and mathematical properties of probability functions."""

import math
import random
import unittest

from math101.probability import (
    bernoulli_pmf,
    binomial_pmf,
    event_probability,
    sample_bernoulli,
)


class ProbabilityTests(unittest.TestCase):
    def test_uniform_finite_event_probability(self) -> None:
        self.assertEqual(event_probability(range(6), [0, 2, 4]), 0.5)
        self.assertEqual(event_probability(["heads", "tails"], []), 0.0)
        self.assertEqual(event_probability(["heads", "tails"], ["heads", "tails"]), 1.0)

    def test_event_input_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            event_probability([], [])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            event_probability([1, 1], [1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            event_probability([1, 2], [1, 1])
        with self.assertRaisesRegex(ValueError, "belong"):
            event_probability([1, 2], [3])
        with self.assertRaisesRegex(TypeError, "hashable"):
            event_probability([[1]], [])

    def test_bernoulli_probabilities_are_complements(self) -> None:
        self.assertEqual(bernoulli_pmf(True, 0.25), 0.25)
        self.assertEqual(bernoulli_pmf(False, 0.25), 0.75)
        self.assertEqual(bernoulli_pmf(True, 1), 1.0)
        with self.assertRaisesRegex(TypeError, "bool"):
            bernoulli_pmf(1, 0.25)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            bernoulli_pmf(True, 1.1)
        with self.assertRaisesRegex(ValueError, "finite"):
            bernoulli_pmf(True, math.nan)

    def test_binomial_known_values_and_support(self) -> None:
        self.assertAlmostEqual(binomial_pmf(2, 3, 0.5), 0.375)
        self.assertEqual(binomial_pmf(-1, 3, 0.5), 0.0)
        self.assertEqual(binomial_pmf(4, 3, 0.5), 0.0)
        self.assertEqual(binomial_pmf(0, 0, 0.4), 1.0)
        self.assertEqual(binomial_pmf(0, 3, 0), 1.0)
        self.assertEqual(binomial_pmf(3, 3, 1), 1.0)

    def test_binomial_distribution_normalizes(self) -> None:
        for probability in (0.1, 0.5, 0.9):
            total = math.fsum(binomial_pmf(k, 12, probability) for k in range(13))
            self.assertTrue(math.isclose(total, 1.0, rel_tol=1e-12))
        large_probability = binomial_pmf(5_000, 10_000, 0.5)
        self.assertTrue(0.0 < large_probability < 1.0)

    def test_binomial_input_contract(self) -> None:
        with self.assertRaisesRegex(TypeError, "trials"):
            binomial_pmf(0, True, 0.5)
        with self.assertRaisesRegex(TypeError, "successes"):
            binomial_pmf(1.5, 3, 0.5)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            binomial_pmf(0, -1, 0.5)

    def test_sampling_is_reproducible_without_global_state_change(self) -> None:
        global_state = random.getstate()
        first = random.Random(42)
        second = random.Random(42)
        first_draws = [sample_bernoulli(0.3, rng=first) for _ in range(20)]
        second_draws = [sample_bernoulli(0.3, rng=second) for _ in range(20)]
        self.assertEqual(first_draws, second_draws)
        self.assertEqual(random.getstate(), global_state)
        self.assertFalse(sample_bernoulli(0, rng=first))
        self.assertTrue(sample_bernoulli(1, rng=first))
        with self.assertRaisesRegex(TypeError, "random.Random"):
            sample_bernoulli(0.3, rng=None)


if __name__ == "__main__":
    unittest.main()
