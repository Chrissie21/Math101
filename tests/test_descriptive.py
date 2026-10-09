"""Behavioral tests for descriptive statistics."""

import math
import statistics
import unittest

from math101.descriptive import (
    mean,
    population_standard_deviation,
    population_variance,
    sample_standard_deviation,
    sample_variance,
)


class DescriptiveTests(unittest.TestCase):
    def test_mean_accepts_one_pass_iterable(self) -> None:
        observations = (number for number in [2, 4, 6])
        self.assertEqual(mean(observations), 4.0)

    def test_variance_agrees_with_standard_library(self) -> None:
        observations = [2, 4, 6, 8]
        self.assertTrue(math.isclose(population_variance(observations), statistics.pvariance(observations)))
        self.assertTrue(math.isclose(sample_variance(observations), statistics.variance(observations)))
        self.assertTrue(math.isclose(population_standard_deviation(observations), statistics.pstdev(observations)))
        self.assertTrue(math.isclose(sample_standard_deviation(observations), statistics.stdev(observations)))

    def test_variance_is_invariant_under_translation(self) -> None:
        observations = [3.0, 5.0, 7.0]
        shifted = [number + 10_000 for number in observations]
        self.assertAlmostEqual(sample_variance(observations), sample_variance(shifted))

    def test_constant_observations_have_zero_variance(self) -> None:
        self.assertEqual(population_variance([5]), 0.0)
        self.assertEqual(sample_variance([5, 5]), 0.0)

    def test_empty_and_short_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            mean([])
        with self.assertRaisesRegex(ValueError, "at least one"):
            population_variance([])
        with self.assertRaisesRegex(ValueError, "at least two"):
            sample_variance([1])

    def test_invalid_observations_are_rejected_with_position(self) -> None:
        with self.assertRaisesRegex(TypeError, "index 1"):
            mean([1, "2"])
        with self.assertRaisesRegex(TypeError, "index 0"):
            mean([True])
        with self.assertRaisesRegex(ValueError, "index 1"):
            mean([1, math.nan])

    def test_unrepresentable_total_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation total exceeds"):
            mean([1e308, 1e308])
        with self.assertRaisesRegex(ValueError, "variance exceeds"):
            population_variance([-1e308, 1e308])


if __name__ == "__main__":
    unittest.main()
