"""Known answers and approximation behavior for numerical calculus."""

import math
import unittest

from math101.calculus import (
    central_difference,
    forward_difference,
    trapezoidal_integral,
)


def square(number: float) -> float:
    return number * number


class CalculusTests(unittest.TestCase):
    def test_differences_approximate_known_derivative(self) -> None:
        self.assertAlmostEqual(forward_difference(square, 2, step=0.1), 4.1)
        self.assertAlmostEqual(central_difference(square, 2, step=0.1), 4.0)
        self.assertAlmostEqual(central_difference(math.sin, 0, step=1e-5), 1.0, places=8)

    def test_smaller_step_improves_forward_difference_here(self) -> None:
        coarse_error = abs(forward_difference(square, 2, step=0.1) - 4.0)
        fine_error = abs(forward_difference(square, 2, step=0.01) - 4.0)
        self.assertLess(fine_error, coarse_error)

    def test_trapezoids_approach_known_integral(self) -> None:
        coarse = trapezoidal_integral(square, 0, 1, intervals=10)
        fine = trapezoidal_integral(square, 0, 1, intervals=100)
        self.assertLess(abs(fine - 1 / 3), abs(coarse - 1 / 3))
        self.assertAlmostEqual(trapezoidal_integral(lambda x: 2 * x, 0, 3, intervals=5), 9.0)

    def test_reversed_and_equal_bounds(self) -> None:
        forward = trapezoidal_integral(square, 0, 1, intervals=100)
        backward = trapezoidal_integral(square, 1, 0, intervals=100)
        self.assertAlmostEqual(backward, -forward)
        self.assertEqual(trapezoidal_integral(square, 2, 2, intervals=1), 0.0)

    def test_invalid_steps_and_intervals(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            forward_difference(math.sin, 0, step=0)
        with self.assertRaisesRegex(ValueError, "too small"):
            central_difference(math.sin, 1e20, step=1e-20)
        with self.assertRaisesRegex(ValueError, "positive"):
            trapezoidal_integral(math.sin, 0, 1, intervals=0)
        with self.assertRaisesRegex(TypeError, "integer"):
            trapezoidal_integral(math.sin, 0, 1, intervals=True)
        with self.assertRaisesRegex(ValueError, "interval width is too small"):
            trapezoidal_integral(math.sin, 0, 5e-324, intervals=10)

    def test_function_and_output_must_be_valid(self) -> None:
        with self.assertRaisesRegex(TypeError, "callable"):
            forward_difference(3, 0, step=0.1)
        with self.assertRaisesRegex(ValueError, "function output"):
            central_difference(lambda x: math.nan, 0, step=0.1)
        with self.assertRaisesRegex(TypeError, "function output"):
            trapezoidal_integral(lambda x: True, 0, 1, intervals=2)


if __name__ == "__main__":
    unittest.main()
