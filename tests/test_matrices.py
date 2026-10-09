"""Behavioral tests for matrix operations."""

import unittest

from math101.matrices import identity_matrix, matrix_add, matrix_multiply, transpose


class MatrixTests(unittest.TestCase):
    def test_addition_and_transpose(self) -> None:
        self.assertEqual(matrix_add([[1, 2]], [[3, 4]]), ((4.0, 6.0),))
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), ((1.0, 4.0), (2.0, 5.0), (3.0, 6.0)))

    def test_multiplication_uses_rows_and_columns(self) -> None:
        self.assertEqual(
            matrix_multiply([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]),
            ((58.0, 64.0), (139.0, 154.0)),
        )

    def test_identity_law(self) -> None:
        matrix = [[2, 3], [5, 7]]
        self.assertEqual(matrix_multiply(matrix, identity_matrix(2)), ((2.0, 3.0), (5.0, 7.0)))
        self.assertEqual(matrix_multiply(identity_matrix(2), matrix), ((2.0, 3.0), (5.0, 7.0)))

    def test_output_does_not_share_mutable_input(self) -> None:
        matrix = [[1, 2]]
        transposed = transpose(matrix)
        matrix[0][0] = 99
        self.assertEqual(transposed, ((1.0,), (2.0,)))

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one row"):
            transpose([])
        with self.assertRaisesRegex(ValueError, "at least one column"):
            transpose([[]])
        with self.assertRaisesRegex(ValueError, "row 1 has 1 columns; expected 2"):
            transpose([[1, 2], [3]])
        with self.assertRaisesRegex(ValueError, "equal shapes"):
            matrix_add([[1]], [[1, 2]])
        with self.assertRaisesRegex(ValueError, "inner dimensions"):
            matrix_multiply([[1, 2]], [[1]])

    def test_invalid_entries_and_identity_size_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"matrix\[0\]\[1\]"):
            transpose([[1, True]])
        with self.assertRaisesRegex(ValueError, "must be finite"):
            transpose([[float("inf")]])
        with self.assertRaisesRegex(ValueError, "positive"):
            identity_matrix(0)
        with self.assertRaises(TypeError):
            identity_matrix(True)
        with self.assertRaisesRegex(ValueError, "sum at"):
            matrix_add([[1e308]], [[1e308]])


if __name__ == "__main__":
    unittest.main()
