"""Basic operations on small, rectangular matrices."""

from collections.abc import Sequence
from typing import TypeAlias

from math101._validation import finite_number, finite_sum, rectangular_matrix

MatrixRows: TypeAlias = tuple[tuple[float, ...], ...]


def matrix_add(
    left: Sequence[Sequence[int | float]], right: Sequence[Sequence[int | float]]
) -> MatrixRows:
    """Add two matrices with the same shape."""
    left_rows = rectangular_matrix(left, name="left matrix")
    right_rows = rectangular_matrix(right, name="right matrix")
    left_shape = _shape(left_rows)
    right_shape = _shape(right_rows)
    if left_shape != right_shape:
        raise ValueError(
            f"matrix addition requires equal shapes; got {left_shape} and {right_shape}"
        )
    return tuple(
        tuple(
            finite_number(left_number + right_number, name=f"sum at [{row_index}][{column_index}]")
            for column_index, (left_number, right_number) in enumerate(zip(left_row, right_row))
        )
        for row_index, (left_row, right_row) in enumerate(zip(left_rows, right_rows))
    )


def matrix_multiply(
    left: Sequence[Sequence[int | float]], right: Sequence[Sequence[int | float]]
) -> MatrixRows:
    """Multiply matrices when left columns equal right rows."""
    left_rows = rectangular_matrix(left, name="left matrix")
    right_rows = rectangular_matrix(right, name="right matrix")
    left_shape = _shape(left_rows)
    right_shape = _shape(right_rows)
    if len(left_rows[0]) != len(right_rows):
        raise ValueError(
            "matrix multiplication requires matching inner dimensions; "
            f"got {left_shape} and {right_shape}"
        )
    right_columns = tuple(zip(*right_rows))
    return tuple(
        tuple(
            finite_sum(
                (
                    left_number * right_number
                    for left_number, right_number in zip(left_row, right_column)
                ),
                name=f"product sum at [{row_index}][{column_index}]",
            )
            for column_index, right_column in enumerate(right_columns)
        )
        for row_index, left_row in enumerate(left_rows)
    )


def transpose(matrix: Sequence[Sequence[int | float]]) -> MatrixRows:
    """Return rows as columns and columns as rows."""
    rows = rectangular_matrix(matrix, name="matrix")
    return tuple(tuple(column) for column in zip(*rows))


def identity_matrix(size: int) -> MatrixRows:
    """Return a square identity matrix of positive size."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(f"size must be an integer, got {type(size).__name__}")
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    return tuple(
        tuple(1.0 if row_index == column_index else 0.0 for column_index in range(size))
        for row_index in range(size)
    )


def _shape(rows: MatrixRows) -> tuple[int, int]:
    return len(rows), len(rows[0])
