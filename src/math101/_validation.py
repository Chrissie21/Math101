"""Shared input checks for public numerical operations."""

from collections.abc import Iterable, Sequence
import math


def finite_number(number: int | float, *, name: str) -> float:
    """Convert an int or float to a finite float."""
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise TypeError(f"{name} must be an int or float, got {type(number).__name__}")

    try:
        converted = float(number)
    except OverflowError as error:
        raise ValueError(f"{name} is too large to represent as a float") from error

    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {number!r}")
    return converted


def finite_sum(numbers: Iterable[float], *, name: str) -> float:
    """Sum finite floats and explain an unrepresentable result."""
    try:
        total = math.fsum(numbers)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} exceeds the finite float range") from error
    return finite_number(total, name=name)


def finite_observations(observations: Iterable[int | float]) -> tuple[float, ...]:
    """Materialize and validate observations from a one-pass iterable."""
    return tuple(
        finite_number(observation, name=f"observation at index {index}")
        for index, observation in enumerate(observations)
    )


def rectangular_matrix(
    matrix: Sequence[Sequence[int | float]], *, name: str
) -> tuple[tuple[float, ...], ...]:
    """Copy a nonempty rectangular matrix into immutable rows."""
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of rows")
    if not matrix:
        raise ValueError(f"{name} must have at least one row")

    rows: list[tuple[float, ...]] = []
    column_count: int | None = None
    for row_index, row in enumerate(matrix):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name} row {row_index} must be a sequence")
        if not row:
            raise ValueError(f"{name} row {row_index} must have at least one column")
        if column_count is not None and len(row) != column_count:
            raise ValueError(
                f"{name} row {row_index} has {len(row)} columns; expected {column_count}"
            )
        column_count = len(row)
        rows.append(
            tuple(
                finite_number(number, name=f"{name}[{row_index}][{column_index}]")
                for column_index, number in enumerate(row)
            )
        )
    return tuple(rows)
