"""Elementary numerical differentiation and integration."""

from collections.abc import Callable, Iterator

from math101._validation import finite_number, finite_sum

RealFunction = Callable[[float], int | float]


def forward_difference(function: RealFunction, at: int | float, *, step: int | float) -> float:
    """Approximate a derivative using values at ``at`` and ``at + step``.

    ``step`` must be finite, positive, and large enough to change ``at`` in
    floating-point arithmetic. Smaller steps do not always improve accuracy.
    """
    _require_function(function)
    point = finite_number(at, name="at")
    interval = _positive_step(step)
    next_point = finite_number(point + interval, name="at + step")
    if next_point == point:
        raise ValueError("step is too small to change at in floating-point arithmetic")

    difference = finite_number(
        _evaluate(function, next_point) - _evaluate(function, point),
        name="function difference",
    )
    return finite_number(difference / interval, name="forward difference")


def central_difference(function: RealFunction, at: int | float, *, step: int | float) -> float:
    """Approximate a derivative using values on either side of ``at``.

    ``step`` must be finite, positive, and large enough to change ``at`` in
    floating-point arithmetic. This is an approximation, not a symbolic
    derivative.
    """
    _require_function(function)
    point = finite_number(at, name="at")
    interval = _positive_step(step)
    left_point = finite_number(point - interval, name="at - step")
    right_point = finite_number(point + interval, name="at + step")
    if left_point == point or right_point == point:
        raise ValueError("step is too small to change at in floating-point arithmetic")

    difference = finite_number(
        _evaluate(function, right_point) - _evaluate(function, left_point),
        name="function difference",
    )
    width = finite_number(2.0 * interval, name="twice step")
    return finite_number(difference / width, name="central difference")


def trapezoidal_integral(
    function: RealFunction,
    lower: int | float,
    upper: int | float,
    *,
    intervals: int,
) -> float:
    """Approximate a definite integral with equal-width trapezoids.

    ``intervals`` must be a positive integer. Equal bounds return zero.
    Reversed bounds produce the negative of the forward integral. The
    function is evaluated at both endpoints and every interior point.
    """
    _require_function(function)
    lower_bound = finite_number(lower, name="lower bound")
    upper_bound = finite_number(upper, name="upper bound")
    if isinstance(intervals, bool) or not isinstance(intervals, int):
        raise TypeError(f"intervals must be an integer, got {type(intervals).__name__}")
    if intervals < 1:
        raise ValueError(f"intervals must be positive, got {intervals}")
    if lower_bound == upper_bound:
        return 0.0

    width = finite_number((upper_bound - lower_bound) / intervals, name="interval width")
    if width == 0.0:
        raise ValueError("interval width is too small for floating-point arithmetic")
    weighted_total = finite_sum(
        _weighted_values(function, lower_bound, upper_bound, width, intervals),
        name="weighted function total",
    )
    return finite_number(width * weighted_total, name="trapezoidal integral")


def _positive_step(step: int | float) -> float:
    interval = finite_number(step, name="step")
    if interval <= 0.0:
        raise ValueError(f"step must be positive, got {interval}")
    return interval


def _require_function(function: RealFunction) -> None:
    if not callable(function):
        raise TypeError("function must be callable")


def _evaluate(function: RealFunction, at: float) -> float:
    return finite_number(function(at), name=f"function output at {at!r}")


def _weighted_values(
    function: RealFunction, lower: float, upper: float, width: float, intervals: int
) -> Iterator[float]:
    yield 0.5 * _evaluate(function, lower)
    for index in range(1, intervals):
        point = finite_number(lower + index * width, name="integration point")
        yield _evaluate(function, point)
    yield 0.5 * _evaluate(function, upper)
