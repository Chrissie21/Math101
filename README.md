# math101

A small, pure Python mathematics package for learning how numerical code and pip packages are built. Install it as `math101` and import it as `math101`.

The first release covers descriptive statistics, matrices, finite probability, and numerical calculus.

## Requirements

Python 3.11 or newer. The library has no runtime dependencies.

## Develop locally

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The editable installation makes changes under `src/math101` available without reinstalling. Run examples from outside the project directory after installing a wheel to check that packaging includes the source files.

## Use it

```python
from math101.descriptive import mean, sample_variance
from math101.matrices import identity_matrix, matrix_multiply
from math101.probability import binomial_pmf, event_probability
from math101.calculus import central_difference, trapezoidal_integral

print(mean([2, 4, 6]))                       # 4.0
print(sample_variance([2, 4, 6]))            # 4.0
print(matrix_multiply([[1, 2]], [[3], [4]])) # ((11.0,),)
print(identity_matrix(2))                    # ((1.0, 0.0), (0.0, 1.0))
print(event_probability(range(6), [0, 2, 4])) # 0.5
print(binomial_pmf(2, 3, 0.5))               # approximately 0.375
print(central_difference(lambda x: x*x, 2, step=0.01)) # approximately 4
print(trapezoidal_integral(lambda x: x*x, 0, 1, intervals=100)) # approximately 1/3
```

Statistics functions accept an iterable of finite `int` or `float` values and return a `float`. Empty data is invalid; sample variance and sample standard deviation require at least two observations. Boolean values are not accepted as numbers.

Matrix functions accept nonempty rectangular sequences of finite `int` or `float` values, with at least one column. They return new tuples of tuples of floats, so later edits to input lists cannot change a result. Addition requires equal shapes; multiplication requires the left column count to equal the right row count. `identity_matrix` requires a positive integer size.

`event_probability` assumes a finite sample space with **equally likely, unique, hashable** outcomes; favorable outcomes must be a subset. `bernoulli_pmf` and `binomial_pmf` take a success probability in `[0, 1]`. Binomial trials are assumed independent and identical. An integer success count outside `0..trials` returns `0.0`; trials must be nonnegative. `sample_bernoulli` requires an explicit `random.Random` generator so a caller can seed it and reproduce a draw. Use the standard library's `math.comb` and `math.perm` for combination and permutation counts.

`forward_difference` and `central_difference` approximate derivatives; `trapezoidal_integral` approximates a definite integral. They require an explicit positive step size or positive integer interval count. Reversed integration bounds return a signed result, and equal bounds return zero. The callable must return finite `int` or `float` values. A smaller step or more intervals does not guarantee better accuracy for every function because of floating-point rounding and function behavior.

Operations use floating-point arithmetic. A calculation whose intermediate or final value cannot be represented as a finite float raises `ValueError`; this is a teaching library rather than an arbitrary-precision or high-performance numerical package.

## Build and check a distribution

Install the build frontend in your development environment, then run:

```sh
python -m build
python -m venv /tmp/math101-wheel-check
/tmp/math101-wheel-check/bin/python -m pip install dist/*.whl
cd /tmp
/tmp/math101-wheel-check/bin/python -c "from math101.descriptive import mean; print(mean([2, 4, 6]))"
```

The project uses the [MIT license](LICENSE). Before a release, check the `math101` distribution name on the target index and review the README, test results, and artifact contents. The [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) covers TestPyPI and PyPI publication.
