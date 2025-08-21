"""
Companion Python tests mirroring the Rust wrapper tests in src/main.rs.

Implements the same objective functions (rosen, rastrigin, rastrigin_fake_long)
and runs four tests with the same bounds and parameters. Requires SciPy and NumPy.

Note on rastrigin_fake_long: The Rust version uses a very large random iteration
count (around 9e8 to 9.9e8) to simulate expensive computations. To keep this
script practical by default, the iteration count is scaled by the environment
variable FAKE_LONG_SCALE (default 0.0001). Set FAKE_LONG_SCALE=1.0 to emulate
the Rust magnitude, but expect very long runtimes.
"""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import shgo


# Match Rust: force single-threaded math libs to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def rosen(x: Sequence[float] | np.ndarray) -> float:
    """Rosenbrock function implemented to match the Rust version."""
    x = np.asarray(x, dtype=float)
    xi = x[:-1]
    xi1 = x[1:]
    return float(np.sum(100.0 * (xi1 - xi * xi) ** 2 + (1.0 - xi) ** 2))


def rastrigin(x: Sequence[float] | np.ndarray, a: float) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(a * n + np.sum(x * x - a * np.cos(2.0 * math.pi * x)))


def rastrigin_fake_long(x: Sequence[float] | np.ndarray, a: float) -> float:
    """Rastrigin with an added, input-dependent dummy load to simulate long runtime.

    The dummy load depends on x so it cannot be optimized away. The contribution
    to the function value is scaled by 1e-20 to avoid changing the landscape.
    """
    x = np.asarray(x, dtype=float)

    # Heavy, input-dependent dummy work. Scale down by default for practicality.
    scale = float(os.getenv("FAKE_LONG_SCALE", "0.0001"))
    base = random.random() * 19_000_000.0 + 11_000_000.0
    iters = max(1, int(base * scale))

    # Per-iteration work depends on x; still keep it simple to avoid extreme cost.
    per_iter = float(np.sum(np.sin(x * x) + np.cos(x)))
    dummy = 0.0
    for _ in range(iters):
        # Use per_iter but mix with a tiny nonlinear op to prevent trivial folding
        dummy += math.sin(per_iter) + per_iter

    # Prevent the optimizer from caring about the dummy work amount
    return rastrigin(x, a) + dummy * 1e-20


def assert_allclose(vec: Iterable[float], target: float, tol: float, name: str) -> None:
    for i, v in enumerate(vec):
        if abs(v - target) >= tol:
            raise AssertionError(f"{name}: result at index {i} is not close to {target}: {v}")


def test_basic_rosen() -> None:
    # 10D Rosenbrock on [0,2]^10, expect x ~ [1,1,...]
    bounds = [(0.0, 2.0)] * 10
    res = shgo(rosen, bounds=bounds)
    print("SHGO result for rosen:", res)
    assert_allclose(res.x, 1.0, 1e-5, "basic_rosen")
    print("Basic rosen test passed")


def test_rastrigin_partial() -> None:
    # 3D Rastrigin with a=10.0 on [-5,5]^3, expect x ~ [0,0,0]
    f = lambda x: rastrigin(x, 10.0)
    bounds = [(-5.0, 5.0)] * 3
    res = shgo(f, bounds=bounds)
    print("SHGO result for rastrigin:", res)
    assert_allclose(res.x, 0.0, 1e-5, "rastrigin_partial")
    print("Rastrigin with closure test passed")


def test_rastrigin_extra_parameters() -> None:
    # Non-default parameters, including a local minimizer with custom options
    f = lambda x: rastrigin(x, 10.0)

    options = {
        # 'maxfev': None,  # Not used directly by SciPy shgo options
        # 'f_min': None,   # Not used directly
        'f_tol': 1e-6,
        # 'maxiter': None, # Not used directly
        # 'maxev': None,   # Not used directly
        # 'maxtime': None, # Not used directly
        # 'minhgrd': None, # Not used directly
        'minimize_every_iter': False,
    }

    minimizer_kwargs = {
        'method': 'COBYQA',  # Mirrors Rust; requires SciPy with COBYQA support
        'options': {
            'tol': 1e-6,
            'maxiter': 200,
        },
    }

    def cb(xk: np.ndarray) -> None:
        print("Callback at x =", xk)

    d = 3
    bounds = [(-5.0, 5.0)] * d
    n = 150
    iters = 10
    res = shgo(
        f,
        bounds=bounds,
        n=n,
        iters=iters,
        callback=cb,
        minimizer_kwargs=minimizer_kwargs,
        options=options,
        sampling_method='sobol',
    )
    print("SHGO result for rastrigin with non-standard params:", res)
    assert_allclose(res.x, 0.0, 1e-5, "rastrigin_extra_parameters")
    print("Rastrigin with non-standard params test passed")


def test_rastrigin_fake_long_partial() -> None:
    # Long-running variant with many workers using a thread pool
    f = lambda x: rastrigin_fake_long(x, 10.0)
    bounds = [(-5.0, 5.0)] * 3

    # Ensure a safe start method for any child processes (even though threads are used)
    try:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except Exception:
        pass

    with ThreadPoolExecutor(max_workers=15) as ex:
        res = shgo(f, bounds=bounds, workers=ex.map)
    print("SHGO result for fake long rastrigin:", res)
    assert_allclose(res.x, 0.0, 1e-5, "rastrigin_fake_long_partial")
    print("Long-running Rastrigin test passed")


def main() -> None:
    # Run all tests by default
    test_basic_rosen()
    test_rastrigin_partial()
    test_rastrigin_extra_parameters()
    test_rastrigin_fake_long_partial()


if __name__ == "__main__":
    main()
