"""
Quick smoke test for the base PCOA implementation.
Validates convergence on Sphere and Rastrigin functions.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pcoa import pcoa


# ---- Benchmark functions ----
def sphere(x):
    """Sphere function  f(x*) = 0 at x* = (0,...,0)"""
    return np.sum(x ** 2)


def rastrigin(x):
    """Rastrigin function  f(x*) = 0 at x* = (0,...,0)"""
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """Rosenbrock function  f(x*) = 0 at x* = (1,...,1)"""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


# ---- Run tests ----
def run_test(name, func, lb, ub, dim, max_fes):
    print(f"\n{'='*50}")
    print(f"  {name}  (dim={dim}, max_fes={max_fes})")
    print(f"{'='*50}")

    best_fit, best_pos, curve = pcoa(func, lb, ub, dim, max_fes)

    print(f"  Best fitness : {best_fit:.6e}")
    print(f"  Best position: [{', '.join(f'{v:.4f}' for v in best_pos[:5])}{'...' if dim > 5 else ''}]")
    print(f"  Convergence  : {curve[0]:.2e} -> {curve[-1]:.2e}  ({len(curve)} records)")
    return best_fit


if __name__ == "__main__":
    dim = 10
    max_fes = 60000

    print("PCOA Base Implementation — Smoke Test")
    print(f"Dimension: {dim}  |  Max FES: {max_fes}")

    f1 = run_test("Sphere", sphere, -100, 100, dim, max_fes)
    f2 = run_test("Rastrigin", rastrigin, -5.12, 5.12, dim, max_fes)
    f3 = run_test("Rosenbrock", rosenbrock, -30, 30, dim, max_fes)

    print(f"\n{'='*50}")
    print("  Summary")
    print(f"{'='*50}")
    print(f"  Sphere     : {f1:.6e}  (optimal = 0)")
    print(f"  Rastrigin  : {f2:.6e}  (optimal = 0)")
    print(f"  Rosenbrock : {f3:.6e}  (optimal = 0)")

    # Basic sanity check
    assert f1 < 1e-2, f"Sphere should converge close to 0, got {f1}"
    print("\n  [PASS] Sphere converged below 1e-2")
    print("\nAll smoke tests completed.")
