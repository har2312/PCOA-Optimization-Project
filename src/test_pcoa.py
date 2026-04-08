"""
Comparison test: Base PCOA vs Modified PCOA (MPCOA)
Tests on Sphere, Rastrigin, and Rosenbrock to see the effect of
Levy flight in the pollination phase.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pcoa import pcoa
from mpcoa import mpcoa


# ---- Benchmark functions ----
def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


# ---- Run comparison ----
def compare(name, func, lb, ub, dim, max_fes, n_runs=5):
    print(f"\n{'='*60}")
    print(f"  {name}  (dim={dim}, max_fes={max_fes}, runs={n_runs})")
    print(f"{'='*60}")

    pcoa_results = []
    mpcoa_results = []

    for run in range(n_runs):
        np.random.seed(run * 42)
        f1, _, _ = pcoa(func, lb, ub, dim, max_fes)
        pcoa_results.append(f1)

        np.random.seed(run * 42)
        f2, _, _ = mpcoa(func, lb, ub, dim, max_fes)
        mpcoa_results.append(f2)

    pcoa_mean = np.mean(pcoa_results)
    pcoa_std = np.std(pcoa_results)
    mpcoa_mean = np.mean(mpcoa_results)
    mpcoa_std = np.std(mpcoa_results)

    print(f"  {'Algorithm':<12} {'Mean':>15} {'Std':>15}")
    print(f"  {'-'*42}")
    print(f"  {'PCOA':<12} {pcoa_mean:>15.6e} {pcoa_std:>15.6e}")
    print(f"  {'MPCOA':<12} {mpcoa_mean:>15.6e} {mpcoa_std:>15.6e}")

    if mpcoa_mean < pcoa_mean:
        improvement = (1 - mpcoa_mean / pcoa_mean) * 100 if pcoa_mean != 0 else 0
        print(f"  >>> MPCOA is BETTER by {improvement:.1f}%")
    else:
        print(f"  >>> PCOA is better on this function")

    return pcoa_mean, mpcoa_mean


if __name__ == "__main__":
    dim = 10
    max_fes = 60000

    print("PCOA vs MPCOA Comparison — Levy Flight Modification")
    print(f"Dimension: {dim}  |  Max FES: {max_fes}")

    results = []
    results.append(("Sphere", *compare("Sphere", sphere, -100, 100, dim, max_fes)))
    results.append(("Rastrigin", *compare("Rastrigin", rastrigin, -5.12, 5.12, dim, max_fes)))
    results.append(("Rosenbrock", *compare("Rosenbrock", rosenbrock, -30, 30, dim, max_fes)))

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Function':<15} {'PCOA':>15} {'MPCOA':>15} {'Winner':>10}")
    print(f"  {'-'*55}")
    for name, p, m in results:
        winner = "MPCOA" if m < p else "PCOA"
        print(f"  {name:<15} {p:>15.4e} {m:>15.4e} {winner:>10}")

    print("\nDone.")
