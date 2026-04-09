"""
Micro-Test Harness for MPCOA v2 Redesign

Runs a quick validation on exactly 2 functions (Sphere + Rastrigin)
with reduced runs to get fast signal on whether the redesign is working.

Tests MPCOA v2 vs PCOA vs PSO simultaneously.
The moment MPCOA beats both on both functions, the math is ready.

Usage:
    python test_mpcoa_v2.py
    python test_mpcoa_v2.py --runs 10 --fes 30000   # faster iteration
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Import the base PCOA from wherever pcoa.py lives
# Adjust this path if your pcoa.py is in a different location
try:
    from pcoa import pcoa
except ImportError:
    # Try the src directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from pcoa import pcoa

from mpcoa import mpcoa

# We'll use a simple PSO implementation for comparison
# so we don't need mealpy as a dependency for quick testing
def simple_pso(obj_func, lb, ub, dim, max_fes, pop_size=30,
               w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
    """Minimal PSO for comparison. Returns (best_fit, best_pos, convergence)."""
    lb_arr = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb)
    ub_arr = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub)
    
    # Initialize
    pos = lb_arr + np.random.rand(pop_size, dim) * (ub_arr - lb_arr)
    vel = np.zeros((pop_size, dim))
    v_max = 0.2 * (ub_arr - lb_arr)
    
    fit = np.array([obj_func(pos[i]) for i in range(pop_size)])
    fes = pop_size
    
    pbest_pos = pos.copy()
    pbest_fit = fit.copy()
    
    gbest_idx = np.argmin(fit)
    gbest_pos = pos[gbest_idx].copy()
    gbest_fit = fit[gbest_idx]
    
    convergence = [gbest_fit]
    max_iter = max_fes // pop_size
    
    for it in range(max_iter):
        w = w_max - (w_max - w_min) * it / max_iter
        
        for i in range(pop_size):
            if fes >= max_fes:
                break
            
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[i] = (w * vel[i] +
                       c1 * r1 * (pbest_pos[i] - pos[i]) +
                       c2 * r2 * (gbest_pos - pos[i]))
            vel[i] = np.clip(vel[i], -v_max, v_max)
            
            pos[i] = np.clip(pos[i] + vel[i], lb_arr, ub_arr)
            fit[i] = obj_func(pos[i])
            fes += 1
            
            if fit[i] < pbest_fit[i]:
                pbest_pos[i] = pos[i].copy()
                pbest_fit[i] = fit[i]
            
            if fit[i] < gbest_fit:
                gbest_fit = fit[i]
                gbest_pos = pos[i].copy()
        
        convergence.append(gbest_fit)
        if fes >= max_fes:
            break
    
    return gbest_fit, gbest_pos, convergence


# ---- Benchmark functions ----
def sphere(x):
    """Unimodal: tests exploitation precision."""
    return np.sum(x ** 2)

def rastrigin(x):
    """Multimodal: tests exploration ability."""
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def run_test(name, func, lb, ub, dim, max_fes, n_runs):
    """Run all 3 algorithms and compare."""
    print(f"\n{'='*65}")
    print(f"  {name}  (dim={dim}, FES={max_fes}, runs={n_runs})")
    print(f"{'='*65}")

    results = {"PCOA": [], "MPCOA_v2": [], "PSO": []}

    for run in range(n_runs):
        seed = run * 42 + 7

        np.random.seed(seed)
        f_pcoa, _, _ = pcoa(func, lb, ub, dim, max_fes)
        results["PCOA"].append(f_pcoa)

        np.random.seed(seed)
        f_mpcoa, _, _ = mpcoa(func, lb, ub, dim, max_fes)
        results["MPCOA_v2"].append(f_mpcoa)

        np.random.seed(seed)
        f_pso, _, _ = simple_pso(func, lb, ub, dim, max_fes)
        results["PSO"].append(f_pso)

    print(f"  {'Algorithm':<12} {'Mean':>15} {'Std':>15} {'Best':>15}")
    print(f"  {'-'*57}")

    means = {}
    for alg, vals in results.items():
        m, s, b = np.mean(vals), np.std(vals), np.min(vals)
        means[alg] = m
        print(f"  {alg:<12} {m:>15.6e} {s:>15.6e} {b:>15.6e}")

    # Verdict
    winner = min(means, key=means.get)
    mpcoa_beats_pcoa = means["MPCOA_v2"] < means["PCOA"]
    mpcoa_beats_pso = means["MPCOA_v2"] < means["PSO"]

    print()
    if mpcoa_beats_pcoa and mpcoa_beats_pso:
        pct_pcoa = (1 - means["MPCOA_v2"] / means["PCOA"]) * 100 if means["PCOA"] != 0 else 0
        pct_pso = (1 - means["MPCOA_v2"] / means["PSO"]) * 100 if means["PSO"] != 0 else 0
        print(f"  ✅ MPCOA_v2 WINS! Beats PCOA by {pct_pcoa:.1f}%, PSO by {pct_pso:.1f}%")
        return True
    elif mpcoa_beats_pcoa:
        print(f"  ⚠️  MPCOA_v2 beats PCOA but loses to PSO — need more tuning")
        return False
    else:
        print(f"  ❌ MPCOA_v2 loses to PCOA — regression detected!")
        return False


def main():
    parser = argparse.ArgumentParser(description="MPCOA v2 Quick Validation")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs (default: 10)")
    parser.add_argument("--fes", type=int, default=60000, help="Max function evaluations (default: 60000)")
    parser.add_argument("--dim", type=int, default=10, help="Dimensionality (default: 10)")
    args = parser.parse_args()

    print("=" * 65)
    print("  MPCOA v2 MICRO-TEST HARNESS")
    print(f"  Dim={args.dim} | FES={args.fes} | Runs={args.runs}")
    print("=" * 65)

    test_cases = [
        ("Sphere (Unimodal)", sphere, -100, 100),
        ("Rastrigin (Multimodal)", rastrigin, -5.12, 5.12),
    ]

    all_pass = True
    for name, func, lb, ub in test_cases:
        passed = run_test(name, func, lb, ub, args.dim, args.fes, args.runs)
        if not passed:
            all_pass = False

    print(f"\n{'='*65}")
    if all_pass:
        print("  🏆 ALL TESTS PASSED — MPCOA v2 dominates both PCOA and PSO!")
        print("  ➡️  Ready to commit and run full benchmark suite.")
    else:
        print("  ⚙️  Some tests failed — tune v2 parameters before committing.")
        print("  Try adjusting: stag_limit, levy_alpha_max, c1_init/c2_init, comm_decay")
    print("=" * 65)


if __name__ == "__main__":
    main()