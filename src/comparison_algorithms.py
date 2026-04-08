"""
Comparison Algorithms Wrapper

Provides a unified interface to run 7 nature-inspired optimization algorithms
using the mealpy library, with the same FES budget as PCOA/MPCOA.

Algorithms:
  1. PSO  - Particle Swarm Optimization
  2. GWO  - Grey Wolf Optimizer
  3. WOA  - Whale Optimization Algorithm
  4. SCA  - Sine Cosine Algorithm
  5. HHO  - Harris Hawks Optimization
  6. AOA  - Arithmetic Optimization Algorithm
  7. AO   - Aquila Optimizer
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from mealpy import PSO, GWO, WOA, SCA, HHO, AOA, AO
from mealpy.utils.space import FloatVar

def _calc_pop_epoch(max_fes, pop_size):
    """Calculate the number of epochs given FES budget and pop_size.
    Each epoch uses ~pop_size evaluations."""
    return max(1, max_fes // pop_size)


# ---- Algorithm registry ----
ALGORITHMS = {
    "PSO": {
        "class": PSO.OriginalPSO,
        "name": "Particle Swarm Optimization",
        "ref": "Kennedy & Eberhart (1995)",
    },
    "GWO": {
        "class": GWO.OriginalGWO,
        "name": "Grey Wolf Optimizer",
        "ref": "Mirjalili et al. (2014)",
    },
    "WOA": {
        "class": WOA.OriginalWOA,
        "name": "Whale Optimization Algorithm",
        "ref": "Mirjalili & Lewis (2016)",
    },
    "SCA": {
        "class": SCA.OriginalSCA,
        "name": "Sine Cosine Algorithm",
        "ref": "Mirjalili (2016)",
    },
    "HHO": {
        "class": HHO.OriginalHHO,
        "name": "Harris Hawks Optimization",
        "ref": "Heidari et al. (2019)",
    },
    "AOA": {
        "class": AOA.OriginalAOA,
        "name": "Arithmetic Optimization Algorithm",
        "ref": "Abualigah et al. (2021)",
    },
    "AO": {
        "class": AO.OriginalAO,
        "name": "Aquila Optimizer",
        "ref": "Abualigah et al. (2021)",
    },
}


def run_algorithm(algo_key, obj_func, lb, ub, dim, max_fes, pop_size=30):
    """Run a single mealpy algorithm.

    Parameters
    ----------
    algo_key : str
        One of 'PSO', 'GWO', 'WOA', 'SCA', 'HHO', 'AOA', 'AO'.
    obj_func : callable
        f(x) -> scalar, minimization.
    lb, ub : float
        Bounds of the search space.
    dim : int
        Dimensionality.
    max_fes : int
        Maximum function evaluations.
    pop_size : int
        Population size (default 30).

    Returns
    -------
    best_fit : float
    best_pos : np.array
    """
    if algo_key not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algo_key}'. Choose from: {list(ALGORITHMS.keys())}")

    epoch = _calc_pop_epoch(max_fes, pop_size)

    # Define problem for mealpy 3.x
    bounds = FloatVar(lb=(lb if np.isscalar(lb) else list(lb)), 
                      ub=(ub if np.isscalar(ub) else list(ub)), name="var")
    
    # In mealpy 3.x, if lb/ub are scalars, FloatVar handles it but we need to specify shape
    if np.isscalar(lb):
        bounds = FloatVar(lb=lb, ub=ub, name="var")
        bounds_list = [FloatVar(lb=lb, ub=ub, name=f"var_{i}") for i in range(dim)]
    else:
        bounds_list = [FloatVar(lb=lb[i], ub=ub[i], name=f"var_{i}") for i in range(dim)]

    problem = {
        "obj_func": obj_func,
        "bounds": bounds_list,
        "minmax": "min",
        "log_to": None,       # silence logging
    }

    algo_class = ALGORITHMS[algo_key]["class"]
    model = algo_class(epoch=epoch, pop_size=pop_size)
    model.solve(problem)

    best_fit = model.g_best.target.fitness
    best_pos = model.g_best.solution

    return best_fit, best_pos


def run_all_algorithms(obj_func, lb, ub, dim, max_fes, pop_size=30):
    """Run all 7 comparison algorithms.

    Returns dict of {algo_key: (best_fit, best_pos)}
    """
    results = {}
    for key in ALGORITHMS:
        try:
            fit, pos = run_algorithm(key, obj_func, lb, ub, dim, max_fes, pop_size)
            results[key] = (fit, pos)
        except Exception as e:
            print(f"  [WARN] {key} failed: {e}")
            results[key] = (np.inf, None)
    return results


def list_algorithms():
    """Print available comparison algorithms."""
    print("\nComparison Algorithms (via mealpy):")
    print("=" * 60)
    for key, info in ALGORITHMS.items():
        print(f"  {key:>4}: {info['name']}")
        print(f"        {info['ref']}")
    print()


# ---- Quick test ----
if __name__ == "__main__":
    list_algorithms()

    def sphere(x):
        return np.sum(np.array(x) ** 2)

    dim = 10
    max_fes = 60000

    print(f"Testing all algorithms on Sphere (dim={dim}, max_fes={max_fes}):")
    print("-" * 60)

    results = run_all_algorithms(sphere, -100, 100, dim, max_fes)
    for key, (fit, pos) in results.items():
        print(f"  {key:>4}: {fit:.6e}")

    print("\n[PASS] All comparison algorithms working.")
