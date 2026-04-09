"""
CEC Benchmark Function Wrapper

Provides a unified interface to load CEC 2014, 2017, 2020, and 2022
benchmark functions via the opfunu library.

Usage:
    from cec_benchmark import get_cec_functions, get_function_info

    # Get all CEC2017 functions (dim=10)
    functions = get_cec_functions("cec2017", dim=10)
    for fid, func, info in functions:
        result = func(np.zeros(10))
        print(f"F{fid}: {info['name']}  bounds=[{info['lb']}, {info['ub']}]")
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import opfunu.cec_based.cec2014 as cec2014_mod
import opfunu.cec_based.cec2017 as cec2017_mod
import opfunu.cec_based.cec2020 as cec2020_mod
import opfunu.cec_based.cec2022 as cec2022_mod
from engineering_problems import get_engineering_functions

# CEC suite configurations
CEC_SUITES = {
    "cec2014": {
        "module": cec2014_mod,
        "class_fmt": "F{fid}2014",
        "func_ids": list(range(1, 31)),       # F1-F30
        "bounds": (-100, 100),
        "note": "30 functions: unimodal (F1-F3), multimodal (F4-F16), hybrid (F17-F22), composition (F23-F30)",
    },
    "cec2017": {
        "module": cec2017_mod,
        "class_fmt": "F{fid}2017",
        "func_ids": [i for i in range(1, 30) if i != 2],  # F1-F29, skip F2
        "bounds": (-100, 100),
        "note": "28 functions (F2 excluded per IEEE guidelines, F30 not in opfunu)",
    },
    "cec2020": {
        "module": cec2020_mod,
        "class_fmt": "F{fid}2020",
        "func_ids": list(range(1, 11)),        # F1-F10
        "bounds": (-100, 100),
        "note": "10 functions for bound-constrained optimization",
    },
    "cec2022": {
        "module": cec2022_mod,
        "class_fmt": "F{fid}2022",
        "func_ids": list(range(1, 13)),        # F1-F12
        "bounds": (-100, 100),
        "note": "12 functions for single-objective optimization",
    },
    "engineering": {
        "module": None,
        "class_fmt": "Eng{fid}",
        "func_ids": list(range(1, 5)),
        "bounds": (0, 100), # Handled dynamically
        "note": "4 standard constrained engineering design problems",
    }
}


def get_cec_function(suite, fid, dim=10):
    """Load a single CEC benchmark function.

    Parameters
    ----------
    suite : str
        One of 'cec2014', 'cec2017', 'cec2020', 'cec2022'.
    fid : int
        Function ID (e.g., 1 for F1).
    dim : int
        Problem dimensionality.

    Returns
    -------
    func : callable
        Function that accepts a numpy array -> scalar.
    info : dict
        Metadata: name, lb, ub, f_bias (known optimal value).
    """
    suite = suite.lower()
    if suite not in CEC_SUITES:
        raise ValueError(f"Unknown suite '{suite}'. Choose from: {list(CEC_SUITES.keys())}")

    cfg = CEC_SUITES[suite]
    class_name = cfg["class_fmt"].format(fid=fid)
    cls = getattr(cfg["module"], class_name, None)

    if cls is None:
        raise RuntimeError(f"Class {class_name} not found in {suite} module")

    func_obj = cls(ndim=dim)

    lb = func_obj.lb.tolist() if hasattr(func_obj.lb, 'tolist') else [cfg["bounds"][0]] * dim
    ub = func_obj.ub.tolist() if hasattr(func_obj.ub, 'tolist') else [cfg["bounds"][1]] * dim

    # Get the optimal (bias) value
    f_bias = getattr(func_obj, 'f_bias', None)
    if f_bias is None:
        f_bias = getattr(func_obj, 'f_global', 0.0)

    info = {
        "name": f"{suite.upper()} F{fid}",
        "lb": lb[0] if len(set(lb)) == 1 else lb,
        "ub": ub[0] if len(set(ub)) == 1 else ub,
        "f_bias": f_bias,
        "dim": dim,
    }

    # Return a simple callable
    def evaluate(x):
        return func_obj.evaluate(x)

    return evaluate, info


def get_cec_functions(suite, dim=10):
    """Load all functions from a CEC suite.

    Returns
    -------
    list of (fid, func, info) tuples
    """
    suite = suite.lower()
    
    if suite == "engineering":
        return get_engineering_functions()
        
    cfg = CEC_SUITES[suite]
    results = []

    for fid in cfg["func_ids"]:
        try:
            func, info = get_cec_function(suite, fid, dim)
            results.append((fid, func, info))
        except Exception as e:
            print(f"  [WARN] Skipping {suite.upper()} F{fid}: {e}")

    return results


def list_suites():
    """Print info about all available CEC suites."""
    print("\nAvailable CEC Benchmark Suites:")
    print("=" * 60)
    for name, cfg in CEC_SUITES.items():
        n = len(cfg["func_ids"])
        print(f"\n  {name.upper()}: {n} functions")
        print(f"    Bounds: [{cfg['bounds'][0]}, {cfg['bounds'][1]}]")
        print(f"    {cfg['note']}")
    print()


# ---- Quick self-test ----
if __name__ == "__main__":
    list_suites()

    # Test loading one function from each suite
    for suite in ["cec2014", "cec2017", "cec2020", "cec2022"]:
        print(f"\n--- Testing {suite.upper()} ---")
        functions = get_cec_functions(suite, dim=10)
        print(f"  Loaded {len(functions)} functions")

        if functions:
            fid, func, info = functions[0]
            x_test = np.zeros(10)
            val = func(x_test)
            print(f"  F{fid}: f(zeros) = {val:.4f}  (bias = {info['f_bias']})")

            fid, func, info = functions[-1]
            val = func(np.zeros(10))
            print(f"  F{fid}: f(zeros) = {val:.4f}  (bias = {info['f_bias']})")

    print("\n[PASS] All CEC suites loaded successfully.")
