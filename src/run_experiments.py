"""
Full Experiment Runner

Runs MPCOA, PCOA, and 7 comparison algorithms across:
- CEC 2014 (30 functions)
- CEC 2017 (29 functions)
- CEC 2020 (10 functions)
- CEC 2022 (12 functions)

Produces raw results (CSV format) for later statistical analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Import algorithms
sys.path.insert(0, os.path.dirname(__file__))
from pcoa import pcoa
from mpcoa import mpcoa
from comparison_algorithms import run_all_algorithms, ALGORITHMS
from cec_benchmark import get_cec_functions, CEC_SUITES

# Experiment Settings
DIM = 10
MAX_FES = 60000
N_RUNS = 30  # Standard for statistical tests (Wilcoxon)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def run_experiment_on_function(func_name, suite_name, func_obj, lb, ub, dim, max_fes, n_runs, out_csv):
    """Run all 9 algorithms `n_runs` times on a single function, saving to CSV incrementally."""
    
    algo_names = ["PCOA", "MPCOA"] + list(ALGORITHMS.keys())
    
    # The func_obj returned by get_cec_functions is already a callable evaluate function
    def obj_wrapper(x):
        return func_obj(x)
        
    for run in range(n_runs):
        print(f"\n    Run {run+1}/{n_runs}... ", end="", flush=True)
        # Always set seed for reproducibility within a run across algorithms
        seed = run * 42
        row_data = {}
        
        # 1. PCOA
        np.random.seed(seed)
        best_f_pcoa, _, _ = pcoa(obj_wrapper, lb, ub, dim, max_fes)
        row_data["PCOA"] = best_f_pcoa
        
        # 2. MPCOA
        np.random.seed(seed)
        best_f_mpcoa, _, _ = mpcoa(obj_wrapper, lb, ub, dim, max_fes)
        row_data["MPCOA"] = best_f_mpcoa
        
        # 3. Mealpy algorithms
        np.random.seed(seed)
        import random
        random.seed(seed)
        mealpy_res = run_all_algorithms(obj_wrapper, lb, ub, dim, max_fes)
        
        for alg_key in ALGORITHMS.keys():
            row_data[alg_key] = mealpy_res[alg_key][0]
            
        # Append this run's data to CSV immediately
        with open(out_csv, "a") as f:
            row = [suite_name, func_name, str(run)]
            for a in algo_names:
                row.append(str(row_data[a]))
            f.write(",".join(row) + "\n")
            
        print("Done.", end="", flush=True)

def main():
    timestamp = datetime.now().strftime("%Y%md_%H%M%S")
    out_csv = os.path.join(RESULTS_DIR, f"raw_results_{DIM}D_{timestamp}.csv")
    
    # Open file, write header
    with open(out_csv, "w") as f:
        algo_names = ["PCOA", "MPCOA"] + list(ALGORITHMS.keys())
        cols = ["Suite", "Function", "Run"] + algo_names
        f.write(",".join(cols) + "\n")
        
    print(f"Starting experiments. Target: {N_RUNS} runs/function. Total Budget: {MAX_FES} FES")
    print(f"Results will be saved incrementally to: {out_csv}")
    print("-" * 60)
    
    for suite_name in CEC_SUITES.keys():
        print(f"\nEvaluating Suite: {suite_name}")
        functions = get_cec_functions(suite_name, dim=DIM)
        for fid, func_obj, info in functions:
            func_name = info['name']
            print(f"  Running {func_name} ...", end="", flush=True)
            
            # Note: For CEC bounds, some functions might have varying bounds, 
            # but standard is [-100, 100]. We can use func_obj bounds.
            # However opfunu generally provides uniform bounds. 
            # In our case cec_benchmark initializes with standard [-100, 100].
            
            try:
                # Opfunu's default domain space 
                # Could be read from func_obj.lb, func_obj.ub
                lb = -100
                ub = 100
                if 'lb' in info and 'ub' in info:
                    lb = info['lb']
                    ub = info['ub']
                
                run_experiment_on_function(
                    func_name, suite_name, func_obj, lb, ub, DIM, MAX_FES, N_RUNS, out_csv
                )
                print("\n  Finished completely.")
                
            except Exception as e:
                print(f" ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Experiments Complete. Data saved to: {out_csv}")

if __name__ == "__main__":
    main()
