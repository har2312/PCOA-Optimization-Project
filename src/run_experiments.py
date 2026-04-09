"""
Full Experiment Runner with Parallel Execution & CLI Distribution

Runs MPCOA, PCOA, and 7 comparison algorithms across:
- CEC 2014, CEC 2017, CEC 2020, CEC 2022
- Engineering Problems Suite

Produces:
1. raw_results CSV for statistical tests.
2. convergence JSON for plotting.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
import concurrent.futures
from multiprocessing import freeze_support

# Import algorithms
sys.path.insert(0, os.path.dirname(__file__))
from pcoa import pcoa
from mpcoa import mpcoa
from comparison_algorithms import run_all_algorithms, ALGORITHMS
from cec_benchmark import get_cec_functions, CEC_SUITES

# Constant Settings
MAX_FES = 60000
N_RUNS = 50 

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def _run_single_execution(seed, run_idx, suite_name, fid, dim, max_fes):
    """Executes exactly 1 run across all 9 algorithms (Used by ProcessPoolExecutor)."""
    
    # Reconstruct the objective function inside the worker process to avoid pickling errors
    functions = get_cec_functions(suite_name, dim=dim)
    # find the matching fid
    func_obj, info = None, None
    for f_id, f_obj, f_info in functions:
        if f_id == fid:
            func_obj = f_obj
            info = f_info
            break
            
    if func_obj is None:
        raise ValueError(f"Function {fid} not found in {suite_name}")
        
    lb = -100
    ub = 100
    if 'lb' in info and 'ub' in info:
        lb = info['lb']
        ub = info['ub']
        
    def obj_wrapper(x):
        x = np.asarray(x).ravel()
        return func_obj(x)

    row_data = {}
    conv_data = {}
    
    # 1. PCOA
    np.random.seed(seed)
    best_f_pcoa, _, conv_pcoa = pcoa(obj_wrapper, lb, ub, dim, max_fes)
    row_data["PCOA"] = best_f_pcoa
    conv_data["PCOA"] = conv_pcoa
    
    # 2. MPCOA
    np.random.seed(seed)
    best_f_mpcoa, _, conv_mpcoa = mpcoa(obj_wrapper, lb, ub, dim, max_fes)
    row_data["MPCOA"] = best_f_mpcoa
    conv_data["MPCOA"] = conv_mpcoa
    
    # 3. Mealpy algorithms
    np.random.seed(seed)
    import random
    random.seed(seed)
    mealpy_res = run_all_algorithms(obj_wrapper, lb, ub, dim, max_fes)
    for alg_key in ALGORITHMS.keys():
        row_data[alg_key] = mealpy_res[alg_key][0]
        # mealpy history stores best_fit for each epoch
        conv_data[alg_key] = mealpy_res[alg_key][2]

    return run_idx, row_data, conv_data

def run_experiment_parallel(func_name, suite_name, fid, info, dim, max_fes, n_runs, out_csv, out_json):
    """Run all 9 algorithms `n_runs` times on a single function concurrently."""
    algo_names = ["PCOA", "MPCOA"] + list(ALGORITHMS.keys())

    # Unpack dimensions if it's dynamic (e.g. engineering)
    if 'dim' in info:
        dim = info['dim']
        
    print(f"\n  Running {func_name} (Parallelizing {n_runs} runs)... ", end="", flush=True)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for run in range(n_runs):
            seed = run * 42
            futures.append(
                executor.submit(_run_single_execution, seed, run, suite_name, fid, dim, max_fes)
            )
            
        for future in concurrent.futures.as_completed(futures):
            run_idx, row_data, conv_data = future.result()
            
            # Immediately append to CSV
            with open(out_csv, "a") as f:
                row = [suite_name, func_name, str(run_idx)]
                for a in algo_names:
                    row.append(str(row_data[a]))
                f.write(",".join(row) + "\n")
                
            # Immediately append to JSON (JSONL format: one valid JSON dictionary per line)
            with open(out_json, "a") as f:
                json_record = {
                    "Suite": suite_name,
                    "Function": func_name,
                    "Run": run_idx,
                    "Convergence": conv_data
                }
                f.write(json.dumps(json_record) + "\n")
                
            print(".", end="", flush=True)
            
    print(" Done.")

def main():
    parser = argparse.ArgumentParser(description="PCOA Experiments Distributed Runner")
    parser.add_argument("--suite", type=str, default="all", choices=["all"] + list(CEC_SUITES.keys()),
                        help="Choose a specific suite to evaluate to divide work among team members.")
    args = parser.parse_args()

    suites_to_run = list(CEC_SUITES.keys()) if args.suite == "all" else [args.suite]
    
    timestamp = datetime.now().strftime("%Y%md_%H%M%S")
    suffix = f"_{args.suite}" if args.suite != "all" else ""
    out_csv = os.path.join(RESULTS_DIR, f"raw_results{suffix}_{timestamp}.csv")
    out_json = os.path.join(RESULTS_DIR, f"convergence{suffix}_{timestamp}.jsonl")
    
    # Open file, write header
    with open(out_csv, "w") as f:
        algo_names = ["PCOA", "MPCOA"] + list(ALGORITHMS.keys())
        cols = ["Suite", "Function", "Run"] + algo_names
        f.write(",".join(cols) + "\n")
        
    # Clear / touch JSONL
    with open(out_json, "w") as f:
        pass
        
    print(f"Starting Distributed Experiments. Target Suite: {args.suite.upper()}")
    print(f"Runs/Function: {N_RUNS} | Total Budget: {MAX_FES} FES")
    print(f"Stats CSV:      {out_csv}")
    print(f"Convergence:    {out_json}")
    print("-" * 60)
    
    dim_default = 10 

    for suite_name in suites_to_run:
        print(f"\nEvaluating Suite: {suite_name}")
        functions = get_cec_functions(suite_name, dim=dim_default)
        for fid, func_obj, info in functions:
            try:
                run_experiment_parallel(
                    info['name'], suite_name, fid, info, dim_default, MAX_FES, N_RUNS, out_csv, out_json
                )
            except Exception as e:
                import traceback
                print(f"\n  [ERROR] Function {info['name']} failed: {e}")
                traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Experiments Complete. Data saved successfully.")

if __name__ == "__main__":
    freeze_support()
    main()
