import os
import time
import numpy as np
import torch
import argparse
import sys

# This allows the script to be run from anywhere and still find the src module
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

from src.smps_reader import SMPSReader
from src.argmax_operation import ArgmaxOperation

def run_benchmark(N: int, M: int, DEVICE: str):
    """
    Runs a single benchmark for the ArgmaxOperation class with given parameters.

    Args:
        N: Number of scenarios to load.
        M: Number of unique dual solutions to generate and add.
        DEVICE: The PyTorch device to use ('cuda' or 'cpu').
    """
    print("--- Starting PyTorch Benchmark ---")
    print(f"Parameters: N={N}, M={M}, Device='{DEVICE}'\n")

    # 1. Initialize SMPS reader and load data
    # ============================================
    print(f"[{time.strftime('%H:%M:%S')}] Loading SMPS data for 'ssn'...")
    try:
        instance_name = "ssn"
        # Assumes 'smps_data' is a subdirectory relative to the project root
        file_dir = os.path.join("smps_data", instance_name)
        core_filepath = os.path.join(file_dir, f"{instance_name}.mps")
        time_filepath = os.path.join(file_dir, f"{instance_name}.tim")
        sto_filepath = os.path.join(file_dir, f"{instance_name}.sto")

        reader = SMPSReader(core_filepath, time_filepath, sto_filepath)
        reader.load_and_extract()
        print(f"[{time.strftime('%H:%M:%S')}] SMPS data loaded successfully.")
    except Exception as e:
        print(f"\nError: Could not load SMPS files. Make sure the 'smps_data/ssn' directory is accessible.")
        print(f"Details: {e}")
        return

    # 2. Initialize ArgmaxOperation
    # ============================================
    argmax_op = ArgmaxOperation.from_smps_reader(
        reader=reader,
        MAX_PI=M,
        MAX_OMEGA=N,
        scenario_batch_size=1000, 
        device=DEVICE
    )

    # 3. Add N scenarios
    # ============================================
    print(f"[{time.strftime('%H:%M:%S')}] Sampling and adding {N} scenarios...")
    # Sample N random scenarios from the stochastic distribution
    sample_pool_rhs_realizations = reader.sample_stochastic_rhs_batch(num_samples=N)
    # Convert to the 'short' format required by the class
    short_delta_r = reader.get_short_delta_r(sample_pool_rhs_realizations)
    argmax_op.add_scenarios(short_delta_r)
    print(f"[{time.strftime('%H:%M:%S')}] {argmax_op.num_scenarios} scenarios added.")

    # 4. Generate and add M unique dual solutions
    # ============================================
    print(f"[{time.strftime('%H:%M:%S')}] Generating and adding {M} unique dual solutions...")
    # Problem dimensions from the reader
    num_stage2_rows = len(reader.row2_indices) # 175 for ssn
    num_stage2_vars = len(reader.y_indices)   # 706 for ssn
    
    # The ssn problem has no bounded variables, so new_rc is empty.
    num_bounded_vars = argmax_op.NUM_BOUNDED_VARS # Should be 0
    
    # Loop until we have successfully added M unique solutions
    while argmax_op.num_pi < M:
        # Generate random basis vectors. Basis statuses are small integers.
        # Gurobi: vbasis (-1, 0, 1, 2), cbasis (-1, 0, 1)
        new_vbasis = np.random.randint(-1, 3, size=num_stage2_vars, dtype=np.int8)
        new_cbasis = np.random.randint(-1, 2, size=num_stage2_rows, dtype=np.int8)
        
        # Generate random pi and rc vectors
        new_pi = np.random.rand(num_stage2_rows).astype(np.float32)
        new_rc = np.array([], dtype=np.float32) # Empty for ssn

        # The add_pi method handles deduplication based on the basis hash
        argmax_op.add_pi(new_pi, new_rc, new_vbasis, new_cbasis)

    print(f"[{time.strftime('%H:%M:%S')}] {argmax_op.num_pi} dual solutions added.")

    # 5. Run and time the new methods
    # ============================================
    # Generate a random first-stage decision vector 'x'
    x_dim = len(reader.x_indices) # 89 for ssn
    x_vector = np.random.rand(x_dim).astype(np.float32)

    # --- Benchmark find_optimal_basis ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting benchmark of find_optimal_basis...")
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    start_time_find = time.perf_counter()
    
    argmax_op.find_optimal_basis(x_vector)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    end_time_find = time.perf_counter()
    elapsed_time_find_ms = (end_time_find - start_time_find) * 1000.0
    print(f"[{time.strftime('%H:%M:%S')}] find_optimal_basis benchmark finished.")

    # --- Benchmark calculate_cut_coefficients ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting benchmark of calculate_cut_coefficients...")
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    start_time_calc = time.perf_counter()

    result = argmax_op.calculate_cut_coefficients()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    end_time_calc = time.perf_counter()
    elapsed_time_calc_ms = (end_time_calc - start_time_calc) * 1000.0
    print(f"[{time.strftime('%H:%M:%S')}] calculate_cut_coefficients benchmark finished.")

    # --- Final Result ---
    print("\n------------------------------------------")
    if result:
        alpha, beta = result
        print("Cut calculation successful.")
        print(f"  alpha: {alpha:.6f}")
        print(f"  beta[0:5]: {beta[:5]}")
    else:
        print("Cut calculation failed.")
    
    print(f"\nExecution time for find_optimal_basis: {elapsed_time_find_ms:.2f} ms")
    print(f"Execution time for calculate_cut_coefficients: {elapsed_time_calc_ms:.2f} ms")
    print("------------------------------------------")
