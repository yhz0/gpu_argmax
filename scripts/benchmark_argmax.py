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
        scenario_batch_size=10000, 
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
    # Generate fake but dimensionally-valid bases (won't be factorized)
    added_count = 0
    while argmax_op.num_pi < M and added_count < M * 20:  # Prevent infinite loop
        # Create dimensionally-correct fake basis:
        # - Exactly 175 basic variables (to match NUM_STAGE2_ROWS)
        # - All constraints non-basic (no slack variables basic)
        new_vbasis = np.full(num_stage2_vars, -1, dtype=np.int8)  # All non-basic initially
        new_cbasis = np.full(num_stage2_rows, -1, dtype=np.int8)   # All non-basic
        
        # Randomly select exactly 175 variables to be basic
        basic_var_indices = np.random.choice(num_stage2_vars, size=num_stage2_rows, replace=False)
        new_vbasis[basic_var_indices] = 0  # Set selected variables as basic
        
        # Generate random pi and rc vectors
        new_pi = np.random.rand(num_stage2_rows).astype(np.float32)
        new_rc = np.array([], dtype=np.float32) # Empty for ssn (no bounded variables)

        # Add the dual solution (basis validity doesn't matter since we won't factorize)
        success = argmax_op.add_pi(new_pi, new_rc, new_vbasis, new_cbasis)
        if not success:
            added_count += 1  # Count failed attempts (duplicates)

    print(f"[{time.strftime('%H:%M:%S')}] {argmax_op.num_pi} dual solutions added.")

    # 5. Skip finalization since we're using fake bases and only testing fast argmax
    # ============================================
    print(f"[{time.strftime('%H:%M:%S')}] Skipping finalization (using fake bases for fast argmax only)...")

    # 6. Run and time the methods
    # ============================================
    # Generate a random first-stage decision vector 'x'
    x_dim = len(reader.x_indices) # 89 for ssn
    x_vector = np.random.rand(x_dim).astype(np.float32)
    
    # Number of benchmark runs for statistical reliability
    NUM_RUNS = 10
    print(f"\n[{time.strftime('%H:%M:%S')}] Running {NUM_RUNS} benchmark iterations...")

    # --- Benchmark find_optimal_basis_fast (Full Method) ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Benchmarking find_optimal_basis_fast (full method)...")
    times_full = []
    
    for run in range(NUM_RUNS):
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        pi_indices, _ = argmax_op.find_optimal_basis_fast(x_vector, touch_lru=False)
        
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000.0
        times_full.append(elapsed_ms)
    
    avg_time_full = np.mean(times_full)
    std_time_full = np.std(times_full)
    print(f"[{time.strftime('%H:%M:%S')}] Full method: {avg_time_full:.2f} ± {std_time_full:.2f} ms")

    # Return structured results for external processing
    return {
        'device': DEVICE,
        'N': N,
        'M': M,
        'full_method_avg_ms': avg_time_full,
        'full_method_std_ms': std_time_full,
    }
