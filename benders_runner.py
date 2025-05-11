import os
from pathlib import Path
import numpy as np
import scipy
import h5py

from smps_reader import SMPSReader
from argmax_operation import ArgmaxOperation
from master import AbstractMasterProblem
from benders import BendersMasterProblem
from second_stage_worker import SecondStageWorker
from parallel_second_stage_worker import ParallelSecondStageWorker
if __name__ == "__main__":
    # --- Configuration & Placeholders ---
    # Adjust if test data is elsewhere
    test_data_path = Path(".")
    # Assuming 'smps_data/cep' is a subdirectory in test_data_path

    # CEP
    # h5_file_path = test_data_path / "cep_100scen_results.h5"
    # smps_base_path = test_data_path / "smps_data/cep"
    # smps_core_file = smps_base_path / "cep.mps"
    # smps_time_file = smps_base_path / "cep.tim"
    # smps_sto_file = smps_base_path / "cep.sto"

    # SSN
    h5_file_path = test_data_path / "ssn_1000scen_results.h5"
    smps_base_path = test_data_path / "smps_data/ssn"
    smps_core_file = smps_base_path / "ssn.mps"
    smps_time_file = smps_base_path / "ssn.tim"
    smps_sto_file = smps_base_path / "ssn.sto"

    # Placeholders for ArgmaxOperation - set these to appropriate values
    MAX_PI_PLACEHOLDER = 100000  # Maximum number of dual solutions to store
    MAX_OMEGA_PLACEHOLDER = 100000  # Maximum number of scenarios to store in ArgmaxOperation
    SCENARIO_BATCH_SIZE_PLACEHOLDER = 1000 # Scenario batch size for ArgmaxOperation

    # Placeholder for sample pool generation - set to desired number of samples
    NUM_SAMPLES_FOR_POOL = 100000

    ETA_LOWER_BOUND = 7.0

    # 1. Load the SMPSReader and call load_and_extract.
    print(f"\n1. Initializing SMPSReader...")
    reader = SMPSReader(
        core_file=str(smps_core_file),
        time_file=str(smps_time_file),
        sto_file=str(smps_sto_file)
    )
    reader.load_and_extract()

    # 2. Use reader to create ArgmaxOperation
    argmax_op = ArgmaxOperation.from_smps_reader(
        reader=reader,
        MAX_PI=MAX_PI_PLACEHOLDER,
        MAX_OMEGA=MAX_OMEGA_PLACEHOLDER,
        scenario_batch_size=SCENARIO_BATCH_SIZE_PLACEHOLDER
    )

    # 3. Use reader to create a BendersMasterProblem and a ParallelSecondStageWorker
    print(f"\n3. Creating BendersMasterProblem and ParallelSecondStageWorker...")
    master_problem = BendersMasterProblem.from_smps_reader(reader)
    master_problem.create_benders_problem(model_name="InitialBendersMaster")
    master_problem.set_eta_lower_bound(ETA_LOWER_BOUND)
    print(f"   ETA_LOWER_BOUND = {ETA_LOWER_BOUND}.")

    num_cpu_workers = os.cpu_count()
    parallel_worker = ParallelSecondStageWorker.from_smps_reader(
        reader=reader,
        num_workers=num_cpu_workers if num_cpu_workers else 1
    )
    print(f"   ParallelSecondStageWorker created with {parallel_worker.num_workers} worker(s).")

    # 4. Use reader to generate a sample pool.
    print(f"\n4. Generating sample pool...")
    sample_pool = reader.sample_stochastic_rhs_batch(num_samples=NUM_SAMPLES_FOR_POOL)
    print(f"   Sample pool of {NUM_SAMPLES_FOR_POOL} scenarios generated successfully.")
    print(f"   Sample pool shape: {sample_pool.shape}")

    short_delta_r = reader.get_short_delta_r(sample_pool)
    print(f"   Short delta_r shape: {short_delta_r.shape}")

    # Add to ArgmaxOperation
    argmax_op.add_scenarios(short_delta_r)

    # 5. Load the dual vertices from the HDF5 file.
    print(f"\n5. Loading basis from HDF5 file...")
    with h5py.File(h5_file_path, 'r') as hf:
        if '/solution/dual/pi_s' in hf and \
            '/basis/vbasis_y_all' in hf and \
            '/basis/cbasis_y_all' in hf:

            pi_s_all = hf['/solution/dual/pi_s'][:]
            vbasis_y_all = hf['/basis/vbasis_y_all'][:]
            cbasis_y_all = hf['/basis/cbasis_y_all'][:]

            added_pi_count = 0
            num_scenarios_in_h5 = pi_s_all.shape[0]
            print(f"   Found {num_scenarios_in_h5} scenario solutions in HDF5 file.")

            # Prepare a template for reduced costs as it's not in HDF5 for this purpose.
            # ArgmaxOperation's add_pi method requires an rc (reduced cost) vector.
            # We use a zero vector placeholder of the correct dimension.
            rc_template = np.zeros(argmax_op.NUM_BOUNDED_VARS, dtype=np.float64)


            # check: 
            for s_idx in range(num_scenarios_in_h5):
                current_pi = pi_s_all[s_idx, :]
                current_vbasis = vbasis_y_all[s_idx, :]
                current_cbasis = cbasis_y_all[s_idx, :]
                
                # Ensure shapes match what add_pi expects, though slicing should handle this.
                # add_pi returns True if successfully added, False if duplicate or full.
                argmax_op.add_pi(current_pi, rc_template.copy(), current_vbasis, current_cbasis)
                added_pi_count += 1
            
            print(f"   Added {added_pi_count} dual solutions to ArgmaxOperation, currently {argmax_op.num_pi} stored.")
        else:
            print("   Error: One or more required datasets not found in HDF5 file.")
            print("   Missing: /solution/dual/pi_s, /basis/vbasis_y_all, or /basis/cbasis_y_all")


    x = None
    TOL = 1e-3
    optimal = False
    iteration_count = 0 # Initialize iteration counter

    print("\n--- Starting Benders Decomposition Loop ---")

    while not optimal:
        iteration_count += 1

        # 1. Solve Master Problem
        x, master_obj, code = master_problem.solve() # Renamed obj to master_obj for clarity

        if code != 2:
            print(f"  Iter {iteration_count}: Error solving master problem. Gurobi status code: {code}")
            break
        
        # Log master objective
        print(f"Iter {iteration_count}: Master Obj = {master_obj:.4f}", end="")

        # 2. Warmstart: Calculate initial cut based on existing duals
        alpha_pre, beta_pre, best_k_index = argmax_op.calculate_cut(x)
        
        # Log unique number of best_k_index from the first argmax_op.calculate_cut
        if best_k_index is not None:
            num_unique_best_k = len(np.unique(best_k_index))
            print(f", Unique Warmstart Duals Used = {num_unique_best_k}", end="")
        else:
            # This case should ideally not happen if best_k_index is guaranteed to exist
            print(", Warmstart best_k_index is None", end="")


        vbasis_batch, cbasis_batch = argmax_op.get_basis(best_k_index)
        
        # 3. Subproblem Solver
        obj_all, y_all, pi_all, rc_all, vbasis_out, cbasis_out = parallel_worker.solve_batch(x, short_delta_r, vbasis_batch, cbasis_batch)

        # 4. Add newly found duals to ArgmaxOperation
        initial_pi_count_in_argmax = argmax_op.num_pi
        for s in range(argmax_op.num_scenarios): # Assuming argmax_op.num_scenarios reflects the scenarios processed
            pi = pi_all[s, :]
            rc = rc_all[s, :]
            vbasis = vbasis_out[s, :]
            cbasis = cbasis_out[s, :]
            argmax_op.add_pi(pi, rc, vbasis, cbasis)
        
        # Log current number of duals in ArgmaxOperation
        print(f", Total Duals in ArgmaxOp = {argmax_op.num_pi}", end="")


        # 5. Run argmax again with all (old + new) duals to determine the strongest cut
        alpha, beta, _ = argmax_op.calculate_cut(x)
        
        # 6. Check for optimality and add cut if necessary
        # Ensure alpha_pre and beta_pre are valid before this calculation
        # (User guaranteed they exist, so direct usage)
        current_cut_value = alpha + beta @ x
        previous_cut_value = alpha_pre + beta_pre @ x # This is E[Q(x, omega_k)] based on pi_k from warmstart

        # This gives a sense of how well the argmax heuristic is
        argmax_gap = current_cut_value - previous_cut_value
        
        # 7. Optimality test
        master_eta_value = master_obj - reader.c @ x
        gap = current_cut_value - master_obj # This is a common way to check Benders gap. Master obj is eta.

        print(f"cut_value = {current_cut_value:.4e}, master_eta = {master_eta_value:.4e}, gap = {gap:.4e}, argmax_gap = {argmax_gap:.4e}", end="")

        optimal = gap < TOL
        if optimal:
            print(" -> Optimal")
            break
        else:
            master_problem.add_optimality_cut(beta, alpha)
            print(" -> Cut Added.")


    if optimal:
        print(f"\n--- Benders Decomposition Converged after {iteration_count} iterations ---")
        print(f"Final Master Problem Objective (c'x + eta): {master_obj:.6f}")
        # You might want to print the final x values as well if they are of interest
        # print(f"Final x solution: {x}")
    else:
        print(f"\n--- Benders Decomposition Stopped after {iteration_count} iterations (not converged or error) ---")

