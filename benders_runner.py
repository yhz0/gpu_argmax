import os
from pathlib import Path
import time
import numpy as np
import scipy
import h5py
import scipy.linalg

from smps_reader import SMPSReader
from argmax_operation import ArgmaxOperation
from benders import BendersMasterProblem
from regularized_benders import RegularizedBendersMasterProblem

from second_stage_worker import SecondStageWorker
from parallel_second_stage_worker import ParallelSecondStageWorker
if __name__ == "__main__":
    # --- Configuration & Placeholders ---
    # Adjust if test data is elsewhere
    test_data_path = Path(".")
    # Assuming 'smps_data/cep' is a subdirectory in test_data_path

    INSTANCE_NAME = "ssn"

    # CEP
    # h5_file_path = test_data_path / "cep_100scen_results.h5"
    # smps_base_path = test_data_path / "smps_data/cep"
    # smps_core_file = smps_base_path / "cep.mps"
    # smps_time_file = smps_base_path / "cep.tim"
    # smps_sto_file = smps_base_path / "cep.sto"

    # SSN
    h5_file_path = test_data_path / "ssn_5000scen_results.h5"
    smps_base_path = test_data_path / "smps_data/ssn"
    smps_core_file = smps_base_path / "ssn.mps"
    smps_time_file = smps_base_path / "ssn.tim"
    smps_sto_file = smps_base_path / "ssn.sto"

    # Placeholders for ArgmaxOperation - set these to appropriate values
    MAX_PI = 1000000  # Maximum number of dual solutions to store
    MAX_OMEGA = 1000000  # Maximum number of scenarios to store in ArgmaxOperation
    SCENARIO_BATCH_SIZE = 10000 # Scenario batch size for ArgmaxOperation

    # Placeholder for sample pool generation - set to desired number of samples
    NUM_SAMPLES_FOR_POOL = 1000000

    ETA_LOWER_BOUND = 0.0

    

    # 1. Load the SMPSReader and call load_and_extract.
    print(f"\n1. Initializing SMPSReader...")
    reader = SMPSReader(
        core_file=str(smps_core_file),
        time_file=str(smps_time_file),
        sto_file=str(smps_sto_file)
    )
    reader.load_and_extract()

    OUTPUT_HDF5_FILE = f"result_{INSTANCE_NAME}.h5"
    # save metadata
    with h5py.File(OUTPUT_HDF5_FILE, 'a') as hf:
        if "/metadata" in hf:
            del hf['/metadata']
        
        hf.create_group('/metadata')
        hf['/metadata'].attrs['instance_name'] = INSTANCE_NAME
        hf['/metadata'].attrs['num_scenarios'] = NUM_SAMPLES_FOR_POOL
        hf['/metadata'].attrs['eta_lower_bound'] = ETA_LOWER_BOUND


    # 2. Use reader to create ArgmaxOperation
    print(f"\n2. Creating ArgmaxOperation...")
    argmax_op = ArgmaxOperation.from_smps_reader(
        reader=reader,
        MAX_PI=MAX_PI,
        MAX_OMEGA=MAX_OMEGA,
        scenario_batch_size=SCENARIO_BATCH_SIZE
    )

    # 3. Use reader to create a BendersMasterProblem and a ParallelSecondStageWorker
    print(f"\n3. Creating BendersMasterProblem and ParallelSecondStageWorker...")

    # regularized
    master_problem:RegularizedBendersMasterProblem = RegularizedBendersMasterProblem.from_smps_reader(reader)
    master_problem.set_regularization_strength(0.1)

    # non regularized
    # master_problem = BendersMasterProblem.from_smps_reader(reader)

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
            x_init = hf['/solution/primal/x'][:]

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

    # 6. Set up log
    try:
        hf_log_file = h5py.File(OUTPUT_HDF5_FILE, 'a') # Open in append mode

        # Delete the /iteration_log group if it exists, to start fresh for this run
        if "iteration_log" in hf_log_file:
            del hf_log_file["iteration_log"]
            print(f"Deleted existing '/iteration_log' group from {OUTPUT_HDF5_FILE}.")

        iter_log_group = hf_log_file.create_group("iteration_log")
        print(f"Created new '/iteration_log' group in {OUTPUT_HDF5_FILE}.")

        # Initialize resizable datasets for logging
        chunk_1D = (1024,) # For scalar history datasets
        
        iter_log_group.create_dataset("iteration_counts", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=chunk_1D)
        iter_log_group.create_dataset("master_obj_history", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=chunk_1D)
        iter_log_group.create_dataset("argmax_cut_height_history", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=chunk_1D)
        iter_log_group.create_dataset("calculated_cut_height_history", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=chunk_1D)
        iter_log_group.create_dataset("master_epigraph_height_history", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=chunk_1D)
        
        x_dim = len(reader.stage1_var_names)
        if x_dim > 0:
            chunk_2D_x = (128, x_dim) 
            iter_log_group.create_dataset("x_vector_history", shape=(0, x_dim), maxshape=(None, x_dim), dtype=x_init.dtype, chunks=chunk_2D_x)
        
        # Keep track of how many rows have been logged
        num_logged_iterations_hdf5 = 0

    except Exception as e_setup:
        print(f"FATAL: Could not set up HDF5 logging: {e_setup}")
        if 'hf_log_file' in locals() and hf_log_file:
            hf_log_file.close()
        raise

    # 7. Initialization
    x = x_init.copy()
    TOL = 1e-3
    optimal = False
    iteration_count = 0 # Initialize iteration counter
    rho = 0.01

    print("\n--- Starting Benders Decomposition Loop ---")

    for t in range(100):
        iteration_count += 1

        # 1. Solve Master Problem
        master_problem.set_regularization_strength(rho)
        master_problem.set_regularization_center(x)
        x_next, master_obj, code = master_problem.solve() # Renamed obj to master_obj for clarity

        print(f"master: dist_moved = {scipy.linalg.norm(x_next - x)}")

        x = x_next.copy()
        
        if code != 2:
            print(f"  Iter {iteration_count}: Error solving master problem. Gurobi status code: {code}")
            break
        
        # Log master objective
        print(f"Iter {iteration_count}: Master Obj = {master_obj:.4f}")

        # 2. Warmstart: Calculate initial cut based on existing duals
        time_start = time.time()
        print("Starting ArgmaxOperation ...", end="")
        alpha_pre, beta_pre, best_k_index = argmax_op.calculate_cut(x)
        argmax_cut_height = alpha_pre + beta_pre @ x
        time_end = time.time()
        print(f"ArgmaxCutHeight = {argmax_cut_height:.4f}, ArgmaxTime = {time_end - time_start:.4f}s")

        # Log unique number of best_k_index from the first argmax_op.calculate_cut
        if best_k_index is not None:
            num_unique_best_k = len(np.unique(best_k_index))
            print(f", Unique Warmstart Duals Used = {num_unique_best_k}", end="")
        else:
            # This case should ideally not happen if best_k_index is guaranteed to exist
            print(", Warmstart best_k_index is None", end="")


        vbasis_batch, cbasis_batch = argmax_op.get_basis(best_k_index)
        
        # 3. Subproblem Solver
        print("Solving Subproblems ...", end="")
        time_start = time.time()
        obj_all, y_all, pi_all, rc_all, vbasis_out, cbasis_out = parallel_worker.solve_batch(x, short_delta_r, vbasis_batch, cbasis_batch)
        time_end = time.time()
        print(f"SubproblemTime = {time_end - time_start:.4f}s")

        # 4. Add newly found duals to ArgmaxOperation
        initial_pi_count_in_argmax = argmax_op.num_pi
        # Create an array of indices from 0 to NUM_SAMPLES_FOR_POOL - 1
        # Randomly choose 10000 indices without replacement
        all_indices = np.arange(NUM_SAMPLES_FOR_POOL)
        chosen_indices = np.random.choice(all_indices, size=10000, replace=False)
        # Loop through the chosen indices and add the corresponding samples
        for s in chosen_indices:
            pi = pi_all[s, :]
            rc = rc_all[s, :]
            vbasis = vbasis_out[s, :]
            cbasis = cbasis_out[s, :]
            argmax_op.add_pi(pi, rc, vbasis, cbasis)

        # Log current number of duals in ArgmaxOperation
        print(f", Total Duals in ArgmaxOp = {argmax_op.num_pi}", end="")

        # 5. Calculate the new cut
        print("Calculating new cut on CPU...", end="")
        time_start = time.time()

        pattern = reader.stochastic_rows_relative_indices
        mean_pi = np.mean(pi_all, axis=0)
        alpha_fixed_part = mean_pi @ reader.r_bar
        short_pi = pi_all[:, pattern]
        variable_part = np.sum(short_pi * short_delta_r, axis=1)
        alpha_variable_part = np.mean(variable_part, axis=0)
        alpha = alpha_fixed_part + alpha_variable_part
        beta = - mean_pi @ reader.C

        # 6. add cut if necessary
        current_cut_height = alpha + beta @ x
        current_master_height = master_problem.calculate_epigraph_value(x)

        print(f", CutHeight = {current_cut_height:.4f}, MasterEpiHeight = {current_master_height:.4f}", end="")
        time_end = time.time()
        print(f"CutTime = {time_end - time_start:.4f}s")

        if current_cut_height > current_master_height + TOL:
            # add cut
            master_problem.add_optimality_cut(beta, alpha)
            rho *= 1.01
            print(f" -> Cut Added because gap = {current_master_height - current_cut_height:.4e}, rho = {rho:.4e}", end="")
        else:
            # No cut added
            print(f" -> No Cut Added.", end="")
        
        # 7. Log iteration data
        try:
            iter_log_master_obj = master_obj
            iter_log_argmax_cut_height = argmax_cut_height
            iter_log_calculated_cut_height = current_cut_height
            iter_log_master_epigraph_height = current_master_height
            
            current_idx_hdf5 = num_logged_iterations_hdf5
            # Resize datasets
            iter_log_group["iteration_counts"].resize((current_idx_hdf5 + 1,))
            iter_log_group["master_obj_history"].resize((current_idx_hdf5 + 1,))
            iter_log_group["argmax_cut_height_history"].resize((current_idx_hdf5 + 1,))
            iter_log_group["calculated_cut_height_history"].resize((current_idx_hdf5 + 1,))
            iter_log_group["master_epigraph_height_history"].resize((current_idx_hdf5 + 1,))
            
            if x_dim > 0:
                iter_log_group["x_vector_history"].resize((current_idx_hdf5 + 1, x_dim))

            # Write data to the new row
            iter_log_group["iteration_counts"][current_idx_hdf5] = iteration_count
            iter_log_group["master_obj_history"][current_idx_hdf5] = iter_log_master_obj
            iter_log_group["argmax_cut_height_history"][current_idx_hdf5] = iter_log_argmax_cut_height
            iter_log_group["calculated_cut_height_history"][current_idx_hdf5] = iter_log_calculated_cut_height
            iter_log_group["master_epigraph_height_history"][current_idx_hdf5] = iter_log_master_epigraph_height
            
            if x_dim > 0:
                iter_log_group["x_vector_history"][current_idx_hdf5, :] = x # Log the current x
            
            num_logged_iterations_hdf5 += 1
            hf_log_file.flush() # Ensure data is written to disk after each iteration

        except Exception as e_log_iter:
            print(f"    Logging Error during iteration {iteration_count} append to HDF5: {e_log_iter}")
            # Continue to next iteration if logging fails for one iteration

