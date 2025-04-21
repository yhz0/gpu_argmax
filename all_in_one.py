# all_in_one.py (Block Matrix Version with Optimized COO Construction & 52 Threads)

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import scipy.sparse as sp # Import SciPy sparse
from typing import Optional, List, Dict, Any
import time # For timing steps

# Ensure the smps_reader.py file is in the same directory
# or accessible via the Python path.
try:
    # Use the updated SMPSReader from the canvas artifact "smps_reader_py"
    from smps_reader import SMPSReader
except ImportError:
    print("ERROR: Could not import SMPSReader.")
    print("Ensure 'smps_reader.py' (with batch sampling) is in the same directory or Python path.")
    exit(1)

# --- Configuration ---
BASE_DIR = "smps_data"
PROBLEM_NAME = "ssn"
FILE_DIR = os.path.join(BASE_DIR, PROBLEM_NAME)
CORE_FILENAME = f"{PROBLEM_NAME}.mps"
TIME_FILENAME = f"{PROBLEM_NAME}.tim"
STO_FILENAME = f"{PROBLEM_NAME}.sto"

# Handle path finding relative to script location or CWD
script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
CORE_FILEPATH = os.path.join(script_dir, FILE_DIR, CORE_FILENAME)
TIME_FILEPATH = os.path.join(script_dir, FILE_DIR, TIME_FILENAME)
STO_FILEPATH = os.path.join(script_dir, FILE_DIR, STO_FILENAME)

NUM_SCENARIOS = 1000 # Keep the large number for testing speed
RANDOM_SEED = 42
NUM_THREADS = 104 # Set desired number of threads

# --- Helper Function for Senses ---
def convert_senses_to_gurobi(sense_chars: np.ndarray) -> np.ndarray:
    """Converts numpy array of sense characters ('<', '=', '>') to Gurobi constants."""
    sense_map = {'<': GRB.LESS_EQUAL, '=': GRB.EQUAL, '>': GRB.GREATER_EQUAL}
    if sense_chars.dtype.kind == 'U':
        # Important: Use object dtype for Gurobi constants
        return np.array([sense_map.get(s, GRB.LESS_EQUAL) for s in sense_chars], dtype=object)
    else:
        # Handle case where input might already be processed or is empty
        if sense_chars.size == 0:
             return np.array([], dtype=object)
        raise TypeError(f"Expected numpy array of strings for senses, got dtype {sense_chars.dtype}")

# --- Main Script ---
if __name__ == "__main__":

    overall_start_time = time.time()

    print("--- Stochastic Programming SAA Builder (Block Matrix Version with Optimized COO Construction) ---")
    print(f"Problem: {PROBLEM_NAME}")
    print(f"Core File: {os.path.abspath(CORE_FILEPATH)}")
    print(f"Time File: {os.path.abspath(TIME_FILEPATH)}")
    print(f"Sto File: {os.path.abspath(STO_FILEPATH)}")
    print(f"Number of SAA Scenarios: {NUM_SCENARIOS}")
    print(f"Number of Threads: {NUM_THREADS}") # Print thread count
    print(f"Random Seed: {RANDOM_SEED}")
    print("-" * 40)


    # 1. Read SMPS Data using SMPSReader
    print("Step 1: Reading SMPS files...")
    step1_start_time = time.time()
    reader: Optional[SMPSReader] = None
    try:
        # Check file existence before instantiation
        if not os.path.exists(CORE_FILEPATH): raise FileNotFoundError(f"Core file not found: {CORE_FILEPATH}")
        if not os.path.exists(TIME_FILEPATH): raise FileNotFoundError(f"Time file not found: {TIME_FILEPATH}")
        if not os.path.exists(STO_FILEPATH): raise FileNotFoundError(f"Sto file not found: {STO_FILEPATH}")

        reader = SMPSReader(
            core_file=CORE_FILEPATH,
            time_file=TIME_FILEPATH,
            sto_file=STO_FILEPATH
        )
        reader.load_and_extract() # This calls validation and prepares for sampling
        step1_end_time = time.time()
        print(f"SMPS data loaded and extracted successfully. (Time: {step1_end_time - step1_start_time:.2f}s)")
    except FileNotFoundError as e:
         print(f"ERROR: {e}")
         exit(1)
    except Exception as e:
        print(f"ERROR during SMPS reading/extraction: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Check if reader or essential data is None after load_and_extract attempt
    if reader is None or reader.r_bar is None:
         print("ERROR: Data extraction did not complete successfully (reader or r_bar is None). Aborting.")
         exit(1)

    # Extract components (ensure they are sparse where appropriate)
    c = reader.c
    A = reader.A.tocsr() if reader.A is not None else None # Ensure CSR
    b = reader.b
    sense1_char = reader.sense1
    lb_x = reader.lb_x
    ub_x = reader.ub_x

    d = reader.d
    C = reader.C.tocsr() if reader.C is not None else None # Ensure CSR
    D = reader.D.tocsr() if reader.D is not None else None # Ensure CSR
    r_bar = reader.r_bar
    sense2_char = reader.sense2
    lb_y = reader.lb_y
    ub_y = reader.ub_y

    stochastic_rows_relative_indices = reader.stochastic_rows_relative_indices
    num_stochastic_elements = len(reader.stochastic_rows_indices_orig)

    num_x = len(c) if c is not None else 0
    num_y = len(d) if d is not None else 0
    num_cons1 = A.shape[0] if A is not None else 0
    num_cons2 = len(reader.row2_indices)

    print(f"Problem Dimensions:")
    print(f"  Stage 1 Vars (x): {num_x}")
    print(f"  Stage 2 Vars (y): {num_y}")
    print(f"  Stage 1 Constraints: {num_cons1}")
    print(f"  Stage 2 Constraints: {num_cons2}")
    print(f"  Validated Stochastic RHS Elements: {num_stochastic_elements}")
    print("-" * 40)


    # 2. Generate Scenarios using BATCH Sampling
    print(f"Step 2: Generating {NUM_SCENARIOS} scenario RHS vectors (Batch approach)...")
    step2_start_time = time.time()
    np.random.seed(RANDOM_SEED) # Set seed for reproducibility
    all_scenario_rhs_matrix = None # Initialize

    if num_stochastic_elements > 0 and r_bar is not None:
        print("  Calling batch sampler...")
        batch_stochastic_parts = reader.sample_stochastic_rhs_batch(NUM_SCENARIOS)
        print(f"  Batch sampling complete. Shape: {batch_stochastic_parts.shape}")

        print("  Constructing full RHS vectors (vectorized)...")
        if r_bar.ndim != 1: raise ValueError(f"r_bar should be 1D, but has shape {r_bar.shape}")
        if len(r_bar) != num_cons2: raise ValueError(f"r_bar length {len(r_bar)} does not match num_cons2 {num_cons2}")

        all_scenario_rhs_matrix = np.tile(r_bar, (NUM_SCENARIOS, 1))

        if len(stochastic_rows_relative_indices) > 0:
            if np.any(stochastic_rows_relative_indices >= all_scenario_rhs_matrix.shape[1]):
                 raise IndexError(f"Stochastic relative indices out of bounds for r_bar columns ({all_scenario_rhs_matrix.shape[1]})")
            if batch_stochastic_parts.shape[1] != len(stochastic_rows_relative_indices):
                 raise ValueError(f"Mismatch between batch sample columns ({batch_stochastic_parts.shape[1]}) and relative indices count ({len(stochastic_rows_relative_indices)})")
            all_scenario_rhs_matrix[:, stochastic_rows_relative_indices] = batch_stochastic_parts
        else:
             print("  Note: No stochastic elements to place into RHS matrix.")

    elif r_bar is not None: # Case with no stochastic elements
         print("  No stochastic elements found. Using deterministic r_bar for all scenarios.")
         all_scenario_rhs_matrix = np.tile(r_bar, (NUM_SCENARIOS, 1))
    elif num_cons2 > 0: # Error case: stage 2 constraints but no r_bar
         print("ERROR: Stage 2 constraints exist but r_bar is None. Cannot generate scenarios.")
         exit(1)
    else: # Case with no stage 2 constraints
         print("  No stage 2 constraints found.")
         # all_scenario_rhs_matrix remains None

    step2_end_time = time.time()
    print(f"Scenario generation complete. (Time: {step2_end_time - step2_start_time:.2f}s)")
    print("-" * 40)


    # 3. Build the SAA Gurobi Model using Block Matrix (Optimized COO Construction)
    print("Step 3: Building the SAA Gurobi model (Optimized COO Construction)...")
    step3_start_time = time.time()
    try:
        with gp.Env(empty=True) as env:
            env.start()
            # --- Set Gurobi Parameters ---
            env.setParam('Threads', NUM_THREADS) # Set desired number of threads

            saa_model = gp.Model(f"SAA_{PROBLEM_NAME}_{NUM_SCENARIOS}scen_OptCOO", env=env) # Updated name

            # --- Define Dimensions ---
            N = NUM_SCENARIOS
            total_vars = num_x + N * num_y
            total_cons = num_cons1 + N * num_cons2

            # --- Prepare Combined Variable Data ---
            print("  Preparing combined variable data (bounds, objective)...")
            # (Bounds and Objective preparation code remains the same)
            lb_x_grb = np.where(np.isneginf(lb_x), -GRB.INFINITY, lb_x) if num_x > 0 else np.array([])
            ub_x_grb = np.where(np.isposinf(ub_x), GRB.INFINITY, ub_x) if num_x > 0 else np.array([])
            lb_y_grb = np.where(np.isneginf(lb_y), -GRB.INFINITY, lb_y) if num_y > 0 else np.array([])
            ub_y_grb = np.where(np.isposinf(ub_y), GRB.INFINITY, ub_y) if num_y > 0 else np.array([])
            lb_list = ([lb_x_grb] if num_x > 0 else []) + ([lb_y_grb] * N if num_y > 0 else [])
            ub_list = ([ub_x_grb] if num_x > 0 else []) + ([ub_y_grb] * N if num_y > 0 else [])
            lb_combined = np.concatenate(lb_list) if lb_list else np.array([])
            ub_combined = np.concatenate(ub_list) if ub_list else np.array([])
            obj_y_scaled = (d / N) if (d is not None and d.size > 0 and N > 0) else np.array([])
            obj_list = ([c] if (c is not None and c.size > 0) else []) + ([obj_y_scaled] * N if obj_y_scaled.size > 0 else [])
            obj_combined = np.concatenate(obj_list) if obj_list else np.array([])

            # --- Add All Variables at Once ---
            print("  Adding all variables...")
            add_vars_start_time = time.time()
            if total_vars > 0:
                all_vars = saa_model.addMVar(shape=total_vars, lb=lb_combined, ub=ub_combined, obj=obj_combined, name="vars")
            else:
                all_vars = None
                print("Warning: No variables in the model.")
            add_vars_end_time = time.time()
            print(f"    Variables added. (Time: {add_vars_end_time - add_vars_start_time:.2f}s)")


            # --- Build Block Matrix (SAA_matrix) using Optimized Direct COO Construction ---
            print("  Building the block constraint matrix via optimized COO...")
            build_matrix_start_time = time.time()
            if total_cons > 0 and total_vars > 0:
                # Initialize lists for final COO components
                rows_list, cols_list, data_list = [], [], []

                # 1. Add A block
                if A is not None and A.nnz > 0:
                    A_coo = A.tocoo()
                    rows_list.append(A_coo.row)
                    cols_list.append(A_coo.col)
                    data_list.append(A_coo.data)
                    print(f"    Added A block (NNZ: {A.nnz})")

                # 2. Prepare C block data (if C exists)
                C_rows_all, C_cols_all, C_data_all = None, None, None
                if C is not None and C.nnz > 0 and N > 0:
                    C_coo = C.tocoo()
                    C_nnz = C.nnz
                    C_data_all = np.tile(C_coo.data, N)
                    row_offsets_C = np.repeat(num_cons1 + np.arange(N, dtype=np.int32) * num_cons2, C_nnz)
                    C_rows_all = np.tile(C_coo.row, N) + row_offsets_C
                    C_cols_all = np.tile(C_coo.col, N)
                    rows_list.append(C_rows_all)
                    cols_list.append(C_cols_all)
                    data_list.append(C_data_all)
                    print(f"    Prepared C blocks (NNZ per block: {C_nnz}, Total NNZ: {C_data_all.size})")


                # 3. Prepare D block data (if D exists)
                D_rows_all, D_cols_all, D_data_all = None, None, None
                if D is not None and D.nnz > 0 and N > 0:
                    D_coo = D.tocoo()
                    D_nnz = D.nnz
                    D_data_all = np.tile(D_coo.data, N)
                    row_offsets_D = np.repeat(num_cons1 + np.arange(N, dtype=np.int32) * num_cons2, D_nnz)
                    D_rows_all = np.tile(D_coo.row, N) + row_offsets_D
                    col_offsets_D = np.repeat(num_x + np.arange(N, dtype=np.int32) * num_y, D_nnz)
                    D_cols_all = np.tile(D_coo.col, N) + col_offsets_D
                    rows_list.append(D_rows_all)
                    cols_list.append(D_cols_all)
                    data_list.append(D_data_all)
                    print(f"    Prepared D blocks (NNZ per block: {D_nnz}, Total NNZ: {D_data_all.size})")

                # Concatenate all parts
                print("    Concatenating final COO arrays...")
                final_rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=np.int32)
                final_cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=np.int32)
                final_data = np.concatenate(data_list) if data_list else np.array([], dtype=np.float64)

                # Create COO matrix
                print("    Creating final COO matrix...")
                saa_coo = sp.coo_matrix((final_data, (final_rows, final_cols)),
                                        shape=(total_cons, total_vars))

                # Convert to CSR matrix
                print("    Converting COO to CSR matrix...")
                convert_start_time = time.time()
                SAA_matrix = saa_coo.tocsr()
                convert_end_time = time.time()
                print(f"    CSR conversion time: {convert_end_time - convert_start_time:.2f}s")

                build_matrix_end_time = time.time()
                print(f"    Block matrix constructed via Optimized COO. Shape: {SAA_matrix.shape}, NNZ: {SAA_matrix.nnz} (Build Time: {build_matrix_end_time - build_matrix_start_time:.2f}s)")

            elif total_cons > 0:
                 print("Warning: Constraints exist, but no variables. SAA Matrix not built.")
                 SAA_matrix = None
            else:
                 print("  No constraints in the model. SAA Matrix not built.")
                 SAA_matrix = None


            # --- Prepare Combined RHS and Sense Vectors ---
            # (This part remains the same)
            print("  Preparing combined RHS and Sense vectors...")
            prep_rhs_sense_start_time = time.time()
            if total_cons > 0:
                if all_scenario_rhs_matrix is not None:
                    scenario_rhs_list_from_matrix = [row for row in all_scenario_rhs_matrix]
                else:
                    scenario_rhs_list_from_matrix = []
                b_safe = b if (b is not None and b.ndim == 1 and num_cons1 > 0) else np.array([])
                rhs_list = ([b_safe] if b_safe.size > 0 else []) + scenario_rhs_list_from_matrix
                if not rhs_list: SAA_rhs = np.array([])
                else:
                    expected_len_b = num_cons1; expected_len_r = num_cons2; valid_rhs = True
                    if b_safe.size > 0 and len(b_safe) != expected_len_b: valid_rhs = False
                    for i, r_s in enumerate(scenario_rhs_list_from_matrix):
                         if len(r_s) != expected_len_r: valid_rhs = False; break
                    if not valid_rhs: raise ValueError("Inconsistent shapes in RHS list.")
                    try: SAA_rhs = np.concatenate(rhs_list)
                    except ValueError as e: print(f"ERROR concatenating RHS: {e}."); raise e
                sense1_grb = convert_senses_to_gurobi(sense1_char if sense1_char is not None else np.array([]))
                sense2_grb = convert_senses_to_gurobi(sense2_char if sense2_char is not None else np.array([]))
                sense1_safe = sense1_grb if (len(sense1_grb) == num_cons1) else np.array([], dtype=object)
                sense2_safe = sense2_grb if (len(sense2_grb) == num_cons2) else np.array([], dtype=object)
                sense_list = ([sense1_safe] if sense1_safe.size > 0 else []) + ([sense2_safe] * N if sense2_safe.size > 0 else [])
                if not sense_list: SAA_sense = np.array([], dtype=object)
                else:
                    try: SAA_sense = np.concatenate(sense_list)
                    except ValueError as e: print(f"ERROR concatenating Senses: {e}."); raise e
                if SAA_matrix is not None and SAA_rhs.shape[0] != total_cons: print(f"ERROR: RHS length mismatch"); exit(1)
                if SAA_matrix is not None and SAA_sense.shape[0] != total_cons: print(f"ERROR: Sense length mismatch"); exit(1)
            prep_rhs_sense_end_time = time.time()
            print(f"    RHS/Sense prepared. (Time: {prep_rhs_sense_end_time - prep_rhs_sense_start_time:.2f}s)")


            # --- Add Constraints using Block Matrix ---
            print("  Adding block constraints to model...")
            add_constr_start_time = time.time()
            if SAA_matrix is not None and all_vars is not None and total_cons > 0:
                try:
                    if SAA_matrix.shape[0] != SAA_rhs.shape[0] or SAA_matrix.shape[0] != SAA_sense.shape[0]: raise ValueError("Mismatch between matrix rows, RHS length, and Sense length.")
                    if SAA_matrix.shape[1] != all_vars.shape[0]: raise ValueError("Mismatch between matrix columns and number of variables.")
                    saa_model.addMConstr(SAA_matrix, all_vars, SAA_sense, SAA_rhs, name="SAA_constraints")
                    add_constr_end_time = time.time()
                    print(f"  Block constraints added successfully. (Time: {add_constr_end_time - add_constr_start_time:.2f}s)")
                except gp.GurobiError as e: print(f"ERROR during addMConstr: {e}"); raise e
                except Exception as e: print(f"Non-Gurobi ERROR during addMConstr: {e}"); raise e
            elif total_cons > 0: print("Skipping addMConstr because Matrix or Variables are missing.")
            else: print("No constraints to add.")


            # Set objective sense
            saa_model.ModelSense = GRB.MINIMIZE
            step3_end_time = time.time()
            print(f"SAA model structure built. (Total Step 3 Time: {step3_end_time - step3_start_time:.2f}s)")
            print("-" * 40)

            # 4. Solve the SAA Model
            print("Step 4: Solving the SAA model...")
            step4_start_time = time.time()
            saa_model.optimize()
            step4_end_time = time.time()
            print(f"Optimization finished. (Time: {step4_end_time - step4_start_time:.2f}s)")
            print("-" * 40)

            # 5. Output Results
            print("Step 5: Results")
            # (Result output code remains the same)
            status = saa_model.Status
            if status == GRB.OPTIMAL:
                print("SAA Problem Solved to Optimality.")
                print(f"Optimal Objective Value: {saa_model.ObjVal:.6f}")
                if num_x > 0 and all_vars is not None:
                    print("Stage 1 Decision Variables (x):")
                    try:
                        x_values = all_vars.X[0:num_x]
                        for i in range(min(num_x, 10)):
                            orig_gurobi_idx = reader.x_indices[i]
                            var_name = reader.index_to_var_name.get(orig_gurobi_idx, f"x_idx_{orig_gurobi_idx}")
                            print(f"  {var_name}: {x_values[i]:.6f}")
                        if num_x > 10: print("  ...")
                    except (gp.GurobiError, AttributeError) as e:
                         print(f"Could not retrieve variable values: {e}")
                else: print("No stage 1 variables to display.")
            elif status == GRB.INFEASIBLE: print("Model is Infeasible.")
            elif status == GRB.UNBOUNDED: print("Model is Unbounded.")
            else: print(f"Optimization finished with status code: {status}")


    except gp.GurobiError as e:
        print(f"ERROR: Gurobi error occurred - Code {e.errno}: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred - {e}")
        import traceback
        traceback.print_exc()

    overall_end_time = time.time()
    print("-" * 40)
    print(f"Script finished. (Total Time: {overall_end_time - overall_start_time:.2f}s)")
