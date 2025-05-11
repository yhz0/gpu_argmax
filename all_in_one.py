# all_in_one.py

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import scipy.sparse as sp # Import SciPy sparse
from typing import Optional, List, Dict, Any, Tuple
import time # For timing steps
import h5py # Import HDF5 library

# Ensure the smps_reader.py file is in the same directory
# or accessible via the Python path.
try:
    # Use the updated SMPSReader from the canvas artifact "smps_reader_py"
    from smps_reader import SMPSReader
except ImportError:
    print("ERROR: Could not import SMPSReader.")
    print("Ensure 'smps_reader.py' (with batch sampling) is in the same directory or Python path.")
    exit(1)

# --- Helper Function (can be outside the class) ---
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

# --- SAA Builder Class ---
class SAABuilder:
    """
    Builds and optionally solves a Sample Average Approximation (SAA) model
    for a two-stage stochastic linear program defined by SMPS files.

    Uses SMPSReader to parse input files and constructs the SAA extensive
    form using a block matrix approach with optimized COO construction.

    Can save generated scenarios, optimal basis, primal solution (x, y_s),
    and second-stage dual multipliers to an HDF5 file.
    """

    def __init__(self, core_filepath: str, time_filepath: str, sto_filepath: str,
                 num_scenarios: int, random_seed: int = 42, num_threads: Optional[int] = None):
        """
        Initializes the SAABuilder.

        Args:
            core_filepath (str): Path to the .cor or .mps file.
            time_filepath (str): Path to the .tim file.
            sto_filepath (str): Path to the .sto file.
            num_scenarios (int): The number of scenarios for the SAA problem.
            random_seed (int): Seed for the random number generator for scenario sampling.
            num_threads (Optional[int]): Number of threads for Gurobi to use. Defaults to Gurobi's default.
        """
        self.core_filepath = core_filepath
        self.time_filepath = time_filepath
        self.sto_filepath = sto_filepath
        self.num_scenarios = num_scenarios
        self.random_seed = random_seed
        self.num_threads = num_threads

        # Attributes populated by methods
        self.reader: Optional[SMPSReader] = None
        self.all_scenario_rhs_matrix: Optional[np.ndarray] = None
        self.saa_model: Optional[gp.Model] = None
        self.all_vars: Optional[gp.MVar] = None # Combined [x, y1, ..., yN] variables
        self.saa_constraints: Optional[gp.MConstr] = None # To store the added constraints for dual access
        self.solution_status: Optional[int] = None
        self.optimal_objective: Optional[float] = None

        # Data to be saved
        self.batch_stochastic_parts: Optional[np.ndarray] = None # Stores only the stochastic RHS parts generated
        self.stage1_solution: Optional[np.ndarray] = None # Primal x
        self.stage2_solution_all: Optional[np.ndarray] = None # Primal y_s for all s (shape N x num_y)
        self.stage2_duals_all: Optional[np.ndarray] = None # Duals pi_s for all s (shape N x num_cons2)
        self.basis_info: Optional[Dict[str, np.ndarray]] = None # Stores basis arrays

        # Extracted dimensions (populated after loading)
        self.num_x = 0
        self.num_y = 0
        self.num_cons1 = 0
        self.num_cons2 = 0
        self.num_stochastic_elements = 0

        # Timing info
        self.timing = {}

        print(f"Initialized SAABuilder for {os.path.basename(core_filepath)} with {num_scenarios} scenarios.")

    def _load_smps_data(self):
        """Loads and extracts data from SMPS files using SMPSReader."""
        print("Step 1: Reading SMPS files...")
        step1_start_time = time.time()
        try:
            # Check file existence before instantiation
            if not os.path.exists(self.core_filepath): raise FileNotFoundError(f"Core file not found: {self.core_filepath}")
            if not os.path.exists(self.time_filepath): raise FileNotFoundError(f"Time file not found: {self.time_filepath}")
            if not os.path.exists(self.sto_filepath): raise FileNotFoundError(f"Sto file not found: {self.sto_filepath}")

            self.reader = SMPSReader(
                core_file=self.core_filepath,
                time_file=self.time_filepath,
                sto_file=self.sto_filepath
            )
            self.reader.load_and_extract() # This calls validation and prepares for sampling

            # Populate dimensions
            self.num_x = len(self.reader.c) if self.reader.c is not None else 0
            self.num_y = len(self.reader.d) if self.reader.d is not None else 0
            self.num_cons1 = self.reader.A.shape[0] if self.reader.A is not None else 0
            self.num_cons2 = len(self.reader.row2_indices)
            # Ensure num_stochastic_elements calculation is robust
            self.num_stochastic_elements = 0
            if self.reader.stochastic_rows_indices_orig is not None:
                 self.num_stochastic_elements = len(self.reader.stochastic_rows_indices_orig)


            step1_end_time = time.time()
            self.timing['load_smps'] = step1_end_time - step1_start_time
            print(f"SMPS data loaded and extracted successfully. (Time: {self.timing['load_smps']:.2f}s)")
            print(f"  Dims: x={self.num_x}, y={self.num_y}, cons1={self.num_cons1}, cons2={self.num_cons2}, stoch_rhs={self.num_stochastic_elements}")

        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            raise # Re-raise to stop execution
        except Exception as e:
            print(f"ERROR during SMPS reading/extraction: {e}")
            import traceback
            traceback.print_exc()
            raise # Re-raise to stop execution

    def _generate_scenarios(self):
        """Generates scenario RHS vectors using batch sampling."""
        if self.reader is None or self.reader.r_bar is None:
             # Handle case where r_bar might be legitimately None (no second stage)
             if self.reader is not None and self.num_cons2 == 0:
                 print("Step 2: No second stage constraints (num_cons2=0), skipping scenario RHS generation.")
                 self.all_scenario_rhs_matrix = None # No RHS needed for second stage
                 self.batch_stochastic_parts = np.array([]).reshape(self.num_scenarios, 0) # Store empty array
                 self.timing['generate_scenarios'] = 0.0
                 return
             else:
                 raise RuntimeError("SMPS data must be loaded before generating scenarios, or r_bar is unexpectedly None.")


        print(f"Step 2: Generating {self.num_scenarios} scenario RHS vectors (Batch approach)...")
        step2_start_time = time.time()
        np.random.seed(self.random_seed) # Set seed for reproducibility

        r_bar = self.reader.r_bar
        stochastic_rows_relative_indices = self.reader.stochastic_rows_relative_indices

        if self.num_stochastic_elements > 0:
            # --- Generate Stochastic Parts ---
            # This is the raw sampled data for the stochastic elements ONLY
            self.batch_stochastic_parts = self.reader.sample_stochastic_rhs_batch(self.num_scenarios)
            # --------------------------------

            if r_bar.ndim != 1: raise ValueError(f"r_bar should be 1D, but has shape {r_bar.shape}")
            if len(r_bar) != self.num_cons2: raise ValueError(f"r_bar length {len(r_bar)} does not match num_cons2 {self.num_cons2}")

            # Initialize full RHS matrix with deterministic part
            self.all_scenario_rhs_matrix = np.tile(r_bar, (self.num_scenarios, 1))

            # Place stochastic parts into the full RHS matrix
            if len(stochastic_rows_relative_indices) > 0:
                if np.any(stochastic_rows_relative_indices >= self.all_scenario_rhs_matrix.shape[1]):
                    raise IndexError(f"Stochastic relative indices out of bounds for r_bar columns ({self.all_scenario_rhs_matrix.shape[1]})")
                if self.batch_stochastic_parts.shape[1] != len(stochastic_rows_relative_indices):
                     raise ValueError(f"Mismatch between batch sample columns ({self.batch_stochastic_parts.shape[1]}) and relative indices count ({len(stochastic_rows_relative_indices)})")

                # Ensure batch_stochastic_parts has the correct number of rows
                if self.batch_stochastic_parts.shape[0] != self.num_scenarios:
                    raise ValueError(f"Generated stochastic parts have {self.batch_stochastic_parts.shape[0]} rows, expected {self.num_scenarios}")

                self.all_scenario_rhs_matrix[:, stochastic_rows_relative_indices] = self.batch_stochastic_parts

        elif r_bar is not None: # Case with no stochastic elements, but r_bar exists
            print("  No stochastic elements found. Using deterministic r_bar for all scenarios.")
            self.all_scenario_rhs_matrix = np.tile(r_bar, (self.num_scenarios, 1))
            self.batch_stochastic_parts = np.array([]).reshape(self.num_scenarios, 0) # Store empty array
        elif self.num_cons2 > 0: # Error case: stage 2 constraints but no r_bar
            raise ValueError("Stage 2 constraints exist but r_bar is None. Cannot generate scenarios.")
        else: # Case with no stage 2 constraints (already handled at the beginning)
            pass # Should not reach here if handled correctly above

        step2_end_time = time.time()
        self.timing['generate_scenarios'] = step2_end_time - step2_start_time
        num_generated = self.all_scenario_rhs_matrix.shape[0] if self.all_scenario_rhs_matrix is not None else 0
        stoch_part_shape = self.batch_stochastic_parts.shape if self.batch_stochastic_parts is not None else "N/A"
        print(f"Scenario generation complete. Generated {num_generated} RHS vectors. Stored stochastic parts shape: {stoch_part_shape}. (Time: {self.timing['generate_scenarios']:.2f}s)")


    def _build_saa_matrix_coo(self) -> Optional[sp.csr_matrix]:
        """Builds the SAA constraint block matrix using optimized COO construction."""
        if self.reader is None:
            raise RuntimeError("SMPS data must be loaded before building the SAA matrix.")

        build_matrix_start_time = time.time()
        N = self.num_scenarios
        total_vars = self.num_x + N * self.num_y
        total_cons = self.num_cons1 + N * self.num_cons2

        if total_cons == 0 or total_vars == 0:
            print("  No constraints or variables - skipping matrix construction.")
            return None

        # Extract matrices (ensure CSR format initially for consistency)
        A = self.reader.A.tocsr() if self.reader.A is not None else None
        C = self.reader.C.tocsr() if self.reader.C is not None else None # Should be T in standard notation, but using C as per code
        D = self.reader.D.tocsr() if self.reader.D is not None else None # Should be W in standard notation, but using D as per code

        # Initialize lists for final COO components
        rows_list, cols_list, data_list = [], [], []

        # 1. Add A block (First stage constraints: Ax <= b)
        if A is not None and A.nnz > 0:
            A_coo = A.tocoo()
            rows_list.append(A_coo.row)
            cols_list.append(A_coo.col)
            data_list.append(A_coo.data)

        # 2. Prepare C block data (Technology matrix T: Tx + Wy <= h_s -> represented as Cx + Dy <= r_s here)
        if C is not None and C.nnz > 0 and N > 0 and self.num_cons2 > 0:
            C_coo = C.tocoo()
            C_nnz = C.nnz
            C_data_all = np.tile(C_coo.data, N)
            row_offsets_C = np.repeat(self.num_cons1 + np.arange(N, dtype=np.int32) * self.num_cons2, C_nnz)
            C_rows_all = np.tile(C_coo.row, N) + row_offsets_C
            C_cols_all = np.tile(C_coo.col, N) # C maps x variables (cols 0 to num_x-1)
            rows_list.append(C_rows_all)
            cols_list.append(C_cols_all)
            data_list.append(C_data_all)

        # 3. Prepare D block data (Recourse matrix W: Tx + Wy <= h_s -> represented as Cx + Dy <= r_s here)
        if D is not None and D.nnz > 0 and N > 0 and self.num_cons2 > 0:
            D_coo = D.tocoo()
            D_nnz = D.nnz
            D_data_all = np.tile(D_coo.data, N)
            row_offsets_D = np.repeat(self.num_cons1 + np.arange(N, dtype=np.int32) * self.num_cons2, D_nnz) # Same row offsets as C
            D_rows_all = np.tile(D_coo.row, N) + row_offsets_D
            col_offsets_D = np.repeat(self.num_x + np.arange(N, dtype=np.int32) * self.num_y, D_nnz) # Column offsets for y_s variables
            D_cols_all = np.tile(D_coo.col, N) + col_offsets_D
            rows_list.append(D_rows_all)
            cols_list.append(D_cols_all)
            data_list.append(D_data_all)

        # Concatenate all parts
        final_rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=np.int32)
        final_cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=np.int32)
        final_data = np.concatenate(data_list) if data_list else np.array([], dtype=np.float64)

        # Create COO matrix
        saa_coo = sp.coo_matrix((final_data, (final_rows, final_cols)),
                                shape=(total_cons, total_vars))

        # Convert to CSR matrix (efficient format for Gurobi)
        SAA_matrix_csr = saa_coo.tocsr()
        build_matrix_end_time = time.time()
        self.timing['build_saa_matrix'] = build_matrix_end_time - build_matrix_start_time
        print(f"  Block matrix constructed via Optimized COO. Shape: {SAA_matrix_csr.shape}, NNZ: {SAA_matrix_csr.nnz} (Time: {self.timing['build_saa_matrix']:.2f}s)")

        return SAA_matrix_csr


    def _prepare_rhs_sense(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepares the combined RHS and Sense vectors for the SAA model."""
        if self.reader is None:
            raise RuntimeError("SMPS data must be loaded before preparing RHS/Sense.")

        prep_rhs_sense_start_time = time.time()
        N = self.num_scenarios
        total_cons = self.num_cons1 + N * self.num_cons2
        SAA_rhs, SAA_sense = None, None

        if total_cons > 0:
            # RHS
            b = self.reader.b # First stage RHS
            b_safe = b if (b is not None and b.ndim == 1 and self.num_cons1 > 0) else np.array([])

            # Use the generated all_scenario_rhs_matrix if it exists
            if self.all_scenario_rhs_matrix is not None and self.num_cons2 > 0:
                 # Check if the shape matches N x num_cons2
                if self.all_scenario_rhs_matrix.shape != (N, self.num_cons2):
                    raise ValueError(f"Shape mismatch for all_scenario_rhs_matrix: expected ({N}, {self.num_cons2}), got {self.all_scenario_rhs_matrix.shape}")
                # Flatten the scenario RHS matrix for concatenation
                scenario_rhs_flat = self.all_scenario_rhs_matrix.flatten()
            elif N > 0 and self.num_cons2 > 0: # Expected second stage RHS but wasn't generated
                 raise ValueError("Second stage constraints exist, but scenario RHS matrix is missing.")
            else: # No second stage constraints or N=0
                 scenario_rhs_flat = np.array([])


            rhs_list = [b_safe, scenario_rhs_flat]
            rhs_list_filtered = [arr for arr in rhs_list if arr.size > 0]


            if not rhs_list_filtered:
                 print("  Warning: RHS vector is empty.")
                 SAA_rhs = np.array([])
            else:
                try:
                    SAA_rhs = np.concatenate(rhs_list_filtered)
                    if len(SAA_rhs) != total_cons:
                        print(f"Warning: Concatenated RHS length {len(SAA_rhs)} != expected total constraints {total_cons}.")
                        # Decide if this is an error or just a warning based on problem structure
                        # raise ValueError("RHS length mismatch after concatenation.")
                except ValueError as e: print(f"ERROR concatenating RHS: {e}. Check component shapes."); raise e


            # Sense
            sense1_char = self.reader.sense1
            sense2_char = self.reader.sense2
            sense1_grb = convert_senses_to_gurobi(sense1_char if sense1_char is not None else np.array([]))
            sense2_grb = convert_senses_to_gurobi(sense2_char if sense2_char is not None else np.array([]))
            sense1_safe = sense1_grb if (len(sense1_grb) == self.num_cons1) else np.array([], dtype=object)
            sense2_safe = sense2_grb if (len(sense2_grb) == self.num_cons2) else np.array([], dtype=object)

            # Repeat sense2 N times if it exists
            sense2_repeated = np.tile(sense2_safe, N) if sense2_safe.size > 0 and N > 0 else np.array([], dtype=object)

            sense_list = [sense1_safe, sense2_repeated]
            sense_list_filtered = [arr for arr in sense_list if arr.size > 0]


            if not sense_list_filtered:
                 print("  Warning: Sense vector is empty.")
                 SAA_sense = np.array([], dtype=object)
            else:
                try:
                    SAA_sense = np.concatenate(sense_list_filtered)
                    if len(SAA_sense) != total_cons:
                         print(f"Warning: Concatenated Sense length {len(SAA_sense)} != expected total constraints {total_cons}.")
                         # raise ValueError("Sense length mismatch after concatenation.")
                except ValueError as e: print(f"ERROR concatenating Senses: {e}. Check component shapes."); raise e

        prep_rhs_sense_end_time = time.time()
        self.timing['prep_rhs_sense'] = prep_rhs_sense_end_time - prep_rhs_sense_start_time
        # print(f"   RHS/Sense prepared. (Time: {self.timing['prep_rhs_sense']:.2f}s)") # Less verbose

        return SAA_rhs, SAA_sense

    def build_model(self, suppress_gurobi_output: bool = False):
        """Builds the Gurobi SAA model object."""
        if self.reader is None:
            self._load_smps_data()
        # Generate scenarios if needed (e.g., if num_cons2 > 0 and they haven't been generated)
        if self.num_cons2 > 0 and self.all_scenario_rhs_matrix is None:
             self._generate_scenarios()
        elif self.num_cons2 == 0 and self.all_scenario_rhs_matrix is None: # Handle no second stage case explicitly
             self._generate_scenarios() # This should just set things to None/empty

        print("Step 3: Building the SAA Gurobi model...")
        step3_start_time = time.time()
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0 if suppress_gurobi_output else 1) # Control Gurobi logs
                env.start()
                # Set Gurobi Threads if specified
                if self.num_threads is not None:
                    env.setParam('Threads', self.num_threads)

                model_name = f"SAA_{os.path.basename(self.core_filepath).split('.')[0]}_{self.num_scenarios}scen_OptCOO"
                self.saa_model = gp.Model(model_name, env=env)

                # --- Define Dimensions ---
                N = self.num_scenarios
                total_vars = self.num_x + N * self.num_y
                total_cons = self.num_cons1 + N * self.num_cons2

                # --- Prepare Combined Variable Data ---
                # print("  Preparing combined variable data...") # Less verbose
                lb_x_grb = np.where(np.isneginf(self.reader.lb_x), -GRB.INFINITY, self.reader.lb_x) if self.num_x > 0 else np.array([])
                ub_x_grb = np.where(np.isposinf(self.reader.ub_x), GRB.INFINITY, self.reader.ub_x) if self.num_x > 0 else np.array([])
                lb_y_grb = np.where(np.isneginf(self.reader.lb_y), -GRB.INFINITY, self.reader.lb_y) if self.num_y > 0 else np.array([])
                ub_y_grb = np.where(np.isposinf(self.reader.ub_y), GRB.INFINITY, self.reader.ub_y) if self.num_y > 0 else np.array([])
                lb_list = ([lb_x_grb] if self.num_x > 0 else []) + ([lb_y_grb] * N if self.num_y > 0 else [])
                ub_list = ([ub_x_grb] if self.num_x > 0 else []) + ([ub_y_grb] * N if self.num_y > 0 else [])
                lb_combined = np.concatenate(lb_list) if lb_list else np.array([])
                ub_combined = np.concatenate(ub_list) if ub_list else np.array([])
                obj_y_scaled = (self.reader.d / N) if (self.reader.d is not None and self.reader.d.size > 0 and N > 0) else np.array([])
                obj_list = ([self.reader.c] if (self.reader.c is not None and self.reader.c.size > 0) else []) + ([obj_y_scaled] * N if obj_y_scaled.size > 0 else [])
                obj_combined = np.concatenate(obj_list) if obj_list else np.array([])

                # --- Add All Variables at Once ---
                # print("  Adding all variables...") # Less verbose
                add_vars_start_time = time.time()
                if total_vars > 0:
                    # Ensure obj_combined, lb_combined, ub_combined have the correct final length
                    if len(obj_combined) != total_vars: raise ValueError(f"Objective vector length mismatch: expected {total_vars}, got {len(obj_combined)}")
                    if len(lb_combined) != total_vars: raise ValueError(f"Lower bound vector length mismatch: expected {total_vars}, got {len(lb_combined)}")
                    if len(ub_combined) != total_vars: raise ValueError(f"Upper bound vector length mismatch: expected {total_vars}, got {len(ub_combined)}")

                    self.all_vars = self.saa_model.addMVar(shape=total_vars, lb=lb_combined, ub=ub_combined, obj=obj_combined, name="vars")
                else:
                    self.all_vars = None
                    print("Warning: No variables in the model.")
                add_vars_end_time = time.time()
                self.timing['add_variables'] = add_vars_end_time - add_vars_start_time
                # print(f"   Variables added. (Time: {self.timing['add_variables']:.2f}s)") # Less verbose

                # --- Build Block Matrix (SAA_matrix) ---
                SAA_matrix = self._build_saa_matrix_coo() # Calls internal method

                # --- Prepare Combined RHS and Sense Vectors ---
                SAA_rhs, SAA_sense = self._prepare_rhs_sense() # Calls internal method

                # --- Add Constraints using Block Matrix ---
                print("  Adding block constraints to model...")
                add_constr_start_time = time.time()
                if SAA_matrix is not None and self.all_vars is not None and total_cons > 0:
                    # Final check before adding constraints
                    if SAA_rhs is None or SAA_sense is None:
                        raise RuntimeError("RHS or Sense vector is None, cannot add constraints.")
                    if SAA_matrix.shape[0] != SAA_rhs.shape[0]: raise ValueError(f"Mismatch between matrix rows ({SAA_matrix.shape[0]}) and RHS length ({SAA_rhs.shape[0]})")
                    if SAA_matrix.shape[0] != SAA_sense.shape[0]: raise ValueError(f"Mismatch between matrix rows ({SAA_matrix.shape[0]}) and Sense length ({SAA_sense.shape[0]})")
                    if SAA_matrix.shape[1] != self.all_vars.shape[0]: raise ValueError(f"Mismatch between matrix columns ({SAA_matrix.shape[1]}) and number of variables ({self.all_vars.shape[0]})")

                    # Store the MConstr object to get duals later
                    self.saa_constraints = self.saa_model.addMConstr(SAA_matrix, self.all_vars, SAA_sense, SAA_rhs, name="SAA_constraints")
                    #-----------------------------------------------------

                    add_constr_end_time = time.time()
                    self.timing['add_constraints'] = add_constr_end_time - add_constr_start_time
                    print(f"  Block constraints added successfully. (Time: {self.timing['add_constraints']:.2f}s)")
                elif total_cons > 0:
                    print("Warning: Skipping addMConstr because Matrix or Variables are missing.")
                    self.saa_constraints = None
                else:
                    print("  No constraints to add.")
                    self.saa_constraints = None


                # Set objective sense
                self.saa_model.ModelSense = GRB.MINIMIZE
                step3_end_time = time.time()
                self.timing['build_model_total'] = step3_end_time - step3_start_time
                print(f"SAA model structure built. (Total Step 3 Time: {self.timing['build_model_total']:.2f}s)")

        except gp.GurobiError as e:
            print(f"ERROR: Gurobi error occurred during model building - Code {e.errno}: {e}")
            raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during model building - {e}")
            import traceback
            traceback.print_exc()
            raise

    def solve(self):
        """Solves the constructed SAA Gurobi model and extracts results if optimal."""
        if self.saa_model is None:
            # Attempt to build the model first
            print("Model not built yet, attempting to build...")
            self.build_model()
            if self.saa_model is None: # Check again if build failed
                raise RuntimeError("SAA model could not be built successfully.")

        print("Step 4: Solving the SAA model...")
        step4_start_time = time.time()
        # Reset solution/dual/basis attributes before solve
        self.optimal_objective = None
        self.stage1_solution = None
        self.stage2_solution_all = None
        self.stage2_duals_all = None
        self.basis_info = None

        try:
            self.saa_model.optimize()
            self.solution_status = self.saa_model.Status

            if self.solution_status == GRB.OPTIMAL:
                print("  Optimal solution found.")
                self.optimal_objective = self.saa_model.ObjVal

                # Ensure solution values are available
                if self.saa_model.SolCount > 0 and self.all_vars is not None:
                    full_solution = self.all_vars.X
                    N = self.num_scenarios

                    # 1. Extract Stage 1 Solution (x)
                    if self.num_x > 0:
                        self.stage1_solution = full_solution[0:self.num_x]
                    else:
                        self.stage1_solution = np.array([])

                    # 2. Extract Stage 2 Solution (y_s for all s)
                    if self.num_y > 0 and N > 0:
                        start_idx_y = self.num_x
                        end_idx_y = start_idx_y + N * self.num_y
                        y_s_flat = full_solution[start_idx_y : end_idx_y]
                        self.stage2_solution_all = y_s_flat.reshape((N, self.num_y))
                    else:
                        # Consistent shape even if no y vars or scenarios
                        self.stage2_solution_all = np.array([]).reshape(N, self.num_y)


                    # 3. Extract Dual Multipliers for Stage 2 Constraints (pi_s)
                    if self.num_cons2 > 0 and N > 0 and self.saa_constraints is not None:
                        try:
                            all_duals = np.array(self.saa_constraints.Pi) * self.num_scenarios # Scale by number of scenarios
                            start_idx_pi = self.num_cons1
                            end_idx_pi = start_idx_pi + N * self.num_cons2
                            # Verify length before slicing
                            if len(all_duals) >= end_idx_pi:
                                duals_y_flat = all_duals[start_idx_pi : end_idx_pi]
                                self.stage2_duals_all = duals_y_flat.reshape((N, self.num_cons2))
                            else:
                                print(f"Warning: Dual vector length ({len(all_duals)}) is less than expected end index ({end_idx_pi}). Cannot extract stage 2 duals.")
                                self.stage2_duals_all = np.array([]).reshape(N, self.num_cons2) # Empty placeholder

                        except gp.GurobiError as e:
                             print(f"Warning: Gurobi error getting duals (Pi attribute): {e}. Duals not extracted.")
                             self.stage2_duals_all = np.array([]).reshape(N, self.num_cons2) # Empty placeholder
                        except AttributeError:
                             print(f"Warning: Could not get duals. 'Pi' attribute not available for MConstr (maybe model type?). Duals not extracted.")
                             self.stage2_duals_all = np.array([]).reshape(N, self.num_cons2) # Empty placeholder

                    else:
                         # Consistent shape even if no cons2 or scenarios or MConstr object
                         self.stage2_duals_all = np.array([]).reshape(N, self.num_cons2)

                else:
                    print("Warning: Optimal status reported but no solution found (SolCount=0) or variables missing.")
                    # Ensure attributes reflect lack of solution
                    self.stage1_solution = None
                    self.stage2_solution_all = None
                    self.stage2_duals_all = None

            else: # Not optimal
                print(f"  Optimization finished with status: {self.solution_status}")
                self.optimal_objective = None
                self.stage1_solution = None
                self.stage2_solution_all = None
                self.stage2_duals_all = None


        except gp.GurobiError as e:
            print(f"ERROR: Gurobi error occurred during optimization - Code {e.errno}: {e}")
            # Store status even if error occurred during solve (e.g., interrupted)
            if hasattr(self.saa_model, 'Status'):
                self.solution_status = self.saa_model.Status
            # Do not raise here, allow reporting/saving attempt if possible
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during optimization - {e}")
            # Do not raise here
        finally: # Ensure timing is recorded even if errors occur
            step4_end_time = time.time()
            self.timing['solve_model'] = step4_end_time - step4_start_time
            print(f"Optimization finished. (Time: {self.timing['solve_model']:.2f}s)")


    def report_results(self):
        """Prints a summary of the optimization results."""
        print("-" * 40)
        print("Step 5: Results Summary")
        if self.solution_status is None:
            print("Model has not been solved yet.")
            return

        status = self.solution_status
        status_map = {
            GRB.OPTIMAL: "Optimal",
            GRB.INFEASIBLE: "Infeasible",
            GRB.UNBOUNDED: "Unbounded",
            GRB.INF_OR_UNBD: "Infeasible or Unbounded",
            GRB.LOADED: "Loaded",
            GRB.SUBOPTIMAL: "Suboptimal",
            GRB.ITERATION_LIMIT: "Iteration Limit Reached",
            GRB.NODE_LIMIT: "Node Limit Reached",
            GRB.TIME_LIMIT: "Time Limit Reached",
            GRB.SOLUTION_LIMIT: "Solution Limit Reached",
            GRB.INTERRUPTED: "Interrupted",
            GRB.NUMERIC: "Numeric Error",
            # Add others if needed
        }
        print(f"Status: {status_map.get(status, f'Gurobi Status Code {status}')}")

        if status == GRB.OPTIMAL:
            if self.optimal_objective is not None:
                print(f"Optimal Objective Value: {self.optimal_objective:.6f}")
            else:
                print("Optimal Objective Value: Not Available (Error during extraction?)")

            # Report Stage 1 Solution
            if self.stage1_solution is not None and self.num_x > 0:
                print("Stage 1 Decision Variables (x) (First 10):")
                for i in range(min(self.num_x, 10)):
                    # Ensure reader and indices are available for name lookup
                    var_name = f"x[{i}]" # Default name
                    if self.reader and self.reader.x_indices is not None and i < len(self.reader.x_indices):
                        orig_gurobi_idx = self.reader.x_indices[i]
                        var_name = self.reader.index_to_var_name.get(orig_gurobi_idx, f"x_idx_{orig_gurobi_idx}")
                    print(f"  {var_name}: {self.stage1_solution[i]:.6f}")
                if self.num_x > 10: print("  ...")
            elif self.num_x > 0:
                 print("Stage 1 solution not available (check SolCount).")
            else:
                 print("No Stage 1 variables in the model.")

            # Report Stage 2 Solution (Example: first 5 vars of first 3 scenarios)
            if self.stage2_solution_all is not None and self.num_y > 0 and self.num_scenarios > 0:
                 print("Stage 2 Decision Variables (y_s) (Examples):")
                 max_scen_report = min(self.num_scenarios, 3)
                 max_y_report = min(self.num_y, 5)
                 for s in range(max_scen_report):
                     print(f"  Scenario {s+1} (First {max_y_report} vars):")
                     for j in range(max_y_report):
                         # Name lookup for y is harder without direct mapping in reader, use indices
                         print(f"    y[{s},{j}]: {self.stage2_solution_all[s, j]:.6f}")
                     if self.num_y > max_y_report: print("    ...")
                 if self.num_scenarios > max_scen_report: print("  ...")
            elif self.num_y > 0:
                 print("Stage 2 solutions not available.")
            else:
                 print("No Stage 2 variables in the model.")

            # Report Stage 2 Duals (Example: first 5 cons of first 3 scenarios)
            if self.stage2_duals_all is not None and self.num_cons2 > 0 and self.num_scenarios > 0:
                 print("Stage 2 Dual Multipliers (pi_s) (Examples):")
                 max_scen_report = min(self.num_scenarios, 3)
                 max_c_report = min(self.num_cons2, 5)
                 for s in range(max_scen_report):
                     print(f"  Scenario {s+1} (First {max_c_report} duals):")
                     for k in range(max_c_report):
                         # Name lookup requires mapping row indices back, use indices
                         print(f"    pi[{s},{k}]: {self.stage2_duals_all[s, k]:.6f}")
                     if self.num_cons2 > max_c_report: print("    ...")
                 if self.num_scenarios > max_scen_report: print("  ...")
            elif self.num_cons2 > 0:
                print("Stage 2 dual multipliers not available.")
            else:
                print("No Stage 2 constraints in the model.")

        elif status == GRB.INFEASIBLE:
            print("Model is Infeasible.")
            # Optionally: compute and report IIS (Irreducible Inconsistent Subsystem)
            # if self.saa_model:
            #     print("Computing IIS...")
            #     self.saa_model.computeIIS()
            #     iis_constr_indices = [c.index for c in self.saa_model.getConstrs() if c.IISConstr]
            #     iis_var_indices = [v.index for v in self.saa_model.getVars() if v.IISLB > 0 or v.IISUB > 0]
            #     print(f"IIS involves {len(iis_constr_indices)} constraints and {len(iis_var_indices)} variable bounds.")
            #     # More detailed IIS reporting could be added here
        elif status == GRB.UNBOUNDED:
            print("Model is Unbounded.")
        # Add more specific messages for other statuses if needed
        print("-" * 40)


    def extract_basis(self):
        """
        Extracts the optimal basis information after solving the SAA model
        and stores it in self.basis_info.

        Requires the model to have been solved to optimality (Status=OPTIMAL).
        Sets self.basis_info to None otherwise.
        """
        self.basis_info = None # Reset
        if self.saa_model is None:
            print("ERROR: Model not built. Cannot get basis.")
            return
        if self.solution_status != GRB.OPTIMAL:
            print(f"INFO: Model status is {self.solution_status}, not Optimal. Cannot get basis.")
            return
        if self.all_vars is None:
            print("ERROR: Model variables not defined. Cannot get basis.")
            return

        print("Step 6: Extracting optimal basis information...")
        extract_basis_start_time = time.time()
        try:
            # Get basis status for all constraints and variables
            # Need to get actual Var and Constr objects first if using addMVar/addMConstr
            all_gurobi_vars = self.saa_model.getVars()
            all_gurobi_constrs = self.saa_model.getConstrs()

            # Check if the number of retrieved vars/constrs matches expectations
            N = self.num_scenarios
            total_cons_expected = self.num_cons1 + N * self.num_cons2
            total_vars_expected = self.num_x + N * self.num_y

            if len(all_gurobi_constrs) != total_cons_expected:
                 print(f"Warning: Number of Gurobi constraints ({len(all_gurobi_constrs)}) doesn't match expected total ({total_cons_expected}). Basis extraction might be incorrect.")
            if len(all_gurobi_vars) != total_vars_expected:
                 print(f"Warning: Number of Gurobi variables ({len(all_gurobi_vars)}) doesn't match expected total ({total_vars_expected}). Basis extraction might be incorrect.")

            cbasis_all = np.array(self.saa_model.getAttr(GRB.Attr.CBasis, all_gurobi_constrs), dtype=np.int8)
            vbasis_all = np.array(self.saa_model.getAttr(GRB.Attr.VBasis, all_gurobi_vars), dtype=np.int8)

            # --- Extract Stage 1 Basis ---
            cbasis_x = cbasis_all[0 : self.num_cons1] if self.num_cons1 > 0 else np.array([], dtype=np.int8)
            vbasis_x = vbasis_all[0 : self.num_x] if self.num_x > 0 else np.array([], dtype=np.int8)

            # --- Extract Stage 2 Basis ---
            # Default empty arrays with correct dimensions for HDF5 saving consistency
            cbasis_y_all = np.array([], dtype=np.int8).reshape(N, 0)
            vbasis_y_all = np.array([], dtype=np.int8).reshape(N, 0)

            start_idx_c = self.num_cons1
            end_idx_c = start_idx_c + N * self.num_cons2
            if N > 0 and self.num_cons2 > 0 and end_idx_c <= len(cbasis_all):
                cbasis_y_flat = cbasis_all[start_idx_c : end_idx_c]
                cbasis_y_all = cbasis_y_flat.reshape((N, self.num_cons2))
            elif N * self.num_cons2 > 0: # If expected shape > 0 but slicing failed
                print("Warning: Could not extract stage 2 constraint basis (index/length issue?). Saving empty array.")


            start_idx_v = self.num_x
            end_idx_v = start_idx_v + N * self.num_y
            if N > 0 and self.num_y > 0 and end_idx_v <= len(vbasis_all):
                vbasis_y_flat = vbasis_all[start_idx_v : end_idx_v]
                vbasis_y_all = vbasis_y_flat.reshape((N, self.num_y))
            elif N * self.num_y > 0: # If expected shape > 0 but slicing failed
                print("Warning: Could not extract stage 2 variable basis (index/length issue?). Saving empty array.")

            # Store in the dictionary attribute
            self.basis_info = {
                'cbasis_x': cbasis_x,
                'vbasis_x': vbasis_x,
                'cbasis_y_all': cbasis_y_all,
                'vbasis_y_all': vbasis_y_all
            }
            extract_basis_end_time = time.time()
            self.timing['extract_basis'] = extract_basis_end_time - extract_basis_start_time
            print(f"Basis extraction complete. (Time: {self.timing['extract_basis']:.2f}s)")


        except gp.GurobiError as e:
            print(f"ERROR: Gurobi error getting basis attributes: {e}")
            self.basis_info = None # Ensure it's None on error
        except Exception as e:
            print(f"ERROR: Unexpected error during basis extraction: {e}")
            self.basis_info = None # Ensure it's None on error


    def save_results_to_hdf5(self, filepath: str):
        """
        Saves the generated/calculated data to an HDF5 file.

        Includes:
        - Generated stochastic RHS parts (if any)
        - Optimal primal solution (x, y_s) (if found)
        - Optimal second-stage duals (pi_s) (if found)
        - Optimal basis information (if found)
        - Key problem dimensions and metadata

        Args:
            filepath (str): The path to the HDF5 file to be created.
        """
        print(f"Step 7: Saving results to HDF5 file: {filepath}...")
        save_start_time = time.time()
        try:
            with h5py.File(filepath, 'w') as f:
                # --- Metadata ---
                meta_grp = f.create_group("metadata")
                meta_grp.attrs['problem_name'] = os.path.basename(self.core_filepath).split('.')[0]
                meta_grp.attrs['num_scenarios'] = self.num_scenarios
                meta_grp.attrs['random_seed'] = self.random_seed
                meta_grp.attrs['num_x'] = self.num_x
                meta_grp.attrs['num_y'] = self.num_y
                meta_grp.attrs['num_cons1'] = self.num_cons1
                meta_grp.attrs['num_cons2'] = self.num_cons2
                meta_grp.attrs['num_stochastic_elements'] = self.num_stochastic_elements
                meta_grp.attrs['solution_status_code'] = self.solution_status if self.solution_status is not None else -1
                if self.optimal_objective is not None:
                    meta_grp.attrs['optimal_objective'] = self.optimal_objective

                # --- Scenarios ---
                scen_grp = f.create_group("scenarios")
                if self.batch_stochastic_parts is not None:
                    # Shape should be (N, num_stochastic_elements)
                    scen_grp.create_dataset("stochastic_rhs_parts", data=self.batch_stochastic_parts, compression="gzip")
                    print(f"  Saved stochastic_rhs_parts with shape {self.batch_stochastic_parts.shape}")
                else:
                    print("  No stochastic RHS parts to save.")
                    scen_grp.create_dataset("stochastic_rhs_parts", data=np.array([]).reshape(self.num_scenarios, 0)) # Save empty placeholder


                # --- Solution ---
                sol_grp = f.create_group("solution")
                prim_grp = sol_grp.create_group("primal")
                dual_grp = sol_grp.create_group("dual")

                if self.solution_status == GRB.OPTIMAL:
                    # Primal x
                    if self.stage1_solution is not None:
                        prim_grp.create_dataset("x", data=self.stage1_solution, compression="gzip")
                        print(f"  Saved primal solution x with shape {self.stage1_solution.shape}")
                    else:
                        prim_grp.create_dataset("x", data=np.array([])) # Empty placeholder
                    # Primal y_s
                    if self.stage2_solution_all is not None:
                         # Shape should be (N, num_y)
                        prim_grp.create_dataset("y_s", data=self.stage2_solution_all, compression="gzip")
                        print(f"  Saved primal solution y_s with shape {self.stage2_solution_all.shape}")
                    else:
                        prim_grp.create_dataset("y_s", data=np.array([]).reshape(self.num_scenarios, self.num_y)) # Empty placeholder
                    # Dual pi_s
                    if self.stage2_duals_all is not None:
                         # Shape should be (N, num_cons2)
                        dual_grp.create_dataset("pi_s", data=self.stage2_duals_all, compression="gzip")
                        print(f"  Saved dual multipliers pi_s with shape {self.stage2_duals_all.shape}")
                    else:
                        dual_grp.create_dataset("pi_s", data=np.array([]).reshape(self.num_scenarios, self.num_cons2)) # Empty placeholder

                else:
                     print("  No optimal solution found, skipping saving primal/dual solution values.")
                     # Save empty datasets to maintain structure
                     prim_grp.create_dataset("x", data=np.array([]))
                     prim_grp.create_dataset("y_s", data=np.array([]).reshape(self.num_scenarios, self.num_y))
                     dual_grp.create_dataset("pi_s", data=np.array([]).reshape(self.num_scenarios, self.num_cons2))


                # --- Basis ---
                basis_grp = f.create_group("basis")
                if self.basis_info is not None:
                    for key, value in self.basis_info.items():
                        if value is not None:
                             basis_grp.create_dataset(key, data=value, compression="gzip")
                             print(f"  Saved basis {key} with shape {value.shape}")
                        else:
                             # Determine expected shape for empty placeholder if needed
                             expected_shape = (0,) # Default
                             if key == 'cbasis_x': expected_shape = (self.num_cons1,)
                             elif key == 'vbasis_x': expected_shape = (self.num_x,)
                             elif key == 'cbasis_y_all': expected_shape = (self.num_scenarios, self.num_cons2)
                             elif key == 'vbasis_y_all': expected_shape = (self.num_scenarios, self.num_y)
                             basis_grp.create_dataset(key, data=np.array([], dtype=np.int8).reshape(expected_shape))
                             print(f"  Basis {key} was None or empty, saved empty placeholder.")

                else:
                    print("  No basis information available to save (model not optimal or error during extraction).")
                    # Save empty datasets to maintain structure
                    basis_grp.create_dataset('cbasis_x', data=np.array([], dtype=np.int8).reshape(self.num_cons1,))
                    basis_grp.create_dataset('vbasis_x', data=np.array([], dtype=np.int8).reshape(self.num_x,))
                    basis_grp.create_dataset('cbasis_y_all', data=np.array([], dtype=np.int8).reshape(self.num_scenarios, self.num_cons2))
                    basis_grp.create_dataset('vbasis_y_all', data=np.array([], dtype=np.int8).reshape(self.num_scenarios, self.num_y))


            save_end_time = time.time()
            self.timing['save_hdf5'] = save_end_time - save_start_time
            print(f"Results saved successfully. (Time: {self.timing['save_hdf5']:.2f}s)")

        except ImportError:
             print("ERROR: h5py library not found. Cannot save to HDF5. Install it using 'pip install h5py'")
             self.timing['save_hdf5'] = 0.0
        except Exception as e:
             print(f"ERROR: Failed to save results to HDF5 file '{filepath}': {e}")
             import traceback
             traceback.print_exc()
             self.timing['save_hdf5'] = time.time() - save_start_time # Record time even on failure


    def run_pipeline(self, solve_model=True, save_hdf5=False, hdf5_filepath="saa_results.h5"):
        """
        Runs the full pipeline: Load -> Generate Scenarios -> Build -> Solve -> Report -> Extract Basis -> Save HDF5.

        Args:
            solve_model (bool): If True, solves the model, reports results, and extracts basis.
            save_hdf5 (bool): If True, saves the results to an HDF5 file after solving (if optimal).
            hdf5_filepath (str): Path for the output HDF5 file if save_hdf5 is True.
        """
        pipeline_start = time.time()
        try:
            self._load_smps_data()
            self._generate_scenarios()
            self.build_model() # Builds the Gurobi model structure

            if solve_model:
                self.solve() # Solves the model, extracts solution/duals if optimal
                self.report_results() # Prints summary to console
                self.extract_basis() # Extracts basis if optimal

                if save_hdf5:
                     # Only attempt saving if solve was attempted
                     if self.solution_status is not None: # Check if solve finished (even if not optimal)
                         self.save_results_to_hdf5(hdf5_filepath)
                     else:
                         print("Skipping HDF5 save because the model solve was not completed.")

            else:
                 print("Skipping solve, report, basis extraction, and HDF5 saving as per configuration.")

        except (FileNotFoundError, ValueError, RuntimeError, gp.GurobiError) as e:
             print(f"\n--- PIPELINE FAILED ---")
             print(f"Error during pipeline execution: {e}")
             # Optionally re-raise if you want the script to terminate with non-zero exit code
             # raise e
        except Exception as e:
             print(f"\n--- PIPELINE FAILED ---")
             print(f"An unexpected error occurred during pipeline execution: {e}")
             import traceback
             traceback.print_exc()
             # raise e
        finally:
            pipeline_end = time.time()
            self.timing['full_pipeline'] = pipeline_end - pipeline_start
            print(f"\nPipeline execution finished. (Total Pipeline Time: {self.timing['full_pipeline']:.2f}s)")


# --- Main Execution Block ---
if __name__ == "__main__":

    overall_start_time_main = time.time()

    # --- Configuration ---
    base_dir_main = "smps_data"
    problem_name_main = "ssn"  # Example problem name
    num_scenarios_main = 5000 # Reduced for quicker testing
    num_threads_main = None # Use Gurobi default or specify (e.g., 4)
    random_seed_main = 42
    output_hdf5_filename = f"{problem_name_main}_{num_scenarios_main}scen_results.h5"

    # --- Setup Paths ---
    script_dir_main = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    file_dir_main = os.path.join(script_dir_main, base_dir_main, problem_name_main)
    core_filepath_main = os.path.join(file_dir_main, f"{problem_name_main}.mps") # Assume .mps, change if .cor
    time_filepath_main = os.path.join(file_dir_main, f"{problem_name_main}.tim")
    sto_filepath_main = os.path.join(file_dir_main, f"{problem_name_main}.sto")
    hdf5_output_path = os.path.join(script_dir_main, output_hdf5_filename) # Save HDF5 in script dir

    # --- Check if input files exist ---
    if not os.path.exists(core_filepath_main):
        print(f"ERROR: Core file not found at {core_filepath_main}")
        exit(1)
    if not os.path.exists(time_filepath_main):
        print(f"ERROR: Time file not found at {time_filepath_main}")
        # Some problems might not have .tim if structure is fully in .mps
        # Decide if this is critical. For now, let's assume it is if sto is present.
        if os.path.exists(sto_filepath_main):
             print("...and .sto file exists, so .tim is likely needed.")
             exit(1)
        else:
             print("...but no .sto file either. Assuming a deterministic problem or different format.")
    if not os.path.exists(sto_filepath_main):
        print(f"WARNING: Sto file not found at {sto_filepath_main}. Assuming deterministic problem or simple recourse.")
        # Allow continuation, but SAABuilder might behave unexpectedly if stochasticity is implied elsewhere.


    # --- Instantiate and Run ---
    print(f"--- Starting SAA for {problem_name_main} ---")
    print(f"Input files directory: {file_dir_main}")
    print(f"Number of scenarios: {num_scenarios_main}")
    print(f"Output HDF5 file: {hdf5_output_path}")
    print("-" * 40)


    saa_builder = SAABuilder(
        core_filepath=core_filepath_main,
        time_filepath=time_filepath_main,
        sto_filepath=sto_filepath_main,
        num_scenarios=num_scenarios_main,
        random_seed=random_seed_main,
        num_threads=num_threads_main
    )

    # Execute the full process including solve and saving HDF5
    saa_builder.run_pipeline(solve_model=True, save_hdf5=True, hdf5_filepath=hdf5_output_path)

    # --- Print Timings ---
    print("\n--- Timing Summary ---")
    # Sort timings for better readability
    sorted_timing = sorted(saa_builder.timing.items(), key=lambda item: item[1], reverse=True)
    for step, duration in sorted_timing:
        # Indent sub-steps for clarity
        indent = "  " if step not in ['load_smps', 'generate_scenarios', 'build_model_total', 'solve_model', 'extract_basis', 'save_hdf5', 'full_pipeline'] else ""
        print(f"  {indent}{step}: {duration:.3f}s")


    overall_end_time_main = time.time()
    print("-" * 40)
    print(f"Script finished. (Total Wall Clock Time: {overall_end_time_main - overall_start_time_main:.2f}s)")