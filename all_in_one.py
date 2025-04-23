# all_in_one.py (Class-Based SAA Builder)

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import scipy.sparse as sp # Import SciPy sparse
from typing import Optional, List, Dict, Any, Tuple
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
class SAABuilder:``
    """
    Builds and optionally solves a Sample Average Approximation (SAA) model
    for a two-stage stochastic linear program defined by SMPS files.

    Uses SMPSReader to parse input files and constructs the SAA extensive
    form using a block matrix approach with optimized COO construction.
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
        self.solution_status: Optional[int] = None
        self.optimal_objective: Optional[float] = None
        self.stage1_solution: Optional[np.ndarray] = None

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
            raise RuntimeError("SMPS data must be loaded before generating scenarios.")

        print(f"Step 2: Generating {self.num_scenarios} scenario RHS vectors (Batch approach)...")
        step2_start_time = time.time()
        np.random.seed(self.random_seed) # Set seed for reproducibility

        r_bar = self.reader.r_bar
        stochastic_rows_relative_indices = self.reader.stochastic_rows_relative_indices

        if self.num_stochastic_elements > 0:
            batch_stochastic_parts = self.reader.sample_stochastic_rhs_batch(self.num_scenarios)

            if r_bar.ndim != 1: raise ValueError(f"r_bar should be 1D, but has shape {r_bar.shape}")
            if len(r_bar) != self.num_cons2: raise ValueError(f"r_bar length {len(r_bar)} does not match num_cons2 {self.num_cons2}")

            self.all_scenario_rhs_matrix = np.tile(r_bar, (self.num_scenarios, 1))

            if len(stochastic_rows_relative_indices) > 0:
                if np.any(stochastic_rows_relative_indices >= self.all_scenario_rhs_matrix.shape[1]):
                     raise IndexError(f"Stochastic relative indices out of bounds for r_bar columns ({self.all_scenario_rhs_matrix.shape[1]})")
                if batch_stochastic_parts.shape[1] != len(stochastic_rows_relative_indices):
                     raise ValueError(f"Mismatch between batch sample columns ({batch_stochastic_parts.shape[1]}) and relative indices count ({len(stochastic_rows_relative_indices)})")
                self.all_scenario_rhs_matrix[:, stochastic_rows_relative_indices] = batch_stochastic_parts

        elif r_bar is not None: # Case with no stochastic elements
             print("  No stochastic elements found. Using deterministic r_bar for all scenarios.")
             self.all_scenario_rhs_matrix = np.tile(r_bar, (self.num_scenarios, 1))
        elif self.num_cons2 > 0: # Error case: stage 2 constraints but no r_bar
             raise ValueError("Stage 2 constraints exist but r_bar is None. Cannot generate scenarios.")
        else: # Case with no stage 2 constraints
             print("  No stage 2 constraints found.")
             self.all_scenario_rhs_matrix = None # Explicitly set to None

        step2_end_time = time.time()
        self.timing['generate_scenarios'] = step2_end_time - step2_start_time
        num_generated = self.all_scenario_rhs_matrix.shape[0] if self.all_scenario_rhs_matrix is not None else 0
        print(f"Scenario generation complete. Generated {num_generated} RHS vectors. (Time: {self.timing['generate_scenarios']:.2f}s)")

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
        C = self.reader.C.tocsr() if self.reader.C is not None else None
        D = self.reader.D.tocsr() if self.reader.D is not None else None

        # Initialize lists for final COO components
        rows_list, cols_list, data_list = [], [], []

        # 1. Add A block
        if A is not None and A.nnz > 0:
            A_coo = A.tocoo()
            rows_list.append(A_coo.row)
            cols_list.append(A_coo.col)
            data_list.append(A_coo.data)

        # 2. Prepare C block data (if C exists)
        if C is not None and C.nnz > 0 and N > 0:
            C_coo = C.tocoo()
            C_nnz = C.nnz
            C_data_all = np.tile(C_coo.data, N)
            row_offsets_C = np.repeat(self.num_cons1 + np.arange(N, dtype=np.int32) * self.num_cons2, C_nnz)
            C_rows_all = np.tile(C_coo.row, N) + row_offsets_C
            C_cols_all = np.tile(C_coo.col, N)
            rows_list.append(C_rows_all)
            cols_list.append(C_cols_all)
            data_list.append(C_data_all)

        # 3. Prepare D block data (if D exists)
        if D is not None and D.nnz > 0 and N > 0:
            D_coo = D.tocoo()
            D_nnz = D.nnz
            D_data_all = np.tile(D_coo.data, N)
            row_offsets_D = np.repeat(self.num_cons1 + np.arange(N, dtype=np.int32) * self.num_cons2, D_nnz) # Same row offsets as C
            D_rows_all = np.tile(D_coo.row, N) + row_offsets_D
            col_offsets_D = np.repeat(self.num_x + np.arange(N, dtype=np.int32) * self.num_y, D_nnz) # Column offsets for y_s
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
            if self.all_scenario_rhs_matrix is not None:
                # Convert the generated matrix back to a list of rows
                scenario_rhs_list_from_matrix = [row for row in self.all_scenario_rhs_matrix]
            else:
                scenario_rhs_list_from_matrix = []

            b = self.reader.b
            b_safe = b if (b is not None and b.ndim == 1 and self.num_cons1 > 0) else np.array([])
            rhs_list = ([b_safe] if b_safe.size > 0 else []) + scenario_rhs_list_from_matrix

            if not rhs_list:
                print("  Warning: RHS vector is empty.")
                SAA_rhs = np.array([])
            else:
                # Validate shapes before concatenating
                valid_rhs = True
                if b_safe.size > 0 and len(b_safe) != self.num_cons1: valid_rhs = False
                for i, r_s in enumerate(scenario_rhs_list_from_matrix):
                     if len(r_s) != self.num_cons2: valid_rhs = False; break
                if not valid_rhs: raise ValueError("Inconsistent shapes in RHS list components.")
                try:
                    SAA_rhs = np.concatenate(rhs_list)
                except ValueError as e: print(f"ERROR concatenating RHS: {e}."); raise e

            # Sense
            sense1_char = self.reader.sense1
            sense2_char = self.reader.sense2
            sense1_grb = convert_senses_to_gurobi(sense1_char if sense1_char is not None else np.array([]))
            sense2_grb = convert_senses_to_gurobi(sense2_char if sense2_char is not None else np.array([]))
            sense1_safe = sense1_grb if (len(sense1_grb) == self.num_cons1) else np.array([], dtype=object)
            sense2_safe = sense2_grb if (len(sense2_grb) == self.num_cons2) else np.array([], dtype=object)
            sense_list = ([sense1_safe] if sense1_safe.size > 0 else []) + ([sense2_safe] * N if sense2_safe.size > 0 else [])

            if not sense_list:
                 print("  Warning: Sense vector is empty.")
                 SAA_sense = np.array([], dtype=object)
            else:
                try:
                    SAA_sense = np.concatenate(sense_list)
                except ValueError as e: print(f"ERROR concatenating Senses: {e}."); raise e

        prep_rhs_sense_end_time = time.time()
        self.timing['prep_rhs_sense'] = prep_rhs_sense_end_time - prep_rhs_sense_start_time
        # print(f"    RHS/Sense prepared. (Time: {self.timing['prep_rhs_sense']:.2f}s)") # Less verbose

        return SAA_rhs, SAA_sense

    def build_model(self, suppress_gurobi_output: bool = True):
        """Builds the Gurobi SAA model object."""
        if self.reader is None:
            self._load_smps_data()
        if self.all_scenario_rhs_matrix is None and self.num_cons2 > 0: # Check if RHS needed but not generated
             self._generate_scenarios()

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
                    self.all_vars = self.saa_model.addMVar(shape=total_vars, lb=lb_combined, ub=ub_combined, obj=obj_combined, name="vars")
                else:
                    self.all_vars = None
                    print("Warning: No variables in the model.")
                add_vars_end_time = time.time()
                self.timing['add_variables'] = add_vars_end_time - add_vars_start_time
                # print(f"    Variables added. (Time: {self.timing['add_variables']:.2f}s)") # Less verbose

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
                    if SAA_matrix.shape[0] != SAA_rhs.shape[0] or SAA_matrix.shape[0] != SAA_sense.shape[0]: raise ValueError("Mismatch between matrix rows, RHS length, and Sense length.")
                    if SAA_matrix.shape[1] != self.all_vars.shape[0]: raise ValueError("Mismatch between matrix columns and number of variables.")

                    self.saa_model.addMConstr(SAA_matrix, self.all_vars, SAA_sense, SAA_rhs, name="SAA_constraints")
                    add_constr_end_time = time.time()
                    self.timing['add_constraints'] = add_constr_end_time - add_constr_start_time
                    print(f"  Block constraints added successfully. (Time: {self.timing['add_constraints']:.2f}s)")
                elif total_cons > 0:
                    print("Warning: Skipping addMConstr because Matrix or Variables are missing.")
                else:
                    print("  No constraints to add.")

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
        """Solves the constructed SAA Gurobi model."""
        if self.saa_model is None:
            # Attempt to build the model first
            self.build_model()
            if self.saa_model is None: # Check again if build failed
                 raise RuntimeError("SAA model has not been built successfully.")

        print("Step 4: Solving the SAA model...")
        step4_start_time = time.time()
        try:
            self.saa_model.optimize()
            self.solution_status = self.saa_model.Status
            if self.solution_status == GRB.OPTIMAL:
                self.optimal_objective = self.saa_model.ObjVal
                if self.num_x > 0 and self.all_vars is not None:
                    # Ensure solution values are available before accessing .X
                    if self.saa_model.SolCount > 0:
                         self.stage1_solution = self.all_vars.X[0:self.num_x]
                    else:
                         print("Warning: Optimal status reported but no solution found (SolCount=0).")
                         self.stage1_solution = None
            else:
                self.optimal_objective = None
                self.stage1_solution = None

        except gp.GurobiError as e:
            print(f"ERROR: Gurobi error occurred during optimization - Code {e.errno}: {e}")
            # Store status even if error occurred during solve (e.g., interrupted)
            if hasattr(self.saa_model, 'Status'):
                 self.solution_status = self.saa_model.Status
            raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during optimization - {e}")
            raise
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
        if status == GRB.OPTIMAL:
            print("Status: Optimal")
            if self.optimal_objective is not None:
                print(f"Optimal Objective Value: {self.optimal_objective:.6f}")
            else:
                print("Optimal Objective Value: Not Available") # Should not happen if status is Optimal

            if self.stage1_solution is not None:
                print("Stage 1 Decision Variables (x) (First 10):")
                for i in range(min(self.num_x, 10)):
                    # Ensure reader and indices are available
                    if self.reader and self.reader.x_indices is not None and i < len(self.reader.x_indices):
                        orig_gurobi_idx = self.reader.x_indices[i]
                        var_name = self.reader.index_to_var_name.get(orig_gurobi_idx, f"x_idx_{orig_gurobi_idx}")
                        print(f"  {var_name}: {self.stage1_solution[i]:.6f}")
                    else:
                         print(f"  x[{i}]: {self.stage1_solution[i]:.6f} (Name lookup failed)")
                if self.num_x > 10: print("  ...")
            else:
                print("Stage 1 solution not available (check SolCount or if num_x > 0).")
        elif status == GRB.INFEASIBLE:
            print("Status: Model is Infeasible.")
        elif status == GRB.UNBOUNDED:
            print("Status: Model is Unbounded.")
        elif status == GRB.INF_OR_UNBD:
             print("Status: Model is Infeasible or Unbounded.")
        # Add more status codes as needed
        else:
            print(f"Status: Optimization finished with Gurobi status code: {status}")

    def get_basis(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieves the optimal basis information after solving the SAA model.

        Requires the model to have been solved to optimality.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing:
            - cbasis_x: Basis status for stage 1 constraints (1D array).
            - vbasis_x: Basis status for stage 1 variables (1D array).
            - cbasis_y_all: Basis status for stage 2 constraints for all scenarios (2D array, shape N x num_cons2).
            - vbasis_y_all: Basis status for stage 2 variables for all scenarios (2D array, shape N x num_y).
            Returns None if the model is not solved or not optimal.
        """
        if self.saa_model is None:
            print("ERROR: Model not built. Cannot get basis.")
            return None
        if self.solution_status != GRB.OPTIMAL:
            print(f"ERROR: Model status is {self.solution_status}, not Optimal. Cannot get basis.")
            return None
        if self.all_vars is None:
             print("ERROR: Model variables not defined. Cannot get basis.")
             return None

        print("Extracting optimal basis information...")
        try:
            # Get basis status for all constraints and variables
            cbasis_all = np.array(self.saa_model.getAttr(GRB.Attr.CBasis, self.saa_model.getConstrs()), dtype=np.int8)
            vbasis_all = np.array(self.saa_model.getAttr(GRB.Attr.VBasis, self.saa_model.getVars()), dtype=np.int8)

            N = self.num_scenarios
            total_cons_expected = self.num_cons1 + N * self.num_cons2
            total_vars_expected = self.num_x + N * self.num_y

            # --- Verification ---
            if len(cbasis_all) != total_cons_expected:
                print(f"Warning: Length of cbasis ({len(cbasis_all)}) does not match expected total constraints ({total_cons_expected}).")
                # Decide how to handle: return None, raise error, or proceed with caution?
                # Proceeding cautiously for now.
            if len(vbasis_all) != total_vars_expected:
                print(f"Warning: Length of vbasis ({len(vbasis_all)}) does not match expected total variables ({total_vars_expected}).")
                # Proceeding cautiously.

            # --- Extract Stage 1 Basis ---
            cbasis_x = cbasis_all[0 : self.num_cons1] if self.num_cons1 > 0 else np.array([], dtype=np.int8)
            vbasis_x = vbasis_all[0 : self.num_x] if self.num_x > 0 else np.array([], dtype=np.int8)

            # --- Extract Stage 2 Basis ---
            cbasis_y_all = np.array([], dtype=np.int8).reshape(N, 0) # Default empty
            vbasis_y_all = np.array([], dtype=np.int8).reshape(N, 0) # Default empty

            start_idx_c = self.num_cons1
            end_idx_c = start_idx_c + N * self.num_cons2
            if N > 0 and self.num_cons2 > 0 and end_idx_c <= len(cbasis_all):
                cbasis_y_flat = cbasis_all[start_idx_c : end_idx_c]
                cbasis_y_all = cbasis_y_flat.reshape((N, self.num_cons2))
            elif N * self.num_cons2 > 0: # If expected shape > 0 but slicing failed
                 print("Warning: Could not extract stage 2 constraint basis (index issue?).")


            start_idx_v = self.num_x
            end_idx_v = start_idx_v + N * self.num_y
            if N > 0 and self.num_y > 0 and end_idx_v <= len(vbasis_all):
                vbasis_y_flat = vbasis_all[start_idx_v : end_idx_v]
                vbasis_y_all = vbasis_y_flat.reshape((N, self.num_y))
            elif N * self.num_y > 0: # If expected shape > 0 but slicing failed
                 print("Warning: Could not extract stage 2 variable basis (index issue?).")

            print("Basis extraction complete.")
            return cbasis_x, vbasis_x, cbasis_y_all, vbasis_y_all

        except gp.GurobiError as e:
            print(f"ERROR: Gurobi error getting basis attributes: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Unexpected error during basis extraction: {e}")
            return None


    def run_pipeline(self, solve_model=True):
        """
        Runs the pipeline steps.

        Args:
            solve_model (bool): If True, solves the model and reports results.
                                If False, only loads data and builds the model structure.
        """
        try:
             self._load_smps_data()
             self._generate_scenarios()
             self.build_model()
             if solve_model:
                 self.solve()
                 self.report_results()
                 # Example of getting basis after solve
                 basis_info = self.get_basis()
                 if basis_info:
                      cbasis_x, vbasis_x, cbasis_y_all, vbasis_y_all = basis_info
                      print("\n--- Basis Information Summary ---")
                      print(f"  Stage 1 CBasis shape: {cbasis_x.shape}")
                      print(f"  Stage 1 VBasis shape: {vbasis_x.shape}")
                      print(f"  Stage 2 CBasis shape (N x n_cons2): {cbasis_y_all.shape}")
                      print(f"  Stage 2 VBasis shape (N x n_y): {vbasis_y_all.shape}")
                      # Example: Print first few basis statuses
                      print(f"  First 5 Stage 1 CBasis: {cbasis_x[:5]}")
                      print(f"  First 5 Stage 1 VBasis: {vbasis_x[:5]}")
                      print(f"  First Scenario, First 5 Stage 2 CBasis: {cbasis_y_all[0, :5]}")
                      print(f"  First Scenario, First 5 Stage 2 VBasis: {vbasis_y_all[0, :5]}")
                      print("---------------------------------")

        except Exception as e:
             print(f"\n--- Pipeline failed ---")
             print(f"Error: {e.with_traceback()}")
             print("------------------------")

# --- Main Execution Block ---
if __name__ == "__main__":

    overall_start_time_main = time.time()

    # --- Configuration ---
    base_dir_main = "smps_data"
    problem_name_main = "ssn"
    num_scenarios_main = 1000
    num_threads_main = 32
    random_seed_main = 42

    # --- Setup Paths ---
    script_dir_main = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    file_dir_main = os.path.join(script_dir_main, base_dir_main, problem_name_main)
    core_filepath_main = os.path.join(file_dir_main, f"{problem_name_main}.mps")
    time_filepath_main = os.path.join(file_dir_main, f"{problem_name_main}.tim")
    sto_filepath_main = os.path.join(file_dir_main, f"{problem_name_main}.sto")

    # --- Instantiate and Run ---
    saa_builder = SAABuilder(
        core_filepath=core_filepath_main,
        time_filepath=time_filepath_main,
        sto_filepath=sto_filepath_main,
        num_scenarios=num_scenarios_main,
        random_seed=random_seed_main,
        num_threads=num_threads_main
    )

    saa_builder.run_pipeline(solve_model=True) # Execute the full process including solve

    # --- Print Timings ---
    print("\n--- Timing Summary ---")
    for step, duration in saa_builder.timing.items():
        print(f"  {step}: {duration:.2f}s")

    overall_end_time_main = time.time()
    print("-" * 40)
    print(f"Script finished. (Total Time: {overall_end_time_main - overall_start_time_main:.2f}s)")