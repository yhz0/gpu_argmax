# build_saa.py

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import scipy.sparse as sp
from typing import Optional, Dict, Tuple
import time
import h5py
import sys
import logging
import traceback

# --- Setup Logging ---
# Configure logger to be used throughout the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This allows the script to be run from anywhere and still find the src module
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

try:
    from src.smps_reader import SMPSReader
    from src.parallel_second_stage_worker import ParallelSecondStageWorker
except ImportError:
    logger.error("Could not import SMPSReader from src.")
    logger.error("Ensure 'smps_reader.py' is in the 'src' directory and the script is run from the project root.")
    exit(1)

# --- Helper Function ---
def convert_senses_to_gurobi(sense_chars: np.ndarray) -> np.ndarray:
    """Converts numpy array of sense characters ('<', '=', '>') to Gurobi constants."""
    sense_map = {'<': GRB.LESS_EQUAL, '=': GRB.EQUAL, '>': GRB.GREATER_EQUAL}
    if sense_chars.dtype.kind == 'U':
        return np.array([sense_map.get(s, GRB.LESS_EQUAL) for s in sense_chars], dtype=object)
    if sense_chars.size == 0:
        return np.array([], dtype=object)
    raise TypeError(f"Expected numpy array of strings for senses, got dtype {sense_chars.dtype}")

# --- SAA Builder Class ---
class SAABuilder:
    """
    Builds and optionally solves a Sample Average Approximation (SAA) model
    for a two-stage stochastic linear program defined by SMPS files.
    """

    def __init__(self, core_filepath: str, time_filepath: str, sto_filepath: str,
                 num_scenarios: int, random_seed: int = 42, num_threads: Optional[int] = None):
        """Initializes the SAABuilder."""
        self.core_filepath = core_filepath
        self.time_filepath = time_filepath
        self.sto_filepath = sto_filepath
        self.num_scenarios = num_scenarios
        self.random_seed = int(random_seed) if random_seed is not None else 42
        self.num_threads = num_threads

        self.reader: Optional[SMPSReader] = None
        self.all_scenario_rhs_matrix: Optional[np.ndarray] = None
        self.saa_model: Optional[gp.Model] = None
        self.all_vars: Optional[gp.MVar] = None
        self.saa_constraints: Optional[gp.MConstr] = None
        self.solution_status: Optional[int] = None
        self.optimal_objective: Optional[float] = None

        self.batch_stochastic_parts: Optional[np.ndarray] = None
        self.stage1_solution: Optional[np.ndarray] = None
        self.stage2_solution_all: Optional[np.ndarray] = None
        self.stage2_duals_all: Optional[np.ndarray] = None
        self.basis_info: Optional[Dict[str, np.ndarray]] = None

        # Attributes for cleaned-up results from parallel worker
        self.stage2_duals_all_clean: Optional[np.ndarray] = None
        self.stage2_rc_all_clean: Optional[np.ndarray] = None
        self.vbasis_y_all_clean: Optional[np.ndarray] = None
        self.cbasis_y_all_clean: Optional[np.ndarray] = None

        self.num_x = 0
        self.num_y = 0
        self.num_cons1 = 0
        self.num_cons2 = 0
        self.num_stochastic_elements = 0
        self.timing = {}

        logger.info(f"Initialized SAABuilder for {os.path.basename(core_filepath)} with {num_scenarios} scenarios.")

    def _load_smps_data(self):
        """Loads and extracts data from SMPS files using SMPSReader."""
        logger.info("Step 1: Reading SMPS files...")
        step1_start_time = time.time()
        try:
            for f_path in [self.core_filepath, self.time_filepath, self.sto_filepath]:
                if not os.path.exists(f_path):
                    raise FileNotFoundError(f"Required file not found: {f_path}")

            self.reader = SMPSReader(self.core_filepath, self.time_filepath, self.sto_filepath)
            self.reader.load_and_extract()

            self.num_x = len(self.reader.c)
            self.num_y = len(self.reader.d)
            self.num_cons1 = self.reader.A.shape[0]
            self.num_cons2 = len(self.reader.row2_indices)
            self.num_stochastic_elements = len(self.reader.stochastic_rows_indices_orig)

            self.timing['load_smps'] = time.time() - step1_start_time
            logger.info(f"SMPS data loaded successfully. (Time: {self.timing['load_smps']:.2f}s)")
            logger.info(f"  Dims: x={self.num_x}, y={self.num_y}, cons1={self.num_cons1}, cons2={self.num_cons2}, stoch_rhs={self.num_stochastic_elements}")

        except (FileNotFoundError, Exception) as e:
            logger.error(f"Error during SMPS reading/extraction: {e}")
            logger.error(traceback.format_exc())
            raise

    def _generate_scenarios(self):
        """Generates scenario RHS vectors using batch sampling."""
        if self.reader is None:
            raise RuntimeError("SMPS data must be loaded before generating scenarios.")

        if self.num_cons2 == 0:
            logger.info("Step 2: No second stage constraints, skipping scenario RHS generation.")
            self.all_scenario_rhs_matrix = None
            self.batch_stochastic_parts = np.array([]).reshape(self.num_scenarios, 0)
            self.timing['generate_scenarios'] = 0.0
            return

        logger.info(f"Step 2: Generating {self.num_scenarios} scenario RHS vectors...")
        step2_start_time = time.time()
        np.random.seed(self.random_seed)

        r_bar = self.reader.r_bar

        self.all_scenario_rhs_matrix = np.tile(r_bar, (self.num_scenarios, 1))

        if self.num_stochastic_elements > 0:
            self.batch_stochastic_parts = self.reader.sample_stochastic_rhs_batch(self.num_scenarios)
            stochastic_indices = self.reader.stochastic_rows_relative_indices
            self.all_scenario_rhs_matrix[:, stochastic_indices] = self.batch_stochastic_parts
        else:
            logger.info("  No stochastic elements found. Using deterministic r_bar for all scenarios.")
            self.batch_stochastic_parts = np.array([]).reshape(self.num_scenarios, 0)

        self.timing['generate_scenarios'] = time.time() - step2_start_time
        logger.info(f"Scenario generation complete. (Time: {self.timing['generate_scenarios']:.2f}s)")

    def _build_saa_matrix_coo(self) -> Optional[sp.csr_matrix]:
        """Builds the SAA constraint block matrix using optimized COO construction."""
        if self.reader is None:
            raise RuntimeError("SMPS data must be loaded before building the SAA matrix.")

        N = self.num_scenarios
        total_vars = self.num_x + N * self.num_y
        total_cons = self.num_cons1 + N * self.num_cons2

        if total_cons == 0 or total_vars == 0:
            logger.warning("No constraints or variables - skipping matrix construction.")
            return None

        build_start = time.time()
        rows, cols, data = [], [], []

        A = self.reader.A.tocoo()
        if A.nnz > 0:
            rows.append(A.row)
            cols.append(A.col)
            data.append(A.data)

        C = self.reader.C.tocoo()
        if C.nnz > 0 and N > 0 and self.num_cons2 > 0:
            row_offsets = np.repeat(self.num_cons1 + np.arange(N, dtype=np.int32) * self.num_cons2, C.nnz)
            rows.append(np.tile(C.row, N) + row_offsets)
            cols.append(np.tile(C.col, N))
            data.append(np.tile(C.data, N))

        D = self.reader.D.tocoo()
        if D.nnz > 0 and N > 0 and self.num_cons2 > 0:
            row_offsets = np.repeat(self.num_cons1 + np.arange(N, dtype=np.int32) * self.num_cons2, D.nnz)
            col_offsets = np.repeat(self.num_x + np.arange(N, dtype=np.int32) * self.num_y, D.nnz)
            rows.append(np.tile(D.row, N) + row_offsets)
            cols.append(np.tile(D.col, N) + col_offsets)
            data.append(np.tile(D.data, N))

        final_rows = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
        final_cols = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
        final_data = np.concatenate(data) if data else np.array([], dtype=np.float64)

        saa_matrix = sp.coo_matrix((final_data, (final_rows, final_cols)), shape=(total_cons, total_vars)).tocsr()
        self.timing['build_saa_matrix'] = time.time() - build_start
        logger.info(f"  Block matrix constructed. Shape: {saa_matrix.shape}, NNZ: {saa_matrix.nnz} (Time: {self.timing['build_saa_matrix']:.2f}s)")
        return saa_matrix

    def _prepare_rhs_sense(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepares the combined RHS and Sense vectors for the SAA model."""
        if self.reader is None:
            raise RuntimeError("SMPS data must be loaded before preparing RHS/Sense.")

        N = self.num_scenarios
        total_cons = self.num_cons1 + N * self.num_cons2
        if total_cons == 0:
            return np.array([]), np.array([])

        b = self.reader.b
        rhs_parts = [b]
        if self.all_scenario_rhs_matrix is not None and self.num_cons2 > 0:
            rhs_parts.append(self.all_scenario_rhs_matrix.flatten())
        SAA_rhs = np.concatenate(rhs_parts)

        sense1 = convert_senses_to_gurobi(self.reader.sense1)
        sense2 = convert_senses_to_gurobi(self.reader.sense2)
        sense_parts = [sense1]
        if sense2.size > 0 and N > 0:
            sense_parts.append(np.tile(sense2, N))
        SAA_sense = np.concatenate(sense_parts)

        return SAA_rhs, SAA_sense

    def _prepare_variables(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepares combined objective, lower bounds, and upper bounds for all variables."""
        N = self.num_scenarios
        
        lb_x = np.where(np.isneginf(self.reader.lb_x), -GRB.INFINITY, self.reader.lb_x) if self.num_x > 0 else np.array([])
        ub_x = np.where(np.isposinf(self.reader.ub_x), GRB.INFINITY, self.reader.ub_x) if self.num_x > 0 else np.array([])
        lb_y = np.where(np.isneginf(self.reader.lb_y), -GRB.INFINITY, self.reader.lb_y) if self.num_y > 0 else np.array([])
        ub_y = np.where(np.isposinf(self.reader.ub_y), GRB.INFINITY, self.reader.ub_y) if self.num_y > 0 else np.array([])

        lb_combined = np.concatenate([lb_x] + [lb_y] * N) if self.num_x + self.num_y > 0 else np.array([])
        ub_combined = np.concatenate([ub_x] + [ub_y] * N) if self.num_x + self.num_y > 0 else np.array([])

        obj_c = self.reader.c
        obj_d = (self.reader.d / N) if self.reader.d.size > 0 and N > 0 else np.array([])
        obj_combined = np.concatenate([obj_c] + [obj_d] * N) if self.num_x + self.num_y > 0 else np.array([])
        
        return obj_combined, lb_combined, ub_combined

    def build_model(self, suppress_gurobi_output: bool = False):
        """Builds the Gurobi SAA model object."""
        if self.reader is None: self._load_smps_data()
        if self.all_scenario_rhs_matrix is None: self._generate_scenarios()

        logger.info("Step 3: Building the SAA Gurobi model...")
        step3_start_time = time.time()
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0 if suppress_gurobi_output else 1)
                if self.num_threads is not None: env.setParam('Threads', self.num_threads)
                env.start()

                model_name = f"SAA_{os.path.basename(self.core_filepath).split('.')[0]}_{self.num_scenarios}scen"
                self.saa_model = gp.Model(model_name, env=env)

                total_vars = self.num_x + self.num_scenarios * self.num_y
                if total_vars > 0:
                    obj, lb, ub = self._prepare_variables()
                    self.all_vars = self.saa_model.addMVar(shape=total_vars, lb=lb, ub=ub, obj=obj, name="vars")

                SAA_matrix = self._build_saa_matrix_coo()
                if SAA_matrix is not None and self.all_vars is not None:
                    SAA_rhs, SAA_sense = self._prepare_rhs_sense()
                    self.saa_constraints = self.saa_model.addMConstr(SAA_matrix, self.all_vars, SAA_sense, SAA_rhs, name="SAA_constraints")

                self.saa_model.ModelSense = GRB.MINIMIZE
                self.timing['build_model_total'] = time.time() - step3_start_time
                logger.info(f"SAA model structure built. (Time: {self.timing['build_model_total']:.2f}s)")

        except (gp.GurobiError, Exception) as e:
            logger.error(f"Error during model building: {e}")
            logger.error(traceback.format_exc())
            raise

    def solve(self):
        """Solves the SAA model and extracts results."""
        if self.saa_model is None:
            raise RuntimeError("Model must be built before solving.")

        logger.info("Step 4: Solving the SAA model...")
        step4_start_time = time.time()
        try:
            self.saa_model.optimize()
            self.solution_status = self.saa_model.Status

            if self.solution_status == GRB.OPTIMAL:
                logger.info("  Optimal solution found.")
                self.optimal_objective = self.saa_model.ObjVal
                self._extract_solution()
            else:
                logger.warning(f"  Optimization finished with status: {self.solution_status}")

        except (gp.GurobiError, Exception) as e:
            logger.error(f"Error during optimization: {e}")
            if hasattr(self.saa_model, 'Status'): self.solution_status = self.saa_model.Status
        finally:
            self.timing['solve_model'] = time.time() - step4_start_time
            logger.info(f"Optimization finished. (Time: {self.timing['solve_model']:.2f}s)")

    def _extract_solution(self):
        """Extracts primal and dual solutions from an optimal model."""
        if self.saa_model.SolCount == 0 or self.all_vars is None:
            logger.warning("Optimal status reported but no solution found.")
            return

        full_solution = self.all_vars.X
        N = self.num_scenarios

        if self.num_x > 0:
            self.stage1_solution = full_solution[:self.num_x]
        if self.num_y > 0 and N > 0:
            self.stage2_solution_all = full_solution[self.num_x:].reshape((N, self.num_y))

        if self.num_cons2 > 0 and N > 0 and self.saa_constraints is not None:
            try:
                all_duals = np.array(self.saa_constraints.Pi) * N
                self.stage2_duals_all = all_duals[self.num_cons1:].reshape((N, self.num_cons2))
            except (gp.GurobiError, AttributeError) as e:
                logger.warning(f"Could not extract duals: {e}")

    def report_results(self):
        """Prints a summary of the optimization results."""
        logger.info("-" * 40)
        logger.info("Step 5: Results Summary")
        if self.solution_status is None:
            logger.warning("Model has not been solved yet.")
            return

        status_map = {
            GRB.OPTIMAL: "Optimal", GRB.INFEASIBLE: "Infeasible", GRB.UNBOUNDED: "Unbounded",
            GRB.TIME_LIMIT: "Time Limit Reached", GRB.INTERRUPTED: "Interrupted",
        }
        status_str = status_map.get(self.solution_status, f"Gurobi Status Code {self.solution_status}")
        logger.info(f"Status: {status_str}")

        if self.solution_status == GRB.OPTIMAL:
            logger.info(f"Optimal Objective Value: {self.optimal_objective:.6f}")
            self._report_primal_results()
            self._report_dual_results()
        elif self.solution_status == GRB.INFEASIBLE:
            logger.warning("Model is Infeasible. Consider computing IIS for analysis.")
        elif self.solution_status == GRB.UNBOUNDED:
            logger.warning("Model is Unbounded.")
        logger.info("-" * 40)

    def _report_primal_results(self):
        """Helper to report primal solution values."""
        if self.stage1_solution is not None and self.num_x > 0:
            logger.info("Stage 1 Decision Variables (x) (First 5):")
            for i in range(min(self.num_x, 5)):
                logger.info(f"  x[{i}]: {self.stage1_solution[i]:.6f}")
        if self.stage2_solution_all is not None and self.num_y > 0:
            logger.info("Stage 2 Decision Variables (y_s) (Examples from first scenario):")
            for j in range(min(self.num_y, 5)):
                logger.info(f"  y[0,{j}]: {self.stage2_solution_all[0, j]:.6f}")

    def _report_dual_results(self):
        """Helper to report dual solution values."""
        if self.stage2_duals_all is not None and self.num_cons2 > 0:
            logger.info("Stage 2 Dual Multipliers (pi_s) (Examples from first scenario):")
            for k in range(min(self.num_cons2, 5)):
                logger.info(f"  pi[0,{k}]: {self.stage2_duals_all[0, k]:.6f}")

    def extract_basis(self):
        """Extracts the optimal basis information."""
        if self.solution_status != GRB.OPTIMAL:
            logger.info(f"Cannot get basis for non-optimal status ({self.solution_status}).")
            return

        logger.info("Step 6: Extracting optimal basis information...")
        extract_basis_start_time = time.time()
        try:
            constrs = self.saa_model.getConstrs()
            variables = self.saa_model.getVars()
            cbasis = np.array(self.saa_model.getAttr(GRB.Attr.CBasis, constrs), dtype=np.int8)
            vbasis = np.array(self.saa_model.getAttr(GRB.Attr.VBasis, variables), dtype=np.int8)

            N = self.num_scenarios
            self.basis_info = {
                'cbasis_y_all': cbasis[self.num_cons1:].reshape((N, self.num_cons2)) if self.num_cons2 > 0 else np.array([]).reshape(N, 0),
                'vbasis_y_all': vbasis[self.num_x:].reshape((N, self.num_y)) if self.num_y > 0 else np.array([]).reshape(N, 0)
            }
            self.timing['extract_basis'] = time.time() - extract_basis_start_time
            logger.info(f"Basis extraction complete. (Time: {self.timing['extract_basis']:.2f}s)")
        except (gp.GurobiError, Exception) as e:
            logger.error(f"Error during basis extraction: {e}")
            self.basis_info = None

    def save_results_to_hdf5(self, filepath: str):
        """Saves the generated/calculated data to an HDF5 file."""
        logger.info(f"Step 7: Saving results to HDF5 file: {filepath}...")
        save_start_time = time.time()
        try:
            with h5py.File(filepath, 'w') as f:
                self._save_hdf5_metadata(f)
                self._save_hdf5_scenarios(f)
                self._save_hdf5_solution(f)
                self._save_hdf5_basis(f)
            self.timing['save_hdf5'] = time.time() - save_start_time
            logger.info(f"Results saved successfully. (Time: {self.timing['save_hdf5']:.2f}s)")
        except ImportError:
            logger.error("h5py library not found. Cannot save to HDF5. Install with 'pip install h5py'")
        except Exception as e:
            logger.error(f"Failed to save to HDF5 file '{filepath}': {e}")
            logger.error(traceback.format_exc())

    def _save_hdf5_metadata(self, h5file):
        """Helper to save metadata to HDF5 file."""
        meta_grp = h5file.create_group("metadata")
        meta_grp.attrs['problem_name'] = os.path.basename(self.core_filepath).split('.')[0]
        meta_grp.attrs['num_scenarios'] = self.num_scenarios
        meta_grp.attrs['random_seed'] = self.random_seed
        meta_grp.attrs['num_x'] = self.num_x
        meta_grp.attrs['num_y'] = self.num_y
        meta_grp.attrs['num_cons1'] = self.num_cons1
        meta_grp.attrs['num_cons2'] = self.num_cons2
        meta_grp.attrs['solution_status_code'] = self.solution_status if self.solution_status is not None else -1
        if self.optimal_objective is not None:
            meta_grp.attrs['optimal_objective'] = self.optimal_objective

    def _save_hdf5_scenarios(self, h5file):
        """Helper to save scenario data to HDF5 file."""
        scen_grp = h5file.create_group("scenarios")
        data = self.batch_stochastic_parts if self.batch_stochastic_parts is not None else np.array([])
        scen_grp.create_dataset("stochastic_rhs_parts", data=data, compression="gzip")

    def _save_hdf5_solution(self, h5file):
        """Helper to save primal/dual solution to HDF5 file."""
        sol_grp = h5file.create_group("solution")
        if self.solution_status == GRB.OPTIMAL:
            primal_grp = sol_grp.create_group("primal")
            primal_grp.create_dataset("x", data=self.stage1_solution if self.stage1_solution is not None else np.array([]), compression="gzip")
            primal_grp.create_dataset("y_s", data=self.stage2_solution_all if self.stage2_solution_all is not None else np.array([]), compression="gzip")

            dual_grp = sol_grp.create_group("dual")
            if self.stage2_duals_all_clean is not None:
                logger.info("Saving cleaned duals (pi_s) from parallel worker.")
                dual_grp.create_dataset("pi_s", data=self.stage2_duals_all_clean, compression="gzip")
            else: # Fallback to original duals if clean ones are not available
                logger.warning("Cleaned duals not available, falling back to original SAA duals.")
                dual_grp.create_dataset("pi_s", data=self.stage2_duals_all if self.stage2_duals_all is not None else np.array([]), compression="gzip")

            if self.stage2_rc_all_clean is not None:
                logger.info("Saving cleaned reduced costs (rc) from parallel worker.")
                dual_grp.create_dataset("rc", data=self.stage2_rc_all_clean, compression="gzip")
            else:
                logger.info("No reduced cost information available, saving empty rc dataset.")
                dual_grp.create_dataset("rc", data=np.array([]))
        else:
            logger.info("No optimal solution found, saving empty solution datasets.")
            sol_grp.create_dataset("primal/x", data=np.array([]))
            sol_grp.create_dataset("primal/y_s", data=np.array([]))
            sol_grp.create_dataset("dual/pi_s", data=np.array([]))
            sol_grp.create_dataset("dual/rc", data=np.array([]))

    def _save_hdf5_basis(self, h5file):
        """Helper to save basis information to HDF5 file."""
        basis_grp = h5file.create_group("basis")
        if self.cbasis_y_all_clean is not None and self.vbasis_y_all_clean is not None:
            logger.info("Saving cleaned basis information from parallel worker.")
            basis_grp.create_dataset('cbasis_y_all', data=self.cbasis_y_all_clean, compression="gzip")
            basis_grp.create_dataset('vbasis_y_all', data=self.vbasis_y_all_clean, compression="gzip")
        else:
            logger.info("No cleaned basis information available, saving empty basis datasets.")
            basis_grp.create_dataset('cbasis_y_all', data=np.array([]))
            basis_grp.create_dataset('vbasis_y_all', data=np.array([]))

    def _resolve_second_stage_in_parallel(self):
        """
        Uses ParallelSecondStageWorker to re-solve all second-stage problems
        to get clean duals, reduced costs, and basis information.
        """
        if self.solution_status != GRB.OPTIMAL:
            logger.warning("Skipping parallel re-solve because SAA was not optimal.")
            return
        if self.reader is None or self.stage1_solution is None:
            logger.error("Cannot re-solve: SMPS reader not initialized or no stage 1 solution.")
            return
        if self.num_cons2 == 0:
            logger.info("No second-stage problems to re-solve.")
            return

        logger.info("Step 6a: Re-solving second-stage problems in parallel for clean results...")
        resolve_start_time = time.time()

        num_workers = self.num_threads if self.num_threads is not None and self.num_threads > 0 else os.cpu_count() or 1
        
        try:
            with ParallelSecondStageWorker.from_smps_reader(self.reader, num_workers) as worker:
                # We need the deviation from the mean RHS (delta_r) for the worker
                delta_r_batch = self.reader.get_short_delta_r(self.batch_stochastic_parts)

                # The `solve_batch` method will return all required information
                _, _, pi_sol, rc_sol, vbasis, cbasis, _ = worker.solve_batch(
                    x=self.stage1_solution,
                    short_delta_r_batch=delta_r_batch,
                    nontrivial_rc_only=True # As per requirement
                )

                # Store the clean results
                self.stage2_duals_all_clean = pi_sol
                self.stage2_rc_all_clean = rc_sol
                self.vbasis_y_all_clean = vbasis
                self.cbasis_y_all_clean = cbasis

            self.timing['resolve_parallel'] = time.time() - resolve_start_time
            logger.info(f"Parallel re-solve complete. (Time: {self.timing['resolve_parallel']:.2f}s)")

        except Exception as e:
            logger.error(f"Error during parallel re-solve of second stage: {e}")
            logger.error(traceback.format_exc())


    def run_pipeline(self, solve_model=True, save_hdf5=False, hdf5_filepath="saa_results.h5"):
        """Runs the full pipeline: Load -> Generate -> Build -> Solve -> Report -> Save."""
        pipeline_start = time.time()
        try:
            self.build_model()
            if solve_model:
                self.solve()
                self.report_results()
                # self.extract_basis() # No longer needed as we get basis from parallel worker

                if self.solution_status == GRB.OPTIMAL:
                    self._resolve_second_stage_in_parallel()

                if save_hdf5 and self.solution_status is not None:
                    self.save_results_to_hdf5(hdf5_filepath)
        except Exception as e:
            logger.error(f"\n--- PIPELINE FAILED: {e} ---")
            logger.error(traceback.format_exc())
        finally:
            self.timing['full_pipeline'] = time.time() - pipeline_start
            logger.info(f"\nPipeline execution finished. (Total Time: {self.timing['full_pipeline']:.2f}s)")
