import os
import time
from typing import Optional, Tuple
import numpy as np
import scipy.linalg # Used in original script, ensure it's available
import h5py # For loading initial basis
import logging
from pathlib import Path
import sys

# This allows the script to be run from anywhere and still find the src module
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

from src.smps_reader import SMPSReader
from src.argmax_operation import ArgmaxOperation
# from src.benders import BendersMasterProblem # For type hinting if Regularized inherits from it
from src.regularized_benders import RegularizedBendersMasterProblem
from src.parallel_second_stage_worker import ParallelSecondStageWorker

class BendersSolver:
    """
    Orchestrates the Benders decomposition algorithm using a regularized master problem,
    a parallelized second-stage solver, and a GPU-accelerated argmax operation for cut selection.
    """
    def __init__(self, config, logger_name='BendersSolver'):
        """
        Initializes the BendersSolver with a configuration dictionary.

        Args:
            config (dict): A dictionary containing all the necessary parameters for the solver.
            logger_name (str): The name for the logger instance.
        """
        self.config = config
        self.logger = logging.getLogger(logger_name)

        self.reader: SMPSReader = None
        self.argmax_op: ArgmaxOperation = None
        self.master_problem: RegularizedBendersMasterProblem = None
        self.parallel_worker: ParallelSecondStageWorker = None
        
        self.x_init: np.ndarray = None
        self.x_candidate: np.ndarray = None
        self.x_incumbent: np.ndarray = None
        self.rho: float = None
        self.short_delta_r: np.ndarray = None # This is the (h_omega - h_bar) sample pool
        self.c_vector: np.ndarray = None # First-stage cost vector

    def _setup_problem(self):
        """
        Initializes and configures all components required for the Benders decomposition algorithm.
        """
        self.logger.info("--- Starting Problem Setup ---")
        # 1. Load SMPS Data
        self.reader = SMPSReader(
            core_file=self.config['smps_core_file'],
            time_file=self.config['smps_time_file'],
            sto_file=self.config['smps_sto_file']
        )
        self.reader.load_and_extract()
        # Assuming self.reader.c is the first-stage cost vector after load_and_extract()
        self.c_vector = self.reader.c 
        self.logger.info("SMPSReader loaded and data extracted.")

        # 2. Initialize ArgmaxOperation
        self.argmax_op = ArgmaxOperation.from_smps_reader(
            reader=self.reader,
            MAX_PI=self.config['MAX_PI'],
            MAX_OMEGA=self.config['MAX_OMEGA'],
            scenario_batch_size=self.config['SCENARIO_BATCH_SIZE']
        )
        self.logger.info("ArgmaxOperation initialized.")

        # 3. Initialize BendersMasterProblem
        self.master_problem = RegularizedBendersMasterProblem.from_smps_reader(self.reader)
        self.logger.info("RegularizedBendersMasterProblem initialized.")
        
        self.master_problem.create_benders_problem(model_name="InitialBendersMaster")
        self.master_problem.set_eta_lower_bound(self.config['ETA_LOWER_BOUND'])
        self.logger.info(f"Master problem created. ETA lower bound set to {self.config['ETA_LOWER_BOUND']}.")

        # 4. Initialize ParallelSecondStageWorker
        default_num_workers = os.cpu_count() if os.cpu_count() else 1
        num_workers_for_parallel = self.config.get('num_workers', default_num_workers)
        self.parallel_worker = ParallelSecondStageWorker.from_smps_reader(
            reader=self.reader,
            num_workers=num_workers_for_parallel
        )
        self.logger.info(f"ParallelSecondStageWorker initialized with {self.parallel_worker.num_workers} workers.")

        # 5. Generate Sample Pool (short_delta_r)
        # set seed for reproducibility
        np.random.seed(42)
        sample_pool_rhs_realizations = self.reader.sample_stochastic_rhs_batch(
            num_samples=self.config['NUM_SAMPLES_FOR_POOL']
        )
        self.short_delta_r = self.reader.get_short_delta_r(sample_pool_rhs_realizations)
        self.argmax_op.add_scenarios(self.short_delta_r) 
        self.logger.info(f"Sample pool of {self.short_delta_r.shape[0]} scenarios (short_delta_r) generated and added to ArgmaxOperation.")

        # 6. Load Initial Basis and x_init from input HDF5
        h5_path = self.config['input_h5_basis_file']
        try:
            with h5py.File(h5_path, 'r') as hf:
                if '/solution/primal/x' in hf:
                    self.x_init = hf['/solution/primal/x'][:]
                    self.logger.info(f"Loaded x_init from {h5_path} with shape {self.x_init.shape}")
                else:
                    self.logger.warning(f"x_init not found in {h5_path}. Defaulting to zeros.")
                    num_x_vars = len(self.reader.stage1_var_names) # Ensure stage1_var_names is available
                    self.x_init = np.zeros(num_x_vars)

                if '/solution/dual/pi_s' in hf and \
                   '/basis/vbasis_y_all' in hf and \
                   '/basis/cbasis_y_all' in hf:
                    pi_s_all = hf['/solution/dual/pi_s'][:]
                    vbasis_y_all = hf['/basis/vbasis_y_all'][:]
                    cbasis_y_all = hf['/basis/cbasis_y_all'][:]
                    
                    rc_dim_argop = self.argmax_op.NUM_BOUNDED_VARS 
                    rc_template = np.zeros(rc_dim_argop) 

                    added_count = 0
                    for i in range(pi_s_all.shape[0]):
                        if self.argmax_op.add_pi(pi_s_all[i], rc_template.copy(), vbasis_y_all[i], cbasis_y_all[i]):
                            added_count +=1
                    self.logger.info(f"Loaded and added {added_count} initial dual solutions to ArgmaxOperation from {h5_path}.")
                else:
                    self.logger.warning(f"Initial duals/basis not fully found in {h5_path}. ArgmaxOperation may start with no preloaded duals.")

        except Exception as e:
            self.logger.error(f"Error loading initial basis from {h5_path}: {e}", exc_info=True)
            raise # Ensure the error is raised to stop execution if loading fails
            # num_x_vars = len(self.reader.stage1_var_names)
            # self.x_init = np.zeros(num_x_vars) 
            # self.logger.warning(f"Defaulting x_init to zeros due to HDF5 loading error.")
        
        if self.x_init is None or self.x_init.shape[0] != self.c_vector.shape[0]:
            self.logger.warning(
                f"x_init shape mismatch (is {self.x_init.shape if self.x_init is not None else 'None'}) "
                f"or not loaded. Defaulting to zeros of len {self.c_vector.shape[0]}."
            )
            self.x_init = np.zeros(self.c_vector.shape[0])
        
        self.x_incumbent = self.x_init.copy()
        self.logger.info("--- Problem Setup Complete ---")

    def _solve_master_problem(self):
        """
        Solves the regularized master problem and returns the solution and timing information.
        """
        start_time = time.time()
        # isinstance check is fine, but since it's always RegularizedBendersMasterProblem:
        self.master_problem.set_regularization_strength(self.rho)
        self.master_problem.set_regularization_center(self.x_incumbent)
        
        x_next, master_total_obj_val, status_code = self.master_problem.solve()
        solve_time = time.time() - start_time
        return x_next, master_total_obj_val, status_code, solve_time

    def _perform_argmax_operation(self, current_x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Performs the argmax operation to calculate the estimated cost and select the best dual solutions.

        Args:
            current_x: The current first-stage decision vector.

        Returns:
            A tuple containing the estimated cost, scores, indices of best solutions, and execution time.
        """
        start_time = time.time()
        self.argmax_op.find_best_k(current_x)
        cut_info = self.argmax_op.calculate_cut_coefficients()

        if cut_info is None:
            self.logger.warning("ArgmaxOperation.calculate_cut_coefficients returned None. Argmax cost will be NaN.")
            argmax_estim_q_x = np.nan
            best_k_scores = np.array([], dtype=float)
            best_k_index = np.array([], dtype=int)
        else:
            alpha_pre, beta_pre = cut_info
            argmax_estim_q_x = alpha_pre + beta_pre @ current_x
            best_k_scores, best_k_index = self.argmax_op.get_best_k_results()

        solve_time = time.time() - start_time
        return argmax_estim_q_x, best_k_scores, best_k_index, solve_time

    def _solve_subproblems(self, current_x: np.ndarray, vbasis_batch: Optional[np.ndarray], cbasis_batch: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solves a batch of subproblems in parallel using the current solution vector and optional basis information.

        Args:
            current_x (np.ndarray): The current solution vector for the master problem.
            vbasis_batch (Optional[np.ndarray]): Optional batch of variable basis statuses for warm-starting the solver.
            cbasis_batch (Optional[np.ndarray]): Optional batch of constraint basis statuses for warm-starting the solver.

        Returns:
            Tuple[
                np.ndarray,  # obj_all: Objective values for all subproblems.
                np.ndarray,  # pi_all: Dual variables for all subproblems.
                np.ndarray,  # rc_all: Reduced costs for all subproblems.
                np.ndarray,  # vbasis_out: Output variable basis statuses after solving.
                np.ndarray,  # cbasis_out: Output constraint basis statuses after solving.
                np.ndarray,  # simplex_iter_count_all: Number of simplex iterations for each subproblem.
                float        # solve_time: Total time taken to solve all subproblems.
            ]
        """
        start_time = time.time()
        obj_all, y_all, pi_all, rc_all, vbasis_out, cbasis_out, simplex_iter_count_all = \
            self.parallel_worker.solve_batch(current_x, self.short_delta_r, vbasis_batch, cbasis_batch)
        solve_time = time.time() - start_time
        return obj_all, pi_all, rc_all, vbasis_out, cbasis_out, simplex_iter_count_all, solve_time

    def _update_argmax_duals(self, pi_all, rc_all, vbasis_out, cbasis_out, scores_difference=None):
        """
        Updates the ArgmaxOperation with new dual solutions based on score differences.

        Args:
            pi_all: All dual variables from the subproblems.
            rc_all: All reduced costs from the subproblems.
            vbasis_out: All variable basis statuses from the subproblems.
            cbasis_out: All constraint basis statuses from the subproblems.
            scores_difference: The difference between subproblem and argmax scores.

        Returns:
            A tuple containing the coverage fraction, the number of added duals, and the execution time.
        """
        start_time = time.time()
        num_to_add = self.config.get('num_duals_to_add_per_iteration', 10000)
        ARGMAX_CUTOFF_TOLERANCE = self.config.get('argmax_tol_cutoff', 1e-4)
        num_available_duals = pi_all.shape[0]
        
        # Calculate coverage fraction: if the scores difference is not None,
        # and that pi has score difference smaller than the cutoff, we don't add it.
        # We say it is covered.
        if scores_difference is not None:
            coverage_fraction = np.sum(scores_difference < ARGMAX_CUTOFF_TOLERANCE) / num_available_duals
            self.logger.debug(f"Coverage fraction: {coverage_fraction:.4f} (cutoff: {ARGMAX_CUTOFF_TOLERANCE})")
        else:
            coverage_fraction = None

        if num_available_duals == 0:
            self.logger.debug("No duals available from subproblems to add to ArgmaxOperation.")
            return None, 0, time.time() - start_time

        actual_num_to_add = min(num_to_add, num_available_duals)
        
        if scores_difference is not None:
            # Obtain the indices of the duals with the largest score differences
            sorted_indices = np.argsort(scores_difference)[::-1]
            chosen_indices = sorted_indices[:actual_num_to_add]
        else:
            # If no scores_difference is provided, randomly select duals
            chosen_indices = np.random.choice(
                np.arange(num_available_duals), 
                size=actual_num_to_add, 
                replace=False
            )
        
        added_count = 0
        for s_idx in chosen_indices:
            # Only add if the difference is larger than the cutoff
            if scores_difference is not None and scores_difference[s_idx] < ARGMAX_CUTOFF_TOLERANCE:
                continue
            if self.argmax_op.add_pi(pi_all[s_idx], rc_all[s_idx], vbasis_out[s_idx], cbasis_out[s_idx]):
                added_count += 1
        update_time = time.time() - start_time
        return coverage_fraction, added_count, update_time

    def _calculate_and_add_cut(self, current_x, pi_all_from_subproblems):
        """
        Calculates a new Benders cut and adds it to the master problem if it's violated.

        Args:
            current_x: The current first-stage decision vector.
            pi_all_from_subproblems: All dual variables from the subproblems.

        Returns:
            A tuple containing the actual cost, a boolean indicating if a cut was added, and the execution time.
        """
        start_time = time.time()
        
        if pi_all_from_subproblems.shape[0] == 0:
            self.logger.warning("Cannot calculate cut, no duals from subproblems.")
            return -np.inf, False, time.time() - start_time

        mean_pi = np.mean(pi_all_from_subproblems, axis=0)
        
        # Ensure reader attributes are available
        alpha_fixed_part = mean_pi @ self.reader.r_bar 
        pattern = self.reader.stochastic_rows_relative_indices
        short_pi_s = pi_all_from_subproblems[:, pattern]
        alpha_variable_part_per_scenario = np.sum(short_pi_s * self.short_delta_r, axis=1)
        alpha_variable_part_mean = np.mean(alpha_variable_part_per_scenario)
        
        alpha = alpha_fixed_part + alpha_variable_part_mean
        beta = -mean_pi @ self.reader.C 

        subproblem_actual_q_x = alpha + beta @ current_x

        current_master_epigraph_eta_at_x = self.master_problem.calculate_epigraph_value(current_x)
        
        cut_added = False
        if subproblem_actual_q_x > current_master_epigraph_eta_at_x + self.config['tolerance']:
            self.master_problem.add_optimality_cut(beta, alpha)
            cut_added = True

        calc_time = time.time() - start_time
        return subproblem_actual_q_x, cut_added, calc_time

    def _log_iteration_data(self, iteration_metrics):
        """
        Logs the metrics for a single iteration of the Benders decomposition algorithm.

        Args:
            iteration_metrics (dict): A dictionary containing the metrics for the iteration.
        """
        log_metrics = {}
        for key, value in iteration_metrics.items():
            if isinstance(value, np.ndarray):
                log_metrics[key] = value.tolist() 
            else:
                log_metrics[key] = value
        
        # UPDATED LOGGING TERMS
        log_parts = [f"Iter: {log_metrics.get('iteration', 'N/A'):>3}"]
        log_parts.append(f"Success: {'Y' if log_metrics.get('successful_step', False) else 'N'}")
        log_parts.append(f"FCD: {log_metrics.get('fcd', float('inf')):.4f}")
        log_parts.append(f"CostMaster: {log_metrics.get('cost_from_master', float('nan')):.4e}")
        log_parts.append(f"CostCut: {log_metrics.get('cost_from_cut', float('nan')):.4e}")
        log_parts.append(f"Gap: {log_metrics.get('gap', float('nan')):.4e}")
        log_parts.append(f"CostArgmax: {log_metrics.get('cost_from_argmax', float('nan')):.4e}")
        log_parts.append(f"Rho: {log_metrics.get('rho', float('nan')):.2e}")
        log_parts.append(f"CutAdded: {'Y' if log_metrics.get('cut_added') else 'N'}")
        log_parts.append(f"TimeIter: {log_metrics.get('total_iteration_time', 0):.2f}s")
        log_parts.append(f"(M:{log_metrics.get('master_solve_time',0):.2f}s, "
                         f"A:{log_metrics.get('argmax_op_time',0):.2f}s, "
                         f"S:{log_metrics.get('subproblem_solve_time',0):.2f}s, "
                         f"C:{log_metrics.get('cut_calculation_time',0):.2f}s)")
        log_parts.append(f"ArgmaxNumPi: {log_metrics.get('argmax_num_pi', 0)}")
        log_parts.append(f"CoverageFraction: {log_metrics.get('coverage_fraction', 0):.4f}")
        self.logger.info(" | ".join(log_parts))

    def run(self):
        """
        Executes the Benders decomposition algorithm with an incumbent-based strategy.
        """
        self._setup_problem()

        if self.x_incumbent is None:
            self.logger.error("x_incumbent is None after setup. Cannot proceed.")
            return

        self.rho = self.config['initial_rho']
        gamma = self.config['gamma']
        rho_decrease_factor = self.config['rho_decrease_factor']
        rho_increase_factor = self.config['rho_increase_factor']
        min_rho = self.config['min_rho']
        
        self.logger.info("--- Starting Benders Decomposition Loop (Incumbent Strategy) ---")
        max_iterations = self.config['max_iterations']

        for iter_count in range(1, max_iterations + 1):
            iter_time_start = time.time()

            # 1. Solve Master Problem
            x_candidate, master_total_obj_candidate, master_status, master_time = self._solve_master_problem()
            
            if master_status != 2:
                self.logger.error(f"Iteration {iter_count}: Master problem solve failed with status {master_status}. Stopping.")
                break
            
            self.x_candidate = x_candidate

            # Calculate perceived decrease
            perceived_decrease = self.master_problem.calculate_estimated_objective(self.x_incumbent) - \
                                 self.master_problem.calculate_estimated_objective(self.x_candidate)

            if perceived_decrease < 0.0:
                self.logger.warning(f"Iteration {iter_count}: Perceived decrease is negative ({perceived_decrease:.4f}). Numerical issues may be present.")
                perceived_decrease = 0.0
            
            # 2. Warmstart: Calculate cut with existing duals in ArgmaxOperation
            argmax_estim_q_x, best_k_scores, best_k_indices_for_basis, argmax_time = self._perform_argmax_operation(self.x_candidate)
    
            vbasis_batch, cbasis_batch = self.argmax_op.get_basis(best_k_indices_for_basis)

            # 3. Subproblem Solver
            obj_all_sp, pi_all_sp, rc_all_sp, vbasis_out_sp, cbasis_out_sp, simplex_iter_count, subproblem_time = \
                self._solve_subproblems(self.x_candidate, vbasis_batch, cbasis_batch)
            
            mean_simplex_iter = np.mean(simplex_iter_count)
            num_zero_iter = np.sum(simplex_iter_count == 0)

            # 4. Add newly found duals to ArgmaxOperation
            if best_k_scores.size > 0:
                score_differences = obj_all_sp - best_k_scores
            else:
                score_differences = None
            
            coverage_fraction, duals_added_count, duals_update_time = self._update_argmax_duals(pi_all_sp, rc_all_sp, vbasis_out_sp, cbasis_out_sp, score_differences)
            self.logger.debug(f"Iter {iter_count}: Added {duals_added_count} new duals to ArgmaxOp. Total duals: {self.argmax_op.num_pi}.")

            # 5. Calculate the new cut and actual decrease
            subproblem_actual_q_x, cut_added, cut_calc_time = self._calculate_and_add_cut(self.x_candidate, pi_all_sp)
            
            actual_decrease = self.master_problem.calculate_estimated_objective(self.x_incumbent) - \
                              self.master_problem.calculate_estimated_objective(self.x_candidate)

            # 6. Update incumbent and rho based on progress
            successful_step = actual_decrease >= gamma * perceived_decrease
            if successful_step:
                self.logger.debug(f"Iter {iter_count}: Successful step. Updating incumbent and decreasing rho.")
                self.x_incumbent = self.x_candidate.copy()
                self.rho = max(min_rho, self.rho * rho_decrease_factor)
            else:
                self.logger.debug(f"Iter {iter_count}: Unsuccessful step. Increasing rho.")
                self.rho = self.rho * rho_increase_factor
            
            fcd = actual_decrease / perceived_decrease if perceived_decrease != 0 else float('inf')

            # --- Collect Metrics for Logging ---
            c_transpose_x = self.c_vector @ self.x_candidate
            cost_from_master = master_total_obj_candidate
            cost_from_cut = c_transpose_x + subproblem_actual_q_x if subproblem_actual_q_x > -np.inf else float('inf')
            cost_from_argmax = c_transpose_x + argmax_estim_q_x
            gap = cost_from_cut - cost_from_master
            total_iteration_time = time.time() - iter_time_start

            iteration_metrics = {
                "iteration": iter_count,
                "master_solve_time": master_time,
                "argmax_op_time": argmax_time,
                "subproblem_solve_time": subproblem_time,
                "duals_update_time": duals_update_time,
                "cut_calculation_time": cut_calc_time,
                "total_iteration_time": total_iteration_time,
                "first_stage_cost_cx": c_transpose_x,
                "cost_from_master": cost_from_master,
                "cost_from_argmax": cost_from_argmax,
                "subproblem_actual_q_x": subproblem_actual_q_x,
                "cost_from_cut": cost_from_cut,
                "rho": self.rho,
                "cut_added": cut_added,
                "gap": gap,
                "mean_simplex_iter": mean_simplex_iter,
                "num_zero_iter": num_zero_iter,
                "x_norm": scipy.linalg.norm(self.x_candidate),
                "argmax_num_pi": self.argmax_op.num_pi,
                "coverage_fraction": coverage_fraction,
                "perceived_decrease": perceived_decrease,
                "actual_decrease": actual_decrease,
                "fcd": fcd,
                "successful_step": successful_step
            }
            self._log_iteration_data(iteration_metrics)

            if iter_count == max_iterations:
                self.logger.info("Reached maximum iterations.")

        self.logger.info("--- Benders Decomposition Loop Finished ---")
        self.cleanup()

    def cleanup(self):
        """
        Cleans up resources used by the solver.
        """
        self.logger.info("BendersSolver cleanup called.")
        # Implement any specific resource cleanup if needed, e.g., for parallel_worker if it has a shutdown method.
