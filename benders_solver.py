import os
import time
from typing import Optional, Tuple
import numpy as np
import scipy.linalg # Used in original script, ensure it's available
import h5py # For loading initial basis
import logging
from pathlib import Path

# Assuming your custom modules are in the same directory or accessible via PYTHONPATH
from smps_reader import SMPSReader
from argmax_operation import ArgmaxOperation
# from benders import BendersMasterProblem # For type hinting if Regularized inherits from it
from regularized_benders import RegularizedBendersMasterProblem
from parallel_second_stage_worker import ParallelSecondStageWorker

class BendersSolver:
    def __init__(self, config, logger_name='BendersSolver'):
        self.config = config
        self.logger = logging.getLogger(logger_name)

        self.reader: SMPSReader = None
        self.argmax_op: ArgmaxOperation = None
        self.master_problem: RegularizedBendersMasterProblem = None
        self.parallel_worker: ParallelSecondStageWorker = None
        
        self.x_init: np.ndarray = None
        self.x: np.ndarray = None
        self.rho: float = None
        self.short_delta_r: np.ndarray = None # This is the (h_omega - h_bar) sample pool
        self.c_vector: np.ndarray = None # First-stage cost vector

    def _setup_problem(self):
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
            num_x_vars = len(self.reader.stage1_var_names) if hasattr(self.reader, 'stage1_var_names') and self.reader.stage1_var_names else self.c_vector.shape[0]
            self.x_init = np.zeros(num_x_vars) 
            self.logger.warning(f"Defaulting x_init to zeros due to HDF5 loading error.")
        
        if self.x_init is None or self.x_init.shape[0] != self.c_vector.shape[0]:
            self.logger.warning(
                f"x_init shape mismatch (is {self.x_init.shape if self.x_init is not None else 'None'}) "
                f"or not loaded. Defaulting to zeros of len {self.c_vector.shape[0]}."
            )
            self.x_init = np.zeros(self.c_vector.shape[0])
        
        self.logger.info("--- Problem Setup Complete ---")

    def _solve_master_problem(self):
        start_time = time.time()
        # isinstance check is fine, but since it's always RegularizedBendersMasterProblem:
        self.master_problem.set_regularization_strength(self.rho)
        self.master_problem.set_regularization_center(self.x)
        
        x_next, master_total_obj_val, status_code = self.master_problem.solve()
        solve_time = time.time() - start_time
        return x_next, master_total_obj_val, status_code, solve_time

    def _perform_argmax_operation(self, current_x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
        start_time = time.time()
        cut_info = self.argmax_op.calculate_cut(current_x)
        
        if cut_info is None:
            self.logger.info("ArgmaxOperation.calculate_cut returned None (e.g., MAX_PI=0 or no duals/scenarios). Argmax cost will be NaN.")
            argmax_estim_q_x = np.nan
            best_k_scores = np.array([], dtype=float)
            best_k_index = np.array([], dtype=int) # Ensure it's an array for get_basis
        else:
            alpha_pre, beta_pre, best_k_scores, best_k_index = cut_info
            argmax_estim_q_x = alpha_pre + beta_pre @ current_x 
        
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
            self.rho *= self.config.get('rho_increase_factor', 1.01)
            cut_added = True
            self.logger.debug(f"Cut added. New rho: {self.rho:.4e}")
        else:
            self.rho *= self.config.get('rho_decrease_factor', 0.98)
            min_rho = self.config.get('min_rho', 1e-5)
            if self.rho < min_rho: self.rho = min_rho
            self.logger.debug(f"No cut added. New rho: {self.rho:.4e}")

        calc_time = time.time() - start_time
        return subproblem_actual_q_x, cut_added, calc_time

    def _log_iteration_data(self, iteration_metrics):
        log_metrics = {}
        for key, value in iteration_metrics.items():
            if isinstance(value, np.ndarray):
                log_metrics[key] = value.tolist() 
            else:
                log_metrics[key] = value
        
        # UPDATED LOGGING TERMS
        log_parts = [f"Iter: {log_metrics.get('iteration', 'N/A'):>3}"]
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
        log_parts.append(f"SimplexIter: {log_metrics.get('mean_simplex_iter', 0):.2f} (0-iter: {log_metrics.get('num_zero_iter', 0)})")
        log_parts.append(f"ArgmaxNumPi: {log_metrics.get('argmax_num_pi', 0)}")
        self.logger.info(" | ".join(log_parts))

    def run(self):
        self._setup_problem()

        if self.x_init is None: # Should have been handled by _setup_problem
             self.logger.error("x_init is None after setup. Cannot proceed.")
             return

        self.x = self.x_init.copy()
        self.rho = self.config['initial_rho']
        
        self.logger.info("--- Starting Benders Decomposition Loop ---")
        max_iterations = self.config['max_iterations']

        for iter_count in range(1, max_iterations + 1):
            iter_time_start = time.time()

            # 1. Solve Master Problem
            x_candidate, master_total_obj_candidate, master_status, master_time = self._solve_master_problem()
            
            if master_status != 2: 
                self.logger.error(f"Iteration {iter_count}: Master problem solve failed with status {master_status}. Stopping.")
                break
            
            dist_moved = scipy.linalg.norm(x_candidate - self.x)
            self.x = x_candidate.copy()
            self.logger.debug(f"Iter {iter_count}: Master solved. Dist_moved: {dist_moved:.4e}")

            # 2. Warmstart: Calculate cut with existing duals in ArgmaxOperation
            argmax_estim_q_x, best_k_scores, best_k_indices_for_basis, argmax_time = self._perform_argmax_operation(self.x)
    
            vbasis_batch, cbasis_batch = self.argmax_op.get_basis(best_k_indices_for_basis)
            if best_k_indices_for_basis.size > 0:
                 self.logger.debug(f"Iter {iter_count}: Argmax op used {len(np.unique(best_k_indices_for_basis))} unique duals for warmstart basis from {best_k_indices_for_basis.size} selected scenarios.")
            else:
                 self.logger.debug(f"Iter {iter_count}: Argmax op provided no basis (best_k_indices empty).")

            # 3. Subproblem Solver
            obj_all_sp, pi_all_sp, rc_all_sp, vbasis_out_sp, cbasis_out_sp, simplex_iter_count, subproblem_time = \
                self._solve_subproblems(self.x, vbasis_batch, cbasis_batch)
            
            # Log simplex iteration counts
            # Count mean simplex iterations for all subproblems, and number subproblems that were immediately solved with a warmstart (simplex_iter_count == 0)
            mean_simplex_iter = np.mean(simplex_iter_count)
            num_zero_iter = np.sum(simplex_iter_count == 0)

            # 4. Calculate score differences and Add newly found duals to ArgmaxOperation
            # Note that solving subproblem solvers raise the score. So the difference is the subproblem score minus the argmax score.
            # As a heuristic, we can use the best k scores from the argmax operation to determine which duals to add.
            score_differences = obj_all_sp - best_k_scores
            coverage_fraction, duals_added_count, duals_update_time = self._update_argmax_duals(pi_all_sp, rc_all_sp, vbasis_out_sp, cbasis_out_sp, score_differences)
            self.logger.debug(f"Iter {iter_count}: Added {duals_added_count} new duals to ArgmaxOp. Total duals: {self.argmax_op.num_pi}. Time: {duals_update_time:.2f}s")

            # 5. Calculate the new cut based on ALL subproblem solutions
            subproblem_actual_q_x, cut_added, cut_calc_time = self._calculate_and_add_cut(self.x, pi_all_sp)

            # --- Collect Metrics for Logging (with UPDATED KEYS) ---
            c_transpose_x = self.c_vector @ self.x
            
            cost_from_master = master_total_obj_candidate 
            cost_from_cut = c_transpose_x + subproblem_actual_q_x if subproblem_actual_q_x > -np.inf else float('inf')
            cost_from_argmax = c_transpose_x + argmax_estim_q_x

            # Gap calculation remains the same conceptually, using the new named values
            # Typically, the master problem provides a lower bound on the true objective,
            # and the subproblem evaluation provides an upper bound (or an estimate of one if not all scenarios solved).
            # So, gap = cost_from_cut - cost_from_master
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
                "x_norm": scipy.linalg.norm(self.x),
                "argmax_num_pi": self.argmax_op.num_pi,
                "coverage_fraction": coverage_fraction,
            }
            self._log_iteration_data(iteration_metrics)

            if iter_count == max_iterations:
                self.logger.info("Reached maximum iterations.")

        self.logger.info("--- Benders Decomposition Loop Finished ---")
        self.cleanup() # Call cleanup

    def cleanup(self): # Added cleanup method
        self.logger.info("BendersSolver cleanup called.")
        # Implement any specific resource cleanup if needed, e.g., for parallel_worker if it has a shutdown method.


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- Define Configuration for BendersSolver ---

    solver_config = {
        'smps_core_file': "./smps_data/ssn/ssn.mps",
        'smps_time_file': "./smps_data/ssn/ssn.tim",
        'smps_sto_file': "./smps_data/ssn/ssn.sto",
        'input_h5_basis_file': './ssn_5000scen_results.h5',
        'MAX_PI': 500000,
        'MAX_OMEGA': 100000, 
        'SCENARIO_BATCH_SIZE': 1000, 
        'NUM_SAMPLES_FOR_POOL': 100000, 
        'ETA_LOWER_BOUND': 0.0,
        'initial_rho': 0.1,
        'rho_increase_factor': 1.05,
        'rho_decrease_factor': 0.95,
        'min_rho': 1e-6,
        'tolerance': 1e-4,
        'max_iterations': 20, 
        'num_duals_to_add_per_iteration': 20000,
        'argmax_tol_cutoff': 1e-4,
        # 'num_workers': 31, # Example: os.cpu_count() if os.cpu_count() else 1,
        'instance_name': "ssn_example_new_log"
    }


    # Example configuration for CEP
    # solver_config = {
    #     'smps_core_file': "./smps_data/cep/cep.mps",
    #     'smps_time_file': "./smps_data/cep/cep.tim",
    #     'smps_sto_file': "./smps_data/cep/cep.sto",
    #     'input_h5_basis_file': './cep_100scen_results.h5',
    #     'MAX_PI': 10000,
    #     'MAX_OMEGA': 100000, 
    #     'SCENARIO_BATCH_SIZE': 1000, 
    #     'NUM_SAMPLES_FOR_POOL': 100000, 
    #     'ETA_LOWER_BOUND': 0.0,
    #     'initial_rho': 0.1,
    #     'rho_increase_factor': 1.05,
    #     'rho_decrease_factor': 0.95,
    #     'min_rho': 1e-6,
    #     'tolerance': 1e-4,
    #     'max_iterations': 20, 
    #     'num_duals_to_add_per_iteration': 1000, 
    #     # 'num_workers': 4, # Example: os.cpu_count() if os.cpu_count() else 1,
    #     'instance_name': "cep_example_new_log"
    # }


    solver = BendersSolver(config=solver_config, logger_name='MyBenders')
    
    try:
        solver.run()
    except Exception as e:
        logging.getLogger('MyBenders').error(f"An error occurred during solver execution: {e}", exc_info=True)
    finally:
        # Ensure cleanup is called even if run() fails
        if 'solver' in locals() and hasattr(solver, 'cleanup'):
            solver.cleanup()


    logging.info("--- Main script finished ---")