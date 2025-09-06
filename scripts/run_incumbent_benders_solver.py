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
from src.control_variate_validator import ControlVariateValidator

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
        self.validator: ControlVariateValidator = None
        
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
            scenario_batch_size=self.config['SCENARIO_BATCH_SIZE'],
            NUM_CANDIDATES=self.config.get('NUM_CANDIDATES', 8),
            enable_optimality_check=False   # TODO: remove after debugging
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

        # 4.5. Initialize ControlVariateValidator
        self.validator = ControlVariateValidator(
            smps_reader=self.reader,
            argmax_op=self.argmax_op,
            parallel_worker=self.parallel_worker
        )
        self.logger.info("ControlVariateValidator initialized.")

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

            self.logger.info("Factorizing initial basis.")
            self.argmax_op.finalize_dual_additions()
        
        except Exception as e:
            self.logger.error(f"Error loading initial basis from {h5_path}: {e}", exc_info=True)
            raise # Ensure the error is raised to stop execution if loading fails
        
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

    def _perform_argmax_operation(self, current_x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Performs the argmax operation to calculate the estimated cost, select the best dual solutions,
        and check for primal feasibility (optimality).

        Args:
            current_x: The current first-stage decision vector.

        Returns:
            A tuple containing:
            - argmax_estim_q_x (float): The estimated cost from the argmax operation.
            - best_k_scores (np.ndarray): The scores of the best dual for each scenario.
            - best_k_index (np.ndarray): The indices of the best dual for each scenario.
            - is_optimal (np.ndarray): A boolean array indicating if each scenario is optimal.
            - solve_time (float): The execution time.
        """
        start_time = time.time()
        
        # Use fast argmax to get pi indices for all scenarios
        pi_indices, _ = self.argmax_op.find_optimal_basis_fast(current_x)
        
        # Calculate cut coefficients using the pi indices
        alpha_pre, beta_pre = self.argmax_op.calculate_cut_coefficients(pi_indices)
        argmax_estim_q_x = alpha_pre + beta_pre @ current_x
        
        # For compatibility, return empty arrays for individual scenario results
        # (the old API provided per-scenario scores, but new API focuses on cut generation)
        best_k_scores = np.array([])
        best_k_index = pi_indices  # Return the pi indices used
        is_optimal = np.array([])  # Fast mode doesn't do feasibility checking
        
        solve_time = time.time() - start_time
        return argmax_estim_q_x, best_k_scores, best_k_index, is_optimal, solve_time

    def _solve_subproblems(self, current_x: np.ndarray, short_delta_r_batch: np.ndarray, vbasis_batch: Optional[np.ndarray], cbasis_batch: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solves a batch of subproblems in parallel.

        Args:
            current_x (np.ndarray): The current solution vector for the master problem.
            short_delta_r_batch (np.ndarray): Batch of scenario-specific RHS modifications to solve.
            vbasis_batch (Optional[np.ndarray]): Optional batch of variable basis statuses for warm-starting.
            cbasis_batch (Optional[np.ndarray]): Optional batch of constraint basis statuses for warm-starting.

        Returns:
            A tuple containing results for the solved scenarios.
        """
        start_time = time.time()
        obj_all, _, pi_all, rc_all, vbasis_out, cbasis_out, simplex_iter_count_all = \
            self.parallel_worker.solve_batch(
                current_x, short_delta_r_batch, vbasis_batch, cbasis_batch
            )
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
            # Removed cutoff logic - add all selected duals regardless of score improvement
            # if scores_difference is not None and scores_difference[s_idx] < ARGMAX_CUTOFF_TOLERANCE:
            #     continue
            if self.argmax_op.add_pi(pi_all[s_idx], rc_all[s_idx], vbasis_out[s_idx], cbasis_out[s_idx]):
                added_count += 1
        
        # Finalize basis addition and Factor basis
        self.argmax_op.finalize_dual_additions()

        update_time = time.time() - start_time
        return coverage_fraction, added_count, update_time

    def _calculate_and_add_cut(self, current_x: np.ndarray, pi_indices: np.ndarray) -> Tuple[float, bool, float]:
        """
        Calculates a new Benders cut using argmax_op with provided pi indices
        and adds it to the master problem if violated.

        Args:
            current_x: The current first-stage decision vector.
            pi_indices: Array of pi indices to use for each scenario.

        Returns:
            A tuple containing the actual cost, a boolean indicating if a cut was added, and the execution time.
        """
        start_time = time.time()

        # Calculate cut coefficients using provided pi indices
        alpha, beta = self.argmax_op.calculate_cut_coefficients(pi_indices)
        subproblem_actual_q_x = alpha + beta @ current_x

        current_master_epigraph_eta_at_x = self.master_problem.calculate_epigraph_value(current_x)
        
        cut_added = False
        if subproblem_actual_q_x > current_master_epigraph_eta_at_x + self.config['tolerance']:
            self.master_problem.add_optimality_cut(beta, alpha)
            cut_added = True

        calc_time = time.time() - start_time
        return subproblem_actual_q_x, cut_added, calc_time

    def _select_lp_scenarios(self, iteration: int) -> np.ndarray:
        """
        Select scenarios for LP solving based on configured strategy.
        
        Args:
            iteration: Current iteration number (1-based)
            
        Returns:
            Array of scenario indices to solve with LP solver
        """
        num_lp_scenarios = self.config.get('num_lp_scenarios_per_iteration', min(1000, self.short_delta_r.shape[0]))
        strategy = self.config.get('lp_scenario_selection_strategy', 'systematic')
        
        # Ensure we don't exceed total number of scenarios
        num_lp_scenarios = min(num_lp_scenarios, self.short_delta_r.shape[0])
        
        if strategy == 'systematic':
            return self._systematic_scenario_selection(iteration, num_lp_scenarios)
        elif strategy == 'random':
            return self._random_scenario_selection(num_lp_scenarios)
        else:
            self.logger.warning(f"Unknown scenario selection strategy '{strategy}', defaulting to systematic")
            return self._systematic_scenario_selection(iteration, num_lp_scenarios)

    def _systematic_scenario_selection(self, iteration: int, num_to_select: int) -> np.ndarray:
        """
        Rotate through scenarios systematically to ensure all scenarios are eventually processed.
        Assumes total_scenarios is divisible by num_to_select for simplicity.
        
        Args:
            iteration: Current iteration number (1-based)
            num_to_select: Number of scenarios to select
            
        Returns:
            Array of selected scenario indices
        """
        total_scenarios = self.short_delta_r.shape[0]
        
        # Calculate starting index with systematic rotation
        start_idx = ((iteration - 1) * num_to_select) % total_scenarios
        
        # Simple case assuming divisibility
        return np.arange(start_idx, start_idx + num_to_select)

    def _random_scenario_selection(self, num_to_select: int) -> np.ndarray:
        """
        Randomly select scenarios for LP solving.
        
        Args:
            num_to_select: Number of scenarios to select
            
        Returns:
            Array of selected scenario indices
        """
        total_scenarios = self.short_delta_r.shape[0]
        return np.random.choice(total_scenarios, size=num_to_select, replace=False)



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
        # log_parts.append(f"CostCut: {log_metrics.get('cost_from_cut', float('nan')):.4e}")
        # log_parts.append(f"Gap: {log_metrics.get('gap', float('nan')):.4e}")
        log_parts.append(f"CostArgmax: {log_metrics.get('cost_from_argmax', float('nan')):.4e}")
        log_parts.append(f"Rho: {log_metrics.get('rho', float('nan')):.2e}")
        log_parts.append(f"CutAdded: {'Y' if log_metrics.get('cut_added') else 'N'}")
        log_parts.append(f"TimeIter: {log_metrics.get('total_iteration_time', 0):.2f}s")
        log_parts.append(f"(M:{log_metrics.get('master_solve_time',0):.2f}s, "
                         f"W:{log_metrics.get('warmstart_time',0):.2f}s, "
                         f"A:{log_metrics.get('argmax_op_time',0):.2f}s, "
                         f"S:{log_metrics.get('subproblem_solve_time',0):.2f}s, "
                         f"D:{log_metrics.get('duals_update_time',0):.2f}s, "
                         f"C:{log_metrics.get('cut_calculation_time',0):.2f}s)")
        log_parts.append(f"ArgmaxNumPi: {log_metrics.get('argmax_num_pi', 0)}")
        log_parts.append(f"LPScenarios: {log_metrics.get('num_lp_scenarios', 0)}")
        log_parts.append(f"CoverageFraction: {log_metrics.get('coverage_fraction', 0):.4f}")
        log_parts.append(f"Optimal%: {log_metrics.get('optimal_fraction', 0) * 100:.2f}%")
        log_parts.append(f"delta_x_norm: {log_metrics.get('delta_x_norm', 0):.4f}")
        self.logger.info(" | ".join(log_parts))

    def _validate_incumbent_solution(self):
        """
        Validates the incumbent solution using control variate variance reduction.
        Outputs brief summary to console and detailed results to log file.
        """
        if self.validator is None or self.x_incumbent is None:
            self.logger.warning("Cannot validate: validator not initialized or no incumbent solution.")
            return
        
        # Get validation parameters from config with large default values
        N1 = self.config.get('validation_N1', 50000)  # Number of LP solves
        N2 = self.config.get('validation_N2', min(1000000, self.argmax_op.MAX_OMEGA))  # Number of GPU evaluations
        confidence_level = self.config.get('validation_confidence_level', 0.95)
        validation_seed = self.config.get('validation_seed', 12345)
        
        self.logger.info("--- Starting Solution Validation ---")
        
        # Brief console summary
        self.logger.info(f"Validating incumbent solution using control variate method...")
        self.logger.info(f"Configuration: N1={N1:,} LP solves, N2={N2:,} GPU evaluations, {confidence_level*100}% confidence")
        
        try:
            validation_start_time = time.time()
            result = self.validator.validate_solution(
                x=self.x_incumbent,
                N1=N1,
                N2=N2,
                confidence_level=confidence_level,
                seed=validation_seed
            )
            validation_time = time.time() - validation_start_time
            
            # Brief console output
            self.logger.info(f"Validation completed successfully in {validation_time:.2f}s")
            self.logger.info(f"Expected total cost (c^T x + E[Q(x,ω)]): {result.expected_total_cost:.6f} [{result.confidence_interval_lower:.6f}, {result.confidence_interval_upper:.6f}]")
            self.logger.info(f"Total cost CI width: {result.confidence_interval_width:.6f}")
            if result.variance_reduction_ratio is not None:
                self.logger.info(f"Variance reduction achieved: {result.variance_reduction_ratio*100:.2f}%")
            
            # Write detailed results to log file
            log_dir = Path(self.config.get('log_dir', 'logs'))
            log_dir.mkdir(exist_ok=True)
            
            # Generate log filename with timestamp
            problem_name = getattr(self.reader, 'problem_name', 'unknown')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"validation_{problem_name}_{timestamp}.log"
            log_path = log_dir / log_filename
            
            with open(log_path, 'w') as f:
                f.write("=== Control Variate Validation Results ===\n\n")
                f.write(f"Validation completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Problem: {problem_name}\n")
                f.write(f"Total validation time: {validation_time:.2f}s\n\n")
                
                f.write("=== Configuration ===\n")
                f.write(f"N1 (LP solves): {N1:,}\n")
                f.write(f"N2 (GPU evaluations): {N2:,}\n")
                f.write(f"Confidence level: {confidence_level*100}%\n")
                f.write(f"Random seed: {validation_seed}\n\n")
                
                f.write("=== Incumbent Solution ===\n")
                f.write(f"x_incumbent: {result.x_incumbent.tolist()}\n\n")
                
                f.write("=== Main Results ===\n")
                f.write(f"Expected total cost E[c^T x + Q(x,ω)]: {result.expected_total_cost:.10f}\n")
                f.write(f"Standard error: {result.standard_error:.10f}\n")
                f.write(f"Confidence interval ({confidence_level*100}%): [{result.confidence_interval_lower:.10f}, {result.confidence_interval_upper:.10f}]\n")
                f.write(f"Confidence interval width: {result.confidence_interval_width:.10f}\n\n")
                
                f.write("=== Cost Breakdown ===\n")
                f.write(f"First-stage cost (c^T x): {result.first_stage_cost:.10f}\n")
                f.write(f"Expected second-stage cost E[Q(x,ω)]: {result.expected_second_stage_cost:.10f}\n\n")
                
                f.write("=== Control Variate Statistics ===\n")
                f.write(f"μ_Q̂ (GPU expectation): {result.mu_q_hat:.10f}\n")
                f.write(f"Δ (correction term): {result.delta:.10f}\n")
                f.write(f"Sample standard deviation of Q̂: {result.sample_std_q_hat:.10f}\n")
                f.write(f"Sample standard deviation of Δ: {result.sample_std_delta:.10f}\n")
                if result.variance_reduction_ratio is not None:
                    f.write(f"Variance reduction ratio: {result.variance_reduction_ratio:.6f} ({result.variance_reduction_ratio*100:.2f}%)\n")
                else:
                    f.write("Variance reduction ratio: N/A (MC comparison not computed)\n")
                
                f.write(f"\n=== Sample Sizes ===\n")
                f.write(f"N1 (exact LP solves): {result.n1:,}\n")
                f.write(f"N2 (GPU approximations): {result.n2:,}\n")
                f.write(f"Total function evaluations: {result.n1 + result.n2:,}\n")
                
                f.write(f"\n=== Performance ===\n")
                f.write(f"Total validation time: {validation_time:.2f}s\n")
                f.write(f"Time per evaluation: {validation_time/(result.n1 + result.n2)*1000:.3f}ms\n")
            
            self.logger.info(f"Detailed validation results written to: {log_path}")
            
        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}", exc_info=True)
            self.logger.warning("Continuing without validation...")

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
            
            delta_x_norm = np.linalg.norm(self.x_candidate - self.x_incumbent)
            

            # Calculate perceived decrease
            perceived_decrease = self.master_problem.calculate_estimated_objective(self.x_incumbent) - \
                                 self.master_problem.calculate_estimated_objective(self.x_candidate)

            if perceived_decrease < -1e-3:
                self.logger.warning(f"Iteration {iter_count}: Perceived decrease is negative ({perceived_decrease:.4f}). Numerical issues may be present.")
                perceived_decrease = 0.0
            
            # 2. Select scenarios for LP solving
            selected_lp_scenarios = self._select_lp_scenarios(iter_count)
            self.logger.debug(f"Iter {iter_count}: Selected {len(selected_lp_scenarios)} scenarios for LP solving "
                           f"(strategy: {self.config.get('lp_scenario_selection_strategy', 'systematic')})")
            
            # 3. Warmstart preparation: Run argmax with top-k on selected scenarios only
            warmstart_start_time = time.time()
            selected_best_scores, selected_best_indices, selected_is_optimal = \
                self.argmax_op.find_optimal_basis_with_subset(self.x_candidate, selected_lp_scenarios)
            
            
            # Get warmstart basis for selected scenarios 
            vbasis_batch, cbasis_batch = self.argmax_op.get_basis(selected_best_indices)
            warmstart_time = time.time() - warmstart_start_time
            
            optimal_fraction = np.mean(selected_is_optimal) if len(selected_is_optimal) > 0 else 1.0
            self.logger.debug(f"Iter {iter_count}: Warmstart found {optimal_fraction*100:.2f}% feasible among selected scenarios.")

            # 4. LP Solving: Filter out already optimal scenarios and solve only those needing LP
            scenarios_needing_lp_mask = ~selected_is_optimal
            scenarios_needing_lp_indices = np.where(scenarios_needing_lp_mask)[0]  # Positions in selected arrays
            
            if len(scenarios_needing_lp_indices) > 0:
                # Get absolute scenario indices for scenarios needing LP solving
                scenarios_needing_lp = selected_lp_scenarios[scenarios_needing_lp_mask]
                
                # Create filtered input arrays for the parallel worker
                short_delta_r_filtered = self.short_delta_r[scenarios_needing_lp]
                vbasis_filtered = vbasis_batch[scenarios_needing_lp_indices] if vbasis_batch is not None else None
                cbasis_filtered = cbasis_batch[scenarios_needing_lp_indices] if cbasis_batch is not None else None
                
                # Solve only the scenarios needing LP
                obj_lp, pi_lp, rc_lp, vbasis_out_lp, cbasis_out_lp, _, subproblem_time = \
                    self._solve_subproblems(self.x_candidate, short_delta_r_filtered, vbasis_filtered, cbasis_filtered)
                
                self.logger.debug(f"Iter {iter_count}: Solved {len(scenarios_needing_lp)} scenarios with LP "
                                f"({len(selected_lp_scenarios) - len(scenarios_needing_lp)} already optimal)")
            else:
                # All scenarios are already optimal, no LP solving needed
                obj_lp = np.array([])
                pi_lp = np.empty((0, 0))
                rc_lp = np.empty((0, 0))
                vbasis_out_lp = np.empty((0, 0), dtype=np.int8)
                cbasis_out_lp = np.empty((0, 0), dtype=np.int8)
                subproblem_time = 0.0
                self.logger.debug(f"Iter {iter_count}: All {len(selected_lp_scenarios)} scenarios already optimal, skipping LP")

            # 5. Add all new duals from LP solving to ArgmaxOperation
            if len(scenarios_needing_lp_indices) > 0:
                # Score differences only for scenarios that needed LP solving  
                lp_score_differences = obj_lp - selected_best_scores[scenarios_needing_lp_mask]
                coverage_fraction, duals_added_count, duals_update_time = self._update_argmax_duals(
                    pi_lp, rc_lp, vbasis_out_lp, cbasis_out_lp, lp_score_differences
                )
                self.logger.debug(f"Iter {iter_count}: Added {duals_added_count} new duals to ArgmaxOp. Total duals: {self.argmax_op.num_pi}.")
            else:
                coverage_fraction, duals_added_count, duals_update_time = 1.0, 0, 0.0

            # 6. Fast cut generation: Run argmax (fast mode) on ALL scenarios
            argmax_start_time = time.time()
            pi_indices_all_scenarios, _ = self.argmax_op.find_optimal_basis_fast(self.x_candidate)
            argmax_time = time.time() - argmax_start_time

            # 7. Calculate cut using pure argmax (no overrides needed)
            subproblem_actual_q_x, cut_added, cut_calc_time = self._calculate_and_add_cut(
                self.x_candidate, pi_indices_all_scenarios
            )
            
            # Get argmax estimation for logging (calculate directly from pi_indices)
            alpha_argmax, beta_argmax = self.argmax_op.calculate_cut_coefficients(pi_indices_all_scenarios)
            argmax_estim_q_x = alpha_argmax + beta_argmax @ self.x_candidate
            
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
                max_rho = self.config.get('max_rho', 1e5)
                self.rho = min(max_rho, self.rho * rho_increase_factor)
            
            fcd = actual_decrease / perceived_decrease if perceived_decrease != 0 else float('inf')

            # --- Collect Metrics for Logging ---
            c_transpose_x = self.c_vector @ self.x_candidate
            cost_from_master = master_total_obj_candidate
            # cost_from_cut = c_transpose_x + subproblem_actual_q_x if subproblem_actual_q_x > -np.inf else float('inf')
            cost_from_argmax = c_transpose_x + argmax_estim_q_x
            # gap = cost_from_cut - cost_from_master
            total_iteration_time = time.time() - iter_time_start

            iteration_metrics = {
                "iteration": iter_count,
                "master_solve_time": master_time,
                "warmstart_time": warmstart_time,
                "argmax_op_time": argmax_time,
                "subproblem_solve_time": subproblem_time,
                "duals_update_time": duals_update_time,
                "cut_calculation_time": cut_calc_time,
                "total_iteration_time": total_iteration_time,
                "first_stage_cost_cx": c_transpose_x,
                "cost_from_master": cost_from_master,
                "cost_from_argmax": cost_from_argmax,
                "subproblem_actual_q_x": subproblem_actual_q_x,
                # "cost_from_cut": cost_from_cut,
                "rho": self.rho,
                "cut_added": cut_added,
                # "gap": gap,
                "x_norm": scipy.linalg.norm(self.x_candidate),
                "argmax_num_pi": self.argmax_op.num_pi,
                "coverage_fraction": coverage_fraction,
                "optimal_fraction": optimal_fraction,
                "perceived_decrease": perceived_decrease,
                "actual_decrease": actual_decrease,
                "fcd": fcd,
                "successful_step": successful_step,
                "num_lp_scenarios": len(selected_lp_scenarios),
                "lp_selection_strategy": self.config.get('lp_scenario_selection_strategy', 'systematic'),
                "delta_x_norm": delta_x_norm
            }
            self._log_iteration_data(iteration_metrics)

            if iter_count == max_iterations:
                self.logger.info("Reached maximum iterations.")

        self.logger.info("--- Benders Decomposition Loop Finished ---")
        
        # Validate the incumbent solution
        if self.config.get('enable_validation', True):
            self._validate_incumbent_solution()
        
        self.cleanup()

    def cleanup(self):
        """
        Cleans up resources used by the solver.
        """
        self.logger.info("BendersSolver cleanup called.")
        # Implement any specific resource cleanup if needed, e.g., for parallel_worker if it has a shutdown method.
