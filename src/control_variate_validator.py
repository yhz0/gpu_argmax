import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING
from scipy import stats

if TYPE_CHECKING:
    from .argmax_operation import ArgmaxOperation
    from .parallel_second_stage_worker import ParallelSecondStageWorker
    from .smps_reader import SMPSReader

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Results from control variate validation of a candidate solution.
    
    Attributes:
        point_estimate: Final estimate of E[Q(x̄,ω̃)]
        confidence_interval: Tuple of (lower_bound, upper_bound) for confidence interval
        standard_error: Standard error of the estimate
        mu_q_hat: Control variate expectation μ_Q̂ from large sample
        delta: Correction term Δ from difference between exact and approximate values
        variance_reduction_ratio: Ratio showing variance reduction achieved (Var(D)/Var(Q))
        sample_std_q_hat: Sample standard deviation s_Q̂ from control variate
        sample_std_delta: Sample standard deviation s_D from correction terms
        n1: Small sample size used for exact LP computation
        n2: Large sample size used for control variate computation
        confidence_level: Confidence level used (e.g., 0.95 for 95%)
        failed_lp_solves: Number of failed LP solves in the small sample
        total_computation_time: Total time taken for validation
        gpu_computation_time: Time spent on GPU computations
        lp_computation_time: Time spent on LP solves
        x_incumbent: First-stage solution that was validated
        first_stage_cost_vector: First-stage cost vector c
    """
    point_estimate: float
    confidence_interval: Tuple[float, float]
    standard_error: float
    mu_q_hat: float
    delta: float
    variance_reduction_ratio: float
    sample_std_q_hat: float
    sample_std_delta: float
    n1: int
    n2: int
    confidence_level: float
    failed_lp_solves: int
    total_computation_time: float
    gpu_computation_time: float
    lp_computation_time: float
    x_incumbent: np.ndarray
    first_stage_cost_vector: np.ndarray
    
    @property
    def first_stage_cost(self) -> float:
        """First-stage cost c^T x"""
        return float(self.first_stage_cost_vector @ self.x_incumbent)
    
    @property
    def expected_second_stage_cost(self) -> float:
        """Expected second-stage cost E[Q(x,ω)]"""
        return self.point_estimate
    
    @property
    def expected_total_cost(self) -> float:
        """Expected total cost c^T x + E[Q(x,ω)]"""
        return self.first_stage_cost + self.expected_second_stage_cost
    
    @property
    def confidence_interval_lower(self) -> float:
        """Lower bound of confidence interval"""
        return self.confidence_interval[0] + self.first_stage_cost
    
    @property
    def confidence_interval_upper(self) -> float:
        """Upper bound of confidence interval"""
        return self.confidence_interval[1] + self.first_stage_cost
    
    @property
    def confidence_interval_width(self) -> float:
        """Width of confidence interval"""
        return self.confidence_interval[1] - self.confidence_interval[0]


class ControlVariateValidator:
    """
    Implements GPU-accelerated solution validation for stochastic programming using 
    control variate variance reduction technique as described in the paper.
    
    This class combines ArgmaxOperation (GPU computations), ParallelSecondStageWorker 
    (exact LP solving), and SMPSReader (scenario generation) to efficiently estimate
    the expected second-stage value function E[Q(x̄,ω̃)] with reduced variance.
    """
    
    def __init__(self, 
                 argmax_op: 'ArgmaxOperation',
                 parallel_worker: 'ParallelSecondStageWorker', 
                 smps_reader: 'SMPSReader'):
        """
        Initialize the ControlVariateValidator.
        
        Args:
            argmax_op: ArgmaxOperation instance for GPU-accelerated dual computations
            parallel_worker: ParallelSecondStageWorker for exact LP solving
            smps_reader: SMPSReader for scenario generation from stochastic data
        """
        if not hasattr(argmax_op, 'find_optimal_basis_fast'):
            raise ValueError("ArgmaxOperation must have find_optimal_basis_fast method")
        if not hasattr(parallel_worker, 'solve_batch'):
            raise ValueError("ParallelSecondStageWorker must have solve_batch method")
        if not hasattr(smps_reader, 'sample_stochastic_rhs_batch'):
            raise ValueError("SMPSReader must have sample_stochastic_rhs_batch method")
        if not hasattr(smps_reader, 'get_short_delta_r'):
            raise ValueError("SMPSReader must have get_short_delta_r method")
            
        self.argmax_op = argmax_op
        self.parallel_worker = parallel_worker
        self.smps_reader = smps_reader
        
        # Verify data is loaded
        if not hasattr(smps_reader, '_data_loaded') or not smps_reader._data_loaded:
            raise ValueError("SMPSReader data must be loaded using load_and_extract() before validation")
            
        logger.info("ControlVariateValidator initialized successfully")

    def validate_solution(self, 
                         x: np.ndarray, 
                         N1: int, 
                         N2: int,
                         confidence_level: float = 0.95,
                         seed: Optional[int] = None) -> ValidationResult:
        """
        Validates a candidate solution using control variate variance reduction.
        
        Implements Algorithm 1 from the paper: GPU-Accelerated Validation with Control Variates.
        
        Args:
            x: The first-stage candidate solution vector
            N1: Small sample size for exact LP computation (correction term)
            N2: Large sample size for control variate computation (should be >> N1)
            confidence_level: Confidence level for interval (default 0.95 for 95%)
            seed: Optional random seed for reproducible scenario generation
            
        Returns:
            ValidationResult containing the estimate, confidence interval, and diagnostics
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If validation process fails
        """
        start_time = time.time()
        
        # Input validation
        self._validate_inputs(x, N1, N2, confidence_level)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Random seed set to {seed} for reproducible results")
            
        logger.info(f"Starting control variate validation with N1={N1}, N2={N2}, confidence={confidence_level}")
        
        # Step 1: Generate independent scenario samples
        logger.info("Generating independent scenario samples...")
        sampling_start = time.time()
        short_delta_r_omega1, short_delta_r_omega2 = self._generate_scenario_samples(N1, N2)
        sampling_time = time.time() - sampling_start
        
        # Step 2: GPU control variate computation (large sample N2)
        logger.info(f"Computing control variate estimates using {N2} scenarios...")
        gpu_start = time.time()
        mu_q_hat, sample_std_q_hat, q_hat_omega1 = self._compute_control_variate_estimates(
            x, short_delta_r_omega1, short_delta_r_omega2)
        gpu_time = time.time() - gpu_start
        
        # Step 3: Exact LP computation (small sample N1)
        logger.info(f"Computing exact LP values using {N1} scenarios...")
        lp_start = time.time()
        q_exact_omega1, failed_lp_count = self._compute_exact_values(x, short_delta_r_omega1)
        lp_time = time.time() - lp_start
        
        # Step 4: Control variate calculation and correction
        logger.info("Computing correction term and final estimate...")
        delta, sample_std_delta, variance_reduction_ratio = self._compute_correction_term(
            q_exact_omega1, q_hat_omega1)
        
        # Step 5: Final estimate and confidence interval
        point_estimate, confidence_interval, standard_error = self._compute_final_statistics(
            mu_q_hat, delta, sample_std_q_hat, sample_std_delta, 
            N1, N2, confidence_level)
        
        total_time = time.time() - start_time
        
        result = ValidationResult(
            point_estimate=point_estimate,
            confidence_interval=confidence_interval,
            standard_error=standard_error,
            mu_q_hat=mu_q_hat,
            delta=delta,
            variance_reduction_ratio=variance_reduction_ratio,
            sample_std_q_hat=sample_std_q_hat,
            sample_std_delta=sample_std_delta,
            n1=N1,
            n2=N2,
            confidence_level=confidence_level,
            failed_lp_solves=failed_lp_count,
            total_computation_time=total_time,
            gpu_computation_time=gpu_time,
            lp_computation_time=lp_time,
            x_incumbent=x.copy(),
            first_stage_cost_vector=self.smps_reader.c.copy()
        )
        
        logger.info(f"Validation complete in {total_time:.2f}s. Second-stage value function E[Q(x,ω)]: {point_estimate:.4f}")
        logger.info(f"Second-stage CI ({confidence_level*100}%): [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        logger.info(f"Second-stage CI width: {confidence_interval[1] - confidence_interval[0]:.4f}")
        logger.info(f"Timing breakdown:")
        logger.info(f"  Scenario generation: {sampling_time:.2f}s ({sampling_time/total_time*100:.1f}%)")
        logger.info(f"  GPU computation: {gpu_time:.2f}s ({gpu_time/total_time*100:.1f}%)")
        logger.info(f"  LP computation: {lp_time:.2f}s ({lp_time/total_time*100:.1f}%)")
        other_time = total_time - sampling_time - gpu_time - lp_time
        logger.info(f"  Other (correction & stats): {other_time:.2f}s ({other_time/total_time*100:.1f}%)")
        
        return result

    def _validate_inputs(self, x: np.ndarray, N1: int, N2: int, confidence_level: float) -> None:
        """Validate input parameters."""
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError("x must be a 1D numpy array")
        if x.shape[0] != self.argmax_op.X_DIM:
            raise ValueError(f"x dimension {x.shape[0]} doesn't match expected {self.argmax_op.X_DIM}")
        if not isinstance(N1, int) or N1 <= 0:
            raise ValueError("N1 must be a positive integer")
        if not isinstance(N2, int) or N2 <= 0:
            raise ValueError("N2 must be a positive integer")
        if N2 <= N1:
            raise ValueError("N2 must be significantly larger than N1 for effective variance reduction")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError("confidence_level must be between 0 and 1")
        if self.argmax_op.num_pi == 0:
            raise ValueError("ArgmaxOperation has no stored dual solutions. Cannot perform validation.")

    def _generate_scenario_samples(self, N1: int, N2: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two independent sets of scenario samples."""
        # Generate first sample set (small, for correction term)
        sample_rhs_omega1 = self.smps_reader.sample_stochastic_rhs_batch(N1)
        short_delta_r_omega1 = self.smps_reader.get_short_delta_r(sample_rhs_omega1)
        
        # Generate second sample set (large, for control variate)
        sample_rhs_omega2 = self.smps_reader.sample_stochastic_rhs_batch(N2)
        short_delta_r_omega2 = self.smps_reader.get_short_delta_r(sample_rhs_omega2)
        
        logger.debug(f"Generated scenario samples: Ω1 shape {short_delta_r_omega1.shape}, Ω2 shape {short_delta_r_omega2.shape}")
        
        return short_delta_r_omega1, short_delta_r_omega2

    def _compute_control_variate_batched(self,
                                        x: np.ndarray,
                                        short_delta_r_scenarios: np.ndarray) -> np.ndarray:
        """
        Compute control variate estimates in batches for large scenario sets.
        
        Args:
            x: First-stage solution vector
            short_delta_r_scenarios: Scenario data array of shape (N, R_SPARSE_LEN)
            
        Returns:
            Array of control variate values (scores) for all scenarios
        """
        batch_size = self.argmax_op.MAX_OMEGA
        num_scenarios = short_delta_r_scenarios.shape[0]
        
        if num_scenarios <= batch_size:
            # Single batch processing
            # Disable TF32 for higher precision in validation
            self.argmax_op._configure_tf32(False)
            
            self.argmax_op.clear_scenarios()
            self.argmax_op.add_scenarios(short_delta_r_scenarios)
            _, scores = self.argmax_op.find_optimal_basis_fast(x, touch_lru=False)
            return scores.astype(np.float64)
        
        # Multi-batch processing
        logger.info(f"Processing {num_scenarios} scenarios in batches of {batch_size}")
        all_scores = []
        
        for batch_start in range(0, num_scenarios, batch_size):
            batch_end = min(batch_start + batch_size, num_scenarios)
            batch_scenarios = short_delta_r_scenarios[batch_start:batch_end]
            
            logger.debug(f"Processing batch {batch_start//batch_size + 1}: scenarios {batch_start+1}-{batch_end}")
            
            # Process this batch
            self.argmax_op.clear_scenarios()
            self.argmax_op.add_scenarios(batch_scenarios)
            _, batch_scores = self.argmax_op.find_optimal_basis_fast(x, touch_lru=False)
            
            all_scores.append(batch_scores.astype(np.float64))
        
        # Concatenate all batch results
        return np.concatenate(all_scores)

    def _compute_control_variate_estimates(self, 
                                          x: np.ndarray, 
                                          short_delta_r_omega1: np.ndarray,
                                          short_delta_r_omega2: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Compute control variate estimates using GPU acceleration with batching support.
        
        Args:
            x: First-stage solution vector
            short_delta_r_omega1: Small sample scenarios for correction computation
            short_delta_r_omega2: Large sample scenarios for control variate
            
        Returns:
            Tuple of (mu_q_hat, sample_std_q_hat, q_hat_omega1)
            - mu_q_hat: Mean of control variate from large sample
            - sample_std_q_hat: Sample standard deviation from large sample
            - q_hat_omega1: Control variate values for small sample (needed for correction)
        """
        N1 = short_delta_r_omega1.shape[0]
        N2 = short_delta_r_omega2.shape[0]
        max_omega = self.argmax_op.MAX_OMEGA
        
        # Log batch processing information
        if N2 > max_omega:
            logger.info(f"N2={N2} exceeds MAX_OMEGA={max_omega}. Using batched processing for large sample.")
        if N1 > max_omega:
            logger.info(f"N1={N1} exceeds MAX_OMEGA={max_omega}. Using batched processing for small sample.")
        
        # Compute control variate for large sample N2 (for μ_Q̂) using batching
        logger.debug(f"Computing control variate for N2={N2} scenarios")
        q_hat_scores_omega2_fp64 = self._compute_control_variate_batched(x, short_delta_r_omega2)
        
        # Calculate μ_Q̂ using double precision arithmetic
        mu_q_hat = np.mean(q_hat_scores_omega2_fp64)
        sample_std_q_hat = np.std(q_hat_scores_omega2_fp64, ddof=1)  # Sample standard deviation (N-1)
        
        logger.debug(f"Control variate: μ_Q̂ = {mu_q_hat:.6f}, s_Q̂ = {sample_std_q_hat:.6f}")
        
        # Compute control variate for small sample N1 (for correction term) using batching
        logger.debug(f"Computing control variate for N1={N1} scenarios")
        q_hat_omega1 = self._compute_control_variate_batched(x, short_delta_r_omega1)
        
        logger.debug(f"Computed Q̂ values: N2 sample mean = {mu_q_hat:.6f}, N1 sample size = {len(q_hat_omega1)}")
        
        return mu_q_hat, sample_std_q_hat, q_hat_omega1

    def _compute_exact_values(self, x: np.ndarray, short_delta_r_omega1: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute exact Q(x̄,ω) values using parallel LP solving.
        
        Args:
            x: First-stage solution vector
            short_delta_r_omega1: Small sample scenarios
            
        Returns:
            Tuple of (q_exact_omega1, failed_count)
            - q_exact_omega1: Array of exact objective values
            - failed_count: Number of failed LP solves
        """
        # Solve batch of LP problems to get exact Q(x̄,ω) values
        obj_values, _, _, _, _, _, _ = self.parallel_worker.solve_batch(
            x=x,
            short_delta_r_batch=short_delta_r_omega1,
            nontrivial_rc_only=False  # We don't need reduced costs for validation
        )
        
        # Count failed solves (NaN values)
        failed_mask = np.isnan(obj_values)
        failed_count = np.sum(failed_mask)
        
        if failed_count > 0:
            logger.warning(f"{failed_count} out of {len(obj_values)} LP solves failed")
            # For failed solves, we'll exclude them from the correction calculation
            # This is a reasonable approach as long as failure rate is low
        
        # Convert to double precision for numerical stability
        q_exact_omega1 = obj_values.astype(np.float64)
        
        logger.debug(f"Exact LP computation: {len(q_exact_omega1)} solves, {failed_count} failed")
        
        return q_exact_omega1, failed_count

    def _compute_correction_term(self, 
                                q_exact_omega1: np.ndarray, 
                                q_hat_omega1: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute the correction term Δ and variance reduction statistics.
        
        Args:
            q_exact_omega1: Exact Q(x̄,ω) values from LP solves
            q_hat_omega1: Approximate Q̂(x̄,ω) values from GPU computation
            
        Returns:
            Tuple of (delta, sample_std_delta, variance_reduction_ratio)
            - delta: Correction term Δ = E[Q - Q̂]
            - sample_std_delta: Sample standard deviation of differences
            - variance_reduction_ratio: Var(Q-Q̂)/Var(Q) showing reduction achieved
        """
        # Handle failed LP solves by excluding them from correction calculation
        valid_mask = ~np.isnan(q_exact_omega1)
        q_exact_valid = q_exact_omega1[valid_mask]
        q_hat_valid = q_hat_omega1[valid_mask]
        
        if len(q_exact_valid) == 0:
            raise RuntimeError("All LP solves failed. Cannot compute correction term.")
        
        if len(q_exact_valid) < len(q_exact_omega1):
            logger.warning(f"Using {len(q_exact_valid)} valid scenarios out of {len(q_exact_omega1)} for correction")
        
        # Monte Carlo statistics from exact LP values only
        q_exact_mean = np.mean(q_exact_valid)
        q_exact_std = np.std(q_exact_valid, ddof=1) if len(q_exact_valid) > 1 else 0.0
        q_exact_se = q_exact_std / np.sqrt(len(q_exact_valid)) if len(q_exact_valid) > 1 else 0.0
        
        # What would be the 95% CI if using only exact LP values (traditional Monte Carlo)
        if len(q_exact_valid) > 1:
            from scipy import stats
            alpha = 0.05  # For 95% confidence
            t_critical = stats.t.ppf(1.0 - alpha/2, df=len(q_exact_valid)-1)  # Use t-distribution for small samples
            mc_margin = t_critical * q_exact_se
            mc_ci_lower = q_exact_mean - mc_margin
            mc_ci_upper = q_exact_mean + mc_margin
        else:
            mc_ci_lower = mc_ci_upper = q_exact_mean
            
        logger.info(f"Monte Carlo statistics from exact LP values:")
        logger.info(f"  Mean: {q_exact_mean:.6f}")
        logger.info(f"  Standard deviation: {q_exact_std:.6f}")
        logger.info(f"  Sample size: {len(q_exact_valid)}")
        logger.info(f"  Standard error: {q_exact_se:.6f}")
        if len(q_exact_valid) > 1:
            logger.info(f"  Traditional MC CI (95%): [{mc_ci_lower:.6f}, {mc_ci_upper:.6f}]")
            logger.info(f"  Traditional MC CI width: {mc_ci_upper - mc_ci_lower:.6f}")
        else:
            logger.info(f"  Traditional MC CI: Not available (sample size = 1)")
        
        # Calculate differences D = Q(x̄,ω) - Q̂(x̄,ω)
        differences = q_exact_valid - q_hat_valid
        
        # Correction term: Δ = (1/N₁) Σ D_i
        delta = np.mean(differences)
        
        # Sample standard deviation of differences
        sample_std_delta = np.std(differences, ddof=1) if len(differences) > 1 else 0.0
        
        # Calculate variance reduction ratio: Var(D)/Var(Q) 
        # Note: We still compute variances for the ratio calculation as it's a variance-based metric
        sample_variance_q = np.var(q_exact_valid, ddof=1) if len(q_exact_valid) > 1 else 1.0
        sample_variance_delta = sample_std_delta ** 2  # Convert back to variance for ratio
        variance_reduction_ratio = sample_variance_delta / sample_variance_q if sample_variance_q > 0 else 0.0
        
        # Debug output to understand the differences
        logger.info(f"Difference statistics:")
        logger.info(f"  Min difference: {np.min(differences):.6f}")
        logger.info(f"  Max difference: {np.max(differences):.6f}")
        logger.info(f"  Mean difference (Δ): {delta:.6f}")
        logger.info(f"  Std of differences: {sample_std_delta:.6f}")
        logger.info(f"  Sample size for correction: {len(differences)}")
        logger.info(f"  Q standard deviation: {np.sqrt(sample_variance_q):.6f}")
        logger.info(f"  D standard deviation: {sample_std_delta:.6f}")
        
        logger.debug(f"Correction term: Δ = {delta:.6f}, s_D = {sample_std_delta:.6f}")
        logger.debug(f"Variance reduction ratio: {variance_reduction_ratio:.6f} (lower is better)")
        
        return delta, sample_std_delta, variance_reduction_ratio

    def _compute_final_statistics(self, 
                                 mu_q_hat: float, 
                                 delta: float,
                                 sample_std_q_hat: float,
                                 sample_std_delta: float,
                                 N1: int,
                                 N2: int,
                                 confidence_level: float) -> Tuple[float, Tuple[float, float], float]:
        """
        Compute final estimate and confidence interval.
        
        Args:
            mu_q_hat: Control variate expectation
            delta: Correction term
            sample_std_q_hat: Sample standard deviation of control variate
            sample_std_delta: Sample standard deviation of correction terms
            N1: Small sample size
            N2: Large sample size
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            Tuple of (point_estimate, confidence_interval, standard_error)
        """
        # Final point estimate: E[Q(x̄,ω̃)] ≈ μ_Q̂ + Δ
        point_estimate = mu_q_hat + delta
        
        # Standard error: SE = √(s²_Q̂/N₂ + s²_D/N₁)
        # Convert standard deviations back to variances for SE calculation
        standard_error = np.sqrt((sample_std_q_hat ** 2) / N2 + (sample_std_delta ** 2) / N1)
        
        # Confidence interval using normal distribution
        alpha = 1.0 - confidence_level
        z_critical = stats.norm.ppf(1.0 - alpha/2)
        margin_error = z_critical * standard_error
        
        confidence_interval = (
            point_estimate - margin_error,
            point_estimate + margin_error
        )
        
        logger.debug(f"Final statistics: estimate = {point_estimate:.6f}, SE = {standard_error:.6f}")
        logger.debug(f"Confidence interval: [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
        
        return point_estimate, confidence_interval, standard_error