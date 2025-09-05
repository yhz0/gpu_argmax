import numpy as np
import time
from typing import Tuple, Optional
from dataclasses import dataclass

from .smps_reader import SMPSReader
from .parallel_second_stage_worker import ParallelSecondStageWorker
from .argmax_operation import ArgmaxOperation


@dataclass
class ValidationResult:
    """Result object containing validation statistics and outcomes."""
    expected_objective: float
    confidence_interval: Tuple[float, float]
    num_samples: int
    converged: bool
    first_stage_obj: float
    second_stage_obj_mean: float
    stopped_due_to_zero_variance: bool
    total_time: float


class Validator:
    """
    Validator for evaluating expected objective values of two-stage stochastic programs
    using Monte Carlo sampling with confidence intervals.
    
    Given a first-stage solution x, evaluates the expected total objective:
    E[f(x,ω)] = c'x + E[Q(x,ω)]
    
    where Q(x,ω) is the second-stage objective for scenario ω.
    """
    
    def __init__(self, reader: SMPSReader, num_workers: int = 1, 
                 validation_batch_size: int = 1000, seed: int = 443):
        """
        Initialize the Validator.
        
        Args:
            reader: Loaded SMPSReader instance containing problem data
            num_workers: Number of parallel workers for second-stage solves
            validation_batch_size: Number of scenarios to sample per batch
            seed: Random seed for reproducible sampling
        """
        if not reader._data_loaded:
            raise RuntimeError("SMPSReader must be loaded before creating Validator")
            
        self.reader = reader
        self.num_workers = num_workers
        self.validation_batch_size = validation_batch_size
        self.seed = seed
        
        # Set random seed for reproducible sampling
        np.random.seed(seed)
        
        # Initialize parallel worker (will be created when needed)
        self._parallel_worker: Optional[ParallelSecondStageWorker] = None
        
    def _get_parallel_worker(self) -> ParallelSecondStageWorker:
        """Lazy initialization of parallel worker."""
        if self._parallel_worker is None:
            self._parallel_worker = ParallelSecondStageWorker.from_smps_reader(
                self.reader, self.num_workers
            )
        return self._parallel_worker
        
    def evaluate_expected_objective(self, x: np.ndarray, 
                                  target_ci_width_relative: float = 0.01) -> ValidationResult:
        """
        Evaluate the expected objective value for a given first-stage solution x.
        
        Samples scenarios in batches, solves second-stage problems, and computes
        confidence intervals until convergence criteria are met.
        
        Args:
            x: First-stage solution vector
            target_ci_width_relative: Target relative width of confidence interval (default 1%)
            
        Returns:
            ValidationResult containing objective estimate and statistics
        """
        # Validate input
        if len(x) != len(self.reader.c):
            raise ValueError(f"x dimension ({len(x)}) doesn't match number of first-stage variables ({len(self.reader.c)})")
            
        # Calculate first-stage objective
        first_stage_obj = np.dot(self.reader.c, x)
        
        # Initialize running statistics
        total_samples = 0
        sum_second_stage = 0.0
        sum_squares_second_stage = 0.0
        
        # Convergence tracking
        converged = False
        stopped_due_to_zero_variance = False
        min_samples = 100
        zero_variance_threshold = 10000
        
        # Timing
        start_time = time.time()
        
        print(f"Starting validation with batch size {self.validation_batch_size}")
        print(f"Target relative CI width: {target_ci_width_relative:.1%}")
        print(f"First-stage objective: {first_stage_obj:.6f}")
        print("-" * 80)
        
        batch_num = 0
        
        while not converged:
            batch_num += 1
            batch_start_time = time.time()
            
            # Sample scenarios for this batch
            stochastic_rhs_batch = self.reader.sample_stochastic_rhs_batch(self.validation_batch_size)
            short_delta_r_batch = self.reader.get_short_delta_r(stochastic_rhs_batch)
            
            # Solve second-stage problems
            parallel_worker = self._get_parallel_worker()
            obj_values, _, _, _, _, _, _ = parallel_worker.solve_batch(
                x, short_delta_r_batch, nontrivial_rc_only=True
            )
            
            # Update running statistics
            batch_second_stage_objs = obj_values
            valid_objs = batch_second_stage_objs[~np.isnan(batch_second_stage_objs)]
            
            if len(valid_objs) == 0:
                raise RuntimeError(f"All scenarios in batch {batch_num} failed to solve")
                
            # Update running sums
            batch_sum = np.sum(valid_objs)
            batch_sum_squares = np.sum(valid_objs ** 2)
            
            sum_second_stage += batch_sum
            sum_squares_second_stage += batch_sum_squares
            total_samples += len(valid_objs)
            
            # Calculate current statistics
            mean_second_stage = sum_second_stage / total_samples
            
            if total_samples > 1:
                variance = (sum_squares_second_stage - (sum_second_stage ** 2) / total_samples) / (total_samples - 1)
                variance = max(variance, 0.0)  # Ensure non-negative due to numerical precision
                std_error = np.sqrt(variance / total_samples)
                
                # Normal distribution confidence interval (95%)
                z_score = 1.96
                ci_half_width = z_score * std_error
                ci_lower = mean_second_stage - ci_half_width
                ci_upper = mean_second_stage + ci_half_width
                
                # Calculate relative CI width
                if abs(mean_second_stage) > 1e-12:
                    relative_ci_width = (2 * ci_half_width) / abs(mean_second_stage)
                else:
                    relative_ci_width = float('inf')
                    
                # Estimate samples needed for convergence
                if abs(mean_second_stage) > 1e-12 and variance > 1e-12:
                    samples_needed = int(np.ceil((z_score * np.sqrt(variance) / (target_ci_width_relative * abs(mean_second_stage))) ** 2))
                else:
                    samples_needed = total_samples
                    
            else:
                variance = 0.0
                ci_lower = ci_upper = mean_second_stage
                relative_ci_width = float('inf')
                samples_needed = float('inf')
            
            batch_time = time.time() - batch_start_time
            total_objective = first_stage_obj + mean_second_stage
            
            # Progress reporting
            print(f"Batch {batch_num:3d} | Time: {batch_time:6.2f}s | Samples: {total_samples:8d}")
            print(f"         | Mean: {mean_second_stage:12.6f} | Variance: {variance:12.6e}")
            print(f"         | Total obj: {total_objective:12.6f} | CI: [{ci_lower:12.6f}, {ci_upper:12.6f}]")
            print(f"         | Rel CI width: {relative_ci_width:8.4f} | Est samples needed: {samples_needed:8d}")
            print("-" * 80)
            
            # Check convergence criteria
            if total_samples >= min_samples:
                # Check for zero variance early stopping
                if variance < 1e-12 and total_samples > zero_variance_threshold:
                    stopped_due_to_zero_variance = True
                    converged = True
                    print(f"Stopped due to zero variance with {total_samples} samples")
                    
                # Check relative CI width convergence
                elif relative_ci_width < target_ci_width_relative:
                    converged = True
                    print(f"Converged: relative CI width {relative_ci_width:.4f} < {target_ci_width_relative:.4f}")
        
        total_time = time.time() - start_time
        
        # Final confidence interval
        final_ci = (first_stage_obj + ci_lower, first_stage_obj + ci_upper)
        
        print(f"\nValidation complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final expected objective: {total_objective:.6f}")
        print(f"Final confidence interval: [{final_ci[0]:.6f}, {final_ci[1]:.6f}]")
        
        return ValidationResult(
            expected_objective=total_objective,
            confidence_interval=final_ci,
            num_samples=total_samples,
            converged=converged and not stopped_due_to_zero_variance,
            first_stage_obj=first_stage_obj,
            second_stage_obj_mean=mean_second_stage,
            stopped_due_to_zero_variance=stopped_due_to_zero_variance,
            total_time=total_time
        )
    
    def evaluate_expected_objective_with_control_variate(
        self, x: np.ndarray, argmax_op: ArgmaxOperation,
        N1: int = 1000, N2: int = 100000,
        target_ci_width_relative: float = 0.01
    ) -> ValidationResult:
        """
        Evaluate expected objective using control variate variance reduction.
        
        Implements Algorithm from variance_reduction_validation.tex:
        - Sample two independent sets Ω₁ (N1 scenarios) and Ω₂ (N2 scenarios)  
        - Use GPU to compute control variate μ̂_Q from large set Ω₂
        - Solve LPs only for small set Ω₁ to compute correction term Δ
        - Combine: E[Q(x,ω)] ≈ μ̂_Q + Δ
        
        This method leverages the GPU-accelerated argmax operation as a control variate
        to dramatically reduce the variance of Monte Carlo estimation. The control
        variate ̂Q(x,ω;Π̂) is computed efficiently on GPU using the stored dual vertex
        pool, while only a small number of full LP solves are needed for the correction.
        
        Args:
            x: First-stage solution vector
            argmax_op: ArgmaxOperation instance with populated dual vertex pool
            N1: Number of scenarios for LP solving (small sample, default 1000)
            N2: Number of scenarios for control variate estimation (large sample, default 100000)
            target_ci_width_relative: Target relative width of confidence interval (default 1%)
            
        Returns:
            ValidationResult containing objective estimate with control variate statistics
            
        Note:
            - argmax_op.clear_scenarios() will be called to reset scenario data
            - Uses double precision for summation operations to ensure numerical stability
            - Standard error calculation: SE = sqrt(s²_Q̂/N2 + s²_D/N1)
        """
        # TODO: Validate inputs
        if len(x) != len(self.reader.c):
            raise ValueError(f"x dimension ({len(x)}) doesn't match number of first-stage variables ({len(self.reader.c)})")
        
        # TODO: Calculate first-stage objective
        first_stage_obj = np.dot(self.reader.c, x)
        
        # TODO: Sample two independent scenario sets Ω₁ and Ω₂
        print(f"Sampling {N1} scenarios for LP solving (Ω₁) and {N2} scenarios for control variate (Ω₂)")
        
        # TODO: GPU Phase - Compute control variates for both sets
        # - Clear scenarios: argmax_op.clear_scenarios()
        # - Add Ω₂ scenarios: argmax_op.add_scenarios(scenarios_omega2)
        # - Get control variate values: _, scores_omega2 = argmax_op.find_optimal_basis_fast(x, return_scores=True)
        # - Calculate μ̂_Q = mean(scores_omega2)
        
        # TODO: Repeat for Ω₁ scenarios
        # - Clear scenarios: argmax_op.clear_scenarios()  
        # - Add Ω₁ scenarios: argmax_op.add_scenarios(scenarios_omega1)
        # - Get control variate values: _, scores_omega1 = argmax_op.find_optimal_basis_fast(x, return_scores=True)
        
        # TODO: CPU Phase - Solve LPs for Ω₁ scenarios only
        # - Use ParallelSecondStageWorker to solve LPs for scenarios_omega1
        # - Calculate correction term: Δ = mean(Q_true - Q_hat) for Ω₁
        
        # TODO: Combine estimates
        # - Final estimate: μ̂_Q + Δ
        # - Calculate variances: s²_Q̂ (from Ω₂), s²_D (from differences in Ω₁)
        # - Combined standard error: SE = sqrt(s²_Q̂/N2 + s²_D/N1)
        # - Confidence interval: estimate ± z_α/2 * SE
        
        # TODO: Return ValidationResult with control variate statistics
        raise NotImplementedError("Control variate validation not yet implemented")
    
    def close(self):
        """Clean up resources."""
        if self._parallel_worker is not None:
            self._parallel_worker.close()
            self._parallel_worker = None
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb  # Mark as intentionally unused
        self.close()
        
    def __del__(self):
        self.close()