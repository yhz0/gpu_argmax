import unittest
import os
import numpy as np
import h5py
import time
import logging
from typing import Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.smps_reader import SMPSReader
from src.argmax_operation import ArgmaxOperation
from src.parallel_second_stage_worker import ParallelSecondStageWorker
from src.control_variate_validator import ControlVariateValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestControlVariateValidator(unittest.TestCase):
    """
    Test ControlVariateValidator using the CEP problem instance and pre-computed SAA results.
    
    This test compares simple Monte Carlo validation against control variate validation
    to demonstrate variance reduction effectiveness and verify correctness.
    """

    @classmethod
    def setUpClass(cls):
        """Load CEP problem and populate ArgmaxOperation with pre-computed dual solutions."""
        print("\n" + "="*70)
        print("Setting up ControlVariateValidator Test using CEP Problem")
        print("="*70)
        
        # File paths
        cls.problem_dir = os.path.join("smps_data", "cep")
        cls.h5_file = "cep_100scen_results.h5"
        
        # Verify files exist
        required_files = [
            os.path.join(cls.problem_dir, "cep.mps"),
            os.path.join(cls.problem_dir, "cep.tim"), 
            os.path.join(cls.problem_dir, "cep.sto"),
            cls.h5_file
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        print("All required files found.")
        
        # Load SMPS problem
        print("Loading SMPS problem...")
        cls.reader = SMPSReader(
            core_file=os.path.join(cls.problem_dir, "cep.mps"),
            time_file=os.path.join(cls.problem_dir, "cep.tim"),
            sto_file=os.path.join(cls.problem_dir, "cep.sto")
        )
        cls.reader.load_and_extract()
        print(f"SMPS problem loaded: {len(cls.reader.x_indices)} x vars, {len(cls.reader.y_indices)} y vars")
        
        # Load pre-computed SAA results from HDF5
        print(f"Loading pre-computed results from HDF5: {os.path.basename(cls.h5_file)}")
        cls._load_hdf5_data()
        
        # Set up ArgmaxOperation and populate with dual solutions
        print("Setting up ArgmaxOperation...")
        cls._setup_argmax_operation()
        
        # Set up ParallelSecondStageWorker
        print("Setting up ParallelSecondStageWorker...")
        cls.parallel_worker = ParallelSecondStageWorker.from_smps_reader(
            reader=cls.reader,
            num_workers=4
        )
        
        # Create ControlVariateValidator
        print("Creating ControlVariateValidator...")
        cls.validator = ControlVariateValidator(
            argmax_op=cls.argmax_op,
            parallel_worker=cls.parallel_worker,
            smps_reader=cls.reader
        )
        
        print(f"Setup complete. ArgmaxOperation has {cls.argmax_op.num_pi} dual solutions stored.")
        print("-" * 70)

    @classmethod
    def _load_hdf5_data(cls):
        """Load pre-computed dual solutions and other data from HDF5 file."""
        with h5py.File(cls.h5_file, "r") as f:
            # Load metadata
            cls.num_scenarios_h5 = f['/metadata'].attrs['num_scenarios']
            print(f"HDF5 contains results for {cls.num_scenarios_h5} scenarios")
            
            # Load solution data
            cls.x_optimal = f['solution/primal/x'][:]  # Optimal first-stage solution
            cls.pi_s = f['/solution/dual/pi_s'][:]     # Dual variables per scenario
            cls.vbasis_y_all = f['/basis/vbasis_y_all'][:]  # Variable basis statuses  
            cls.cbasis_y_all = f['/basis/cbasis_y_all'][:]  # Constraint basis statuses
            cls.stochastic_rhs_parts = f['/scenarios/stochastic_rhs_parts'][:]  # Stochastic RHS
            
            print(f"Loaded optimal x solution: shape {cls.x_optimal.shape}")
            print(f"Loaded dual solutions: {cls.pi_s.shape[0]} scenarios, {cls.pi_s.shape[1]} constraints")
            print(f"Loaded basis information: {cls.vbasis_y_all.shape[0]} scenarios")

    @classmethod
    def _setup_argmax_operation(cls):
        """Set up ArgmaxOperation and populate with dual solutions from HDF5."""
        # Create ArgmaxOperation
        cls.argmax_op = ArgmaxOperation.from_smps_reader(
            reader=cls.reader,
            MAX_PI=200,  # Should be enough for the HDF5 scenarios
            MAX_OMEGA=20000,  # Room for larger test scenarios
            enable_optimality_check=False  # Disable for performance
        )
        
        print("Populating ArgmaxOperation with dual solutions from HDF5...")
        
        # We need to extract reduced costs from the basis information
        # For now, we'll use zero reduced costs as a placeholder since we're focusing on pi values
        num_bounded_vars = cls.argmax_op.NUM_BOUNDED_VARS
        
        added_count = 0
        for i in range(cls.num_scenarios_h5):
            pi_i = cls.pi_s[i, :]  # Dual variables for scenario i
            rc_i = np.zeros(num_bounded_vars, dtype=np.float32)  # Placeholder reduced costs
            vbasis_i = cls.vbasis_y_all[i, :].astype(np.int8)  # Variable basis status
            cbasis_i = cls.cbasis_y_all[i, :].astype(np.int8)  # Constraint basis status
            
            # Add to ArgmaxOperation (may deduplicate based on basis)
            was_added = cls.argmax_op.add_pi(pi_i, rc_i, vbasis_i, cbasis_i)
            if was_added:
                added_count += 1
        
        print(f"Added {added_count} unique dual solutions to ArgmaxOperation")

    def test_control_variate_validation_vs_monte_carlo(self):
        """
        Main test: Compare simple Monte Carlo vs control variate validation.
        Demonstrates variance reduction and validates correctness.
        """
        print("\n" + "="*70) 
        print("Testing Control Variate Validation vs Monte Carlo")
        print("="*70)
        
        # Test parameters
        total_scenarios = 10000
        N1 = 2000  # Small sample for correction term
        N2 = 8000  # Large sample for control variate
        confidence_level = 0.95
        test_seed = 12345
        
        print(f"Test parameters:")
        print(f"  Total scenarios: {total_scenarios}")
        print(f"  Control variate split: N1={N1}, N2={N2}")
        print(f"  Confidence level: {confidence_level}")
        print(f"  Random seed: {test_seed}")
        print()
        
        # Generate test scenarios
        print("Generating test scenarios...")
        np.random.seed(test_seed)
        test_rhs = self.reader.sample_stochastic_rhs_batch(total_scenarios)
        test_short_delta_r = self.reader.get_short_delta_r(test_rhs)
        print(f"Generated {total_scenarios} scenarios: shape {test_short_delta_r.shape}")
        
        # Run Simple Monte Carlo validation
        print("\n" + "-"*50)
        print("1. SIMPLE MONTE CARLO VALIDATION")
        print("-"*50)
        mc_result = self._run_monte_carlo_validation(test_short_delta_r)
        
        # Run Control Variate validation using same scenarios
        print("\n" + "-"*50)
        print("2. CONTROL VARIATE VALIDATION")
        print("-"*50)
        cv_result = self._run_control_variate_validation(N1, N2, confidence_level, test_seed)
        
        # Compare and analyze results
        print("\n" + "-"*50)
        print("3. RESULTS COMPARISON AND ANALYSIS")
        print("-"*50)
        self._analyze_results(mc_result, cv_result)
        
        print("\n" + "="*70)
        print("Test completed successfully!")
        print("="*70)

    def test_limited_dual_vertex_coverage(self):
        """
        Test control variate validation with limited dual vertex coverage.
        Uses only the first 2 dual extreme points.
        """
        print("\n" + "="*80)
        print("Testing Control Variate with Limited Dual Vertex Coverage (First 2 Points)")
        print("="*80)
        
        # Create new ArgmaxOperation with limited dual solutions
        print("Setting up ArgmaxOperation with limited dual vertex set...")
        limited_argmax_op = ArgmaxOperation.from_smps_reader(
            reader=self.reader,
            MAX_PI=50,   # Smaller capacity
            MAX_OMEGA=20000,
            enable_optimality_check=False
        )
        
        # Add only the first 2 dual solutions
        num_bounded_vars = limited_argmax_op.NUM_BOUNDED_VARS
        added_count = 0
        max_to_add = 2
        
        for i in range(min(max_to_add, self.num_scenarios_h5)):
            pi_i = self.pi_s[i, :]  
            rc_i = np.zeros(num_bounded_vars, dtype=np.float32)  
            vbasis_i = self.vbasis_y_all[i, :].astype(np.int8)  
            cbasis_i = self.cbasis_y_all[i, :].astype(np.int8)  
            
            was_added = limited_argmax_op.add_pi(pi_i, rc_i, vbasis_i, cbasis_i)
            if was_added:
                added_count += 1
        
        print(f"Added {added_count} dual solutions to limited ArgmaxOperation")
        
        # Create limited validator
        limited_validator = ControlVariateValidator(
            argmax_op=limited_argmax_op,
            parallel_worker=self.parallel_worker,
            smps_reader=self.reader
        )
        
        # Test parameters - fair comparison with same number of LP solves
        monte_carlo_scenarios = 1000  # LP solves for Monte Carlo
        N1 = 1000   # Same number of LP solves for control variate
        N2 = 10000  # Large number of fast GPU evaluations
        confidence_level = 0.95
        test_seed = 54321
        
        print(f"\nTest parameters:")
        print(f"  Monte Carlo LP solves: {monte_carlo_scenarios}")
        print(f"  Control variate: N1={N1} LP solves + N2={N2} GPU evaluations")
        print(f"  Dual vertex coverage: {added_count} extreme points")
        print()
        
        # Generate test scenarios for Monte Carlo (1000 scenarios)
        print("Generating Monte Carlo test scenarios...")
        np.random.seed(test_seed)
        mc_test_rhs = self.reader.sample_stochastic_rhs_batch(monte_carlo_scenarios)
        mc_test_short_delta_r = self.reader.get_short_delta_r(mc_test_rhs)
        print(f"Generated {monte_carlo_scenarios} scenarios for Monte Carlo: shape {mc_test_short_delta_r.shape}")
        
        # Run Monte Carlo validation
        print("\n" + "-"*60)
        print("1. MONTE CARLO VALIDATION (1000 LP solves)")
        print("-"*60)
        mc_result = self._run_monte_carlo_validation(mc_test_short_delta_r)
        
        # Run Control Variate validation with limited dual coverage
        print("\n" + "-"*60) 
        print("2. CONTROL VARIATE VALIDATION (1000 LP + 10000 GPU)")
        print("-"*60)
        
        # Use the limited validator instead of the full one
        print("Running control variate validation with limited dual vertex set...")
        cv_result = limited_validator.validate_solution(
            x=self.x_optimal,
            N1=N1,
            N2=N2,
            confidence_level=confidence_level,
            seed=test_seed
        )
        
        print(f"Control Variate Results (Limited Coverage):")
        print(f"  Point estimate: {cv_result.point_estimate:.6f}")
        print(f"  Control variate μ_Q̂: {cv_result.mu_q_hat:.6f}")
        print(f"  Correction term Δ: {cv_result.delta:.6f}")
        print(f"  Standard error: {cv_result.standard_error:.6f}")
        print(f"  95% CI: [{cv_result.confidence_interval[0]:.6f}, {cv_result.confidence_interval[1]:.6f}]")
        print(f"  Variance reduction ratio: {cv_result.variance_reduction_ratio:.6f}")
        print(f"  Failed LP solves: {cv_result.failed_lp_solves}")
        print(f"  Total time: {cv_result.total_computation_time:.2f}s")
        print(f"    GPU time: {cv_result.gpu_computation_time:.2f}s")
        print(f"    LP time: {cv_result.lp_computation_time:.2f}s")
        
        # Compare results
        print("\n" + "-"*60)
        print("3. COMPARISON: Limited vs Full Dual Coverage")
        print("-"*60)
        
        # Create new ArgmaxOperation with full dual coverage for comparison
        print("Re-initializing ArgmaxOperation with full dual coverage for comparison...")
        full_argmax_op = ArgmaxOperation.from_smps_reader(
            reader=self.reader,
            MAX_PI=200,
            MAX_OMEGA=20000,
            enable_optimality_check=False
        )
        
        # Add all available dual solutions
        full_added_count = 0
        for i in range(self.num_scenarios_h5):
            pi_i = self.pi_s[i, :]  
            rc_i = np.zeros(full_argmax_op.NUM_BOUNDED_VARS, dtype=np.float32)  
            vbasis_i = self.vbasis_y_all[i, :].astype(np.int8)  
            cbasis_i = self.cbasis_y_all[i, :].astype(np.int8)  
            
            was_added = full_argmax_op.add_pi(pi_i, rc_i, vbasis_i, cbasis_i)
            if was_added:
                full_added_count += 1
        
        print(f"Added {full_added_count} dual solutions to full ArgmaxOperation")
        
        # Create full validator
        full_validator = ControlVariateValidator(
            argmax_op=full_argmax_op,
            parallel_worker=self.parallel_worker,
            smps_reader=self.reader
        )
        
        print("Running control variate with FULL dual coverage for comparison...")
        cv_full_result = full_validator.validate_solution(
            x=self.x_optimal,
            N1=N1,
            N2=N2,
            confidence_level=confidence_level,
            seed=test_seed
        )
        
        self._analyze_limited_vs_full_coverage(mc_result, cv_result, cv_full_result, added_count, full_added_count)
        
        print("\n" + "="*80)
        print("Limited dual vertex coverage test completed!")
        print("="*80)

    def _analyze_limited_vs_full_coverage(self, mc_result: dict, cv_limited, cv_full, num_limited_points: int, num_full_points: int):
        """Compare results between limited and full dual vertex coverage."""
        print("Comparison: Limited vs Full Dual Vertex Coverage")
        print()
        
        # Point estimate comparison
        print(f"Point Estimates:")
        print(f"  Monte Carlo:              {mc_result['point_estimate']:.6f}")
        print(f"  Control Variate ({num_limited_points} pts):  {cv_limited.point_estimate:.6f}")
        print(f"  Control Variate ({num_full_points} pts): {cv_full.point_estimate:.6f}")
        
        limited_diff = abs(mc_result['point_estimate'] - cv_limited.point_estimate)
        full_diff = abs(mc_result['point_estimate'] - cv_full.point_estimate)
        print(f"  |MC - CV({num_limited_points})|:              {limited_diff:.6f}")
        print(f"  |MC - CV({num_full_points})|:             {full_diff:.6f}")
        print()
        
        # Variance reduction comparison
        print(f"Variance Analysis:")
        print(f"  Monte Carlo variance:      {mc_result['sample_variance']:.0f}")
        print(f"  CV({num_limited_points}) s_D:                {cv_limited.sample_std_delta:.6f}")
        print(f"  CV({num_full_points}) s_D:               {cv_full.sample_std_delta:.6f}")
        print(f"  CV({num_limited_points}) reduction ratio:     {cv_limited.variance_reduction_ratio:.6f}")
        print(f"  CV({num_full_points}) reduction ratio:    {cv_full.variance_reduction_ratio:.6f}")
        print()
        
        # Correction term comparison
        print(f"Correction Terms:")
        print(f"  CV({num_limited_points}) correction Δ:        {cv_limited.delta:.6f}")
        print(f"  CV({num_full_points}) correction Δ:       {cv_full.delta:.6f}")
        correction_ratio = abs(cv_limited.delta / cv_full.delta) if cv_full.delta != 0 else float('inf')
        print(f"  |Δ({num_limited_points})| / |Δ({num_full_points})|:          {correction_ratio:.2f}")
        print()
        
        # Confidence interval width comparison
        mc_ci_width = mc_result['confidence_interval'][1] - mc_result['confidence_interval'][0]
        cv_limited_ci_width = cv_limited.confidence_interval[1] - cv_limited.confidence_interval[0]
        cv_full_ci_width = cv_full.confidence_interval[1] - cv_full.confidence_interval[0]
        
        limited_improvement = 100 * (1 - cv_limited_ci_width / mc_ci_width)
        full_improvement = 100 * (1 - cv_full_ci_width / mc_ci_width)
        
        print(f"Confidence Intervals:")
        print(f"  MC CI width:               {mc_ci_width:.2f}")
        print(f"  CV({num_limited_points}) CI width:            {cv_limited_ci_width:.2f} ({limited_improvement:.1f}% vs MC)")
        print(f"  CV({num_full_points}) CI width:           {cv_full_ci_width:.2f} ({full_improvement:.1f}% vs MC)")

    def _run_monte_carlo_validation(self, test_scenarios: np.ndarray) -> dict:
        """Run simple Monte Carlo validation and return results."""
        print("Solving all scenarios with ParallelSecondStageWorker...")
        
        start_time = time.time()
        obj_values, _, _, _, _, _, _ = self.parallel_worker.solve_batch(
            x=self.x_optimal,
            short_delta_r_batch=test_scenarios,
            nontrivial_rc_only=False
        )
        lp_time = time.time() - start_time
        
        # Check for failed solves
        failed_mask = np.isnan(obj_values)
        num_failed = np.sum(failed_mask)
        if num_failed > 0:
            raise RuntimeError(f"Monte Carlo validation failed: {num_failed} LP solves were not optimal")
        
        # Calculate statistics using double precision
        obj_values_fp64 = obj_values.astype(np.float64)
        mc_mean = np.mean(obj_values_fp64)
        mc_variance = np.var(obj_values_fp64, ddof=1)
        mc_std_error = np.sqrt(mc_variance / len(obj_values_fp64))
        
        # Confidence interval
        z_95 = 1.96  # 95% confidence
        mc_ci = (mc_mean - z_95 * mc_std_error, mc_mean + z_95 * mc_std_error)
        
        result = {
            'point_estimate': mc_mean,
            'sample_variance': mc_variance,
            'standard_error': mc_std_error,
            'confidence_interval': mc_ci,
            'num_scenarios': len(obj_values_fp64),
            'computation_time': lp_time,
            'objective_values': obj_values_fp64  # Keep for analysis
        }
        
        print(f"Monte Carlo Results:")
        print(f"  Point estimate: {mc_mean:.6f}")
        print(f"  Sample variance: {mc_variance:.6f}")
        print(f"  Standard error: {mc_std_error:.6f}")
        print(f"  95% CI: [{mc_ci[0]:.6f}, {mc_ci[1]:.6f}]")
        print(f"  Computation time: {lp_time:.2f} seconds")
        
        return result

    def _run_control_variate_validation(self, N1: int, N2: int, confidence_level: float, seed: int) -> dict:
        """Run control variate validation and return results."""
        print("Running control variate validation...")
        
        result = self.validator.validate_solution(
            x=self.x_optimal,
            N1=N1,
            N2=N2,
            confidence_level=confidence_level,
            seed=seed
        )
        
        print(f"Control Variate Results:")
        print(f"  Point estimate: {result.point_estimate:.6f}")
        print(f"  Control variate μ_Q̂: {result.mu_q_hat:.6f}")
        print(f"  Correction term Δ: {result.delta:.6f}")
        print(f"  Standard error: {result.standard_error:.6f}")
        print(f"  95% CI: [{result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f}]")
        print(f"  Variance reduction ratio: {result.variance_reduction_ratio:.6f}")
        print(f"  Failed LP solves: {result.failed_lp_solves}")
        print(f"  Total time: {result.total_computation_time:.2f}s")
        print(f"    GPU time: {result.gpu_computation_time:.2f}s")
        print(f"    LP time: {result.lp_computation_time:.2f}s")
        
        return result

    def _analyze_results(self, mc_result: dict, cv_result) -> None:
        """Compare and analyze Monte Carlo vs Control Variate results."""
        print("Comparative Analysis:")
        print()
        
        # Point estimate comparison
        estimate_diff = abs(mc_result['point_estimate'] - cv_result.point_estimate)
        estimate_diff_pct = 100 * estimate_diff / abs(mc_result['point_estimate'])
        print(f"Point Estimate Comparison:")
        print(f"  Monte Carlo:     {mc_result['point_estimate']:.6f}")
        print(f"  Control Variate: {cv_result.point_estimate:.6f}")
        print(f"  Absolute diff:   {estimate_diff:.6f} ({estimate_diff_pct:.3f}%)")
        print()
        
        # Confidence interval width comparison
        mc_ci_width = mc_result['confidence_interval'][1] - mc_result['confidence_interval'][0]
        cv_ci_width = cv_result.confidence_interval[1] - cv_result.confidence_interval[0]
        ci_improvement = 100 * (1 - cv_ci_width / mc_ci_width)
        
        print(f"Confidence Interval Width:")
        print(f"  Monte Carlo:     {mc_ci_width:.6f}")
        print(f"  Control Variate: {cv_ci_width:.6f}")
        print(f"  Improvement:     {ci_improvement:.1f}% (narrower is better)")
        print()
        
        # Variance analysis
        print(f"Variance Analysis:")
        print(f"  Monte Carlo variance:        {mc_result['sample_variance']:.6f}")
        print(f"  Control variate s_Q̂:        {cv_result.sample_std_q_hat:.6f}")
        print(f"  Control variate s_D:        {cv_result.sample_std_delta:.6f}")
        print(f"  Variance reduction ratio:    {cv_result.variance_reduction_ratio:.6f}")
        if cv_result.variance_reduction_ratio < 0.5:
            print("  ✓ Good variance reduction achieved (ratio < 0.5)")
        else:
            print("  ⚠ Limited variance reduction (ratio >= 0.5)")
        print()
        
        # Component verification
        print(f"Control Variate Component Verification:")
        print(f"  μ_Q̂ + Δ = {cv_result.mu_q_hat:.6f} + {cv_result.delta:.6f} = {cv_result.point_estimate:.6f}")
        print(f"  Monte Carlo estimate: {mc_result['point_estimate']:.6f}")
        component_diff = abs((cv_result.mu_q_hat + cv_result.delta) - mc_result['point_estimate'])
        print(f"  Component accuracy: {component_diff:.6f} difference")
        print()
        
        # Performance comparison
        print(f"Performance Comparison:")
        print(f"  Monte Carlo time:    {mc_result['computation_time']:.2f}s (all LP solves)")
        print(f"  Control Variate time: {cv_result.total_computation_time:.2f}s")
        print(f"    GPU portion:       {cv_result.gpu_computation_time:.2f}s ({100*cv_result.gpu_computation_time/cv_result.total_computation_time:.1f}%)")
        print(f"    LP portion:        {cv_result.lp_computation_time:.2f}s ({100*cv_result.lp_computation_time/cv_result.total_computation_time:.1f}%)")
        
        speedup = mc_result['computation_time'] / cv_result.total_computation_time
        if speedup > 1:
            print(f"  ⚠ Control variate slower by {speedup:.2f}x (expected for small samples)")
        else:
            print(f"  ✓ Control variate faster by {1/speedup:.2f}x")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        print(f"\nCleaning up resources...")
        if hasattr(cls, 'parallel_worker'):
            cls.parallel_worker.close()
        print("Cleanup complete.")


if __name__ == '__main__':
    unittest.main(verbosity=2)