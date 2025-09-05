import unittest
import numpy as np
import h5py
import os
import sys
import torch

# This allows the script to be run from anywhere and still find the src module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(PROJECT_ROOT)

try:
    from src.smps_reader import SMPSReader
    from src.argmax_operation import ArgmaxOperation
    from src.second_stage_worker import SecondStageWorker
except ImportError as e:
    print(f"ERROR: Could not import a required module: {e}")
    raise

# --- Configuration ---
PROB_NAME = "cep"
SMPS_DATA_DIR = os.path.join(PROJECT_ROOT, "smps_data", PROB_NAME)
H5_RESULTS_FILE = os.path.join(PROJECT_ROOT, "cep_100scen_results.h5")

# SMPS file paths
CORE_FILENAME = f"{PROB_NAME}.mps"
TIME_FILENAME = f"{PROB_NAME}.tim"
STO_FILENAME = f"{PROB_NAME}.sto"
core_filepath = os.path.join(SMPS_DATA_DIR, CORE_FILENAME)
time_filepath = os.path.join(SMPS_DATA_DIR, TIME_FILENAME)
sto_filepath = os.path.join(SMPS_DATA_DIR, STO_FILENAME)


class TestOptimalityCheck(unittest.TestCase):
    """
    Tests the optimality checking functionality of the ArgmaxOperation class.
    """
    reader = None
    x_sol = None
    num_scenarios = None
    pi_s = None
    stochastic_rhs_parts = None
    cbasis_y_all = None
    vbasis_y_all = None

    @classmethod
    def setUpClass(cls):
        """
        Load all necessary data from SMPS and HDF5 files once.
        """
        print("\nSetting up TestOptimalityCheck...")

        # --- Verify required files exist ---
        required_files = [core_filepath, time_filepath, sto_filepath, H5_RESULTS_FILE]
        if not all(os.path.exists(f) for f in required_files):
            raise FileNotFoundError(f"Missing required test data files.")

        # --- Load SMPS data ---
        cls.reader = SMPSReader(core_filepath, time_filepath, sto_filepath)
        cls.reader.load_and_extract()

        # --- Load pre-computed results from HDF5 ---
        with h5py.File(H5_RESULTS_FILE, "r") as f:
            cls.num_scenarios = f['/metadata'].attrs['num_scenarios']
            cls.pi_s = f['/solution/dual/pi_s'][:]
            cls.stochastic_rhs_parts = f['/scenarios/stochastic_rhs_parts'][:]
            cls.cbasis_y_all = f['/basis/cbasis_y_all'][:]
            cls.vbasis_y_all = f['/basis/vbasis_y_all'][:]
            cls.x_sol = f['solution/primal/x'][:]
        
        print("Setup complete.")

    def test_optimality_checking_procedure(self):
        """
        Implements the full 10-step procedure for testing optimality checking.
        """
        # Step 2: Initialize ArgmaxOperation with optimality check enabled
        argmax_op = ArgmaxOperation.from_smps_reader(
            self.reader,
            MAX_PI=self.num_scenarios,
            MAX_OMEGA=self.num_scenarios,
            device='cpu',
            optimality_dtype=torch.float64
        )
        
        # Load all scenarios
        short_delta_r = self.reader.get_short_delta_r(self.stochastic_rhs_parts)
        argmax_op.add_scenarios(short_delta_r)

        # Step 3: Solve all scenario subproblems to get ground truth objectives
        print("Solving subproblems for ground truth objectives...")
        worker = SecondStageWorker.from_smps_reader(self.reader)
        worker.set_x(self.x_sol)
        ground_truth_objectives = np.zeros(self.num_scenarios)
        for s in range(self.num_scenarios):
            worker.set_scenario(short_delta_r[s, :])
            result = worker.solve()
            self.assertIsNotNone(result, f"Solver failed for scenario {s}")
            ground_truth_objectives[s] = result[0]
        worker.close()
        print("Ground truth objectives calculated.")

        # Step 4: Add only the FIRST 3 dual solutions
        num_duals_to_add = 3
        for s in range(num_duals_to_add):
            argmax_op.add_pi(self.pi_s[s, :], np.array([]), self.vbasis_y_all[s, :], self.cbasis_y_all[s, :])

        # Step 4.5: Finalize dual additions to process pending factorizations
        argmax_op.finalize_dual_additions()

        # Step 5: Run argmax procedure with optimality checking
        all_scenario_indices = np.arange(self.num_scenarios)
        scores, indices, is_optimal = argmax_op.find_optimal_basis_with_subset(
            self.x_sol, all_scenario_indices, primal_feas_tol=1e-4)
        
        # Step 6: Sanity check the scores of the first 3 scenarios

        print("Performing sanity check on first 3 scenarios...")
        np.testing.assert_allclose(
            scores[:num_duals_to_add],
            ground_truth_objectives[:num_duals_to_add],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Scores of first 3 scenarios do not match ground truth objectives"
        )
        print("Sanity check passed.")

        # Step 7: Compare scores of all scenarios with objectives for ground truth optimality
        # A scenario's argmax result is "optimal" if its score matches the true objective
        ground_truth_optimality = np.isclose(scores, ground_truth_objectives, rtol=1e-5, atol=1e-5)
        # Step 8 & 9: Assertions based on the new `is_optimal` flag
        print("Asserting optimality results...")
        # 1. Assert that the first `num_duals_to_add` scenarios are marked as optimal,
        #    since their exact dual solutions were provided.
        self.assertTrue(
            np.all(is_optimal[:num_duals_to_add]),
            "First 3 scenarios were not all marked as optimal by the new method."
        )
        # 2. Assert that the optimality status for all scenarios matches the ground truth.
        np.testing.assert_array_equal(
            is_optimal,
            ground_truth_optimality,
            err_msg="Optimality results from find_optimal_basis_with_subset do not match the ground truth."
        )
        print("All optimality assertions passed.")
        
        print("Test completed successfully.")


if __name__ == '__main__':
    unittest.main()
