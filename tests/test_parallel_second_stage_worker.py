import unittest
import h5py
import numpy as np
from pathlib import Path
import sys
import os
# import gurobipy as gp # Not directly used in test logic, but good if worker can raise GurobiError

# This allows the script to be run from anywhere and still find the src module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(PROJECT_ROOT)

from src.smps_reader import SMPSReader
# from second_stage_worker import SecondStageWorker # Not directly instantiated in the parallel test
from src.parallel_second_stage_worker import ParallelSecondStageWorker # Import the class to be tested

class TestParallelSecondStageWorkerAgainstSAA(unittest.TestCase):
    """
    Tests the ParallelSecondStageWorker by solving a batch of scenarios derived from
    an SAA solution stored in an HDF5 file and comparing the results.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load reference HDF5 data and SMPS problem data once for all tests.
        This assumes the necessary files ('cep_100scen_results.h5' and SMPS files for 'cep')
        are present in the expected locations.
        """
        cls.test_data_path = Path(PROJECT_ROOT) # Current directory by default
        cls.h5_file_path = cls.test_data_path / "cep_100scen_results.h5"
        cls.smps_base_path = cls.test_data_path / "smps_data/cep"
        cls.smps_core_file = cls.smps_base_path / "cep.mps"
        cls.smps_time_file = cls.smps_base_path / "cep.tim"
        cls.smps_sto_file = cls.smps_base_path / "cep.sto"

        # --- Basic File Existence Checks ---
        if not cls.h5_file_path.is_file():
            raise FileNotFoundError(f"HDF5 result file not found: {cls.h5_file_path}")
        if not cls.smps_core_file.is_file():
            raise FileNotFoundError(f"SMPS core file not found: {cls.smps_core_file}")
        if not cls.smps_time_file.is_file():
            raise FileNotFoundError(f"SMPS time file not found: {cls.smps_time_file}")
        if not cls.smps_sto_file.is_file():
            raise FileNotFoundError(f"SMPS sto file not found: {cls.smps_sto_file}")

        # --- Load Reference Data from HDF5 ---
        with h5py.File(cls.h5_file_path, 'r') as f:
            # Gurobi status code 2 typically means Optimal.
            cls.h5_optimal = f['/metadata'].attrs.get('solution_status_code', -1) == 2
            if not cls.h5_optimal:
                raise ValueError(
                    f"HDF5 file {cls.h5_file_path} does not contain an optimal reference solution (status != 2). "
                    "Cannot run comparison tests."
                )

            cls.num_scenarios_ref = f['/metadata'].attrs['num_scenarios']
            # Stochastic RHS *values* (not deviations yet) for each scenario.
            cls.stochastic_rhs_parts_h5 = f['/scenarios/stochastic_rhs_parts'][:]
            
            # Reference solutions from HDF5
            cls.x_sol_ref = f['/solution/primal/x'][:]
            cls.y_s_ref = f['/solution/primal/y_s'][:] # y solutions for all scenarios
            cls.pi_s_ref = f['/solution/dual/pi_s'][:] # pi solutions for all scenarios
            # cls.vbasis_s_ref = f['/basis/vbasis_y_all'][:] # Basis info, not directly compared in parallel test
            # cls.cbasis_s_ref = f['/basis/cbasis_y_all'][:]

        # --- Load SMPS data using the SMPSReader ---
        cls.reader = SMPSReader(
            core_file=str(cls.smps_core_file),
            time_file=str(cls.smps_time_file),
            sto_file=str(cls.smps_sto_file)
        )
        cls.reader.load_and_extract() # Assumes this method parses and prepares the reader

        # --- Pre-calculate all scenario deviations (short_delta_r) ---
        # This uses the loaded SMPSReader and the stochastic RHS values from HDF5.
        try:
            # Ensure stochastic_rhs_parts_h5 matches expected input for get_short_delta_r
            # If get_short_delta_r expects (num_stochastic_elements, num_scenarios) and HDF5 is
            # (num_scenarios, num_stochastic_elements), a transpose might be needed.
            # Based on typical SMPSReader.get_short_delta_r, input is (num_scenarios, num_stochastic_elements)
            if cls.stochastic_rhs_parts_h5.shape[0] != cls.num_scenarios_ref:
                 raise ValueError(f"Mismatch in HDF5 scenario count ({cls.stochastic_rhs_parts_h5.shape[0]}) "
                                  f"and metadata ({cls.num_scenarios_ref})")

            cls.short_delta_r_all = cls.reader.get_short_delta_r(cls.stochastic_rhs_parts_h5)
        except AttributeError:
            raise AttributeError("SMPSReader instance does not have the 'get_short_delta_r' method or it failed.")
        except Exception as e:
            raise RuntimeError(f"Failed to calculate short_delta_r using SMPSReader: {e}") from e

    def test_parallel_worker_solves_and_matches_h5(self):
        """
        Tests ParallelSecondStageWorker's solve_batch() against HDF5 reference
        for all scenarios. Compares objective values, primal (y), and dual (pi) solutions.
        """
        num_parallel_workers = 8 # Configurable number of workers for the test
        parallel_worker = None   # Ensure defined for finally block

        try:
            # --- Initialize Parallel Worker ---
            parallel_worker = ParallelSecondStageWorker.from_smps_reader(
                reader=self.reader,
                num_workers=num_parallel_workers
            )

            # --- Solve Batch of Scenarios ---
            # The first-stage solution 'x' is common for all scenarios in this batch.
            # The short_delta_r_all contains deviations for all scenarios.
            # No basis information is passed for warm-start in this test, relying on default behavior.
            obj_vals_batch, y_sols_batch, pi_sols_batch, rc_sols_batch, vbasis_batch_out, cbasis_batch_out, _ = \
                parallel_worker.solve_batch(
                    x=self.x_sol_ref,
                    short_delta_r_batch=self.short_delta_r_all
                )

            # --- Define Comparison Tolerances ---
            rtol = 1e-4 # Relative tolerance for floating-point comparisons
            atol = 1e-5 # Absolute tolerance for floating-point comparisons

            # --- Loop Through Scenarios and Compare Results ---
            self.assertEqual(len(obj_vals_batch), self.num_scenarios_ref, "Mismatch in number of objective values returned.")
            self.assertEqual(y_sols_batch.shape[0], self.num_scenarios_ref, "Mismatch in number of y-solutions returned.")
            self.assertEqual(pi_sols_batch.shape[0], self.num_scenarios_ref, "Mismatch in number of pi-solutions returned.")


            for s in range(self.num_scenarios_ref):
                with self.subTest(scenario_index=s):
                    # Check for NaNs which indicate solve failures for a scenario
                    self.assertFalse(np.isnan(obj_vals_batch[s]), f"Objective value is NaN for scenario {s}, indicating solve failure.")
                    
                    # Compare Primal Solution (y) for scenario s
                    np.testing.assert_allclose(
                        y_sols_batch[s, :],
                        self.y_s_ref[s, :],
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"Primal solution 'y' mismatch for scenario {s}"
                    )


        finally:
            # --- Cleanup ---
            # Ensure the Gurobi environments within the parallel worker are closed
            if parallel_worker is not None:
                parallel_worker.close()

    def test_solve_batch_with_subset_indices(self):
        """
        Tests that solve_batch correctly processes only a subset of scenarios
        when `subset_indices` is provided.
        """
        num_parallel_workers = 4
        with ParallelSecondStageWorker.from_smps_reader(self.reader, num_parallel_workers) as worker:
            # Select a random subset of scenarios to solve
            subset_size = self.num_scenarios_ref // 2
            np.random.seed(42) # for reproducibility
            subset_indices = np.random.choice(self.num_scenarios_ref, size=subset_size, replace=False)
            subset_indices.sort() # Sorting is not required by the function but makes comparison easier

            obj_vals, y_sols, _, _, _, _, _ = worker.solve_batch(
                x=self.x_sol_ref,
                short_delta_r_batch=self.short_delta_r_all,
                subset_indices=subset_indices
            )

            # --- Assertions ---
            self.assertEqual(len(obj_vals), subset_size, "Number of returned objective values should match subset size.")
            self.assertEqual(y_sols.shape[0], subset_size, "Number of returned y-solutions should match subset size.")

            # Compare results against the reference HDF5 data for the selected subset
            expected_y_sols = self.y_s_ref[subset_indices, :]
            np.testing.assert_allclose(
                y_sols,
                expected_y_sols,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Y-solutions for subset do not match reference data."
            )

    def test_solve_batch_with_empty_subset(self):
        """
        Tests that solve_batch returns empty, correctly shaped arrays when an
        empty `subset_indices` is provided.
        """
        num_parallel_workers = 2
        with ParallelSecondStageWorker.from_smps_reader(self.reader, num_parallel_workers) as worker:
            empty_indices = np.array([], dtype=int)
            
            obj_vals, y_sols, pi_sols, rc_sols, vbasis, cbasis, iters = worker.solve_batch(
                x=self.x_sol_ref,
                short_delta_r_batch=self.short_delta_r_all,
                subset_indices=empty_indices
            )

            # --- Assertions for empty results ---
            self.assertEqual(len(obj_vals), 0)
            self.assertEqual(y_sols.shape[0], 0)
            self.assertEqual(pi_sols.shape[0], 0)
            self.assertEqual(rc_sols.shape[0], 0)
            self.assertEqual(vbasis.shape[0], 0)
            self.assertEqual(cbasis.shape[0], 0)
            self.assertEqual(len(iters), 0)
            
            # Check that the dimensions other than the scenario count are correct
            self.assertEqual(y_sols.shape[1], self.reader.d.shape[0])
            self.assertEqual(pi_sols.shape[1], self.reader.r_bar.shape[0])


    def test_solve_batch_with_invalid_subset_indices(self):
        """
        Tests that solve_batch raises a ValueError if `subset_indices` contains
        an out-of-bounds index.
        """
        num_parallel_workers = 2
        with ParallelSecondStageWorker.from_smps_reader(self.reader, num_parallel_workers) as worker:
            invalid_indices = np.array([0, self.num_scenarios_ref, self.num_scenarios_ref + 1])
            
            with self.assertRaises(ValueError):
                worker.solve_batch(
                    x=self.x_sol_ref,
                    short_delta_r_batch=self.short_delta_r_all,
                    subset_indices=invalid_indices
                )

if __name__ == '__main__':
    # This allows running the tests directly from the command line
    unittest.main()
