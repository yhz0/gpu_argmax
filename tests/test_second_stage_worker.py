import unittest
import h5py
import numpy as np
from pathlib import Path
import sys
import gurobipy as gp # Needed for potential errors during worker init/solve
import os

# This allows the script to be run from anywhere and still find the src module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(PROJECT_ROOT)

from src.smps_reader import SMPSReader
from src.second_stage_worker import SecondStageWorker

class TestSecondStageWorkerAgainstSAA(unittest.TestCase):
    """
    Tests the SecondStageWorker by solving individual scenarios derived from
    an SAA solution stored in an HDF5 file and comparing results.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load reference HDF5 data and SMPS problem data once for all tests.
        This assumes the necessary files are present.
        """
        cls.test_data_path = Path(PROJECT_ROOT) # Adjust if test data is elsewhere
        cls.h5_file_path = cls.test_data_path / "cep_100scen_results.h5"
        cls.smps_base_path = cls.test_data_path / "smps_data/cep" # Base dir for SMPS files
        cls.smps_core_file = cls.smps_base_path / "cep.mps"
        cls.smps_time_file = cls.smps_base_path / "cep.tim"
        cls.smps_sto_file = cls.smps_base_path / "cep.sto"

        # --- Basic File Existence Checks ---
        if not cls.h5_file_path.is_file():
            raise FileNotFoundError(f"HDF5 result file not found: {cls.h5_file_path}")
        if not cls.smps_core_file.is_file():
            raise FileNotFoundError(f"SMPS core file not found: {cls.smps_core_file}")
        # Add checks for .tim and .sto if SMPSReader strictly requires them
        if not cls.smps_time_file.is_file():
             raise FileNotFoundError(f"SMPS time file not found: {cls.smps_time_file}")
        if not cls.smps_sto_file.is_file():
             raise FileNotFoundError(f"SMPS sto file not found: {cls.smps_sto_file}")


        # --- Load Reference Data from HDF5 ---
        # Using 'with' ensures the file is closed properly
        with h5py.File(cls.h5_file_path, 'r') as f:
            # Check if solution was optimal in HDF5 (essential for comparison)
            # Assume Gurobi status code 2 means Optimal
            cls.h5_optimal = f['/metadata'].attrs.get('solution_status_code', -1) == 2
            if not cls.h5_optimal:
                 # Fail setup early if reference data isn't usable
                 raise ValueError(f"HDF5 file {cls.h5_file_path} does not contain an optimal solution (status != 2). Cannot run comparison tests.")

            # Load data needed for the tests
            cls.num_scenarios = f['/metadata'].attrs['num_scenarios']
            # Stochastic RHS *values* (not deviations yet)
            cls.stochastic_rhs_parts_h5 = f['/scenarios/stochastic_rhs_parts'][:]
            # Reference solutions
            cls.x_sol_ref = f['/solution/primal/x'][:]
            cls.y_s_ref = f['/solution/primal/y_s'][:]
            cls.pi_s_ref = f['/solution/dual/pi_s'][:]
            # Reference basis
            cls.vbasis_s_ref = f['/basis/vbasis_y_all'][:]
            cls.cbasis_s_ref = f['/basis/cbasis_y_all'][:]

        # --- Load SMPS data using the SMPSReader ---
        # This assumes SMPSReader constructor takes paths and parses the data.
        # Adjust if SMPSReader has a separate .load() or .parse() method.
        cls.reader = SMPSReader(
            core_file=str(cls.smps_core_file),
            time_file=str(cls.smps_time_file),
            sto_file=str(cls.smps_sto_file)
        )
        cls.reader.load_and_extract()

        # --- Pre-calculate all scenario deviations (short_delta_r) ---
        # Use the loaded reader and the stochastic RHS values from HDF5
        try:
            cls.short_delta_r_all = cls.reader.get_short_delta_r(cls.stochastic_rhs_parts_h5)
        except AttributeError:
             raise AttributeError("SMPSReader instance does not have the required 'get_short_delta_r' method.")
        except Exception as e:
             raise RuntimeError(f"Failed to calculate short_delta_r using reader: {e}") from e


    def test_worker_solves_and_matches_h5(self):
        """
        Tests worker's solve(), get_basis() against HDF5 reference for each scenario.
        """
        worker = None # Ensure worker is defined for finally block cleanup
        try:
            # --- Initialize Worker ---
            # Use the factory method with the pre-loaded reader
            worker = SecondStageWorker.from_smps_reader(self.reader)

            # --- Set First-Stage Solution ---
            worker.set_x(self.x_sol_ref)

            # --- Define Comparison Tolerances ---
            rtol = 1e-3 # Relative tolerance for floats
            atol = 1e-4 # Absolute tolerance for floats

            # --- Loop Through Scenarios ---
            for s in range(self.num_scenarios):
                # Use subtest to clearly identify failing scenario number
                with self.subTest(scenario_index=s):
                    current_short_delta_r = self.short_delta_r_all[s, :]

                    # --- Set Scenario in Worker ---
                    worker.set_scenario(current_short_delta_r)

                    # --- Solve Scenario Subproblem ---
                    # Using default solve parameters (dual simplex, single thread)
                    # Assumes HDF5 results were generated compatibly
                    result = worker.solve()

                    # --- Assert Solve Success ---
                    self.assertIsNotNone(result, f"Worker solve returned None (non-optimal) for scenario {s}.")
                    # If solve failed, skip further comparisons for this scenario
                    if result is None:
                        continue

                    _obj_val, y_sol_worker, pi_sol_worker, _rc_sol_worker = result

                    # --- Get Basis from Worker ---
                    basis_result = worker.get_basis()
                    self.assertIsNotNone(basis_result, f"Worker get_basis returned None for scenario {s}.")
                    if basis_result is None:
                        continue # Skip basis comparison if unavailable

                    vbasis_worker, cbasis_worker = basis_result

                    # --- Compare Primal Solution (y) ---
                    np.testing.assert_allclose(
                        y_sol_worker,
                        self.y_s_ref[s, :],
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"Primal solution 'y' mismatch for scenario {s}"
                    )

        finally:
            # --- Cleanup ---
            # Ensure the Gurobi environment within the worker is closed
            if worker is not None:
                worker.close()

    def test_basis_set_sucessful(self):
        """
        Tests if the basis can be set successfully in the worker.
        """
        worker = None

        # Initialize Worker
        worker = SecondStageWorker.from_smps_reader(self.reader)

        # Set x and scenario
        worker.set_x(self.x_sol_ref)
        current_short_delta_r = self.short_delta_r_all[0, :]
        worker.set_scenario(current_short_delta_r)

        # Optimize the model
        result = worker.solve()

        # print number of simplex iterations
        print(f"First solve: {worker.get_iter_count()} simplex iterations")

        # Obtain the basis
        vb, cb = worker.get_basis()
        print(f"vbasis first solve: {vb}")
        print(f"cbasis first solve: {cb}")

        # Dispose worker and restart a new one
        worker.close()
        del worker
        worker = SecondStageWorker.from_smps_reader(self.reader)

        # Set the first-stage solution and scenario again
        worker.set_x(self.x_sol_ref)
        worker.set_scenario(current_short_delta_r)

        # Set the basis
        worker.set_basis(vb, cb)

        # Optimize the model
        result = worker.solve()

        # print number of simplex iterations
        print(f"Second solve: {worker.get_iter_count()} simplex iterations")

        # assert the number of iterations is zero
        self.assertEqual(worker.get_iter_count(), 0, "Basis not set correctly, expected 0 iterations after setting basis.")

        # Cleanup
        worker.close()

if __name__ == '__main__':
    # This allows running the tests from the command line
    unittest.main()
