import unittest
import numpy as np
import h5py
import os

# --- Assumed imports from your project ---
# Ensure these modules are accessible (e.g., in PYTHONPATH or same directory)
try:
    from smps_reader import SMPSReader
    from argmax_operation import ArgmaxOperation
except ImportError as e:
    print(f"ERROR: Could not import SMPSReader or ArgmaxOperation: {e}")
    print("Please ensure smps_reader.py and argmax_operation.py are accessible.")
    raise # Stop execution if essential modules are missing

# --- Configuration ---
# Use __file__ to make paths relative to the location of this test script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROB_NAME = "cep"
SMPS_DATA_DIR = os.path.join(BASE_DIR, "smps_data", PROB_NAME)
# Assuming the H5 results file is in the same directory as this test script
H5_RESULTS_FILE = os.path.join(BASE_DIR, "cep_100scen_results.h5")

# SMPS file paths
CORE_FILENAME = f"{PROB_NAME}.mps"
TIME_FILENAME = f"{PROB_NAME}.tim"
STO_FILENAME = f"{PROB_NAME}.sto"
core_filepath = os.path.join(SMPS_DATA_DIR, CORE_FILENAME)
time_filepath = os.path.join(SMPS_DATA_DIR, TIME_FILENAME)
sto_filepath = os.path.join(SMPS_DATA_DIR, STO_FILENAME)

# --- Test Class ---

class TestArgmaxCalculationCEP(unittest.TestCase):
    """
    Tests the ArgmaxOperation calculation against pre-computed results
    for the CEP problem instance using data loaded via SMPSReader and HDF5.
    """

    # Class attributes to store objects/data loaded in setUpClass
    reader = None
    x_sol = None
    correct_scenario_objective = None
    argmax_op = None
    num_scenarios = None

    @classmethod
    def setUpClass(cls):
        """
        Load necessary data and set up objects once before running the test method.
        This avoids redundant loading for potentially multiple tests in this class.
        """
        print("\nSetting up TestArgmaxCalculationCEP...")

        # --- Verify required files exist ---
        required_files = [core_filepath, time_filepath, sto_filepath, H5_RESULTS_FILE]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            # Fail setup immediately if data is missing
            raise FileNotFoundError(f"Missing required test data files: {', '.join(missing_files)}")

        # --- Load SMPS data ---
        print(f"Loading SMPS data for '{PROB_NAME}'...")
        cls.reader = SMPSReader(core_filepath, time_filepath, sto_filepath)
        cls.reader.load_and_extract()
        r_bar = cls.reader.r_bar # Deterministic part of stage 2 RHS
        C = cls.reader.C         # Technology matrix coupling stage 1 and 2
        print("SMPS data loaded.")

        # --- Load pre-computed results from HDF5 ---
        print(f"Loading results from '{os.path.basename(H5_RESULTS_FILE)}'...")
        with h5py.File(H5_RESULTS_FILE, "r") as f:
            cls.num_scenarios = f['/metadata'].attrs['num_scenarios']
            pi_s = f['/solution/dual/pi_s'][:] # Dual variables (pi) per scenario
            stochastic_rhs_parts = f['/scenarios/stochastic_rhs_parts'][:] # Stochastic RHS values
            cbasis_y_all = f['/basis/cbasis_y_all'][:] # Constraint basis statuses
            vbasis_y_all = f['/basis/vbasis_y_all'][:] # Variable basis statuses
            cls.x_sol = f['solution/primal/x'][:]      # Optimal first-stage solution (x)
        print(f"HDF5 results loaded ({cls.num_scenarios} scenarios).")

        # --- Calculate the 'correct' objective value for comparison ---
        # This replicates the calculation from the original script: E_s [pi_s * (r_s - C*x)]
        print("Calculating 'correct' expected scenario objective value...")
        # Construct full RHS r[s] for each scenario
        r = np.tile(r_bar, (cls.num_scenarios, 1))
        stochastic_indices = cls.reader.stochastic_rows_relative_indices
        if stochastic_indices is not None and len(stochastic_indices) > 0:
            expected_rhs_shape = (cls.num_scenarios, len(stochastic_indices))
            if stochastic_rhs_parts.shape != expected_rhs_shape:
                 raise ValueError(f"Shape mismatch for stochastic_rhs_parts. Expected {expected_rhs_shape}, got {stochastic_rhs_parts.shape}")
            r[:, stochastic_indices] = stochastic_rhs_parts # Update stochastic elements

        Cx_product = C.dot(cls.x_sol)
        difference_term = r - Cx_product # Shape: (num_scenarios, num_stage2_constraints)
        # Calculate the mean objective across scenarios
        cls.correct_scenario_objective = np.sum(pi_s * difference_term, axis=1).mean()
        print(f"Correct scenario objective calculated: {cls.correct_scenario_objective:.6f}")
        del r, difference_term, Cx_product # Free memory

        # --- Initialize and populate the ArgmaxOperation object ---
        print("Initializing and populating ArgmaxOperation...")
        # Get the variations delta_r from the stochastic parts
        short_delta_r = cls.reader.get_short_delta_r(stochastic_rhs_parts)

        # Define capacities (example: allow some buffer)
        max_pi_capacity = 1000
        max_scenario_capacity = 10000

        cls.argmax_op = ArgmaxOperation.from_smps_reader(
            cls.reader, max_pi_capacity, max_scenario_capacity, device='cpu'
        )

        # Add dual solutions (pi) and basis information from HDF5 results
        for s in range(cls.num_scenarios):
            # Assuming add_pi doesn't need basis_x info here, passing empty array
            cls.argmax_op.add_pi(pi_s[s, :], np.array([]), vbasis_y_all[s, :], cbasis_y_all[s, :])

        # Add scenario data (delta_r)
            cls.argmax_op.add_scenarios(short_delta_r)

        print(f"ArgmaxOperation populated. Num Pi: {cls.argmax_op.num_pi}, Num Scenarios: {cls.argmax_op.num_scenarios}")
        print("Setup complete.")

    # --- Test Method ---

    def test_argmax_calculation_matches_expected_objective(self):
        """
        Verify the objective from ArgmaxOperation.calculate_cut matches the reference value.
        """
        # Pre-conditions check (ensure setUpClass succeeded)
        self.assertIsNotNone(self.argmax_op, "ArgmaxOperation object not initialized.")
        self.assertIsNotNone(self.x_sol, "x_sol not loaded.")
        self.assertIsNotNone(self.correct_scenario_objective, "Correct objective not calculated.")

        print("\nRunning test: Argmax calculation vs Expected objective...")
        # --- Perform the core calculation using ArgmaxOperation ---
        # calculate_cut likely returns alpha, beta, and maybe an index
        alpha, beta, _, _ = self.argmax_op.calculate_cut(self.x_sol) # Ignore index if unused
        # Calculate the estimated objective value: alpha + beta^T * x
        estimated_objective = alpha + beta @ self.x_sol

        print(f"  Estimated objective from ArgmaxOperation: {estimated_objective:.6f}")
        print(f"  Expected ('correct') objective value:      {self.correct_scenario_objective:.6f}")

        # --- Assertion ---
        # Compare the estimated value with the pre-calculated 'correct' value
        # using assertAlmostEqual for robust floating-point comparison.
        tolerance = 0.1 # Based on the original script's assertion
        self.assertAlmostEqual(
            estimated_objective,
            self.correct_scenario_objective,
            delta=tolerance,
            msg=(f"Estimated objective {estimated_objective:.6f} differs from expected "
                 f"{self.correct_scenario_objective:.6f} by more than delta={tolerance}")
        )
        print(f"Assertion PASSED: Estimated objective matches expected objective within tolerance {tolerance}.")

# --- Standard boilerplate to run tests ---
if __name__ == '__main__':
    unittest.main()