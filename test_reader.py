import unittest
import os
import numpy as np
import scipy.sparse as sp
from numpy import inf

from smps_reader import SMPSReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes test script is runnable
PROB_NAME = "cep"
FILE_DIR = os.path.join(BASE_DIR, "smps_data", PROB_NAME)

CORE_FILENAME = f"{PROB_NAME}.mps"
TIME_FILENAME = f"{PROB_NAME}.tim"
STO_FILENAME = f"{PROB_NAME}.sto"

class TestSMPSReaderCEP(unittest.TestCase):
    """
    Tests the dimensions of matrices and vectors extracted by SMPSReader
    using the CEP problem instance data.
    """

    reader = None # Class attribute to hold the loaded reader instance

    @classmethod
    def setUpClass(cls):
        """Load the SMPS data once for all tests in this class."""
        core_filepath = os.path.join(FILE_DIR, CORE_FILENAME)
        time_filepath = os.path.join(FILE_DIR, TIME_FILENAME)
        sto_filepath = os.path.join(FILE_DIR, STO_FILENAME)

        cls.reader = SMPSReader(core_filepath, time_filepath, sto_filepath)
        cls.reader.load_and_extract()

        # --- Expected dimensions based on CEP1 problem (n=4 machines, m=3 parts) ---
        # Stage 1 Vars (x): x1, x2, x3, x4, z1, z2, z3, z4
        cls.n_vars_stage1 = 8
        # Stage 2 Vars (y): y11...y34, s1, s2, s3
        cls.n_vars_stage2 = 15 # (3*4 + 3)
        # Stage 1 Constrs: HRSM1, HRSM2, HRSM3, HRSM4, MAI
        cls.n_constr_stage1 = 5
        # Stage 2 Constrs: CAPM1, CAPM2, CAPM3, CAPM4, DEMP1, DEMP2, DEMP3
        cls.n_constr_stage2 = 7

    # --- Test Variable Related Attributes ---

    def test_stage1_var_names_length(self):
        """Check the number of stage 1 variable names."""
        self.assertEqual(len(self.reader.stage1_var_names), self.n_vars_stage1,
                         f"Expected {self.n_vars_stage1} stage 1 var names, got {len(self.reader.stage1_var_names)}")

    def test_stage2_var_names_length(self):
        """Check the number of stage 2 variable names."""
        self.assertEqual(len(self.reader.stage2_var_names), self.n_vars_stage2,
                         f"Expected {self.n_vars_stage2} stage 2 var names, got {len(self.reader.stage2_var_names)}")

    def test_lb_x_shape(self):
        """Check the shape of the stage 1 lower bounds vector."""
        self.assertIsNotNone(self.reader.lb_x, "lb_x attribute should not be None")
        self.assertEqual(self.reader.lb_x.shape, (self.n_vars_stage1,),
                         f"Expected lb_x shape {(self.n_vars_stage1,)}, got {self.reader.lb_x.shape}")

    def test_ub_x_shape(self):
        """Check the shape of the stage 1 upper bounds vector."""
        self.assertIsNotNone(self.reader.ub_x, "ub_x attribute should not be None")
        self.assertEqual(self.reader.ub_x.shape, (self.n_vars_stage1,),
                         f"Expected ub_x shape {(self.n_vars_stage1,)}, got {self.reader.ub_x.shape}")

    def test_lb_y_shape(self):
        """Check the shape of the stage 2 lower bounds vector."""
        self.assertIsNotNone(self.reader.lb_y, "lb_y attribute should not be None")
        self.assertEqual(self.reader.lb_y.shape, (self.n_vars_stage2,),
                         f"Expected lb_y shape {(self.n_vars_stage2,)}, got {self.reader.lb_y.shape}")

    def test_ub_y_shape(self):
        """Check the shape of the stage 2 upper bounds vector."""
        self.assertIsNotNone(self.reader.ub_y, "ub_y attribute should not be None")
        self.assertEqual(self.reader.ub_y.shape, (self.n_vars_stage2,),
                         f"Expected ub_y shape {(self.n_vars_stage2,)}, got {self.reader.ub_y.shape}")

    # --- Test Constraint Related Attributes ---

    def test_stage1_constr_names_length(self):
        """Check the number of stage 1 constraint names."""
        self.assertEqual(len(self.reader.stage1_constr_names), self.n_constr_stage1,
                         f"Expected {self.n_constr_stage1} stage 1 constr names, got {len(self.reader.stage1_constr_names)}")

    def test_stage2_constr_names_length(self):
        """Check the number of stage 2 constraint names."""
        self.assertEqual(len(self.reader.stage2_constr_names), self.n_constr_stage2,
                         f"Expected {self.n_constr_stage2} stage 2 constr names, got {len(self.reader.stage2_constr_names)}")

    def test_A_shape_and_type(self):
        """Check the shape and type of matrix A."""
        self.assertIsNotNone(self.reader.A, "A attribute should not be None")
        # A relates Stage 1 constraints to Stage 1 variables
        expected_shape = (self.n_constr_stage1, self.n_vars_stage1)
        self.assertEqual(self.reader.A.shape, expected_shape,
                         f"Expected A shape {expected_shape}, got {self.reader.A.shape}")
        self.assertTrue(isinstance(self.reader.A, sp.csr_matrix), "A should be a SciPy CSR matrix")

    def test_b_shape(self):
        """Check the shape of the stage 1 RHS vector b."""
        self.assertIsNotNone(self.reader.b, "b attribute should not be None")
        self.assertEqual(self.reader.b.shape, (self.n_constr_stage1,),
                         f"Expected b shape {(self.n_constr_stage1,)}, got {self.reader.b.shape}")

    def test_sense1_shape(self):
        """Check the shape of the stage 1 constraint senses vector."""
        self.assertIsNotNone(self.reader.sense1, "sense1 attribute should not be None")
        self.assertEqual(self.reader.sense1.shape, (self.n_constr_stage1,),
                         f"Expected sense1 shape {(self.n_constr_stage1,)}, got {self.reader.sense1.shape}")

    def test_C_shape_and_type(self):
        """Check the shape and type of matrix C."""
        self.assertIsNotNone(self.reader.C, "C attribute should not be None")
        # C relates Stage 2 constraints to Stage 1 variables
        expected_shape = (self.n_constr_stage2, self.n_vars_stage1)
        self.assertEqual(self.reader.C.shape, expected_shape,
                         f"Expected C shape {expected_shape}, got {self.reader.C.shape}")
        self.assertTrue(isinstance(self.reader.C, sp.csr_matrix), "C should be a SciPy CSR matrix")

    def test_D_shape_and_type(self):
        """Check the shape and type of matrix D."""
        self.assertIsNotNone(self.reader.D, "D attribute should not be None")
        # D relates Stage 2 constraints to Stage 2 variables
        expected_shape = (self.n_constr_stage2, self.n_vars_stage2)
        self.assertEqual(self.reader.D.shape, expected_shape,
                         f"Expected D shape {expected_shape}, got {self.reader.D.shape}")
        self.assertTrue(isinstance(self.reader.D, sp.csr_matrix), "D should be a SciPy CSR matrix")

    def test_r_bar_shape(self):
        """Check the shape of the stage 2 deterministic RHS vector r_bar."""
        self.assertIsNotNone(self.reader.r_bar, "r_bar attribute should not be None")
        self.assertEqual(self.reader.r_bar.shape, (self.n_constr_stage2,),
                         f"Expected r_bar shape {(self.n_constr_stage2,)}, got {self.reader.r_bar.shape}")

    def test_sense2_shape(self):
        """Check the shape of the stage 2 constraint senses vector."""
        self.assertIsNotNone(self.reader.sense2, "sense2 attribute should not be None")
        self.assertEqual(self.reader.sense2.shape, (self.n_constr_stage2,),
                         f"Expected sense2 shape {(self.n_constr_stage2,)}, got {self.reader.sense2.shape}")



if __name__ == '__main__':
    # This allows running the tests directly from the command line
    unittest.main()