import unittest
import numpy as np
import h5py
import os
import sys
import hashlib

# This allows the script to be run from anywhere and still find the src module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(PROJECT_ROOT)

try:
    from src.smps_reader import SMPSReader
    from src.argmax_operation import ArgmaxOperation
except ImportError as e:
    print(f"ERROR: Could not import SMPSReader or ArgmaxOperation: {e}")
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

class TestLruEvictionPolicy(unittest.TestCase):
    """
    Tests the LRU eviction and update mechanisms in ArgmaxOperation.
    """
    reader = None
    unique_bases = []
    dummy_pi = None
    dummy_rc = None

    @classmethod
    def setUpClass(cls):
        """
        Load data and find at least 5 unique bases for testing.
        """
        print("\nSetting up TestLruEvictionPolicy...")

        # --- Verify required files exist ---
        required_files = [core_filepath, time_filepath, sto_filepath, H5_RESULTS_FILE]
        if not all(os.path.exists(f) for f in required_files):
            raise FileNotFoundError(f"Missing required test data files.")

        # --- Load SMPS data ---
        cls.reader = SMPSReader(core_filepath, time_filepath, sto_filepath)
        cls.reader.load_and_extract()

        # --- Load basis data from HDF5 ---
        with h5py.File(H5_RESULTS_FILE, "r") as f:
            cbasis_y_all = f['/basis/cbasis_y_all'][:]
            vbasis_y_all = f['/basis/vbasis_y_all'][:]
        
        # --- Find unique bases ---
        hashes = set()
        for i in range(cbasis_y_all.shape[0]):
            vbasis = vbasis_y_all[i]
            cbasis = cbasis_y_all[i]
            basis_hash = hashlib.sha256(np.concatenate((vbasis, cbasis)).tobytes()).hexdigest()
            if basis_hash not in hashes:
                hashes.add(basis_hash)
                cls.unique_bases.append({'vbasis': vbasis, 'cbasis': cbasis, 'hash': basis_hash})
        
        print(f"Found {len(cls.unique_bases)} unique bases.")
        cls.assertTrue(len(cls.unique_bases) >= 5, "Test requires at least 5 unique bases.")

        # Create dummy duals (pi, rc) for adding to the pool
        cls.dummy_pi = np.zeros(cls.reader.C.shape[0])
        # The shape of rc depends on the number of bounded variables
        num_bounded_vars = np.sum(np.isfinite(cls.reader.ub_y) | (np.abs(cls.reader.lb_y) > 1e-9))
        cls.dummy_rc = np.zeros(num_bounded_vars)


    def test_lru_eviction(self):
        """
        Test basic LRU eviction: add items to a full cache and verify the oldest is evicted.
        """
        print("\nRunning test: Basic LRU Eviction...")
        MAX_PI_TEST = 2
        
        argmax_op = ArgmaxOperation.from_smps_reader(
            self.reader, MAX_PI=MAX_PI_TEST, MAX_OMEGA=10, check_optimality=False
        )

        # 1. Add first two unique bases
        b1 = self.unique_bases[0]
        b2 = self.unique_bases[1]
        self.assertTrue(argmax_op.add_pi(self.dummy_pi, self.dummy_rc, b1['vbasis'], b1['cbasis']))
        self.assertTrue(argmax_op.add_pi(self.dummy_pi, self.dummy_rc, b2['vbasis'], b2['cbasis']))
        
        self.assertEqual(argmax_op.num_pi, 2)
        self.assertIn(b1['hash'], argmax_op.lru_manager)
        self.assertIn(b2['hash'], argmax_op.lru_manager)
        print("  Cache filled. num_pi = 2.")

        # 2. Add a third basis, which should trigger eviction of the first (LRU)
        b3 = self.unique_bases[2]
        self.assertTrue(argmax_op.add_pi(self.dummy_pi, self.dummy_rc, b3['vbasis'], b3['cbasis']))
        
        self.assertEqual(argmax_op.num_pi, 2)
        self.assertNotIn(b1['hash'], argmax_op.lru_manager, "Basis 1 should have been evicted.")
        self.assertIn(b2['hash'], argmax_op.lru_manager)
        self.assertIn(b3['hash'], argmax_op.lru_manager)
        print("  Added 3rd basis. Basis 1 correctly evicted.")

        # 3. Add a fourth basis, evicting the second
        b4 = self.unique_bases[3]
        self.assertTrue(argmax_op.add_pi(self.dummy_pi, self.dummy_rc, b4['vbasis'], b4['cbasis']))

        self.assertEqual(argmax_op.num_pi, 2)
        self.assertNotIn(b2['hash'], argmax_op.lru_manager, "Basis 2 should have been evicted.")
        self.assertIn(b3['hash'], argmax_op.lru_manager)
        self.assertIn(b4['hash'], argmax_op.lru_manager)
        print("  Added 4th basis. Basis 2 correctly evicted.")
        print("  Basic LRU Eviction test PASSED.")

    def test_lru_update_on_access(self):
        """
        Test that touching an item makes it the most recently used and avoids eviction.
        """
        print("\nRunning test: LRU Update on Access...")
        MAX_PI_TEST = 2

        argmax_op = ArgmaxOperation.from_smps_reader(
            self.reader, MAX_PI=MAX_PI_TEST, MAX_OMEGA=10, check_optimality=False
        )

        # 1. Add first two unique bases
        b1 = self.unique_bases[0]
        b2 = self.unique_bases[1]
        argmax_op.add_pi(self.dummy_pi, self.dummy_rc, b1['vbasis'], b1['cbasis'])
        argmax_op.add_pi(self.dummy_pi, self.dummy_rc, b2['vbasis'], b2['cbasis'])
        
        # At this point, b1 is the LRU item.
        print("  Cache filled. b1 is LRU.")

        # 2. "Touch" the first basis (at index 0) to make it the most recently used
        argmax_op.update_lru_on_access(np.array([0]))
        print("  Touched basis at index 0. b2 should now be LRU.")

        # 3. Add a third basis. This should now evict the second basis (b2).
        b3 = self.unique_bases[2]
        argmax_op.add_pi(self.dummy_pi, self.dummy_rc, b3['vbasis'], b3['cbasis'])

        self.assertEqual(argmax_op.num_pi, 2)
        self.assertIn(b1['hash'], argmax_op.lru_manager, "Basis 1 should NOT have been evicted.")
        self.assertNotIn(b2['hash'], argmax_op.lru_manager, "Basis 2 SHOULD have been evicted.")
        self.assertIn(b3['hash'], argmax_op.lru_manager)
        print("  Added 3rd basis. Basis 2 correctly evicted, Basis 1 remains.")
        print("  LRU Update on Access test PASSED.")

if __name__ == '__main__':
    unittest.main()
