import torch
import numpy as np
import scipy.sparse # For type hint
import hashlib
import time
import math
from typing import Tuple, Optional, Union, TYPE_CHECKING
import threading
from cachetools import LRUCache
# from memory_monitor import MemoryMonitor # TODO: remove after debugging

if TYPE_CHECKING:
    from smps_reader import SMPSReader

class ArgmaxOperation:
    """
    Implements GPU-accelerated calculation related to Benders cuts using PyTorch.

    Stores dual vectors (pi, rc) on the target device (GPU/CPU) and associated
    basis information (vbasis, cbasis) on the CPU. Calculates Benders cut
    coefficients alpha and beta based on maximizing the dual objective over
    stored dual solutions for a batch of scenarios.

    Features:
    - Stores (pi, RC) on target PyTorch device, (vbasis, cbasis) on CPU.
    - Deduplicates added solutions based on the (pi, rc) pair.
    - Processes scenarios in batches for memory efficiency.
    - Uses double precision (float64) for key reduction steps for accuracy.
    - Handles sparse transfer matrix C (torch.sparse_csr_tensor).
    - Configurable precision for optimality checking (`torch.float32` or `torch.float64`).
    """

    def __init__(self, NUM_STAGE2_ROWS: int, NUM_STAGE2_VARS: int,
                 X_DIM: int, MAX_PI: int, MAX_OMEGA: int,
                 r_sparse_indices: np.ndarray,
                 r_bar: np.ndarray,
                 sense: np.ndarray,
                 C: scipy.sparse.spmatrix,
                 D: scipy.sparse.spmatrix,
                 lb_y: np.ndarray,
                 ub_y: np.ndarray,
                 scenario_batch_size: int = 10000,
                 device: Optional[Union[str, torch.device]] = None,
                 NUM_CANDIDATES: int = 8,
                 optimality_dtype: torch.dtype = torch.float32
                 ):
        """
        Initializes the ArgmaxOperation class.

        Args:
            NUM_STAGE2_ROWS: Dimension of pi vector and stage 2 constraints.
            NUM_STAGE2_VARS: Dimension of vbasis vector (total stage 2 vars).
            X_DIM: Dimension of x vector (stage 1 vars).
            MAX_PI: Maximum number of (pi, RC, basis) tuples to store.
            MAX_OMEGA: Maximum number of scenarios to store.
            r_sparse_indices: 1D np.ndarray of indices where stage 2 RHS is stochastic.
            r_bar: 1D np.ndarray for the fixed part of stage 2 RHS.
            sense: 1D np.ndarray of constraint senses. (e.g., '<', '=', '>') 
            C: SciPy sparse matrix (stage 2 rows x X_DIM).
            D: SciPy sparse matrix (stage 2 rows x NUM_STAGE2_VARS).
            lb_y: 1D np.ndarray of lower bounds for all stage 2 variables.
            ub_y: 1D np.ndarray of upper bounds for all stage 2 variables.
            scenario_batch_size: Number of scenarios per GPU batch. Defaults to 10,000.
            device: PyTorch device ('cuda', 'cpu', etc.). Auto-detects if None.
            optimality_dtype: The torch.dtype for storing basis factors and related data
                              for optimality checks. Defaults to torch.float32. Use
                              torch.float64 for higher precision if needed.
        """
        print(f"[{time.strftime('%H:%M:%S')}] Initializing ArgmaxOperation...")
        start_time = time.time()

        # --- Determine Device ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[{time.strftime('%H:%M:%S')}] Using device: {self.device}")

        # --- Input Validation ---
        if r_bar.shape != (NUM_STAGE2_ROWS,): raise ValueError("r_bar shape mismatch.")
        if sense.shape != (NUM_STAGE2_ROWS,): raise ValueError("sense shape mismatch.")
        if C.shape != (NUM_STAGE2_ROWS, X_DIM): raise ValueError("C matrix shape mismatch.")
        if lb_y.shape != (NUM_STAGE2_VARS,): raise ValueError("lb_y shape mismatch.")
        if ub_y.shape != (NUM_STAGE2_VARS,): raise ValueError("ub_y shape mismatch.")
        if not isinstance(r_sparse_indices, np.ndarray) or r_sparse_indices.ndim != 1: raise ValueError("r_sparse_indices must be 1D.")
        unique_r_indices = np.unique(r_sparse_indices)
        if len(unique_r_indices) != len(r_sparse_indices):
            print("Warning: r_sparse_indices contains duplicate values. Using unique indices.")
            r_sparse_indices = unique_r_indices
        if np.any(r_sparse_indices < 0) or np.any(r_sparse_indices >= NUM_STAGE2_ROWS): raise ValueError("r_sparse_indices out of bounds.")
        if not scipy.sparse.issparse(C): raise TypeError("C must be a SciPy sparse matrix.")
        if not isinstance(scenario_batch_size, int) or scenario_batch_size <= 0: raise ValueError("scenario_batch_size must be positive.")

        # --- Store D matrix on CPU in CSC format for efficient column slicing ---
        self.D = D.tocsc()

        # --- Dimensions and Capacities ---
        self.sense = sense
        self.NUM_STAGE2_ROWS = NUM_STAGE2_ROWS
        self.NUM_STAGE2_VARS = NUM_STAGE2_VARS
        self.X_DIM = X_DIM
        self.MAX_PI = MAX_PI
        self.MAX_OMEGA = MAX_OMEGA
        self.scenario_batch_size = scenario_batch_size
        self.NUM_CANDIDATES = NUM_CANDIDATES
        self.r_sparse_indices_cpu = np.sort(np.array(r_sparse_indices, dtype=np.int32))
        self.R_SPARSE_LEN = len(self.r_sparse_indices_cpu)

        # --- Counters ---
        self.num_pi = 0
        self.num_scenarios = 0

        # --- LRU Replacement Mechanism ---
        self.lru_manager = LRUCache(maxsize=MAX_PI) # Maps (vbasis, cbasis) pairs to their indices
        self.index_to_hash_map = {}  # Maps index to basis hash for quick lookup

        # --- CPU Data Storage (Only Basis) ---
        print(f"[{time.strftime('%H:%M:%S')}] Allocating CPU memory for basis...")
        basis_dtype_np = np.int8 # Compact storage for basis status
        self.vbasis_cpu = np.zeros((MAX_PI, NUM_STAGE2_VARS), dtype=basis_dtype_np)
        self.cbasis_cpu = np.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=basis_dtype_np)

        # --- Bounded Variable Calculation ---
        self.lb_y_full = lb_y
        self.ub_y_full = ub_y
        has_finite_ub = np.isfinite(ub_y)
        has_finite_lb = np.isfinite(lb_y) & (np.abs(lb_y) > 1e-9)
        self.bounded_mask = has_finite_ub | has_finite_lb
        lb_y_bounded = lb_y[self.bounded_mask]
        ub_y_bounded = ub_y[self.bounded_mask]
        self.NUM_BOUNDED_VARS = len(lb_y_bounded)

        # --- GPU (Device) Data Storage ---
        print(f"[{time.strftime('%H:%M:%S')}] Allocating memory on device {self.device}...")
        torch_dtype = torch.float32 # Default dtype for most device tensors

        self.pi_gpu = torch.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=torch_dtype, device=self.device)
        self.rc_gpu = torch.zeros((MAX_PI, self.NUM_BOUNDED_VARS), dtype=torch_dtype, device=self.device)
        self.short_pi_gpu = torch.zeros((MAX_PI, self.R_SPARSE_LEN), dtype=torch_dtype, device=self.device)
        self.lb_gpu = torch.from_numpy(lb_y_bounded).to(dtype=torch_dtype, device=self.device)
        self.ub_gpu = torch.from_numpy(ub_y_bounded).to(dtype=torch_dtype, device=self.device)
        self.lb_gpu_f32 = self.lb_gpu.to(torch.float32)
        self.ub_gpu_f32 = self.ub_gpu.to(torch.float32)
        self.r_bar_gpu = torch.from_numpy(r_bar).to(dtype=torch_dtype, device=self.device)
        self.r_sparse_indices_gpu = torch.from_numpy(self.r_sparse_indices_cpu).to(dtype=torch.long, device=self.device) # LongTensor for indexing
        self.short_delta_r_gpu = torch.zeros((MAX_OMEGA, self.R_SPARSE_LEN), dtype=torch_dtype, device=self.device)

        # --- Calculate C_gpu_fp64_transpose (Transpose of C, float64) ---
        CT_csr64 = C.transpose().tocsr().astype(np.float64, copy=False)

        self.C_gpu_fp64_transpose = torch.sparse_csr_tensor(
            torch.from_numpy(CT_csr64.indptr).long().to(self.device),
            torch.from_numpy(CT_csr64.indices).long().to(self.device),
            torch.from_numpy(CT_csr64.data).to(dtype=torch.float64, device=self.device),
            size=CT_csr64.shape, dtype=torch.float64, device=self.device
        )

        # --- Original code for C_gpu (float32 CSR) ---
        C_csr32 = C.astype(np.float32, copy=False)
        self.C_gpu = torch.sparse_csr_tensor(
            torch.from_numpy(C_csr32.indptr).long().to(self.device),
            torch.from_numpy(C_csr32.indices).long().to(self.device),
            torch.from_numpy(C_csr32.data).to(dtype=torch.float32, device=self.device),
            size=C_csr32.shape, dtype=torch.float32, device=self.device
        )

        # Data requirements for optimality check 
        # Let z = [y, s] where s is the slack variables s>=0
        device_factors = 'cpu' # Factors are always stored on CPU for now
        # LU Factors of B_j
        self.basis_factors_cpu = torch.zeros((MAX_PI, NUM_STAGE2_ROWS, NUM_STAGE2_ROWS), dtype=optimality_dtype, device=device_factors)
        # Pivots of B_j
        self.basis_pivots_cpu = torch.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=torch.int32, device=device_factors)
        # Basic Variable Lower Bounds of B_j
        self.basis_lb_cpu = torch.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=optimality_dtype, device=device_factors)
        # Basic Variable Upper Bounds of B_j
        self.basis_ub_cpu = torch.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=optimality_dtype, device=device_factors)
        # Non-Basic Contribution of N_j . i.e. N@z_N
        self.non_basic_contribution_cpu = torch.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=optimality_dtype, device=device_factors)

        # --- Threading Lock ---
        self._lock = threading.Lock()

        # --- Placeholders for results from find_optimal_basis ---
        self.best_k_indices_gpu = torch.zeros((MAX_OMEGA,), dtype=torch.long, device=self.device)
        self.best_k_scores_gpu = torch.zeros((MAX_OMEGA,), dtype=torch.float32, device=self.device)
        self.is_verified_optimal_gpu = torch.zeros((MAX_OMEGA,), dtype=torch.bool, device=self.device)

        # Current candidate solution
        self.current_x_gpu = torch.zeros(X_DIM, dtype=torch.float32, device=self.device)

        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Initialization complete ({end_time - start_time:.2f}s).")

    @classmethod
    def from_smps_reader(cls, reader: 'SMPSReader',
                         MAX_PI: int, MAX_OMEGA: int,
                         scenario_batch_size: int = 10000,
                         device: Optional[Union[str, torch.device]] = None,
                         NUM_CANDIDATES: int = 8,
                         optimality_dtype: torch.dtype = torch.float32) -> 'ArgmaxOperation':
        """
        Factory method to create an ArgmaxOperation instance from an SMPSReader.

        Args:
            reader: An initialized and loaded SMPSReader instance.
            MAX_PI: Maximum number of (pi, RC, basis) tuples to store.
            MAX_OMEGA: Maximum number of scenarios to store.
            scenario_batch_size: Number of scenarios per GPU batch. Defaults to 10,000.
            device: PyTorch device ('cuda', 'cpu', etc.). Auto-detects if None.
            NUM_CANDIDATES: The number of top candidates to check for feasibility.
            optimality_dtype: The torch.dtype for storing basis factors. Defaults to torch.float32.

        Returns:
            An initialized ArgmaxOperation instance.

        Raises:
            ValueError: If the reader hasn't been loaded or essential data is missing.
        """
        if not reader._data_loaded:
            raise ValueError("SMPSReader data must be loaded using load_and_extract() before creating ArgmaxOperation.")
        if reader.C is None:
             raise ValueError("SMPSReader did not load matrix C (stage 2 technology matrix).")
        if reader.D is None:
             raise ValueError("SMPSReader did not load matrix D (stage 2 recourse matrix).")
        
        # --- Extract parameters from reader ---
        NUM_STAGE2_ROWS = len(reader.row2_indices)
        NUM_STAGE2_VARS = len(reader.y_indices)
        X_DIM = len(reader.x_indices)
        r_sparse_indices = reader.stochastic_rows_relative_indices
        r_bar = reader.r_bar
        sense = reader.sense2
        C = reader.C
        D = reader.D
        lb_y = reader.lb_y if reader.lb_y is not None else np.array([])
        ub_y = reader.ub_y if reader.ub_y is not None else np.array([])

        # --- Instantiate the class using the extracted parameters ---
        return cls(
            NUM_STAGE2_ROWS=NUM_STAGE2_ROWS,
            NUM_STAGE2_VARS=NUM_STAGE2_VARS,
            X_DIM=X_DIM,
            MAX_PI=MAX_PI,
            MAX_OMEGA=MAX_OMEGA,
            r_sparse_indices=r_sparse_indices,
            r_bar=r_bar,
            sense=sense,
            C=C,
            D=D,
            lb_y=lb_y,
            ub_y=ub_y,
            scenario_batch_size=scenario_batch_size,
            device=device,
            NUM_CANDIDATES=NUM_CANDIDATES,
            optimality_dtype=optimality_dtype
        )

    def _hash_basis_pair(self, vbasis: np.ndarray, cbasis: np.ndarray) -> str:
        basis_np_dtype = np.int8
        vbasis = vbasis.astype(basis_np_dtype, copy=False)
        cbasis = cbasis.astype(basis_np_dtype, copy=False)
        combined = np.concatenate((vbasis, cbasis))
        return hashlib.sha256(combined.tobytes()).hexdigest()

    def add_pi(self, new_pi: np.ndarray, new_rc: np.ndarray,
               new_vbasis: np.ndarray, new_cbasis: np.ndarray) -> bool:
        """
        Adds a new dual solution (pi, rc) and its basis (vbasis, cbasis).
        Stores pi/rc on device, basis on CPU. Deduplicates based on the basis pair (vbasis, cbasis).

        Args:
            new_pi: Constraint dual vector (NumPy).
            new_rc: Reduced cost vector (NumPy).
            new_vbasis: Variable basis status vector (NumPy int8).
            new_cbasis: Constraint basis status vector (NumPy int8).

        Returns:
            True if the solution was added, False if duplicate or storage full.
        """
        # --- Dimension Checks ---
        if new_pi.shape != (self.NUM_STAGE2_ROWS,): raise ValueError("new_pi shape mismatch.")
        if new_rc.shape != (self.NUM_BOUNDED_VARS,): raise ValueError("new_rc shape mismatch.")
        if new_vbasis.shape != (self.NUM_STAGE2_VARS,): raise ValueError("new_vbasis shape mismatch.")
        if new_cbasis.shape != (self.NUM_STAGE2_ROWS,): raise ValueError("new_cbasis shape mismatch.")

        # Ensure consistent dtypes for basis hashing/storage and float conversion for pi/rc
        pi_np_dtype = np.float32 # Match torch_dtype in __init__
        basis_np_dtype = np.int8 # Match CPU storage dtype
        new_pi = new_pi.astype(pi_np_dtype, copy=False)
        new_rc = new_rc.astype(pi_np_dtype, copy=False)
        new_vbasis = new_vbasis.astype(basis_np_dtype, copy=False)
        new_cbasis = new_cbasis.astype(basis_np_dtype, copy=False)

        # --- Acquire Lock for Critical Section ---
        with self._lock:
            # Deduplication based on the (vbasis, cbasis) pair
            current_hash = self._hash_basis_pair(new_vbasis, new_cbasis)
            if current_hash in self.lru_manager:
                # Touch the entry to mark it as recently used
                _ = self.lru_manager[current_hash]
                return False

            # Index determination
            if len(self.lru_manager) >= self.MAX_PI:
                lru_item = self.lru_manager.popitem()  # Remove the least recently used item
                idx = lru_item[1]  # Get the index of the evicted item
                del self.index_to_hash_map[idx]  # Remove from index map
            else:
                idx = len(self.lru_manager)

            # Store basis only on CPU
            self.vbasis_cpu[idx] = new_vbasis
            self.cbasis_cpu[idx] = new_cbasis

            # Extract the 'short' version from pi using the sparse indices (on CPU)
            # This is needed for efficient calculation later, store result on device
            if self.R_SPARSE_LEN > 0:
                short_new_pi = new_pi[self.r_sparse_indices_cpu]
            else:
                short_new_pi = np.array([], dtype=pi_np_dtype)

            # Add pi, rc, and short_pi to target device (GPU)
            # Use non-blocking transfers if on CUDA for potential overlap
            is_cuda = self.device.type == 'cuda'
            self.pi_gpu[idx].copy_(torch.from_numpy(new_pi), non_blocking=is_cuda)
            self.rc_gpu[idx].copy_(torch.from_numpy(new_rc), non_blocking=is_cuda)
            self.short_pi_gpu[idx].copy_(torch.from_numpy(short_new_pi), non_blocking=is_cuda)

            # --- Offline Factorization for Optimality Checking  ---
            # 1. Construct the sparse basis matrix B
            B = self._get_basis_matrix(new_vbasis, new_cbasis)

            try:
                # 2. Factorize B and cache the LU factors and pivots using PyTorch
                B_dense_tensor = torch.from_numpy(B.toarray()).to(
                    dtype=self.basis_factors_cpu.dtype,
                    device=self.basis_factors_cpu.device
                )
                torch.linalg.lu_factor(
                    B_dense_tensor,
                    pivot=True,
                    out=(self.basis_factors_cpu[idx], self.basis_pivots_cpu[idx])
                )
            except torch.linalg.LinAlgError:
                print(f"Warning: Singular basis matrix encountered for new solution {self.num_pi}. "
                      "Marking factors with NaN to fail optimality checks.")
                # Mark with NaN so it never passes the optimality check
                self.basis_factors_cpu[idx].fill_(float('nan'))
                # Pivots can be zero, doesn't matter as NaN factors will propagate
                self.basis_pivots_cpu[idx].zero_()

            # 3. Cache the bounds for the basic variables
            lb_B, ub_B = self._get_basic_variable_bounds(new_vbasis, new_cbasis)
            self.basis_lb_cpu[idx].copy_(torch.from_numpy(lb_B))
            self.basis_ub_cpu[idx].copy_(torch.from_numpy(ub_B))

            # 4. Cache the contribution from non-basic variables (N @ z_N)
            non_basic_contribution = self._get_non_basic_contribution(new_vbasis)
            self.non_basic_contribution_cpu[idx].copy_(torch.from_numpy(non_basic_contribution))

            # --- LRU bookkeeping ---
            # Update LRU manager and index map
            self.lru_manager[current_hash] = idx
            self.index_to_hash_map[idx] = current_hash

            # Keep track of the number of stored solutions
            self.num_pi = len(self.lru_manager)  # Update the count of stored solutions

            # --- Critical Section End ---
            return True
        
    def update_lru_on_access(self, best_k_index: np.ndarray) -> None:
        """
        Updates the LRU cache based on the frequency of winning solutions.

        This method processes the `best_k_index` array, which contains the index
        of the winning dual solution for each scenario. It updates the LRU manager
        by treating solutions that win more frequently as "more recently used."

        The process is as follows:
        1. Count the occurrences of each unique index in `best_k_index`.
        2. Sort these unique indices in ascending order of their counts.
        3. Iterate through the sorted indices and "touch" each one in the
           LRU cache. By touching the most frequent winner last, it becomes
           the most recently used item in the cache.

        Args:
            best_k_index: A 1-D numpy array of indices for the effective dual basis.
        """
        if best_k_index.size == 0:
            return

        # Get unique indices and their counts
        unique_indices, counts = np.unique(best_k_index, return_counts=True)

        # Pair indices with their counts and sort by count in ascending order
        sorted_by_count = sorted(zip(unique_indices, counts), key=lambda item: item[1])

        with self._lock:
            # Touch items in the LRU cache in ascending order of frequency
            for index, _ in sorted_by_count:
                # The index must be converted to a Python int for dictionary key lookup
                py_index = int(index)
                if py_index in self.index_to_hash_map:
                    basis_hash = self.index_to_hash_map[py_index]
                    # Accessing the item marks it as recently used
                    _ = self.lru_manager[basis_hash]


    def add_scenarios(self, new_short_r_delta: np.ndarray) -> bool:
        """
        Adds new scenario data (packed delta_r values for stochastic RHS rows).
        User is expected to provide data with scenarios as rows.

        Args:
            new_short_r_delta: Array of shape (num_new_scenarios, R_SPARSE_LEN).
                               Each row represents a scenario, columns are stochastic elements.
        Returns:
            True if scenarios were added (partially or fully), False otherwise.
        """
        # Check shape[1] for R_SPARSE_LEN, input is (num_scenarios, R_SPARSE_LEN)
        if new_short_r_delta.ndim != 2 or new_short_r_delta.shape[1] != self.R_SPARSE_LEN:
            raise ValueError(f"new_short_r_delta shape mismatch. Expected (num_new_scenarios, {self.R_SPARSE_LEN}), got {new_short_r_delta.shape}")
        
        # num_new_scenarios is now from shape[0]
        num_new_scenarios = new_short_r_delta.shape[0]
        available_slots = self.MAX_OMEGA - self.num_scenarios
        if num_new_scenarios <= 0 or available_slots <= 0:
            # print(f"Debug: No new scenarios to add or no available slots. New: {num_new_scenarios}, Available: {available_slots}")
            return False

        num_to_add = min(num_new_scenarios, available_slots)
        
        new_short_r_delta_to_add = new_short_r_delta # Placeholder for slicing
        if num_to_add < num_new_scenarios:
            print(f"Warning: Exceeds MAX_OMEGA. Adding only {num_to_add} of {num_new_scenarios} new scenarios.")
            # Slicing rows from the input
            new_short_r_delta_to_add = new_short_r_delta[:num_to_add, :]
        else:
            new_short_r_delta_to_add = new_short_r_delta # Use as is if it fits

        start_row_idx = self.num_scenarios 
        end_row_idx = self.num_scenarios + num_to_add

        # Add scenarios to device tensor. Input new_short_r_delta_to_add is already (num_to_add, R_SPARSE_LEN)
        new_scenario_data_gpu = torch.from_numpy(new_short_r_delta_to_add).to(dtype=self.short_delta_r_gpu.dtype, device=self.device)
        self.short_delta_r_gpu[start_row_idx:end_row_idx, :] = new_scenario_data_gpu

        self.num_scenarios += num_to_add
        return True

    def find_optimal_basis(self, x: np.ndarray, touch_lru: bool = True, primal_feas_tol: float = 1e-5) -> None:
        """
        Finds the best, preferrably primally feasible dual solution for each scenario.

        This method integrates score calculation with a robust feasibility check:
        1.  Computes scores for all dual solutions against all scenarios.
        2.  For each scenario, it identifies the top `NUM_CANDIDATES` solutions.
        3.  It then performs a batched primal feasibility check on these candidates.
        4.  The final winner is the highest-scoring candidate that is confirmed
            to be primally feasible.
        5.  If no candidates are feasible, it defaults to the top-scoring one.

        The results (indices and scores) are stored on the device.

        Args:
            x: The first-stage decision vector (NumPy array, size X_DIM).
            touch_lru: Whether to update the LRU cache for the winning solutions.
            primal_feas_tol: The tolerance for checking feasibility bounds.
        """
        if self.num_pi == 0:
            print("Warning: No pi vectors stored. Cannot find optimal basis.")
            return
        if self.num_scenarios == 0:
            print("Warning: No scenarios stored. Cannot find optimal basis.")
            return
        if x.shape != (self.X_DIM,):
            raise ValueError("Input x has incorrect shape.")

        with torch.no_grad():
            # # Initialize memory monitoring
            # monitor = MemoryMonitor(self.device)
            # monitor.set_baseline("find_optimal_basis_start")
            
            # --- Prepare device data views ---
            active_pi_gpu = self.pi_gpu[:self.num_pi]  # shape: (num_pi, NUM_STAGE2_ROWS)
            active_rc_gpu = self.rc_gpu[:self.num_pi]  # shape: (num_pi, NUM_BOUNDED_VARS)
            active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]  # shape: (num_pi, short_delta_r_dim)
            active_scenario_data_gpu = self.short_delta_r_gpu[:self.num_scenarios, :]  # shape: (num_scenarios, short_delta_r_dim)

            self.current_x_gpu.copy_(torch.from_numpy(x))  # shape: (X_DIM,)

            # --- Precompute terms constant across ALL scenarios ---
            Cx_gpu = torch.matmul(self.C_gpu, self.current_x_gpu)  # shape: (NUM_STAGE2_ROWS,)
            h_bar_gpu = self.r_bar_gpu - Cx_gpu  # shape: (NUM_STAGE2_ROWS,)
            pi_h_bar_term_all_k = torch.matmul(active_pi_gpu, h_bar_gpu)  # shape: (num_pi,)
            
            lambda_all_k = torch.clamp(active_rc_gpu, min=0)  # shape: (num_pi, NUM_BOUNDED_VARS)
            mu_all_k = torch.clamp(-active_rc_gpu, min=0)  # shape: (num_pi, NUM_BOUNDED_VARS)
            lambda_l_term_all_k = torch.matmul(lambda_all_k, self.lb_gpu_f32)  # shape: (num_pi,)
            mu_u_term_all_k = torch.matmul(mu_all_k, self.ub_gpu_f32)  # shape: (num_pi,)
            constant_score_part_all_k = pi_h_bar_term_all_k - lambda_l_term_all_k + mu_u_term_all_k  # shape: (num_pi,)

            # --- Pre-allocate expanded tensors once to avoid per-iteration allocation ---
            max_candidates_per_batch = self.scenario_batch_size * min(self.NUM_CANDIDATES, self.num_pi)
            expanded_factors = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS, self.NUM_STAGE2_ROWS),  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS, NUM_STAGE2_ROWS)
                                         dtype=self.basis_factors_cpu.dtype, device=self.device)
            expanded_pivots = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS),  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS)
                                        dtype=self.basis_pivots_cpu.dtype, device=self.device)
            expanded_lb = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS),  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS)
                                    dtype=self.basis_lb_cpu.dtype, device=self.device)
            expanded_ub = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS),  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS)
                                    dtype=self.basis_ub_cpu.dtype, device=self.device)
            expanded_non_basic_contrib = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS),  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS)
                                                   dtype=self.non_basic_contribution_cpu.dtype, device=self.device)
            
            # Pre-allocate all intermediate matrices to eliminate per-iteration allocation
            optimality_dtype = self.basis_factors_cpu.dtype
            delta_r_candidates = torch.zeros((max_candidates_per_batch, self.NUM_STAGE2_ROWS), dtype=optimality_dtype, device=self.device)  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS)
            repeated_delta_r = torch.empty((max_candidates_per_batch, self.short_delta_r_gpu.shape[1]), dtype=optimality_dtype, device=self.device)  # shape: (max_candidates_per_batch, short_delta_r_dim)
            rhs_candidates = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS), dtype=optimality_dtype, device=self.device)  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS)
            solution_z_B = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS, 1), dtype=optimality_dtype, device=self.device)  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS, 1)
            feasible = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS), dtype=torch.bool, device=self.device)  # shape: (max_candidates_per_batch, NUM_STAGE2_ROWS)

            # --- Process scenarios in batches ---
            # monitor.measure("after_preprocessing")
            num_batches = math.ceil(self.num_scenarios / self.scenario_batch_size)
            for i in range(num_batches):
                # batch_label = f"batch_{i+1}_of_{num_batches}"
                # monitor.measure(f"{batch_label}_start")
                
                start_idx = i * self.scenario_batch_size
                end_idx = min((i + 1) * self.scenario_batch_size, self.num_scenarios)
                batch_size = end_idx - start_idx
                
                # --- Step 1: Calculate Scores ---
                batch_scenario_slice = active_scenario_data_gpu[start_idx:end_idx, :]  # shape: (batch_size, short_delta_r_dim)
                scores_batch = torch.matmul(batch_scenario_slice, active_short_pi_gpu.T)  # shape: (batch_size, num_pi)
                scores_batch += constant_score_part_all_k  # In-place addition, shape: (batch_size, num_pi)
                
                # --- Step 2: Find Top Candidates ---
                effective_k = min(self.NUM_CANDIDATES, self.num_pi)
                top_k_scores, top_k_indices = torch.topk(scores_batch, k=effective_k, dim=1)  # shapes: (batch_size, effective_k)

                # --- Step 3 & 4: Batched Feasibility Check ---
                # Flatten the workload: (batch_size, k) -> (batch_size * k)
                candidate_indices_flat = top_k_indices.flatten()  # shape: (batch_size * effective_k,)

                # Gather basis data from CPU cache
                unique_candidate_indices, inverse_map = torch.unique(candidate_indices_flat.cpu(), return_inverse=True)  # shapes: (n_unique,), (batch_size * effective_k,)
                
                factors_batch_cpu = self.basis_factors_cpu[unique_candidate_indices]  # shape: (n_unique, NUM_STAGE2_ROWS, NUM_STAGE2_ROWS)
                pivots_batch_cpu = self.basis_pivots_cpu[unique_candidate_indices]  # shape: (n_unique, NUM_STAGE2_ROWS)
                lb_batch_cpu = self.basis_lb_cpu[unique_candidate_indices]  # shape: (n_unique, NUM_STAGE2_ROWS)
                ub_batch_cpu = self.basis_ub_cpu[unique_candidate_indices]  # shape: (n_unique, NUM_STAGE2_ROWS)
                non_basic_contrib_cpu = self.non_basic_contribution_cpu[unique_candidate_indices]  # shape: (n_unique, NUM_STAGE2_ROWS)
                
                # Move data to GPU and expand to match the flattened workload
                if self.device.type == 'cuda':
                    factors_gpu = factors_batch_cpu.to(self.device)  # shape: (n_unique, NUM_STAGE2_ROWS, NUM_STAGE2_ROWS)
                    pivots_gpu = pivots_batch_cpu.to(self.device)  # shape: (n_unique, NUM_STAGE2_ROWS)
                    lb_gpu = lb_batch_cpu.to(self.device)  # shape: (n_unique, NUM_STAGE2_ROWS)
                    ub_gpu = ub_batch_cpu.to(self.device)  # shape: (n_unique, NUM_STAGE2_ROWS)
                    non_basic_contrib_gpu = non_basic_contrib_cpu.to(self.device)  # shape: (n_unique, NUM_STAGE2_ROWS)
                    inverse_map_gpu = inverse_map.to(self.device)  # shape: (batch_size * effective_k,)
                else:
                    # On CPU, avoid unnecessary copying
                    factors_gpu = factors_batch_cpu  # shape: (n_unique, NUM_STAGE2_ROWS, NUM_STAGE2_ROWS)
                    pivots_gpu = pivots_batch_cpu  # shape: (n_unique, NUM_STAGE2_ROWS)
                    lb_gpu = lb_batch_cpu  # shape: (n_unique, NUM_STAGE2_ROWS)
                    ub_gpu = ub_batch_cpu  # shape: (n_unique, NUM_STAGE2_ROWS)
                    non_basic_contrib_gpu = non_basic_contrib_cpu  # shape: (n_unique, NUM_STAGE2_ROWS)
                    inverse_map_gpu = inverse_map  # shape: (batch_size * effective_k,)

                # Use slices of pre-allocated tensors and populate with index_select (avoids allocation)
                total_candidates = batch_size * effective_k
                expanded_factors_slice = expanded_factors[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS, NUM_STAGE2_ROWS)
                expanded_pivots_slice = expanded_pivots[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS)
                expanded_lb_slice = expanded_lb[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS)
                expanded_ub_slice = expanded_ub[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS)
                expanded_non_basic_contrib_slice = expanded_non_basic_contrib[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS)
                
                torch.index_select(factors_gpu, 0, inverse_map_gpu, out=expanded_factors_slice)
                torch.index_select(pivots_gpu, 0, inverse_map_gpu, out=expanded_pivots_slice)
                torch.index_select(lb_gpu, 0, inverse_map_gpu, out=expanded_lb_slice)
                torch.index_select(ub_gpu, 0, inverse_map_gpu, out=expanded_ub_slice)
                torch.index_select(non_basic_contrib_gpu, 0, inverse_map_gpu, out=expanded_non_basic_contrib_slice)

                # Use slices of pre-allocated intermediate matrices (eliminates per-iteration allocation)
                delta_r_candidates_slice = delta_r_candidates[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS)
                repeated_delta_r_slice = repeated_delta_r[:total_candidates]  # shape: (total_candidates, short_delta_r_dim)
                rhs_candidates_slice = rhs_candidates[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS)
                solution_z_B_slice = solution_z_B[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS, 1)
                feasible_slice = feasible[:total_candidates]  # shape: (total_candidates, NUM_STAGE2_ROWS)
                
                # Clear delta_r_candidates slice (reuse zeros tensor)
                delta_r_candidates_slice.zero_()
                
                # Populate repeated_delta_r slice using tensor views (zero allocation)
                batch_scenario_optimality = batch_scenario_slice.to(optimality_dtype)  # shape: (batch_size, short_delta_r_dim)
                # Reshape slice to (batch_size, effective_k, short_delta_r_dim) then copy batch data to each k
                repeated_view = repeated_delta_r_slice[:total_candidates].view(batch_size, effective_k, -1)  # shape: (batch_size, effective_k, short_delta_r_dim)
                repeated_view.copy_(batch_scenario_optimality.unsqueeze(1).expand(-1, effective_k, -1))  # broadcast across k dimension
                
                # Use index_add_ for sparse-to-dense conversion
                delta_r_candidates_slice.index_add_(1, self.r_sparse_indices_gpu, repeated_delta_r_slice)  # delta_r_candidates_slice shape: (total_candidates, NUM_STAGE2_ROWS)
                
                # Assemble RHS (reuse pre-allocated tensor)
                h_bar_optimality = h_bar_gpu.to(optimality_dtype)  # shape: (NUM_STAGE2_ROWS,)
                rhs_candidates_slice.copy_(h_bar_optimality.unsqueeze(0).expand(total_candidates, -1))  # broadcast to all candidates
                rhs_candidates_slice.add_(delta_r_candidates_slice).sub_(expanded_non_basic_contrib_slice)  # shape: (total_candidates, NUM_STAGE2_ROWS)

                # Solve all systems in one go (device-conditional approach)
                # monitor.measure(f"{batch_label}_before_lu_solve")
                torch.linalg.lu_solve(expanded_factors_slice, expanded_pivots_slice, 
                                    rhs_candidates_slice.unsqueeze(-1), out=solution_z_B_slice)
                # monitor.measure(f"{batch_label}_after_lu_solve")

                # Create 2D view for feasibility check (zero-cost view)
                solution_z_B_2D = solution_z_B_slice.squeeze(-1)  # shape: (total_candidates, NUM_STAGE2_ROWS)
                
                # Check feasibility (reuse pre-allocated tensor)
                feasible_slice.copy_((solution_z_B_2D >= expanded_lb_slice - primal_feas_tol) & (solution_z_B_2D <= expanded_ub_slice + primal_feas_tol))  # shape: (total_candidates, NUM_STAGE2_ROWS)
                is_feasible_flat = torch.all(feasible_slice, dim=1)  # shape: (total_candidates,)
                
                # --- Step 5: Select the First Valid Winner ---
                is_feasible_batch = is_feasible_flat.view(batch_size, effective_k)  # shape: (batch_size, effective_k)
                
                # For each row, find the index of the first 'True'. If all are 'False', argmax returns 0.
                first_feasible_idx = torch.argmax(is_feasible_batch.int(), dim=1)  # shape: (batch_size,)

                # Gather the final winning indices and scores
                best_k_index_batch = top_k_indices.gather(1, first_feasible_idx.unsqueeze(1)).squeeze(1)  # shape: (batch_size,)
                best_k_scores_batch = top_k_scores.gather(1, first_feasible_idx.unsqueeze(1)).squeeze(1)  # shape: (batch_size,)

                # Store the boolean feasibility status of the chosen winner
                final_optimality_status = is_feasible_batch.gather(1, first_feasible_idx.unsqueeze(1)).squeeze(1)
                self.is_verified_optimal_gpu[start_idx:end_idx] = final_optimality_status

                self.best_k_indices_gpu[start_idx:end_idx] = best_k_index_batch
                self.best_k_scores_gpu[start_idx:end_idx] = best_k_scores_batch

                # Clean up temporary tensors to prevent memory leaks
                del factors_gpu, pivots_gpu, lb_gpu, ub_gpu, non_basic_contrib_gpu
                del factors_batch_cpu, pivots_batch_cpu, lb_batch_cpu, ub_batch_cpu, non_basic_contrib_cpu  
                del unique_candidate_indices, inverse_map
                if self.device.type == 'cuda':
                    del inverse_map_gpu
                    torch.cuda.empty_cache()  # Force PyTorch to release cached GPU memory
                
                # monitor.measure(f"{batch_label}_after_cleanup")
            
            # # Final memory summary
            # monitor.measure("all_batches_complete")
            # print("\n=== BATCH MEMORY MONITORING SUMMARY ===")
            # monitor.print_summary()

        if touch_lru:
            best_k_indices_cpu = self.best_k_indices_gpu[:self.num_scenarios].cpu().numpy()
            self.update_lru_on_access(best_k_indices_cpu)

    def calculate_cut_coefficients(self,
                               override_indices: Optional[np.ndarray] = None,
                               override_pi: Optional[np.ndarray] = None,
                               override_rc: Optional[np.ndarray] = None) -> Optional[Tuple[float, np.ndarray]]:
        """
        Calculates Benders cut coefficients (alpha, beta) using pre-computed
        best_k_indices from `find_optimal_basis`. Allows for temporary, non-destructive
        overwriting of pi and rc values for a subset of scenarios.

        Args:
            override_indices: Optional. 1D NumPy array of scenario indices to override.
            override_pi: Optional. 2D NumPy array of new pi vectors for the override indices.
            override_rc: Optional. 2D NumPy array of new rc vectors for the override indices.

        Returns:
            A tuple (alpha, beta) or None if `find_optimal_basis` has not been run.
            - alpha (float): Constant term of the Benders cut (float64).
            - beta (np.ndarray): Coefficient vector of the cut (float64, size X_DIM).
        """
        if self.best_k_indices_gpu is None:
            print("Error: `find_optimal_basis` must be run before calculating coefficients.")
            return None

        # --- Input Validation for Overrides ---
        has_override = override_indices is not None
        if has_override:
            if not (override_pi is not None and override_rc is not None):
                raise ValueError("If overriding, all three must be provided: override_indices, override_pi, override_rc.")
            if not (override_indices.ndim == 1 and override_pi.ndim == 2 and override_rc.ndim == 2):
                raise ValueError("Override arrays have incorrect dimensions.")
            num_overrides = len(override_indices)
            if not (override_pi.shape == (num_overrides, self.NUM_STAGE2_ROWS) and override_rc.shape == (num_overrides, self.NUM_BOUNDED_VARS)):
                raise ValueError("Shape mismatch in override_pi or override_rc.")
            if np.any(override_indices < 0) or np.any(override_indices >= self.num_scenarios):
                raise ValueError("override_indices contains out-of-bounds values.")

        with torch.no_grad():
            active_pi_gpu = self.pi_gpu[:self.num_pi]
            active_rc_gpu = self.rc_gpu[:self.num_pi]
            active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]
            best_k_indices_slice = self.best_k_indices_gpu[:self.num_scenarios]

            # --- Select pi, rc, and short_pi based on best_k_indices ---
            selected_pi = active_pi_gpu[best_k_indices_slice]
            selected_rc = active_rc_gpu[best_k_indices_slice]
            selected_short_pi = active_short_pi_gpu[best_k_indices_slice]

            # --- Apply Overrides (Non-Destructive) ---
            if has_override:
                # Clone tensors to avoid modifying the original data
                selected_pi = selected_pi.clone()
                selected_rc = selected_rc.clone()
                selected_short_pi = selected_short_pi.clone()

                # Transfer override data to GPU
                override_indices_gpu = torch.from_numpy(override_indices).to(self.device, dtype=torch.long)
                override_pi_gpu = torch.from_numpy(override_pi).to(self.device, dtype=selected_pi.dtype)
                override_rc_gpu = torch.from_numpy(override_rc).to(self.device, dtype=selected_rc.dtype)

                # Scatter new values into the cloned tensors
                selected_pi[override_indices_gpu] = override_pi_gpu
                selected_rc[override_indices_gpu] = override_rc_gpu

                # Update the corresponding short_pi values
                if self.R_SPARSE_LEN > 0:
                    override_short_pi_gpu = override_pi_gpu[:, self.r_sparse_indices_gpu]
                    selected_short_pi[override_indices_gpu] = override_short_pi_gpu

            # --- Calculate lambda and mu from the (potentially overridden) rc ---
            selected_lambda = torch.clamp(selected_rc, min=0)
            selected_mu = torch.clamp(-selected_rc, min=0)

            # --- Calculate averages (using float64 for precision) ---
            Avg_Pi = torch.mean(selected_pi.to(torch.float64), dim=0)
            Avg_Lambda = torch.mean(selected_lambda.to(torch.float64), dim=0)
            Avg_Mu = torch.mean(selected_mu.to(torch.float64), dim=0)

            # --- Calculate E[s_i] = E[pi_k*^T * delta_r_i] ---
            active_scenario_data_gpu = self.short_delta_r_gpu[:self.num_scenarios, :]
            s_all_scenarios = (selected_short_pi * active_scenario_data_gpu).sum(dim=1)
            Avg_S = torch.mean(s_all_scenarios.to(torch.float64))

            # The full score is pi^T(r_bar - Cx) + pi^T(delta_r) - lambda^T*l + mu^T*u
            # Averaged version:
            # alpha = Avg_Pi' r_bar + Avg_S - Avg_Lambda' l + Avg_Mu' u
            # beta = -C' Avg_pi

            # --- Final Coefficient Calculation (float64) ---
            beta_gpu = -torch.matmul(self.C_gpu_fp64_transpose, Avg_Pi)

            r_bar_gpu_fp64 = self.r_bar_gpu.to(torch.float64)
            lb_gpu_fp64 = self.lb_gpu.to(torch.float64)
            ub_gpu_fp64 = self.ub_gpu.to(torch.float64)

            alpha_pi_term = torch.dot(Avg_Pi, r_bar_gpu_fp64) + Avg_S
            alpha_lambda_term = torch.dot(Avg_Lambda, lb_gpu_fp64)
            alpha_mu_term = torch.dot(Avg_Mu, ub_gpu_fp64)

            alpha_gpu = alpha_pi_term - alpha_lambda_term + alpha_mu_term

            alpha = alpha_gpu.item()
            beta = beta_gpu.cpu().numpy()

            return alpha, beta

    def get_best_k_results(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieves the best_k scores, indices, and optimality status from the device.

        Returns:
            A tuple (scores, indices, is_optimal) as NumPy arrays, or None if not computed.
            - scores: The objective score of the selected basis for each scenario.
            - indices: The index of the selected basis for each scenario.
            - is_optimal: A boolean array indicating if the selected basis was verified as feasible.
        """
        # The user is responsible for ensuring that `find_optimal_basis` has been called
        # and that the input `x` has not changed since the last call.
        
        # Truncate the results to the actual number of scenarios processed.
        scores_np = self.best_k_scores_gpu[:self.num_scenarios].cpu().numpy()
        indices_np = self.best_k_indices_gpu[:self.num_scenarios].cpu().numpy()
        is_optimal_np = self.is_verified_optimal_gpu[:self.num_scenarios].cpu().numpy()
        
        return scores_np, indices_np, is_optimal_np

    # --- Basis Matrix Related Methods for Optimality Checking ---
    def _get_basis_matrix(self, vbasis: np.ndarray, cbasis: np.ndarray) -> scipy.sparse.spmatrix:
        """
        Constructs the sparse basis matrix (B) for a given basis.
        Columns are ordered: basic variables from D, then basic slack/surplus variables.

        Args:
            vbasis: Variable basis status vector (0=basic, -1=non-basic at LB, -2=non-basic at UB).
            cbasis: Constraint basis status vector (0=basic, -1=non-basic).

        Returns:
            The basis matrix (B) as a SciPy CSC sparse matrix.
        """
        # 1. Identify indices of basic variables and basic slack/surplus variables
        # Gurobi basis status: 0 = basic.
        basic_var_indices = np.where(vbasis == 0)[0]
        basic_slack_indices = np.where(cbasis == 0)[0]

        # 2. Select corresponding columns from the D matrix
        # This is efficient because self.D is in CSC format
        d_cols = self.D[:, basic_var_indices]

        # 3. Construct sparse columns for slack/surplus variables
        num_slacks = len(basic_slack_indices)
        if num_slacks > 0:
            # Determine the sign based on constraint sense ('<' -> +1, '>' -> -1)
            # Note: Gurobi sense characters are '<', '>', '='
            sense_values = np.array([-1 if self.sense[j] == '>' else 1 for j in basic_slack_indices])
            
            # Create a COO matrix for the slack columns
            # Row indices are the slack indices, col indices are 0 to num_slacks-1
            slack_cols = scipy.sparse.coo_matrix(
                (sense_values, (basic_slack_indices, np.arange(num_slacks))),
                shape=(self.NUM_STAGE2_ROWS, num_slacks)
            ).tocsc()

            # 4. Assemble B by horizontally stacking the columns
            B = scipy.sparse.hstack([d_cols, slack_cols], format='csc')
        else:
            B = d_cols.tocsc()

        # Check for degeneracy: the basis matrix must be square.
        if B.shape[1] != self.NUM_STAGE2_ROWS:
            raise ValueError(
                f"Invalid basis: detected degeneracy. Basis matrix has shape {B.shape}, "
                f"but expected {self.NUM_STAGE2_ROWS} columns (basic variables)."
            )

        return B

    def _get_basic_variable_bounds(self, vbasis: np.ndarray, cbasis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the lower and upper bounds for all basic variables specified by the basis.
        The order matches the columns in the basis matrix from _get_basis_matrix.

        Args:
            vbasis: Variable basis status vector (0=basic, -1=non-basic at LB, -2=non-basic at UB).
            cbasis: Constraint basis status vector (0=basic, -1=non-basic).

        Returns:
            A tuple (lb_B, ub_B) of NumPy arrays for the lower and upper bounds.
        """
        # 1. Get bounds for basic y variables
        basic_var_indices = np.where(vbasis == 0)[0]
        lb_y_basic = self.lb_y_full[basic_var_indices]
        ub_y_basic = self.ub_y_full[basic_var_indices]

        # 2. Get bounds for basic slack/surplus variables (always [0, inf))
        num_basic_slacks = np.count_nonzero(cbasis == 0)
        lb_slacks = np.zeros(num_basic_slacks)
        ub_slacks = np.full(num_basic_slacks, np.inf)

        # 3. Concatenate the bounds, maintaining the correct order
        lb_B = np.concatenate([lb_y_basic, lb_slacks])
        ub_B = np.concatenate([ub_y_basic, ub_slacks])

        return lb_B, ub_B

    def _get_non_basic_contribution(self, vbasis: np.ndarray) -> np.ndarray:
        """
        Calculates the contribution of non-basic variables (N @ z_N).

        This is a key component for solving the primal system B @ z_B = h - N @ z_N,
        where z_B are the basic variables and z_N are the non-basic variables
        fixed at their bounds. This function pre-computes the `N @ z_N` term, which
        is constant for a given basis.

        Args:
            vbasis: Variable basis status vector (-1 for non-basic at LB, -2 for UB).

        Returns:
            A NumPy array representing the vector N @ z_N.
        """
        # Identify non-basic variables and their values
        non_basic_at_lb = np.where(vbasis == -1)[0]
        non_basic_at_ub = np.where(vbasis == -2)[0]

        # Initialize z_N with zeros. We only need to fill in the non-zero values.
        z_N = np.zeros(self.NUM_STAGE2_VARS)
        z_N[non_basic_at_lb] = self.lb_y_full[non_basic_at_lb]
        z_N[non_basic_at_ub] = self.ub_y_full[non_basic_at_ub]
        
        # The contribution from non-basic slack variables is zero (they are at their lower bound of 0),
        # so we only need to consider the columns of D corresponding to non-basic y-variables.
        non_basic_indices = np.concatenate([non_basic_at_lb, non_basic_at_ub])
        
        if non_basic_indices.size > 0:
            # Filter z_N to only include the values for non-basic variables
            z_N_filtered = z_N[non_basic_indices]
            # Select corresponding columns from D
            N_filtered = self.D[:, non_basic_indices]
            # Calculate N @ z_N contribution
            non_basic_contribution = N_filtered @ z_N_filtered
        else:
            non_basic_contribution = np.zeros(self.NUM_STAGE2_ROWS)
        
        return non_basic_contribution
    
    # --- Retrieval Methods ---
    def get_basis(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the basis (vbasis, cbasis) arrays from CPU for a given 
        1D NumPy array of indices. Each row in the output arrays corresponds 
        to an index in the input array.

        Args:
            indices: A 1D NumPy array of integer indices for which to retrieve the basis.

        Returns:
            A tuple containing two NumPy arrays:
            - vbasis_batch (np.ndarray): Array of shape (len(indices), NUM_STAGE2_VARS),
                                         containing the selected variable basis statuses.
            - cbasis_batch (np.ndarray): Array of shape (len(indices), NUM_STAGE2_ROWS),
                                         containing the selected constraint basis statuses.
        Raises:
            TypeError: If 'indices' is not a 1D NumPy array of integers.
            ValueError: If any index is out of bounds [0, self.num_pi - 1).
        """
        if not isinstance(indices, np.ndarray) or indices.ndim != 1:
            raise TypeError("Input 'indices' must be a 1D NumPy array.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Elements of 'indices' must be integers.")

        # Check if all requested indices are within the valid range
        # Valid range is [0, self.num_pi - 1]
        are_indices_valid = (indices >= 0) & (indices < self.num_pi)
        
        if not np.all(are_indices_valid):
            invalid_indices = indices[~are_indices_valid]
            raise ValueError(
                f"One or more indices are out of bounds [0, {self.num_pi - 1}]. "
                f"Invalid indices found: {invalid_indices.tolist()}"
            )

        vbasis_batch = self.vbasis_cpu[indices]
        cbasis_batch = self.cbasis_cpu[indices]
        
        return vbasis_batch, cbasis_batch
