import torch
import numpy as np
import scipy.sparse # For type hint
import hashlib
import time
import math
from typing import Tuple, Optional, Union, TYPE_CHECKING
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
    - Optional optimality checking: when disabled, skips basis factorization and
      feasibility verification for improved performance and reduced memory usage.
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
                 optimality_dtype: torch.dtype = torch.float32,
                 factorization_batch_size: int = 1000,
                 enable_optimality_check: bool = True
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
            factorization_batch_size: Number of basis matrices to process in each GPU batch
                                    for LU factorization. Defaults to 1000.
            enable_optimality_check: Whether to enable optimality checking and feasibility 
                                   verification. When False, skips basis factorization and
                                   related memory allocations. Defaults to True.
        """
        print(f"[{time.strftime('%H:%M:%S')}] Initializing ArgmaxOperation...")
        start_time = time.time()

        # --- Determine Device ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        
        print(f"[{time.strftime('%H:%M:%S')}] Using device: {self.device}")

        # --- Enable TF32 for float types on CUDA ---
        if (self.device.type == 'cuda' and 
            optimality_dtype in [torch.float32, torch.float16, torch.bfloat16]):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"[{time.strftime('%H:%M:%S')}] Enabled TF32 for {optimality_dtype}")

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
        self.factorization_batch_size = factorization_batch_size
        self.NUM_CANDIDATES = NUM_CANDIDATES
        self.enable_optimality_check = enable_optimality_check
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
        if self.enable_optimality_check:
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
        else:
            # Set to None when optimality checking is disabled
            self.basis_factors_cpu = None
            self.basis_pivots_cpu = None
            self.basis_lb_cpu = None
            self.basis_ub_cpu = None
            self.non_basic_contribution_cpu = None


        # Current candidate solution
        self.current_x_gpu = torch.zeros(X_DIM, dtype=torch.float32, device=self.device)

        # --- Pending Factorizations for Batch Processing ---
        if self.enable_optimality_check:
            self.pending_vbasis_list = []
            self.pending_cbasis_list = []
            self.pending_cpu_indices = []
            self.has_pending_factorizations = False
            self.D_gpu = None  # Will be initialized when first needed
        else:
            self.pending_vbasis_list = None
            self.pending_cbasis_list = None
            self.pending_cpu_indices = None
            self.has_pending_factorizations = False
            self.D_gpu = None

        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Initialization complete ({end_time - start_time:.2f}s).")

    @classmethod
    def from_smps_reader(cls, reader: 'SMPSReader',
                         MAX_PI: int, MAX_OMEGA: int,
                         scenario_batch_size: int = 10000,
                         device: Optional[Union[str, torch.device]] = None,
                         NUM_CANDIDATES: int = 8,
                         optimality_dtype: torch.dtype = torch.float32,
                         factorization_batch_size: int = 1000,
                         enable_optimality_check: bool = True) -> 'ArgmaxOperation':
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
            factorization_batch_size: Number of basis matrices to process in each GPU batch. Defaults to 1000.
            enable_optimality_check: Whether to enable optimality checking and feasibility 
                                   verification. When False, skips basis factorization and
                                   related memory allocations. Defaults to True.

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
            optimality_dtype=optimality_dtype,
            factorization_batch_size=factorization_batch_size,
            enable_optimality_check=enable_optimality_check
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

        # --- Queue for Batch Factorization ---
        # Skip factorization operations when optimality checking is disabled
        if self.enable_optimality_check:
            # Instead of immediate factorization, add to pending arrays for batch processing
            self.pending_vbasis_list.append(new_vbasis.copy())
            self.pending_cbasis_list.append(new_cbasis.copy())
            self.pending_cpu_indices.append(idx)
            self.has_pending_factorizations = True

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

    def clear_scenarios(self) -> None:
        """
        Clears all stored scenario data to allow fresh scenario loading.
        
        This method resets the scenario counter and zeros out the scenario tensor,
        effectively removing all previously stored scenarios. Useful for validation
        procedures that need to work with different scenario sets.
        
        Note: This does not affect the stored dual solutions (pi, rc, basis data).
        """
        self.num_scenarios = 0
        if hasattr(self, 'short_delta_r_gpu'):
            self.short_delta_r_gpu.zero_()

    def finalize_dual_additions(self):
        """
        Process all pending basis factorizations in GPU batches.
        
        This method must be called after adding dual solutions and before calling
        find_optimal_basis_with_subset() to ensure all basis matrices are factorized.
        When optimality checking is disabled, this method returns immediately.
        """
        if not self.enable_optimality_check:
            return  # Nothing to process when optimality checking is disabled
            
        if not self.has_pending_factorizations:
            return  # Nothing to process
        
        # print(f"[{time.strftime('%H:%M:%S')}] Processing {len(self.pending_vbasis_list)} pending factorizations...")
        start_time = time.time()
        
        # Initialize D matrix on GPU if needed
        if self.D_gpu is None:
            self._initialize_D_gpu()
        
        # Process in batches
        total_processed = 0
        batch_size = self.factorization_batch_size
        
        for batch_start in range(0, len(self.pending_vbasis_list), batch_size):
            batch_end = min(batch_start + batch_size, len(self.pending_vbasis_list))
            
            # Extract batch data
            batch_vbasis = self.pending_vbasis_list[batch_start:batch_end]
            batch_cbasis = self.pending_cbasis_list[batch_start:batch_end]
            batch_cpu_indices = self.pending_cpu_indices[batch_start:batch_end]
            
            # Process batch on GPU
            self._process_factorization_batch(batch_vbasis, batch_cbasis, batch_cpu_indices)
            
            total_processed += len(batch_vbasis)
            
            # Free GPU memory between batches
            torch.cuda.empty_cache()
        
        # Clear pending arrays
        self.pending_vbasis_list.clear()
        self.pending_cbasis_list.clear()
        self.pending_cpu_indices.clear()
        self.has_pending_factorizations = False
        
        end_time = time.time()
        # print(f"[{time.strftime('%H:%M:%S')}] Completed {total_processed} factorizations in {end_time - start_time:.2f}s")

    def _compute_scores_batch_core(self, batch_scenario_slice: torch.Tensor, 
                                   active_short_pi_gpu: torch.Tensor, 
                                   constant_score_part_all_k: torch.Tensor) -> torch.Tensor:
        """
        Core score computation logic extracted for reuse across different methods.
        
        Args:
            batch_scenario_slice: Scenario data for current batch, shape (batch_size, short_delta_r_dim)
            active_short_pi_gpu: Active short pi vectors, shape (num_pi, short_delta_r_dim) 
            constant_score_part_all_k: Precomputed constant terms, shape (num_pi,)
            
        Returns:
            scores_batch: Computed scores, shape (batch_size, num_pi)
        """
        scores_batch = torch.matmul(batch_scenario_slice, active_short_pi_gpu.T)  # shape: (batch_size, num_pi)
        scores_batch += constant_score_part_all_k  # In-place addition, shape: (batch_size, num_pi)
        return scores_batch

    def find_optimal_basis_fast(self, x: np.ndarray, touch_lru: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast argmax operation without top-k candidates or feasibility checking.
        Computes only the best-scoring dual solution for each scenario.
        
        This method is optimized for cut generation where we only need the argmax
        result and don't require basis information or feasibility verification.
        
        Args:
            x: The first-stage decision vector (NumPy array, size X_DIM).
            touch_lru: Whether to update the LRU cache for the winning solutions.
            
        Returns:
            A tuple containing:
            - pi_indices: Array of shape (num_scenarios,) containing the index of the 
                         best dual solution for each scenario.
            - best_scores: Array of shape (num_scenarios,) containing the scores of the
                          best dual solution for each scenario.
        """
        if self.num_pi == 0:
            raise RuntimeError("No pi vectors stored. Cannot find optimal basis.")
        if self.num_scenarios == 0:
            raise RuntimeError("No scenarios stored. Cannot find optimal basis.")
        if x.shape != (self.X_DIM,):
            raise ValueError("Input x has incorrect shape.")

        pi_indices = np.zeros(self.num_scenarios, dtype=np.int64)
        best_scores = np.zeros(self.num_scenarios, dtype=np.float32)

        with torch.no_grad():
            # --- Prepare device data views ---
            active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]  # shape: (num_pi, short_delta_r_dim)
            active_scenario_data_gpu = self.short_delta_r_gpu[:self.num_scenarios, :]  # shape: (num_scenarios, short_delta_r_dim)

            self.current_x_gpu.copy_(torch.from_numpy(x))  # shape: (X_DIM,)

            # --- Precompute terms constant across ALL scenarios ---
            Cx_gpu = torch.matmul(self.C_gpu, self.current_x_gpu)  # shape: (NUM_STAGE2_ROWS,)
            h_bar_gpu = self.r_bar_gpu - Cx_gpu  # shape: (NUM_STAGE2_ROWS,)
            
            active_pi_gpu = self.pi_gpu[:self.num_pi]  # shape: (num_pi, NUM_STAGE2_ROWS)
            active_rc_gpu = self.rc_gpu[:self.num_pi]  # shape: (num_pi, NUM_BOUNDED_VARS)
            
            pi_h_bar_term_all_k = torch.matmul(active_pi_gpu, h_bar_gpu)  # shape: (num_pi,)
            
            lambda_all_k = torch.clamp(active_rc_gpu, min=0)  # shape: (num_pi, NUM_BOUNDED_VARS)
            mu_all_k = torch.clamp(-active_rc_gpu, min=0)  # shape: (num_pi, NUM_BOUNDED_VARS)
            lambda_l_term_all_k = torch.matmul(lambda_all_k, self.lb_gpu_f32)  # shape: (num_pi,)
            mu_u_term_all_k = torch.matmul(mu_all_k, self.ub_gpu_f32)  # shape: (num_pi,)
            constant_score_part_all_k = pi_h_bar_term_all_k - lambda_l_term_all_k + mu_u_term_all_k  # shape: (num_pi,)

            # --- Process scenarios in batches (fast mode - only argmax) ---
            num_batches = math.ceil(self.num_scenarios / self.scenario_batch_size)
            for i in range(num_batches):
                start_idx = i * self.scenario_batch_size
                end_idx = min((i + 1) * self.scenario_batch_size, self.num_scenarios)
                
                # --- Calculate Scores ---
                batch_scenario_slice = active_scenario_data_gpu[start_idx:end_idx, :]  # shape: (batch_size, short_delta_r_dim)
                scores_batch = self._compute_scores_batch_core(batch_scenario_slice, active_short_pi_gpu, constant_score_part_all_k)
                
                # --- Find Best (Argmax Only) ---
                best_scores_batch, best_indices_batch = torch.max(scores_batch, dim=1)  # shapes: (batch_size,)
                
                # --- Store Results ---
                pi_indices[start_idx:end_idx] = best_indices_batch.cpu().numpy()
                best_scores[start_idx:end_idx] = best_scores_batch.cpu().numpy()

        if touch_lru:
            self.update_lru_on_access(pi_indices)

        return pi_indices, best_scores

    def find_optimal_basis_with_subset(self, x: np.ndarray, scenario_indices: np.ndarray, 
                                       touch_lru: bool = True, primal_feas_tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Finds the best dual solution for a specific subset of scenarios with top-k candidates
        and feasibility checking. This method is optimized for warmstarting LP solvers.
        
        When optimality checking is disabled, this method returns the argmax indices without
        feasibility verification and sets all is_optimal flags to False.
        
        Args:
            x: The first-stage decision vector (NumPy array, size X_DIM).
            scenario_indices: 1D array of scenario indices to process.
            touch_lru: Whether to update the LRU cache for the winning solutions.
            primal_feas_tol: The tolerance for checking feasibility bounds.
            
        Returns:
            A tuple containing:
            - best_k_scores: The scores of the best dual for each processed scenario
            - best_k_indices: The indices of the best dual for each processed scenario  
            - is_optimal: Boolean array indicating feasibility for each processed scenario
                         (always False when optimality checking is disabled)
        """
        if self.enable_optimality_check and self.has_pending_factorizations:
            raise RuntimeError(
                "Cannot perform optimal basis finding with pending factorizations. "
                "Please call finalize_dual_additions() first to process all pending basis factorizations."
            )
        
        if self.num_pi == 0:
            print("Warning: No pi vectors stored. Cannot find optimal basis.")
            return np.array([]), np.array([]), np.array([])
        if len(scenario_indices) == 0:
            print("Warning: No scenarios provided. Cannot find optimal basis.")
            return np.array([]), np.array([]), np.array([])
        if x.shape != (self.X_DIM,):
            raise ValueError("Input x has incorrect shape.")
        
        # Validate scenario indices
        if np.any(scenario_indices < 0) or np.any(scenario_indices >= self.num_scenarios):
            raise ValueError("scenario_indices contains out-of-bounds values.")

        # Fast path when optimality checking is disabled
        if not self.enable_optimality_check:
            return self._find_optimal_basis_fast_path(x, scenario_indices, touch_lru)

        # For now, process all scenarios (filtering can be added later when we have optimal scenario tracking)
        scenario_indices_to_process = scenario_indices.copy()
        num_selected_scenarios = len(scenario_indices_to_process)
        
        # Pre-allocate result arrays for selected scenarios
        selected_best_scores = np.zeros(num_selected_scenarios, dtype=np.float32)
        selected_best_indices = np.zeros(num_selected_scenarios, dtype=np.int64)
        selected_is_optimal = np.zeros(num_selected_scenarios, dtype=bool)

        with torch.no_grad():
            # --- Prepare device data views ---
            active_pi_gpu = self.pi_gpu[:self.num_pi]  # shape: (num_pi, NUM_STAGE2_ROWS)
            active_rc_gpu = self.rc_gpu[:self.num_pi]  # shape: (num_pi, NUM_BOUNDED_VARS)
            active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]  # shape: (num_pi, short_delta_r_dim)

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

            # --- Pre-allocate tensors for feasibility checking (reuse from main method) ---
            max_candidates_per_batch = self.scenario_batch_size * min(self.NUM_CANDIDATES, self.num_pi)
            expanded_factors = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS, self.NUM_STAGE2_ROWS),
                                         dtype=self.basis_factors_cpu.dtype, device=self.device)
            expanded_pivots = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS),
                                        dtype=self.basis_pivots_cpu.dtype, device=self.device)
            expanded_lb = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS),
                                    dtype=self.basis_lb_cpu.dtype, device=self.device)
            expanded_ub = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS),
                                    dtype=self.basis_ub_cpu.dtype, device=self.device)
            expanded_non_basic_contrib = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS),
                                                   dtype=self.non_basic_contribution_cpu.dtype, device=self.device)
            
            # Pre-allocate intermediate matrices
            optimality_dtype = self.basis_factors_cpu.dtype
            delta_r_candidates = torch.zeros((max_candidates_per_batch, self.NUM_STAGE2_ROWS), dtype=optimality_dtype, device=self.device)
            repeated_delta_r = torch.empty((max_candidates_per_batch, self.short_delta_r_gpu.shape[1]), dtype=optimality_dtype, device=self.device)
            rhs_candidates = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS), dtype=optimality_dtype, device=self.device)
            solution_z_B = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS, 1), dtype=optimality_dtype, device=self.device)
            feasible = torch.empty((max_candidates_per_batch, self.NUM_STAGE2_ROWS), dtype=torch.bool, device=self.device)

            # --- Process selected scenarios in batches ---
            num_batches = math.ceil(num_selected_scenarios / self.scenario_batch_size)
            
            for batch_idx in range(num_batches):
                # Determine batch boundaries for selected scenarios
                batch_start = batch_idx * self.scenario_batch_size
                batch_end = min((batch_idx + 1) * self.scenario_batch_size, num_selected_scenarios)
                batch_size = batch_end - batch_start
                
                # Get scenario indices for this batch
                current_batch_scenario_indices = scenario_indices_to_process[batch_start:batch_end]
                
                # Extract scenario data for current batch
                batch_scenario_slice = self.short_delta_r_gpu[current_batch_scenario_indices, :]  # shape: (batch_size, short_delta_r_dim)
                
                # --- Calculate Scores ---
                scores_batch = self._compute_scores_batch_core(batch_scenario_slice, active_short_pi_gpu, constant_score_part_all_k)
                
                # --- Find Top Candidates ---
                effective_k = min(self.NUM_CANDIDATES, self.num_pi)
                top_k_scores, top_k_indices = torch.topk(scores_batch, k=effective_k, dim=1)  # shapes: (batch_size, effective_k)

                # --- Batched Feasibility Check (same as original method) ---
                # Flatten the workload: (batch_size, k) -> (batch_size * k)
                candidate_indices_flat = top_k_indices.flatten()  # shape: (batch_size * effective_k,)

                # Gather basis data from CPU cache
                if self.device.type == 'cuda':
                    unique_candidate_indices, inverse_map = torch.unique(candidate_indices_flat, return_inverse=True)
                    unique_candidate_indices_cpu = unique_candidate_indices.cpu()
                else:
                    unique_candidate_indices, inverse_map = torch.unique(candidate_indices_flat, return_inverse=True)
                    unique_candidate_indices_cpu = unique_candidate_indices
                
                factors_batch_cpu = self.basis_factors_cpu[unique_candidate_indices_cpu]
                pivots_batch_cpu = self.basis_pivots_cpu[unique_candidate_indices_cpu]
                lb_batch_cpu = self.basis_lb_cpu[unique_candidate_indices_cpu]
                ub_batch_cpu = self.basis_ub_cpu[unique_candidate_indices_cpu]
                non_basic_contrib_cpu = self.non_basic_contribution_cpu[unique_candidate_indices_cpu]
                
                # Move data to GPU and expand to match the flattened workload
                if self.device.type == 'cuda':
                    factors_gpu = factors_batch_cpu.to(self.device)
                    pivots_gpu = pivots_batch_cpu.to(self.device)
                    lb_gpu = lb_batch_cpu.to(self.device)
                    ub_gpu = ub_batch_cpu.to(self.device)
                    non_basic_contrib_gpu = non_basic_contrib_cpu.to(self.device)
                    inverse_map_gpu = inverse_map
                else:
                    factors_gpu = factors_batch_cpu
                    pivots_gpu = pivots_batch_cpu
                    lb_gpu = lb_batch_cpu
                    ub_gpu = ub_batch_cpu
                    non_basic_contrib_gpu = non_basic_contrib_cpu
                    inverse_map_gpu = inverse_map

                # Use slices of pre-allocated tensors
                total_candidates = batch_size * effective_k
                expanded_factors_slice = expanded_factors[:total_candidates]
                expanded_pivots_slice = expanded_pivots[:total_candidates]
                expanded_lb_slice = expanded_lb[:total_candidates]
                expanded_ub_slice = expanded_ub[:total_candidates]
                expanded_non_basic_contrib_slice = expanded_non_basic_contrib[:total_candidates]
                
                torch.index_select(factors_gpu, 0, inverse_map_gpu, out=expanded_factors_slice)
                torch.index_select(pivots_gpu, 0, inverse_map_gpu, out=expanded_pivots_slice)
                torch.index_select(lb_gpu, 0, inverse_map_gpu, out=expanded_lb_slice)
                torch.index_select(ub_gpu, 0, inverse_map_gpu, out=expanded_ub_slice)
                torch.index_select(non_basic_contrib_gpu, 0, inverse_map_gpu, out=expanded_non_basic_contrib_slice)

                # Use slices of pre-allocated intermediate matrices
                delta_r_candidates_slice = delta_r_candidates[:total_candidates]
                repeated_delta_r_slice = repeated_delta_r[:total_candidates]
                rhs_candidates_slice = rhs_candidates[:total_candidates]
                solution_z_B_slice = solution_z_B[:total_candidates]
                feasible_slice = feasible[:total_candidates]
                
                # Clear and populate delta_r
                delta_r_candidates_slice.zero_()
                
                batch_scenario_optimality = batch_scenario_slice.to(optimality_dtype)
                repeated_view = repeated_delta_r_slice[:total_candidates].view(batch_size, effective_k, -1)
                repeated_view.copy_(batch_scenario_optimality.unsqueeze(1).expand(-1, effective_k, -1))
                
                delta_r_candidates_slice.index_add_(1, self.r_sparse_indices_gpu, repeated_delta_r_slice)
                
                # Assemble RHS
                h_bar_optimality = h_bar_gpu.to(optimality_dtype)
                rhs_candidates_slice.copy_(h_bar_optimality.unsqueeze(0).expand(total_candidates, -1))
                rhs_candidates_slice.add_(delta_r_candidates_slice).sub_(expanded_non_basic_contrib_slice)

                # Solve systems
                torch.linalg.lu_solve(expanded_factors_slice, expanded_pivots_slice, 
                                    rhs_candidates_slice.unsqueeze(-1), out=solution_z_B_slice)

                # Check feasibility
                solution_z_B_2D = solution_z_B_slice.squeeze(-1)
                feasible_slice.copy_((solution_z_B_2D >= expanded_lb_slice - primal_feas_tol) & 
                                   (solution_z_B_2D <= expanded_ub_slice + primal_feas_tol))
                is_feasible_flat = torch.all(feasible_slice, dim=1)
                
                # --- Select the First Valid Winner ---
                is_feasible_batch = is_feasible_flat.view(batch_size, effective_k)
                
                # For each row, find the index of the first 'True'. If all are 'False', argmax returns 0.
                first_feasible_idx = torch.argmax(is_feasible_batch.int(), dim=1)
                
                # Gather the final winning indices and scores
                batch_best_indices = top_k_indices.gather(1, first_feasible_idx.unsqueeze(1)).squeeze(1)
                batch_best_scores = top_k_scores.gather(1, first_feasible_idx.unsqueeze(1)).squeeze(1)
                
                # Store the boolean feasibility status of the chosen winner
                final_optimality_status = is_feasible_batch.gather(1, first_feasible_idx.unsqueeze(1)).squeeze(1)
                
                # Store results in the pre-allocated arrays
                selected_best_scores[batch_start:batch_end] = batch_best_scores.cpu().numpy()
                selected_best_indices[batch_start:batch_end] = batch_best_indices.cpu().numpy() 
                selected_is_optimal[batch_start:batch_end] = final_optimality_status.cpu().numpy()

                # Clean up temporary tensors
                del factors_gpu, pivots_gpu, lb_gpu, ub_gpu, non_basic_contrib_gpu
                del factors_batch_cpu, pivots_batch_cpu, lb_batch_cpu, ub_batch_cpu, non_basic_contrib_cpu  
                del unique_candidate_indices, inverse_map, inverse_map_gpu
                if self.device.type == 'cuda':
                    del unique_candidate_indices_cpu

        if touch_lru:
            self.update_lru_on_access(selected_best_indices)

        return selected_best_scores, selected_best_indices, selected_is_optimal

    def _find_optimal_basis_fast_path(self, x: np.ndarray, scenario_indices: np.ndarray, 
                                      touch_lru: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast path for find_optimal_basis_with_subset when optimality checking is disabled.
        
        Returns argmax indices without feasibility checking, and sets all is_optimal to False.
        """
        num_selected_scenarios = len(scenario_indices)
        
        # Pre-allocate result arrays
        selected_best_scores = np.zeros(num_selected_scenarios, dtype=np.float32)
        selected_best_indices = np.zeros(num_selected_scenarios, dtype=np.int64)
        selected_is_optimal = np.zeros(num_selected_scenarios, dtype=bool)  # Always False
        
        with torch.no_grad():
            # --- Prepare device data views ---
            active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]  # shape: (num_pi, short_delta_r_dim)
            
            self.current_x_gpu.copy_(torch.from_numpy(x))  # shape: (X_DIM,)
            
            # --- Precompute terms constant across ALL scenarios ---
            Cx_gpu = torch.matmul(self.C_gpu, self.current_x_gpu)  # shape: (NUM_STAGE2_ROWS,)
            h_bar_gpu = self.r_bar_gpu - Cx_gpu  # shape: (NUM_STAGE2_ROWS,)
            
            active_pi_gpu = self.pi_gpu[:self.num_pi]  # shape: (num_pi, NUM_STAGE2_ROWS)
            active_rc_gpu = self.rc_gpu[:self.num_pi]  # shape: (num_pi, NUM_BOUNDED_VARS)
            
            pi_h_bar_term_all_k = torch.matmul(active_pi_gpu, h_bar_gpu)  # shape: (num_pi,)
            
            lambda_all_k = torch.clamp(active_rc_gpu, min=0)  # shape: (num_pi, NUM_BOUNDED_VARS)
            mu_all_k = torch.clamp(-active_rc_gpu, min=0)  # shape: (num_pi, NUM_BOUNDED_VARS)
            lambda_l_term_all_k = torch.matmul(lambda_all_k, self.lb_gpu_f32)  # shape: (num_pi,)
            mu_u_term_all_k = torch.matmul(mu_all_k, self.ub_gpu_f32)  # shape: (num_pi,)
            constant_score_part_all_k = pi_h_bar_term_all_k - lambda_l_term_all_k + mu_u_term_all_k  # shape: (num_pi,)
            
            # --- Process selected scenarios in batches ---
            num_batches = math.ceil(num_selected_scenarios / self.scenario_batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.scenario_batch_size
                batch_end = min((batch_idx + 1) * self.scenario_batch_size, num_selected_scenarios)
                
                # Get scenario indices for this batch  
                current_batch_scenario_indices = scenario_indices[batch_start:batch_end]
                
                # Extract scenario data for current batch
                batch_scenario_slice = self.short_delta_r_gpu[current_batch_scenario_indices, :]  # shape: (batch_size, short_delta_r_dim)
                
                # --- Calculate Scores and Find Argmax ---
                scores_batch = self._compute_scores_batch_core(batch_scenario_slice, active_short_pi_gpu, constant_score_part_all_k)
                
                # Find best (argmax only)
                best_scores_batch, best_indices_batch = torch.max(scores_batch, dim=1)  # shapes: (batch_size,)
                
                # Store results
                selected_best_scores[batch_start:batch_end] = best_scores_batch.cpu().numpy()
                selected_best_indices[batch_start:batch_end] = best_indices_batch.cpu().numpy()
                # selected_is_optimal remains all False
        
        if touch_lru:
            self.update_lru_on_access(selected_best_indices)
        
        return selected_best_scores, selected_best_indices, selected_is_optimal

    def calculate_cut_coefficients(self, pi_indices: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculates Benders cut coefficients (alpha, beta) using provided pi indices array.
        
        Args:
            pi_indices: Array of shape (num_scenarios,) containing the index of the 
                       dual solution to use for each scenario.
                       
        Returns:
            A tuple (alpha, beta):
            - alpha (float): Constant term of the Benders cut (float64).
            - beta (np.ndarray): Coefficient vector of the cut (float64, size X_DIM).
        """
        if pi_indices.shape != (self.num_scenarios,):
            raise ValueError(f"pi_indices must have shape ({self.num_scenarios},), got {pi_indices.shape}")
        if np.any(pi_indices < 0) or np.any(pi_indices >= self.num_pi):
            raise ValueError("pi_indices contains out-of-bounds values.")

        with torch.no_grad():
            active_pi_gpu = self.pi_gpu[:self.num_pi]
            active_rc_gpu = self.rc_gpu[:self.num_pi]
            active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]
            pi_indices_gpu = torch.from_numpy(pi_indices).to(self.device, dtype=torch.long)

            # --- Select pi, rc, and short_pi based on pi_indices ---
            selected_pi = active_pi_gpu[pi_indices_gpu]
            selected_rc = active_rc_gpu[pi_indices_gpu]
            selected_short_pi = active_short_pi_gpu[pi_indices_gpu]

            # --- Calculate lambda and mu from rc ---
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

    def _initialize_D_gpu(self):
        """Initialize D matrix on GPU for efficient column slicing."""
        print(f"[{time.strftime('%H:%M:%S')}] Initializing D matrix on GPU...")
        # Convert D to dense tensor on GPU for efficient column indexing
        D_dense = torch.from_numpy(self.D.toarray()).to(
            dtype=self.basis_factors_cpu.dtype,
            device=self.device
        )
        self.D_gpu = D_dense
        print(f"[{time.strftime('%H:%M:%S')}] D matrix cached on GPU: shape {self.D_gpu.shape}")

    def _process_factorization_batch(self, batch_vbasis, batch_cbasis, batch_cpu_indices):
        """Process a batch of basis matrices for LU factorization on GPU."""
        batch_size = len(batch_vbasis)
        gpu_matrices = []
        
        # Construct basis matrices on GPU
        for i in range(batch_size):
            B_gpu = self._get_basis_matrix_gpu(batch_vbasis[i], batch_cbasis[i])
            gpu_matrices.append(B_gpu)
        
        # Stack matrices for batch processing
        if batch_size == 1:
            batch_tensor = gpu_matrices[0].unsqueeze(0)
        else:
            batch_tensor = torch.stack(gpu_matrices)
        
        # Batch LU factorization
        factors, pivots = torch.linalg.lu_factor(batch_tensor)
        
        # Write back to CPU storage using vectorized indexing
        cpu_indices_tensor = torch.tensor(batch_cpu_indices)
        self.basis_factors_cpu[cpu_indices_tensor] = factors.cpu()
        self.basis_pivots_cpu[cpu_indices_tensor] = pivots.cpu()
    
    def _get_basis_matrix_gpu(self, vbasis: np.ndarray, cbasis: np.ndarray) -> torch.Tensor:
        """Construct basis matrix directly on GPU."""
        basic_var_indices = np.where(vbasis == 0)[0]
        basic_slack_indices = np.where(cbasis == 0)[0]
        
        # Select columns from D matrix on GPU
        basic_var_indices_gpu = torch.tensor(basic_var_indices, device=self.device, dtype=torch.long)
        d_cols_gpu = self.D_gpu[:, basic_var_indices_gpu]
        
        if len(basic_slack_indices) > 0:
            # Create slack columns on GPU
            sense_values = torch.tensor([
                -1.0 if self.sense[j] == '>' else 1.0 
                for j in basic_slack_indices
            ], device=self.device, dtype=self.D_gpu.dtype)
            
            slack_cols = torch.zeros((self.NUM_STAGE2_ROWS, len(basic_slack_indices)), 
                                   device=self.device, dtype=self.D_gpu.dtype)
            slack_cols[basic_slack_indices, torch.arange(len(basic_slack_indices))] = sense_values
            
            # Concatenate D columns with slack columns
            B_gpu = torch.cat([d_cols_gpu, slack_cols], dim=1)
        else:
            B_gpu = d_cols_gpu
        
        return B_gpu
    
