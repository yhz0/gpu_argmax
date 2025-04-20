import cupy as cp
import numpy as np
import scipy.sparse # For input type hint
import cupyx.scipy.sparse as cpsparse # For GPU sparse matrices
import hashlib
import time # For basic timing feedback (used in init)
import math
from typing import List, Tuple, Dict, Optional, Any
import threading


class ArgmaxOperation:
    """
    Implements GPU-accelerated calculation related to Benders cuts using CuPy.

    Stores and processes constraint duals (pi) and reduced costs (RC) for bounded
    variables on the GPU. Stores corresponding Gurobi basis information (vbasis, cbasis)
    associated with pre-computed dual solutions on the CPU only.

    It calculates the selection score based on the full second-stage dual
    objective and computes the complete Benders alpha and beta coefficients using GPU data.

    Features:
    - Stores (pi, RC) pairs on GPU, (vbasis, cbasis) on CPU.
    - Processes scenarios in batches for memory efficiency.
    - Uses double precision (float64) for reduction steps (summation) for accuracy.
    - Handles sparse C matrix.
    - Assumes fixed sparsity pattern for stochastic RHS (delta_r).
    - Deduplicates added (pi, RC) pairs (basis excluded from hash).

    Attributes:
        NUM_STAGE2_ROWS (int): Dimension of the constraint dual space (pi).
        NUM_BOUNDED_VARS (int): Number of stage 2 variables with non-trivial bounds (dimension of RC).
        NUM_STAGE2_VARS (int): Total number of stage 2 variables (dimension of vbasis).
        X_DIM (int): Dimension of the first-stage variable x.
        MAX_PI (int): Maximum number of (pi, RC, basis) tuples to store.
        MAX_OMEGA (int): Maximum number of scenarios to store.
        R_SPARSE_LEN (int): Number of stochastic stage 2 constraints.
        scenario_batch_size (int): Number of scenarios to process in each GPU batch.
        num_pi (int): Current number of stored (pi, RC, basis) tuples.
        num_scenarios (int): Current number of stored scenarios.
        # CPU Storage
        pi_cpu (np.ndarray): Stored pi vectors (constraint duals) on CPU.
        rc_cpu (np.ndarray): Stored RC vectors (for bounded vars) on CPU.
        vbasis_cpu (np.ndarray): Stored vbasis arrays on CPU (dtype=int8).
        cbasis_cpu (np.ndarray): Stored cbasis arrays on CPU (dtype=int8).
        short_pi_cpu (np.ndarray): Stored pi components corresponding to stochastic rows on CPU.
        r_sparse_indices_cpu (np.ndarray): Indices (relative to stage 2 rows) of stochastic rows on CPU.
        pi_rc_hashes (set): Hashes of stored (pi, RC) pairs for deduplication.
        # GPU Storage
        pi_gpu (cp.ndarray): Stored pi vectors (constraint duals) on GPU.
        rc_gpu (cp.ndarray): Stored RC vectors (for bounded vars) on GPU.
        short_pi_gpu (cp.ndarray): Stored pi components corresponding to stochastic rows on GPU.
        lb_gpu (cp.ndarray): Lower bounds for bounded variables on GPU.
        ub_gpu (cp.ndarray): Upper bounds for bounded variables on GPU.
        r_bar_gpu (cp.ndarray): Fixed part of RHS (for stage 2 rows) on GPU.
        r_sparse_indices_gpu (cp.ndarray): Indices (relative to stage 2 rows) of stochastic rows on GPU.
        short_delta_r_gpu (cp.ndarray): Packed stochastic RHS deviations for scenarios on GPU.
        C_gpu (cpsparse.csr_matrix): Sparse transfer matrix C (stage 2 rows x stage 1 vars) on GPU.
        _lock (threading.Lock): Lock to ensure thread-safe access to add_pi.
    """

    def __init__(self, NUM_STAGE2_ROWS: int, NUM_BOUNDED_VARS: int, NUM_STAGE2_VARS: int,
                 X_DIM: int, MAX_PI: int, MAX_OMEGA: int,
                 r_sparse_indices: np.ndarray, # Indices relative to stage 2 rows (0 to NUM_STAGE2_ROWS-1)
                 r_bar: np.ndarray, # Size NUM_STAGE2_ROWS
                 C: scipy.sparse.spmatrix, # Shape (NUM_STAGE2_ROWS, X_DIM)
                 lb_y_bounded: np.ndarray, # Size NUM_BOUNDED_VARS
                 ub_y_bounded: np.ndarray, # Size NUM_BOUNDED_VARS
                 scenario_batch_size: int = 100000):
        """
        Initializes the ArgmaxOperation class. Allocates memory on CPU and GPU.

        Args:
            NUM_STAGE2_ROWS (int): Dimension of the constraint dual space (pi).
            NUM_BOUNDED_VARS (int): Number of stage 2 variables with non-trivial bounds (dimension of RC).
            NUM_STAGE2_VARS (int): Total number of stage 2 variables (y), dimension of vbasis.
            X_DIM (int): Dimension of the first-stage variable x.
            MAX_PI (int): Maximum number of (pi, RC, basis) tuples to store.
            MAX_OMEGA (int): Maximum number of scenarios to store.
            r_sparse_indices (np.ndarray): 1D array of indices (0 to NUM_STAGE2_ROWS-1) where the stage 2 RHS is stochastic.
            r_bar (np.ndarray): 1D array for the fixed part of the stage 2 RHS (size NUM_STAGE2_ROWS).
            C (scipy.sparse.spmatrix): The transfer matrix C (size NUM_STAGE2_ROWS x X_DIM).
            lb_y_bounded (np.ndarray): Lower bounds for variables corresponding to RC components (size NUM_BOUNDED_VARS).
            ub_y_bounded (np.ndarray): Upper bounds for variables corresponding to RC components (size NUM_BOUNDED_VARS).
            scenario_batch_size (int): Number of scenarios to process in each GPU batch. Defaults to 100,000.
        """
        print(f"[{time.strftime('%H:%M:%S')}] Initializing ArgmaxOperation (Pi+RC GPU, Basis CPU, Sparse C, Batched)...")
        start_time = time.time()

        # --- Input Validation ---
        if r_bar.shape != (NUM_STAGE2_ROWS,): raise ValueError(f"r_bar shape mismatch.")
        if C.shape != (NUM_STAGE2_ROWS, X_DIM): raise ValueError(f"C matrix shape mismatch.")
        if lb_y_bounded.shape != (NUM_BOUNDED_VARS,): raise ValueError(f"lb_y_bounded shape mismatch.")
        if ub_y_bounded.shape != (NUM_BOUNDED_VARS,): raise ValueError(f"ub_y_bounded shape mismatch.")
        if not isinstance(r_sparse_indices, np.ndarray) or r_sparse_indices.ndim != 1: raise ValueError("r_sparse_indices must be 1D.")
        if len(np.unique(r_sparse_indices)) != len(r_sparse_indices):
             print("Warning: r_sparse_indices contains duplicate values. Using unique indices.")
             r_sparse_indices = np.unique(r_sparse_indices)
        if np.any(r_sparse_indices < 0) or np.any(r_sparse_indices >= NUM_STAGE2_ROWS): raise ValueError("r_sparse_indices out of bounds.")
        if not scipy.sparse.issparse(C): raise TypeError("C must be a SciPy sparse matrix.")
        if not isinstance(scenario_batch_size, int) or scenario_batch_size <= 0: raise ValueError("scenario_batch_size must be positive.")


        # --- Dimensions and Capacities ---
        self.NUM_STAGE2_ROWS = NUM_STAGE2_ROWS
        self.NUM_BOUNDED_VARS = NUM_BOUNDED_VARS
        self.NUM_STAGE2_VARS = NUM_STAGE2_VARS
        self.X_DIM = X_DIM
        self.MAX_PI = MAX_PI
        self.MAX_OMEGA = MAX_OMEGA
        self.scenario_batch_size = scenario_batch_size
        self.r_sparse_indices_cpu = np.sort(np.array(r_sparse_indices, dtype=np.int32))
        self.R_SPARSE_LEN = len(self.r_sparse_indices_cpu)

        # --- Counters ---
        self.num_pi = 0 # Number of (pi, RC, basis) tuples stored
        self.num_scenarios = 0

        # --- CPU Data Storage ---
        print(f"[{time.strftime('%H:%M:%S')}] Allocating CPU memory...")
        dtype = np.float32 # Default dtype for pi, rc
        basis_dtype = np.int8 # Compact storage for basis status
        self.pi_cpu = np.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=dtype)
        self.rc_cpu = np.zeros((MAX_PI, NUM_BOUNDED_VARS), dtype=dtype)
        self.vbasis_cpu = np.zeros((MAX_PI, NUM_STAGE2_VARS), dtype=basis_dtype) # Basis on CPU only
        self.cbasis_cpu = np.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=basis_dtype) # Basis on CPU only
        self.short_pi_cpu = np.zeros((MAX_PI, self.R_SPARSE_LEN), dtype=dtype)
        self.pi_rc_hashes = set() # For deduplication of (pi, rc) pairs

        # --- GPU Data Storage ---
        print(f"[{time.strftime('%H:%M:%S')}] Allocating GPU memory on device {cp.cuda.runtime.getDevice()}...")
        cp_dtype = cp.float32 # Default dtype for most GPU arrays
        # Pi, RC vectors stored on GPU
        self.pi_gpu = cp.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=cp_dtype)
        self.rc_gpu = cp.zeros((MAX_PI, NUM_BOUNDED_VARS), dtype=cp_dtype)
        # short_pi stores only the pi components corresponding to stochastic rows
        self.short_pi_gpu = cp.zeros((MAX_PI, self.R_SPARSE_LEN), dtype=cp_dtype)

        # Bounds for RC variables
        self.lb_gpu = cp.asarray(lb_y_bounded, dtype=cp_dtype)
        self.ub_gpu = cp.asarray(ub_y_bounded, dtype=cp_dtype)

        # RHS data
        self.r_bar_gpu = cp.asarray(r_bar, dtype=cp_dtype)
        self.r_sparse_indices_gpu = cp.asarray(self.r_sparse_indices_cpu)

        # Scenario data (short/packed r_delta)
        self.short_delta_r_gpu = cp.zeros((self.R_SPARSE_LEN, MAX_OMEGA), dtype=cp_dtype)

        # C matrix (Sparse)
        print(f"[{time.strftime('%H:%M:%S')}] Converting C to sparse CSR format on GPU...")
        if C.dtype != dtype: C = C.astype(dtype)
        self.C_gpu = cpsparse.csr_matrix(C, dtype=cp_dtype)

        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Initialization complete ({end_time - start_time:.2f}s).")


    def add_pi(self, new_pi: np.ndarray, new_rc: np.ndarray,
               new_vbasis: np.ndarray, new_cbasis: np.ndarray) -> bool:
        """
        Adds a new constraint dual vector (pi), its corresponding reduced cost
        vector (rc), and the associated Gurobi basis vectors (vbasis, cbasis).
        Basis is stored only on CPU. Deduplication is based only on (pi, rc).
        Calculates 'short_pi', and updates CPU/GPU storage for pi and rc.

        Args:
            new_pi (np.ndarray): Constraint dual vector (size NUM_STAGE2_ROWS).
            new_rc (np.ndarray): Reduced cost vector (size NUM_BOUNDED_VARS).
            new_vbasis (np.ndarray): Variable basis status vector (size NUM_STAGE2_VARS).
            new_cbasis (np.ndarray): Constraint basis status vector (size NUM_STAGE2_ROWS).

        Returns:
            bool: True if the tuple was added successfully, False otherwise.
        """
        if self.num_pi >= self.MAX_PI:
            return False

        # --- Dimension Checks ---
        if new_pi.shape != (self.NUM_STAGE2_ROWS,): raise ValueError(f"new_pi shape mismatch.")
        if new_rc.shape != (self.NUM_BOUNDED_VARS,): raise ValueError(f"new_rc shape mismatch.")
        if new_vbasis.shape != (self.NUM_STAGE2_VARS,): raise ValueError(f"new_vbasis shape mismatch.")
        if new_cbasis.shape != (self.NUM_STAGE2_ROWS,): raise ValueError(f"new_cbasis shape mismatch.")

        # Ensure consistent dtypes before hashing/storing
        new_pi = new_pi.astype(self.pi_cpu.dtype, copy=False)
        new_rc = new_rc.astype(self.rc_cpu.dtype, copy=False)
        new_vbasis = new_vbasis.astype(self.vbasis_cpu.dtype, copy=False) # int8
        new_cbasis = new_cbasis.astype(self.cbasis_cpu.dtype, copy=False) # int8


        # --- Acquire Lock for Critical Section ---
        with self._lock:
            # --- Critical Section Start ---
            # --- Deduplication based only on the (pi, rc) pair ---
            try:
                # Concatenate only pi and rc for hashing
                combined = np.concatenate((np.ascontiguousarray(new_pi), np.ascontiguousarray(new_rc)))
                pi_rc_hash = hashlib.sha256(combined.tobytes()).hexdigest()
                if pi_rc_hash in self.pi_rc_hashes: return False
                self.pi_rc_hashes.add(pi_rc_hash)
            except Exception as e:
                print(f"Warning: Could not hash (pi, rc) pair for deduplication. Adding anyway. Error: {e}")

            idx = self.num_pi
            # Store pi and rc on CPU
            self.pi_cpu[idx] = new_pi
            self.rc_cpu[idx] = new_rc
            # Store basis only on CPU
            self.vbasis_cpu[idx] = new_vbasis
            self.cbasis_cpu[idx] = new_cbasis

            # Extract the 'short' version from pi using the relative sparse indices
            if len(self.r_sparse_indices_cpu) > 0:
                short_new_pi = new_pi[self.r_sparse_indices_cpu]
            else:
                short_new_pi = np.array([], dtype=new_pi.dtype)
            self.short_pi_cpu[idx] = short_new_pi

            # Add pi and rc to GPU
            self.pi_gpu[idx] = cp.asarray(new_pi)
            self.rc_gpu[idx] = cp.asarray(new_rc)
            self.short_pi_gpu[idx] = cp.asarray(short_new_pi)

            self.num_pi += 1
            # --- Critical Section End ---
            return True

    def add_scenarios(self, new_short_r_delta: np.ndarray) -> bool:
        """
        Adds new scenario data (packed delta_r values corresponding to r_sparse_indices).

        Args:
            new_short_r_delta (np.ndarray): Array of shape (R_SPARSE_LEN, num_new_scenarios).

        Returns:
            bool: True if scenarios were added successfully (partially or fully), False otherwise.
        """
        if new_short_r_delta.ndim != 2 or new_short_r_delta.shape[0] != self.R_SPARSE_LEN:
             raise ValueError(f"new_short_r_delta shape mismatch.")
        num_new_scenarios = new_short_r_delta.shape[1]
        available_slots = self.MAX_OMEGA - self.num_scenarios
        if num_new_scenarios <= 0: return False
        if available_slots <= 0: return False
        num_to_add = min(num_new_scenarios, available_slots)
        if num_to_add < num_new_scenarios:
             print(f"Warning: Exceeds MAX_OMEGA. Adding only {num_to_add} of {num_new_scenarios} new scenarios.")
             new_short_r_delta = new_short_r_delta[:, :num_to_add]
        start_col = self.num_scenarios
        end_col = self.num_scenarios + num_to_add
        new_short_r_delta_gpu = cp.asarray(new_short_r_delta, dtype=self.short_delta_r_gpu.dtype)
        self.short_delta_r_gpu[:, start_col:end_col] = new_short_r_delta_gpu
        self.num_scenarios += num_to_add
        return True

    def calculate_cut(self, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
        """
        Performs the core calculation for a given x using constraint duals (pi)
        and reduced costs (rc): selects the best (pi_k, rc_k) pair for each
        scenario based on the full dual objective, processes in batches, and
        computes the complete Benders alpha and beta coefficients.

        Args:
            x (np.ndarray): The first-stage decision vector (size X_DIM).

        Returns:
            tuple[float, np.ndarray, np.ndarray] | None:
                - alpha (float): The full constant term alpha = E[pi^T r] - E[lambda^T l] + E[mu^T u] (float64 accuracy).
                - beta (np.ndarray): The coefficient vector beta = -E[C^T pi] (size X_DIM, float64 accuracy).
                - best_k_index (np.ndarray): Indices of the best (pi_k, rc_k) pair for each scenario (size num_scenarios).
                                             Basis info must be retrieved separately using these indices.
            Returns None if there are no pi vectors or no scenarios stored.
        """
        if self.num_pi == 0: print("Error: No pi vectors stored."); return None
        if self.num_scenarios == 0: print("Error: No scenarios stored."); return None
        if x.shape != (self.X_DIM,): raise ValueError(f"x shape mismatch.")

        # --- Prepare GPU data views ---
        active_pi_gpu = self.pi_gpu[:self.num_pi]
        active_rc_gpu = self.rc_gpu[:self.num_pi]
        active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]
        active_short_delta_r_gpu = self.short_delta_r_gpu[:, :self.num_scenarios]
        x_gpu = cp.asarray(x, dtype=self.C_gpu.dtype)

        # --- Precompute terms constant across batches ---
        Cx_gpu = self.C_gpu @ x_gpu # Shape (NUM_STAGE2_ROWS,)

        # Precompute lambda and mu for all stored duals
        lambda_all_k = cp.maximum(0, active_rc_gpu)
        mu_all_k = cp.maximum(0, -active_rc_gpu)
        # Precompute bound terms constant across scenarios
        lb_gpu_f32 = self.lb_gpu.astype(cp.float32, copy=False)
        ub_gpu_f32 = self.ub_gpu.astype(cp.float32, copy=False)
        lambda_l_term_all_k = lambda_all_k @ lb_gpu_f32 # Shape (num_pi,)
        mu_u_term_all_k = mu_all_k @ ub_gpu_f32       # Shape (num_pi,)

        # --- Initialize accumulators and result arrays ---
        total_selected_pi_sum = cp.zeros(self.NUM_STAGE2_ROWS, dtype=cp.float64)
        total_selected_lambda_sum = cp.zeros(self.NUM_BOUNDED_VARS, dtype=cp.float64)
        total_selected_mu_sum = cp.zeros(self.NUM_BOUNDED_VARS, dtype=cp.float64)
        total_s_sum = cp.zeros((), dtype=cp.float64)
        all_best_k_indices = cp.zeros(self.num_scenarios, dtype=cp.int32)

        # --- Process scenarios in batches ---
        num_batches = math.ceil(self.num_scenarios / self.scenario_batch_size)
        print(f"[{time.strftime('%H:%M:%S')}] Processing {self.num_scenarios} scenarios in {num_batches} batches of size {self.scenario_batch_size}...")

        mempool = cp.get_default_memory_pool()

        for i in range(num_batches):
            start_idx = i * self.scenario_batch_size
            end_idx = min((i + 1) * self.scenario_batch_size, self.num_scenarios)
            current_batch_size = end_idx - start_idx

            batch_short_delta_r = active_short_delta_r_gpu[:, start_idx:end_idx]

            # --- Step 1 (Batch): Compute h = r - Cx ---
            delta_r_gpu_batch = cp.zeros((self.NUM_STAGE2_ROWS, current_batch_size), dtype=cp.float32)
            delta_r_gpu_batch[self.r_sparse_indices_gpu, :] = batch_short_delta_r
            r_gpu_batch = self.r_bar_gpu[:, cp.newaxis] + delta_r_gpu_batch
            h_gpu_batch = r_gpu_batch - Cx_gpu[:, cp.newaxis]
            del delta_r_gpu_batch, r_gpu_batch

            # --- Step 2 (Batch): Select Best (pi_k, rc_k) based on full score ---
            # score = pi^T * h - lambda^T * l + mu^T * u
            pi_h_term = active_pi_gpu @ h_gpu_batch # (num_pi, current_batch_size)
            del h_gpu_batch
            # Combine terms using broadcasting
            scores_batch = pi_h_term - lambda_l_term_all_k[:, cp.newaxis] + mu_u_term_all_k[:, cp.newaxis]
            del pi_h_term

            best_k_index_batch = cp.argmax(scores_batch, axis=0)
            del scores_batch

            all_best_k_indices[start_idx:end_idx] = best_k_index_batch

            # --- Step 3 (Batch): Compute s_i = short_pi_k*^T * short_delta_r_i ---
            s_all_gpu_batch = active_short_pi_gpu @ batch_short_delta_r
            del batch_short_delta_r
            s_gpu_batch = s_all_gpu_batch[best_k_index_batch, cp.arange(current_batch_size)]
            del s_all_gpu_batch

            # --- Step 4 (Batch): Accumulate Results ---
            selected_pi_batch = active_pi_gpu[best_k_index_batch]
            selected_lambda_batch = lambda_all_k[best_k_index_batch]
            selected_mu_batch = mu_all_k[best_k_index_batch]
            del best_k_index_batch # No longer needed for this batch

            # Accumulate sums using float64
            total_selected_pi_sum += cp.sum(selected_pi_batch, axis=0, dtype=cp.float64)
            total_selected_lambda_sum += cp.sum(selected_lambda_batch, axis=0, dtype=cp.float64)
            total_selected_mu_sum += cp.sum(selected_mu_batch, axis=0, dtype=cp.float64)
            total_s_sum += cp.sum(s_gpu_batch, dtype=cp.float64)
            del selected_pi_batch, selected_lambda_batch, selected_mu_batch, s_gpu_batch

            # Memory pool freeing inside loop removed based on previous discussion

        # Clean up precomputed arrays
        del lambda_all_k, mu_all_k, lambda_l_term_all_k, mu_u_term_all_k

        # --- Calculate final averages ---
        if self.num_scenarios > 0:
            Avg_Pi = total_selected_pi_sum / self.num_scenarios # float64
            Avg_Lambda = total_selected_lambda_sum / self.num_scenarios # float64
            Avg_Mu = total_selected_mu_sum / self.num_scenarios # float64
            Avg_S = total_s_sum / self.num_scenarios # float64 scalar
        else: # Should not happen
            Avg_Pi = cp.zeros(self.NUM_STAGE2_ROWS, dtype=cp.float64)
            Avg_Lambda = cp.zeros(self.NUM_BOUNDED_VARS, dtype=cp.float64)
            Avg_Mu = cp.zeros(self.NUM_BOUNDED_VARS, dtype=cp.float64)
            Avg_S = cp.zeros((), dtype=cp.float64)

        # --- Step 5: Final Coefficient Calculation ---
        # beta = -C^T * Avg_Pi
        beta_gpu = -self.C_gpu.T @ Avg_Pi # float64

        # alpha = E[pi^T r_bar] + E[s] - E[lambda^T l] + E[mu^T u]
        r_bar_gpu_fp64 = self.r_bar_gpu.astype(cp.float64, copy=False)
        lb_gpu_fp64 = self.lb_gpu.astype(cp.float64, copy=False)
        ub_gpu_fp64 = self.ub_gpu.astype(cp.float64, copy=False)

        alpha_pi_term = Avg_Pi @ r_bar_gpu_fp64 + Avg_S
        alpha_lambda_term = Avg_Lambda @ lb_gpu_fp64
        alpha_mu_term = Avg_Mu @ ub_gpu_fp64

        alpha_gpu = alpha_pi_term - alpha_lambda_term + alpha_mu_term # scalar float64

        # --- Return results (transfer back to CPU) ---
        alpha = alpha_gpu.item() # Full alpha
        beta = cp.asnumpy(beta_gpu) # float64
        best_k_index = cp.asnumpy(all_best_k_indices) # int32

        print(f"[{time.strftime('%H:%M:%S')}] Cut calculation finished.")
        print(f"    Final VRAM used: {mempool.used_bytes() / 1024**3:.2f} GB") # Useful VRAM info

        return alpha, beta, best_k_index

    # --- Retrieval Methods ---
    def get_pi_rc_basis(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Retrieves the stored pi, rc, vbasis (CPU), and cbasis (CPU) for a given index.

        Args:
            index (int): The index of the stored tuple (0 to num_pi - 1).

        Returns:
            tuple | None: A tuple containing (pi, rc, vbasis, cbasis) as NumPy arrays,
                          or None if the index is out of bounds.
        """
        if not 0 <= index < self.num_pi:
            print(f"Error: Index {index} out of bounds (0 to {self.num_pi - 1})")
            return None
        # Retrieve from CPU arrays
        pi_val = self.pi_cpu[index].copy()
        rc_val = self.rc_cpu[index].copy()
        vbasis_val = self.vbasis_cpu[index].copy()
        cbasis_val = self.cbasis_cpu[index].copy()
        return pi_val, rc_val, vbasis_val, cbasis_val

    def get_basis(self, index: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Retrieves only the basis (vbasis, cbasis) from CPU for a given index."""
        if not 0 <= index < self.num_pi:
            print(f"Error: Index {index} out of bounds (0 to {self.num_pi - 1})")
            return None
        vbasis_val = self.vbasis_cpu[index].copy()
        cbasis_val = self.cbasis_cpu[index].copy()
        return vbasis_val, cbasis_val
