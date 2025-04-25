import torch
import numpy as np
import scipy.sparse # For type hint
import hashlib
import time
import math
from typing import Tuple, Optional, Union, TYPE_CHECKING
import threading

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
    - Deduplicates added solutions based on basis (vbasis, cbasis).
    - Processes scenarios in batches for memory efficiency.
    - Uses double precision (float64) for key reduction steps for accuracy.
    - Handles sparse transfer matrix C (torch.sparse_csr_tensor).
    """

    def __init__(self, NUM_STAGE2_ROWS: int, NUM_BOUNDED_VARS: int, NUM_STAGE2_VARS: int,
                 X_DIM: int, MAX_PI: int, MAX_OMEGA: int,
                 r_sparse_indices: np.ndarray,
                 r_bar: np.ndarray,
                 C: scipy.sparse.spmatrix,
                 lb_y_bounded: np.ndarray,
                 ub_y_bounded: np.ndarray,
                 scenario_batch_size: int = 100000,
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initializes the ArgmaxOperation class.

        Args:
            NUM_STAGE2_ROWS: Dimension of pi vector and stage 2 constraints.
            NUM_BOUNDED_VARS: Dimension of rc vector (vars with non-trivial bounds).
            NUM_STAGE2_VARS: Dimension of vbasis vector (total stage 2 vars).
            X_DIM: Dimension of x vector (stage 1 vars).
            MAX_PI: Maximum number of (pi, RC, basis) tuples to store.
            MAX_OMEGA: Maximum number of scenarios to store.
            r_sparse_indices: 1D np.ndarray of indices where stage 2 RHS is stochastic.
            r_bar: 1D np.ndarray for the fixed part of stage 2 RHS.
            C: SciPy sparse matrix (stage 2 rows x X_DIM).
            lb_y_bounded: 1D np.ndarray of lower bounds for bounded variables.
            ub_y_bounded: 1D np.ndarray of upper bounds for bounded variables.
            scenario_batch_size: Number of scenarios per GPU batch. Defaults to 100,000.
            device: PyTorch device ('cuda', 'cpu', etc.). Auto-detects if None.
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
        if C.shape != (NUM_STAGE2_ROWS, X_DIM): raise ValueError("C matrix shape mismatch.")
        if lb_y_bounded.shape != (NUM_BOUNDED_VARS,): raise ValueError("lb_y_bounded shape mismatch.")
        if ub_y_bounded.shape != (NUM_BOUNDED_VARS,): raise ValueError("ub_y_bounded shape mismatch.")
        if not isinstance(r_sparse_indices, np.ndarray) or r_sparse_indices.ndim != 1: raise ValueError("r_sparse_indices must be 1D.")
        unique_r_indices = np.unique(r_sparse_indices)
        if len(unique_r_indices) != len(r_sparse_indices):
            print("Warning: r_sparse_indices contains duplicate values. Using unique indices.")
            r_sparse_indices = unique_r_indices
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
        self.num_pi = 0
        self.num_scenarios = 0

        # --- CPU Data Storage (Only Basis) ---
        print(f"[{time.strftime('%H:%M:%S')}] Allocating CPU memory for basis...")
        basis_dtype_np = np.int8 # Compact storage for basis status
        self.vbasis_cpu = np.zeros((MAX_PI, NUM_STAGE2_VARS), dtype=basis_dtype_np)
        self.cbasis_cpu = np.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=basis_dtype_np)
        self.basis_hashes = set() # For deduplication based on basis

        # --- GPU (Device) Data Storage ---
        print(f"[{time.strftime('%H:%M:%S')}] Allocating memory on device {self.device}...")
        torch_dtype = torch.float32 # Default dtype for most device tensors

        self.pi_gpu = torch.zeros((MAX_PI, NUM_STAGE2_ROWS), dtype=torch_dtype, device=self.device)
        self.rc_gpu = torch.zeros((MAX_PI, NUM_BOUNDED_VARS), dtype=torch_dtype, device=self.device)
        self.short_pi_gpu = torch.zeros((MAX_PI, self.R_SPARSE_LEN), dtype=torch_dtype, device=self.device)
        self.lb_gpu = torch.from_numpy(lb_y_bounded).to(dtype=torch_dtype, device=self.device)
        self.ub_gpu = torch.from_numpy(ub_y_bounded).to(dtype=torch_dtype, device=self.device)
        self.r_bar_gpu = torch.from_numpy(r_bar).to(dtype=torch_dtype, device=self.device)
        self.r_sparse_indices_gpu = torch.from_numpy(self.r_sparse_indices_cpu).to(dtype=torch.long, device=self.device) # LongTensor for indexing
        self.short_delta_r_gpu = torch.zeros((self.R_SPARSE_LEN, MAX_OMEGA), dtype=torch_dtype, device=self.device)

        # --- Convert C matrix to sparse CSR tensor on device ---
        print(f"[{time.strftime('%H:%M:%S')}] Converting C to sparse CSR format on {self.device}...")
        if not scipy.sparse.isspmatrix_csr(C):
            C = C.tocsr()
        if C.dtype != np.float32:
             C = C.astype(np.float32)

        self.C_gpu = torch.sparse_csr_tensor(
            crow_indices=torch.from_numpy(C.indptr).to(dtype=torch.int64, device=self.device), # int64 required by torch
            col_indices=torch.from_numpy(C.indices).to(dtype=torch.int64, device=self.device), # int64 required by torch
            values=torch.from_numpy(C.data).to(dtype=torch_dtype, device=self.device),
            size=C.shape,
            dtype=torch_dtype,
            device=self.device
        )

        # --- Threading Lock ---
        self._lock = threading.Lock()

        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Initialization complete ({end_time - start_time:.2f}s).")

    @classmethod
    def from_smps_reader(cls, reader: 'SMPSReader',
                         MAX_PI: int, MAX_OMEGA: int,
                         scenario_batch_size: int = 100000,
                         device: Optional[Union[str, torch.device]] = None) -> 'ArgmaxOperation':
        """
        Factory method to create an ArgmaxOperation instance from an SMPSReader.

        Args:
            reader: An initialized and loaded SMPSReader instance.
            MAX_PI: Maximum number of (pi, RC, basis) tuples to store.
            MAX_OMEGA: Maximum number of scenarios to store.
            scenario_batch_size: Number of scenarios per GPU batch. Defaults to 100,000.
            device: PyTorch device ('cuda', 'cpu', etc.). Auto-detects if None.

        Returns:
            An initialized ArgmaxOperation instance.

        Raises:
            ValueError: If the reader hasn't been loaded or essential data is missing.
        """
        if not reader._data_loaded:
            raise ValueError("SMPSReader data must be loaded using load_and_extract() before creating ArgmaxOperation.")
        if reader.C is None:
             raise ValueError("SMPSReader did not load matrix C (stage 2 technology matrix).")
        
        # --- Extract parameters from reader ---
        NUM_STAGE2_ROWS = len(reader.row2_indices)
        NUM_STAGE2_VARS = len(reader.y_indices)
        X_DIM = len(reader.x_indices)
        r_sparse_indices = reader.stochastic_rows_relative_indices
        r_bar = reader.r_bar
        C = reader.C

        # Calculate bounded variable info
        if NUM_STAGE2_VARS > 0 and reader.lb_y is not None and reader.ub_y is not None:
            has_finite_ub = np.isfinite(reader.ub_y)
            has_finite_lb = np.isfinite(reader.lb_y) & (np.abs(reader.lb_y) > 1e-9) # Non-zero LB
            bounded_mask = has_finite_ub | has_finite_lb
            lb_y_bounded = reader.lb_y[bounded_mask]
            ub_y_bounded = reader.ub_y[bounded_mask]
            NUM_BOUNDED_VARS = len(lb_y_bounded)
        else:
            lb_y_bounded = np.array([])
            ub_y_bounded = np.array([])
            NUM_BOUNDED_VARS = 0

        # --- Instantiate the class using the extracted parameters ---
        return cls(
            NUM_STAGE2_ROWS=NUM_STAGE2_ROWS,
            NUM_BOUNDED_VARS=NUM_BOUNDED_VARS,
            NUM_STAGE2_VARS=NUM_STAGE2_VARS,
            X_DIM=X_DIM,
            MAX_PI=MAX_PI,
            MAX_OMEGA=MAX_OMEGA,
            r_sparse_indices=r_sparse_indices,
            r_bar=r_bar,
            C=C,
            lb_y_bounded=lb_y_bounded,
            ub_y_bounded=ub_y_bounded,
            scenario_batch_size=scenario_batch_size,
            device=device
        )


    def add_pi(self, new_pi: np.ndarray, new_rc: np.ndarray,
               new_vbasis: np.ndarray, new_cbasis: np.ndarray) -> bool:
        """
        Adds a new dual solution (pi, rc) and its basis (vbasis, cbasis).
        Stores pi/rc on device, basis on CPU. Deduplicates based on basis.

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
            # --- Critical Section Start ---
            if self.num_pi >= self.MAX_PI:
                 # print(f"Warning: MAX_PI ({self.MAX_PI}) reached. Cannot add new solution.")
                 return False

            # --- Deduplication based only on the basis (vbasis, cbasis) pair ---
            try:
                # Concatenate basis vectors for hashing
                combined_basis = np.concatenate((np.ascontiguousarray(new_vbasis),
                                                 np.ascontiguousarray(new_cbasis)))
                basis_hash = hashlib.sha256(combined_basis.tobytes()).hexdigest()
                if basis_hash in self.basis_hashes:
                    # print("Debug: Duplicate basis detected, not adding.")
                    return False
                self.basis_hashes.add(basis_hash)
            except Exception as e:
                print(f"Warning: Could not hash basis pair for deduplication. Adding anyway. Error: {e}")

            idx = self.num_pi

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

            self.num_pi += 1
            # --- Critical Section End ---
            return True

    def add_scenarios(self, new_short_r_delta: np.ndarray) -> bool:
        """
        Adds new scenario data (packed delta_r values for stochastic RHS rows).

        Args:
            new_short_r_delta: Array of shape (R_SPARSE_LEN, num_new_scenarios).

        Returns:
            True if scenarios were added (partially or fully), False otherwise.
        """
        if new_short_r_delta.ndim != 2 or new_short_r_delta.shape[0] != self.R_SPARSE_LEN:
            raise ValueError(f"new_short_r_delta shape mismatch. Expected ({self.R_SPARSE_LEN}, N), got {new_short_r_delta.shape}")
        num_new_scenarios = new_short_r_delta.shape[1]
        available_slots = self.MAX_OMEGA - self.num_scenarios
        if num_new_scenarios <= 0 or available_slots <= 0:
            return False

        num_to_add = min(num_new_scenarios, available_slots)
        if num_to_add < num_new_scenarios:
            print(f"Warning: Exceeds MAX_OMEGA. Adding only {num_to_add} of {num_new_scenarios} new scenarios.")
            new_short_r_delta = new_short_r_delta[:, :num_to_add]

        start_col = self.num_scenarios
        end_col = self.num_scenarios + num_to_add

        # Add scenarios to device tensor
        new_short_r_delta_gpu = torch.from_numpy(new_short_r_delta).to(dtype=self.short_delta_r_gpu.dtype, device=self.device)
        self.short_delta_r_gpu[:, start_col:end_col] = new_short_r_delta_gpu

        self.num_scenarios += num_to_add
        return True

    def calculate_cut(self, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
        """
        Calculates Benders cut coefficients (alpha, beta) for a given x using an
        optimized approach. Selects the best dual solution (pi_k, rc_k) for each
        scenario based on maximizing the dual objective, using device acceleration.

        Optimization: Avoids recalculating pi^T * (r_bar - Cx) for every scenario
        by splitting the score calculation into constant and stochastic parts.

        Args:
            x: The first-stage decision vector (NumPy array, size X_DIM).

        Returns:
            tuple (alpha, beta, best_k_indices) or None if no solutions/scenarios.
            - alpha (float): Constant term of the Benders cut (float64).
            - beta (np.ndarray): Coefficient vector of the cut (float64, size X_DIM).
            - best_k_indices (np.ndarray): Index of the best pi/rc/basis for each
                                           scenario (int32, size num_scenarios).
        """
        if self.num_pi == 0: print("Error: No pi vectors stored."); return None
        if self.num_scenarios == 0: print("Error: No scenarios stored."); return None
        if x.shape != (self.X_DIM,): raise ValueError("x shape mismatch.")

        with torch.no_grad(): # Disable gradient tracking
            # --- Prepare device data views ---
            active_pi_gpu = self.pi_gpu[:self.num_pi]
            active_rc_gpu = self.rc_gpu[:self.num_pi]
            active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]
            active_short_delta_r_gpu = self.short_delta_r_gpu[:, :self.num_scenarios]
            x_gpu = torch.from_numpy(x).to(dtype=self.C_gpu.dtype, device=self.device)

            # --- Precompute terms constant across ALL scenarios ---
            # 1. Calculate h_bar = r_bar - C*x
            Cx_gpu = torch.matmul(self.C_gpu, x_gpu) # (NUM_STAGE2_ROWS,)
            h_bar_gpu = self.r_bar_gpu - Cx_gpu     # (NUM_STAGE2_ROWS,)

            # 2. Calculate pi_k^T * h_bar for all k
            pi_h_bar_term_all_k = torch.matmul(active_pi_gpu, h_bar_gpu) # (num_pi,)

            # 3. Calculate bound terms: - lambda_k^T * l + mu_k^T * u for all k
            lambda_all_k = torch.clamp(active_rc_gpu, min=0)
            mu_all_k = torch.clamp(-active_rc_gpu, min=0)

            # Ensure bounds are float32 for matmul with float32 lambda/mu
            lb_gpu_f32 = self.lb_gpu.to(torch.float32)
            ub_gpu_f32 = self.ub_gpu.to(torch.float32)
            lambda_l_term_all_k = torch.matmul(lambda_all_k, lb_gpu_f32) # (num_pi,)
            mu_u_term_all_k = torch.matmul(mu_all_k, ub_gpu_f32)       # (num_pi,)

            # Combine constant terms: score_constant = pi_h_bar - lambda_l + mu_u
            constant_score_part_all_k = pi_h_bar_term_all_k - lambda_l_term_all_k + mu_u_term_all_k # (num_pi,)

            # --- Initialize accumulators (float64 for precision) and result array ---
            total_selected_pi_sum = torch.zeros(self.NUM_STAGE2_ROWS, dtype=torch.float64, device=self.device)
            total_selected_lambda_sum = torch.zeros(self.NUM_BOUNDED_VARS, dtype=torch.float64, device=self.device)
            total_selected_mu_sum = torch.zeros(self.NUM_BOUNDED_VARS, dtype=torch.float64, device=self.device)
            total_s_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            all_best_k_indices = torch.zeros(self.num_scenarios, dtype=torch.int32, device=self.device)

            # --- Process scenarios in batches ---
            num_batches = math.ceil(self.num_scenarios / self.scenario_batch_size)
            print(f"[{time.strftime('%H:%M:%S')}] Processing {self.num_scenarios} scenarios in {num_batches} batches of size {self.scenario_batch_size}...")

            for i in range(num_batches):
                start_idx = i * self.scenario_batch_size
                end_idx = min((i + 1) * self.scenario_batch_size, self.num_scenarios)
                current_batch_size = end_idx - start_idx

                batch_short_delta_r = active_short_delta_r_gpu[:, start_idx:end_idx] # (R_SPARSE_LEN, current_batch_size)

                # --- Step 1 (Batch): Compute Stochastic Score Part ---
                # stochastic_score_part = pi_k^T * delta_r(omega_i)
                #                      = short_pi_k^T * short_delta_r(omega_i)
                stochastic_score_part_batch = torch.matmul(active_short_pi_gpu, batch_short_delta_r) # (num_pi, current_batch_size)

                # --- Step 2 (Batch): Compute Total Score and Select Best k* ---
                # total_score = constant_score_part (broadcasted) + stochastic_score_part
                scores_batch = constant_score_part_all_k[:, None] + stochastic_score_part_batch # Broadcast constant part
                best_k_index_batch = torch.argmax(scores_batch, dim=0) # Find best pi index per scenario in batch
                del scores_batch # Free memory
                all_best_k_indices[start_idx:end_idx] = best_k_index_batch.to(torch.int32) # Store indices

                # --- Step 3 (Batch): Compute s_i = short_pi_k*^T * short_delta_r_i for the best k* ---
                # We already computed this for all k in stochastic_score_part_batch.
                # We just need to select the values corresponding to the winning k* for each scenario.
                batch_scenario_indices = torch.arange(current_batch_size, device=self.device)
                s_gpu_batch = stochastic_score_part_batch[best_k_index_batch, batch_scenario_indices] # Select s for best k*
                del stochastic_score_part_batch, batch_scenario_indices # Free memory

                # --- Step 4 (Batch): Accumulate Results using float64 ---
                selected_pi_batch = active_pi_gpu[best_k_index_batch]
                selected_lambda_batch = lambda_all_k[best_k_index_batch] # Use precomputed lambda_all_k
                selected_mu_batch = mu_all_k[best_k_index_batch]       # Use precomputed mu_all_k
                del best_k_index_batch # No longer needed for this batch

                total_selected_pi_sum += torch.sum(selected_pi_batch, dim=0, dtype=torch.float64)
                total_selected_lambda_sum += torch.sum(selected_lambda_batch, dim=0, dtype=torch.float64)
                total_selected_mu_sum += torch.sum(selected_mu_batch, dim=0, dtype=torch.float64)
                total_s_sum += torch.sum(s_gpu_batch, dtype=torch.float64)
                del selected_pi_batch, selected_lambda_batch, selected_mu_batch, s_gpu_batch # Free memory

            # --- Cleanup precomputed tensors ---
            del lambda_all_k, mu_all_k, lambda_l_term_all_k, mu_u_term_all_k
            del constant_score_part_all_k, pi_h_bar_term_all_k, h_bar_gpu, Cx_gpu

            # --- Calculate final averages ---
            Avg_Pi = total_selected_pi_sum / self.num_scenarios
            Avg_Lambda = total_selected_lambda_sum / self.num_scenarios
            Avg_Mu = total_selected_mu_sum / self.num_scenarios
            Avg_S = total_s_sum / self.num_scenarios

            # --- Step 5: Final Coefficient Calculation (float64) ---
            # beta = -E[C^T * pi] = -C^T * Avg_Pi
            # Ensure compatible types for sparse matmul (convert C to float64 for safety)
            C_gpu_fp64 = self.C_gpu.to(torch.float64)
            beta_gpu = -torch.matmul(C_gpu_fp64.T, Avg_Pi)
            del C_gpu_fp64

            # alpha = E[pi^T * r_bar] + E[s] - E[lambda^T * l] + E[mu^T * u]
            #       = Avg_Pi^T * r_bar + Avg_S - Avg_Lambda^T * l + Avg_Mu^T * u
            r_bar_gpu_fp64 = self.r_bar_gpu.to(torch.float64)
            lb_gpu_fp64 = self.lb_gpu.to(torch.float64)
            ub_gpu_fp64 = self.ub_gpu.to(torch.float64)

            alpha_pi_term = torch.dot(Avg_Pi, r_bar_gpu_fp64) + Avg_S # Note: Avg_S already incorporates pi^T*delta_r
            alpha_lambda_term = torch.dot(Avg_Lambda, lb_gpu_fp64)
            alpha_mu_term = torch.dot(Avg_Mu, ub_gpu_fp64)

            alpha_gpu = alpha_pi_term - alpha_lambda_term + alpha_mu_term

            # --- Return results (transfer to CPU as NumPy arrays) ---
            alpha = alpha_gpu.item()
            beta = beta_gpu.cpu().numpy()
            best_k_index = all_best_k_indices.cpu().numpy()

            print(f"[{time.strftime('%H:%M:%S')}] Cut calculation finished.")
            if self.device.type == 'cuda':
                # Optional: Add memory reporting if needed
                try:
                    allocated_mem_gb = torch.cuda.memory_allocated(self.device) / 1024**3
                    reserved_mem_gb = torch.cuda.memory_reserved(self.device) / 1024**3
                    print(f"    Device VRAM allocated: {allocated_mem_gb:.2f} GB")
                    print(f"    Device VRAM reserved:  {reserved_mem_gb:.2f} GB")
                except Exception as e:
                    print(f"    Could not get CUDA memory info: {e}")


            return alpha, beta, best_k_index

    # --- Retrieval Methods ---
    def get_basis(self, index: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Retrieves only the basis (vbasis, cbasis) from CPU for a given index."""
        if not 0 <= index < self.num_pi:
            print(f"Error: Index {index} out of bounds (0 to {self.num_pi - 1})")
            return None
        # Ensure thread safety if accessed concurrently with add_pi, though typically called after calculate_cut
        # with self._lock: # Usually not needed here if access pattern is sequential
        vbasis_val = self.vbasis_cpu[index].copy()
        cbasis_val = self.cbasis_cpu[index].copy()
        return vbasis_val, cbasis_val
