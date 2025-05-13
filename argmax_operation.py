import torch
import numpy as np
import scipy.sparse # For type hint
import hashlib
import time
import math
from typing import Tuple, Optional, Union, TYPE_CHECKING
import threading
import h5py
import os

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
    """

    def __init__(self, NUM_STAGE2_ROWS: int, NUM_BOUNDED_VARS: int, NUM_STAGE2_VARS: int,
                 X_DIM: int, MAX_PI: int, MAX_OMEGA: int,
                 r_sparse_indices: np.ndarray,
                 r_bar: np.ndarray,
                 C: scipy.sparse.spmatrix,
                 lb_y_bounded: np.ndarray,
                 ub_y_bounded: np.ndarray,
                 scenario_batch_size: int = 10000,
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
            scenario_batch_size: Number of scenarios per GPU batch. Defaults to 10,000.
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
        self.pi_rc_hashes = set() # For deduplication based on (pi, rc) pair

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

        # --- Threading Lock ---
        self._lock = threading.Lock()

        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Initialization complete ({end_time - start_time:.2f}s).")

    @classmethod
    def from_smps_reader(cls, reader: 'SMPSReader',
                         MAX_PI: int, MAX_OMEGA: int,
                         scenario_batch_size: int = 10000,
                         device: Optional[Union[str, torch.device]] = None) -> 'ArgmaxOperation':
        """
        Factory method to create an ArgmaxOperation instance from an SMPSReader.

        Args:
            reader: An initialized and loaded SMPSReader instance.
            MAX_PI: Maximum number of (pi, RC, basis) tuples to store.
            MAX_OMEGA: Maximum number of scenarios to store.
            scenario_batch_size: Number of scenarios per GPU batch. Defaults to 10,000.
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
        Stores pi/rc on device, basis on CPU. Deduplicates based on the (pi, rc) pair.

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

            # --- Deduplication based on the (pi, rc) pair ---
            try:
                # 1. Round new_pi and new_rc to float16 precision
                # Convert to float16 (this performs rounding to the nearest float16 value)
                pi_reduced_precision = new_pi.astype(np.float16)
                rc_reduced_precision = new_rc.astype(np.float16)

                # 2. Concatenate the precision-reduced pi and rc vectors for hashing
                # Ensure they are contiguous for consistent hashing
                combined_pi_rc = np.concatenate((np.ascontiguousarray(pi_reduced_precision),
                                                np.ascontiguousarray(rc_reduced_precision)))
    
                current_hash = hashlib.sha256(combined_pi_rc.tobytes()).hexdigest()
                if current_hash in self.pi_rc_hashes:
                    # print("Debug: Duplicate (pi, rc) pair detected, not adding.")
                    return False
                self.pi_rc_hashes.add(current_hash)
            except Exception as e:
                print(f"Warning: Could not hash (pi, rc) pair for deduplication. Adding anyway. Error: {e}")

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
        User is expected to provide data with scenarios as rows.

        Args:
            new_short_r_delta: Array of shape (num_new_scenarios, R_SPARSE_LEN).
                               Each row represents a scenario, columns are stochastic elements.
        Returns:
            True if scenarios were added (partially or fully), False otherwise.
        """
        # MODIFIED VALIDATION: Check shape[1] for R_SPARSE_LEN, input is (num_scenarios, R_SPARSE_LEN)
        if new_short_r_delta.ndim != 2 or new_short_r_delta.shape[1] != self.R_SPARSE_LEN:
            raise ValueError(f"new_short_r_delta shape mismatch. Expected (num_new_scenarios, {self.R_SPARSE_LEN}), got {new_short_r_delta.shape}")
        
        # MODIFIED: num_new_scenarios is now from shape[0]
        num_new_scenarios = new_short_r_delta.shape[0]
        available_slots = self.MAX_OMEGA - self.num_scenarios
        if num_new_scenarios <= 0 or available_slots <= 0:
            # print(f"Debug: No new scenarios to add or no available slots. New: {num_new_scenarios}, Available: {available_slots}")
            return False

        num_to_add = min(num_new_scenarios, available_slots)
        
        new_short_r_delta_to_add = new_short_r_delta # Placeholder for slicing
        if num_to_add < num_new_scenarios:
            print(f"Warning: Exceeds MAX_OMEGA. Adding only {num_to_add} of {num_new_scenarios} new scenarios.")
            # MODIFIED: Slicing rows from the input
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

    def calculate_cut(self, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
        """
        Calculates Benders cut coefficients (alpha, beta) for a given x using an
        optimized approach. Selects the best dual solution (pi_k, rc_k) for each
        scenario based on maximizing the dual objective, using device acceleration.

        Optimization: Avoids recalculating pi^T * (r_bar - Cx) for every scenario
        by splitting the score calculation into constant and stochastic parts.
        The stochastic part calculation and argmax are optimized for memory layout.

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
            active_pi_gpu = self.pi_gpu[:self.num_pi]  # Shape (num_pi, NUM_STAGE2_ROWS)
            active_rc_gpu = self.rc_gpu[:self.num_pi]  # Shape (num_pi, NUM_BOUNDED_VARS)
            active_short_pi_gpu = self.short_pi_gpu[:self.num_pi] # Shape (num_pi, R_SPARSE_LEN)
            
            # active_scenario_data_gpu has shape (num_scenarios, R_SPARSE_LEN)
            active_scenario_data_gpu = self.short_delta_r_gpu[:self.num_scenarios, :] 
            x_gpu = torch.from_numpy(x).to(dtype=self.C_gpu.dtype, device=self.device)

            # --- Precompute terms constant across ALL scenarios ---
            # 1. Calculate h_bar = r_bar - C*x
            Cx_gpu = torch.matmul(self.C_gpu, x_gpu) 
            h_bar_gpu = self.r_bar_gpu - Cx_gpu    

            # 2. Calculate pi_k^T * h_bar for all k
            pi_h_bar_term_all_k = torch.matmul(active_pi_gpu, h_bar_gpu) # (num_pi,)

            # 3. Calculate bound terms: - lambda_k^T * l + mu_k^T * u for all k
            lambda_all_k = torch.clamp(active_rc_gpu, min=0)
            mu_all_k = torch.clamp(-active_rc_gpu, min=0)

            lb_gpu_f32 = self.lb_gpu.to(torch.float32)
            ub_gpu_f32 = self.ub_gpu.to(torch.float32)
            lambda_l_term_all_k = torch.matmul(lambda_all_k, lb_gpu_f32) 
            mu_u_term_all_k = torch.matmul(mu_all_k, ub_gpu_f32)      

            constant_score_part_all_k = pi_h_bar_term_all_k - lambda_l_term_all_k + mu_u_term_all_k # (num_pi,)

            # --- Initialize accumulators (float64 for precision) and result array ---
            total_selected_pi_sum = torch.zeros(self.NUM_STAGE2_ROWS, dtype=torch.float64, device=self.device)
            total_selected_lambda_sum = torch.zeros(self.NUM_BOUNDED_VARS, dtype=torch.float64, device=self.device)
            total_selected_mu_sum = torch.zeros(self.NUM_BOUNDED_VARS, dtype=torch.float64, device=self.device)
            total_s_sum = torch.zeros((), dtype=torch.float64, device=self.device) # scalar
            all_best_k_indices = torch.zeros(self.num_scenarios, dtype=torch.int32, device=self.device)

            # --- Process scenarios in batches ---
            num_batches = math.ceil(self.num_scenarios / self.scenario_batch_size)
            # print(f"[{time.strftime('%H:%M:%S')}] Processing {self.num_scenarios} scenarios in {num_batches} batches of size {self.scenario_batch_size}...")

            for i in range(num_batches):
                # start_idx and end_idx now refer to scenario indices (rows)
                start_scenario_idx = i * self.scenario_batch_size
                end_scenario_idx = min((i + 1) * self.scenario_batch_size, self.num_scenarios)
                current_batch_size = end_scenario_idx - start_scenario_idx

                # batch_scenario_slice has shape (current_batch_size, R_SPARSE_LEN)
                batch_scenario_slice = active_scenario_data_gpu[start_scenario_idx:end_scenario_idx, :] 

                # --- Step 1 (Batch): Compute Stochastic Score Part ---
                # stochastic_score_part = delta_r(omega_i)^T * short_pi_k  (for each omega_i, pi_k pair)
                # active_short_pi_gpu.T has shape (R_SPARSE_LEN, num_pi)
                # Resulting stochastic_score_part_batch shape: (current_batch_size, num_pi)
                stochastic_score_part_batch = torch.matmul(batch_scenario_slice, active_short_pi_gpu.T)

                # --- Step 2 (Batch): Compute Total Score and Select Best k* ---
                # constant_score_part_all_k has shape (num_pi,). Broadcasting adds it to each row.
                # scores_batch shape: (current_batch_size, num_pi)
                scores_batch = stochastic_score_part_batch + constant_score_part_all_k 
                # argmax along dim=1 to find best pi_k for each scenario (row)
                best_k_index_batch = torch.argmax(scores_batch, dim=1) # Shape (current_batch_size,)
                del scores_batch 
                all_best_k_indices[start_scenario_idx:end_scenario_idx] = best_k_index_batch.to(torch.int32)

                # --- Step 3 (Batch): Compute s_i = short_pi_k*^T * short_delta_r_i for the best k* ---
                # Need to select the scores from stochastic_score_part_batch that correspond to the best_k_index_batch
                batch_indices_for_s = torch.arange(current_batch_size, device=self.device)
                # Indexing into stochastic_score_part_batch (current_batch_size, num_pi)
                s_gpu_batch = stochastic_score_part_batch[batch_indices_for_s, best_k_index_batch] # Shape (current_batch_size,)
                del stochastic_score_part_batch, batch_indices_for_s # Free memory

                # --- Step 4 (Batch): Accumulate Results using float64 ---
                selected_pi_batch = active_pi_gpu[best_k_index_batch]
                selected_lambda_batch = lambda_all_k[best_k_index_batch] 
                selected_mu_batch = mu_all_k[best_k_index_batch]     
                del best_k_index_batch 

                total_selected_pi_sum += torch.sum(selected_pi_batch, dim=0, dtype=torch.float64)
                total_selected_lambda_sum += torch.sum(selected_lambda_batch, dim=0, dtype=torch.float64)
                total_selected_mu_sum += torch.sum(selected_mu_batch, dim=0, dtype=torch.float64)
                total_s_sum += torch.sum(s_gpu_batch, dtype=torch.float64)
                del selected_pi_batch, selected_lambda_batch, selected_mu_batch, s_gpu_batch 

            # --- Cleanup precomputed tensors ---
            del lambda_all_k, mu_all_k, lambda_l_term_all_k, mu_u_term_all_k
            del constant_score_part_all_k, pi_h_bar_term_all_k, h_bar_gpu, Cx_gpu

            # --- Calculate final averages ---
            Avg_Pi = total_selected_pi_sum / self.num_scenarios
            Avg_Lambda = total_selected_lambda_sum / self.num_scenarios
            Avg_Mu = total_selected_mu_sum / self.num_scenarios
            Avg_S = total_s_sum / self.num_scenarios

            # --- Step 5: Final Coefficient Calculation (float64) ---
            beta_gpu = -torch.matmul(self.C_gpu_fp64_transpose, Avg_Pi)

            r_bar_gpu_fp64 = self.r_bar_gpu.to(torch.float64)
            lb_gpu_fp64 = self.lb_gpu.to(torch.float64)
            ub_gpu_fp64 = self.ub_gpu.to(torch.float64)

            alpha_pi_term = torch.dot(Avg_Pi, r_bar_gpu_fp64) + Avg_S 
            alpha_lambda_term = torch.dot(Avg_Lambda, lb_gpu_fp64)
            alpha_mu_term = torch.dot(Avg_Mu, ub_gpu_fp64)

            alpha_gpu = alpha_pi_term - alpha_lambda_term + alpha_mu_term

            # --- Return results (transfer to CPU as NumPy arrays) ---
            alpha = alpha_gpu.item()
            beta = beta_gpu.cpu().numpy()
            best_k_index_cpu = all_best_k_indices.cpu().numpy()

            # print(f"[{time.strftime('%H:%M:%S')}] Cut calculation finished.")
            if self.device.type == 'cuda':
                try:
                    allocated_mem_gb = torch.cuda.memory_allocated(self.device) / 1024**3
                    reserved_mem_gb = torch.cuda.memory_reserved(self.device) / 1024**3
            #         print(f"    Device VRAM allocated: {allocated_mem_gb:.2f} GB")
                    print(f"    Device VRAM reserved:  {reserved_mem_gb:.2f} GB")
                except Exception as e:
                    print(f"    Could not get CUDA memory info: {e}")

            return alpha, beta, best_k_index_cpu

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

    def manage_pool_data(self, mode: str, filepath: str = "result.hdf5") -> None:
            """
            Saves or loads the current pool of dual solutions (pi, rc) and their
            corresponding basis information (vbasis, cbasis) to/from an HDF5 file
            under the '/argmax_info/' group.

            When loading, the current pool in memory is cleared and replaced by the
            data from the HDF5 file. The `short_pi_gpu` and `pi_rc_hashes` are
            reconstructed based on the loaded data.

            Args:
                mode: "save" to save the current pool, "load" to load from file.
                filepath: The path to the HDF5 file. Defaults to "result.hdf5".

            Raises:
                ValueError: If mode is not 'save' or 'load', or if dimensionality mismatch occurs.
                FileNotFoundError: If mode is 'load' and the filepath does not exist.
                KeyError: If mode is 'load' and expected datasets/attributes are missing.
                Exception: For other HDF5 related I/O errors.
            """

            group_name = "argmax_info"

            with self._lock:
                is_cuda = self.device.type == 'cuda'

                if mode == "save":
                    if self.num_pi == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] Pool is empty. Nothing to save to {filepath} under group '{group_name}'.")
                        # To explicitly mark an empty pool in the file, you could open, ensure group, set num_pi=0,
                        # and delete datasets if they exist. For now, following original logic of not saving if empty.
                        return

                    print(f"[{time.strftime('%H:%M:%S')}] Saving pool data to {filepath} under group '{group_name}' ({self.num_pi} entries)...")
                    try:
                        # Mode 'a': Read/write if exists, create otherwise
                        with h5py.File(filepath, 'a') as hf:
                            # Ensure the main group exists
                            pool_group = hf.require_group(group_name)

                            pool_group.attrs['num_pi'] = self.num_pi
                            pool_group.attrs['NUM_STAGE2_ROWS'] = self.NUM_STAGE2_ROWS
                            pool_group.attrs['NUM_BOUNDED_VARS'] = self.NUM_BOUNDED_VARS
                            pool_group.attrs['NUM_STAGE2_VARS'] = self.NUM_STAGE2_VARS
                            pool_group.attrs['R_SPARSE_LEN'] = self.R_SPARSE_LEN
                            pool_group.attrs['MAX_PI_capacity_at_save'] = self.MAX_PI # Store capacity for info

                            # Prepare data for saving (move GPU data to CPU)
                            pi_to_save = self.pi_gpu[:self.num_pi].cpu().numpy()
                            rc_to_save = self.rc_gpu[:self.num_pi].cpu().numpy()
                            vbasis_to_save = self.vbasis_cpu[:self.num_pi]
                            cbasis_to_save = self.cbasis_cpu[:self.num_pi]

                            # Delete existing datasets in the group before creating new ones
                            # This ensures clean overwrite, especially if new data is smaller.
                            for ds_name in ["pi_pool", "rc_pool", "vbasis_pool", "cbasis_pool"]:
                                if ds_name in pool_group:
                                    del pool_group[ds_name]

                            # Create datasets with chunking and compression
                            pool_group.create_dataset("pi_pool", data=pi_to_save, chunks=True, compression="gzip")
                            pool_group.create_dataset("rc_pool", data=rc_to_save, chunks=True, compression="gzip")
                            pool_group.create_dataset("vbasis_pool", data=vbasis_to_save, chunks=True, compression="gzip")
                            pool_group.create_dataset("cbasis_pool", data=cbasis_to_save, chunks=True, compression="gzip")
                        print(f"[{time.strftime('%H:%M:%S')}] Successfully saved pool data.")
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] Error saving pool data to HDF5: {e}")
                        raise

                elif mode == "load":
                    if not os.path.exists(filepath):
                        print(f"[{time.strftime('%H:%M:%S')}] Error: File not found at {filepath}. Cannot load pool data.")
                        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

                    print(f"[{time.strftime('%H:%M:%S')}] Loading pool data from {filepath} from group '{group_name}'...")
                    try:
                        with h5py.File(filepath, 'r') as hf:
                            if group_name not in hf:
                                print(f"[{time.strftime('%H:%M:%S')}] Group '{group_name}' not found in {filepath}. Pool remains empty.")
                                self.num_pi = 0
                                self.pi_rc_hashes.clear()
                                return

                            pool_group = hf[group_name]
                            num_pi_from_file = pool_group.attrs.get('num_pi', 0)

                            # --- Dimensionality/Consistency Check ---
                            if pool_group.attrs.get('NUM_STAGE2_ROWS') != self.NUM_STAGE2_ROWS or \
                            pool_group.attrs.get('NUM_BOUNDED_VARS') != self.NUM_BOUNDED_VARS or \
                            pool_group.attrs.get('NUM_STAGE2_VARS') != self.NUM_STAGE2_VARS or \
                            pool_group.attrs.get('R_SPARSE_LEN') != self.R_SPARSE_LEN:
                                print(f"[{time.strftime('%H:%M:%S')}] Error: Dimensionality mismatch between HDF5 group '{group_name}' and current instance.")
                                raise ValueError(f"HDF5 group '{group_name}' dimensions do not match current instance configuration.")

                            # --- Clear current pool state before loading ---
                            self.num_pi = 0
                            self.pi_rc_hashes.clear()

                            if num_pi_from_file == 0:
                                print(f"[{time.strftime('%H:%M:%S')}] No data to load from group '{group_name}' (num_pi is 0 in file attributes). Pool is now empty.")
                                return

                            # Determine how many entries to actually load based on current instance's MAX_PI
                            num_pi_to_load = min(num_pi_from_file, self.MAX_PI)
                            if num_pi_from_file > self.MAX_PI:
                                print(f"[{time.strftime('%H:%M:%S')}] Warning: File group '{group_name}' contains {num_pi_from_file} entries, "
                                    f"but current MAX_PI is {self.MAX_PI}. Loading only {self.MAX_PI} entries.")

                            # --- Load data from HDF5 datasets within the group ---
                            # Check if datasets exist before trying to load
                            required_datasets = ["pi_pool", "rc_pool", "vbasis_pool", "cbasis_pool"]
                            for ds_name in required_datasets:
                                if ds_name not in pool_group:
                                    raise KeyError(f"Dataset '{ds_name}' not found in group '{group_name}' in file '{filepath}'.")

                            pi_loaded_np = pool_group["pi_pool"][:num_pi_to_load]
                            rc_loaded_np = pool_group["rc_pool"][:num_pi_to_load]
                            vbasis_loaded_np = pool_group["vbasis_pool"][:num_pi_to_load]
                            cbasis_loaded_np = pool_group["cbasis_pool"][:num_pi_to_load]

                            # --- Populate instance attributes ---
                            pi_tensor_loaded_cpu = torch.from_numpy(pi_loaded_np).to(dtype=self.pi_gpu.dtype)
                            self.pi_gpu[:num_pi_to_load].copy_(pi_tensor_loaded_cpu.to(self.device), non_blocking=is_cuda)

                            rc_tensor_loaded_cpu = torch.from_numpy(rc_loaded_np).to(dtype=self.rc_gpu.dtype)
                            self.rc_gpu[:num_pi_to_load].copy_(rc_tensor_loaded_cpu.to(self.device), non_blocking=is_cuda)

                            self.vbasis_cpu[:num_pi_to_load] = vbasis_loaded_np.astype(self.vbasis_cpu.dtype, copy=False)
                            self.cbasis_cpu[:num_pi_to_load] = cbasis_loaded_np.astype(self.cbasis_cpu.dtype, copy=False)

                            # Reconstruct short_pi_gpu from the loaded pi_gpu data
                            if self.R_SPARSE_LEN > 0 and num_pi_to_load > 0 : # Check num_pi_to_load also
                                loaded_pi_on_device_slice = self.pi_gpu[:num_pi_to_load]
                                # Ensure r_sparse_indices_gpu is on the correct device (it should be from __init__)
                                recomputed_short_pi = loaded_pi_on_device_slice[:, self.r_sparse_indices_gpu]
                                self.short_pi_gpu[:num_pi_to_load].copy_(recomputed_short_pi, non_blocking=is_cuda)
                            elif num_pi_to_load > 0: # R_SPARSE_LEN is 0
                                # Ensure the slice of short_pi_gpu up to num_pi_to_load has shape (num_pi_to_load, 0)
                                # This is implicitly handled by its original allocation and the slice.
                                # If it was non-zero before, it should be zeroed or ensured it reflects (N,0)
                                self.short_pi_gpu[:num_pi_to_load].zero_() # Explicitly zero if necessary for safety
                            
                            # Reconstruct pi_rc_hashes from the loaded pi and rc data
                            pi_np_dtype_for_hash = np.float32 # Match dtype used in add_pi for hashing
                            for i in range(num_pi_to_load):
                                # Use the CPU tensors created just before GPU copy for hashing
                                current_pi_for_hash = pi_tensor_loaded_cpu[i].numpy().astype(pi_np_dtype_for_hash, copy=False)
                                current_rc_for_hash = rc_tensor_loaded_cpu[i].numpy().astype(pi_np_dtype_for_hash, copy=False)
                                try:
                                    pi_reduced_precision = current_pi_for_hash.astype(np.float16)
                                    rc_reduced_precision = current_rc_for_hash.astype(np.float16)
                                    combined_pi_rc = np.concatenate((np.ascontiguousarray(pi_reduced_precision),
                                                                    np.ascontiguousarray(rc_reduced_precision)))
                                    current_hash = hashlib.sha256(combined_pi_rc.tobytes()).hexdigest()
                                    self.pi_rc_hashes.add(current_hash)
                                except Exception as e:
                                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Could not reconstruct hash for loaded entry {i}. Error: {e}")

                            self.num_pi = num_pi_to_load
                        print(f"[{time.strftime('%H:%M:%S')}] Successfully loaded {self.num_pi} entries from pool data under group '{group_name}'.")

                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] Error loading pool data from HDF5 group '{group_name}': {e}")
                        self.num_pi = 0 # Ensure pool is cleared on any load error
                        self.pi_rc_hashes.clear()
                        raise
                else:
                    raise ValueError(f"Invalid mode '{mode}'. Must be 'save' or 'load'.")