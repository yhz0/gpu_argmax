import cupy as cp
import numpy as np
import scipy.sparse # For input type hint and example
import cupyx.scipy.sparse as cpsparse # For GPU sparse matrices
import hashlib
import time # For basic timing feedback
import math

class ArgmaxOperation:
    """
    Implements GPU-accelerated calculation of Benders cut coefficients (alpha, beta)
    using pre-computed dual extreme points (pi) and scenario data via CuPy.

    Processes scenarios in batches to manage GPU memory usage.
    Uses double precision (float64) for reduction steps to improve accuracy.

    Assumes a fixed sparsity pattern for the stochastic part of the RHS (r_delta),
    allowing for optimized dense operations on relevant sub-vectors.
    Maintains copies of pi vectors on both CPU (NumPy) and GPU (CuPy).
    Uses a sparse matrix representation for the C matrix.

    Attributes:
        PI_DIM (int): Dimension of the dual space (length of pi vectors).
        X_DIM (int): Dimension of the first-stage variable x.
        MAX_PI (int): Maximum number of dual extreme points to store.
        MAX_OMEGA (int): Maximum number of scenarios to store.
        R_SPARSE_LEN (int): Number of non-zero indices in r_delta.
        scenario_batch_size (int): Number of scenarios to process in each GPU batch.
        num_pi (int): Current number of stored pi vectors.
        num_scenarios (int): Current number of stored scenarios.
        pi_cpu (np.ndarray): Stored pi vectors on CPU.
        short_pi_cpu (np.ndarray): Stored short pi vectors on CPU.
        r_sparse_indices_cpu (np.ndarray): Indices for sparse part on CPU.
        pi_hashes (set): Hashes of stored pi vectors for deduplication.
        pi_gpu (cp.ndarray): Stored pi vectors on GPU.
        short_pi_gpu (cp.ndarray): Stored short pi vectors on GPU.
        r_bar_gpu (cp.ndarray): Fixed part of RHS on GPU.
        r_sparse_indices_gpu (cp.ndarray): Indices for sparse part on GPU.
        short_delta_r_gpu (cp.ndarray): Packed sparse RHS parts for scenarios on GPU.
        C_gpu (cpsparse.csr_matrix): Sparse transfer matrix C on GPU (CSR format).
    """

    def __init__(self, PI_DIM: int, X_DIM: int, MAX_PI: int, MAX_OMEGA: int,
                 r_sparse_indices: np.ndarray, r_bar: np.ndarray, C: scipy.sparse.spmatrix,
                 scenario_batch_size: int = 250000): # Added batch size parameter
        """
        Initializes the ArgmaxOperation class. Allocates memory on CPU and GPU.

        Args:
            PI_DIM (int): Dimension of the dual space (length of pi vectors).
            X_DIM (int): Dimension of the first-stage variable x.
            MAX_PI (int): Maximum number of dual extreme points to store.
            MAX_OMEGA (int): Maximum number of scenarios to store.
            r_sparse_indices (np.ndarray): 1D array of indices where r_delta is non-zero.
                                           Should contain unique, sorted indices within [0, PI_DIM).
            r_bar (np.ndarray): 1D array for the fixed part of the RHS (size PI_DIM).
            C (scipy.sparse.spmatrix): The transfer matrix (size PI_DIM x X_DIM) as a SciPy sparse matrix
                                       (e.g., csr_matrix, csc_matrix). It will be stored as CSR on GPU.
            scenario_batch_size (int): Number of scenarios to process in each GPU batch. Defaults to 250,000.
        """
        print(f"[{time.strftime('%H:%M:%S')}] Initializing ArgmaxOperation (Sparse C, Batched, FP64 Reduction)...")
        start_time = time.time()

        # --- Input Validation ---
        if not isinstance(r_sparse_indices, np.ndarray) or r_sparse_indices.ndim != 1:
             raise ValueError("r_sparse_indices must be a 1D NumPy array.")
        if len(np.unique(r_sparse_indices)) != len(r_sparse_indices):
             print("Warning: r_sparse_indices contains duplicate values. Using unique indices.")
             r_sparse_indices = np.unique(r_sparse_indices)
        if np.any(r_sparse_indices < 0) or np.any(r_sparse_indices >= PI_DIM):
             raise ValueError("r_sparse_indices contain indices out of bounds for PI_DIM.")
        if r_bar.shape != (PI_DIM,):
             raise ValueError(f"r_bar shape mismatch. Expected ({PI_DIM},), got {r_bar.shape}")
        # Validate sparse matrix C
        if not scipy.sparse.issparse(C):
             raise TypeError("C must be a SciPy sparse matrix (e.g., csr_matrix).")
        if C.shape != (PI_DIM, X_DIM):
            raise ValueError(f"Sparse C matrix shape mismatch. Expected ({PI_DIM}, {X_DIM}), got {C.shape}")
        if not isinstance(scenario_batch_size, int) or scenario_batch_size <= 0:
             raise ValueError("scenario_batch_size must be a positive integer.")


        # --- Dimensions and Capacities ---
        self.PI_DIM = PI_DIM
        self.X_DIM = X_DIM
        self.MAX_PI = MAX_PI
        self.MAX_OMEGA = MAX_OMEGA
        self.scenario_batch_size = scenario_batch_size
        # Ensure indices are sorted for potential future optimizations & consistency
        self.r_sparse_indices_cpu = np.sort(np.array(r_sparse_indices, dtype=np.int32))
        self.R_SPARSE_LEN = len(self.r_sparse_indices_cpu)

        # --- Counters ---
        self.num_pi = 0
        self.num_scenarios = 0

        # --- CPU Data Storage ---
        print(f"[{time.strftime('%H:%M:%S')}] Allocating CPU memory (MAX_PI={MAX_PI}, PI_DIM={PI_DIM})...")
        # Use float32 if precision allows, often faster on GPU
        dtype = np.float32 # Default dtype for most data
        self.pi_cpu = np.zeros((MAX_PI, PI_DIM), dtype=dtype)
        self.short_pi_cpu = np.zeros((MAX_PI, self.R_SPARSE_LEN), dtype=dtype)
        self.pi_hashes = set() # For deduplication

        # --- GPU Data Storage ---
        # Assumes the correct GPU device has been selected *before* this call if done programmatically
        print(f"[{time.strftime('%H:%M:%S')}] Allocating GPU memory on device {cp.cuda.runtime.getDevice()}...")
        cp_dtype = cp.float32 # Default dtype for most GPU arrays
        # Pi vectors (full and short)
        self.pi_gpu = cp.zeros((MAX_PI, PI_DIM), dtype=cp_dtype)
        self.short_pi_gpu = cp.zeros((MAX_PI, self.R_SPARSE_LEN), dtype=cp_dtype)

        # RHS data
        self.r_bar_gpu = cp.asarray(r_bar, dtype=cp_dtype)
        self.r_sparse_indices_gpu = cp.asarray(self.r_sparse_indices_cpu) # Indices on GPU

        # Scenario data (short/packed r_delta)
        # Shape: (num_sparse_indices, num_scenarios)
        self.short_delta_r_gpu = cp.zeros((self.R_SPARSE_LEN, MAX_OMEGA), dtype=cp_dtype)

        # C matrix (Sparse)
        print(f"[{time.strftime('%H:%M:%S')}] Converting C to sparse CSR format on GPU...")
        # Ensure input sparse matrix C has the target dtype before conversion
        if C.dtype != dtype:
             C = C.astype(dtype)
        # Convert the input SciPy sparse matrix to CuPy CSR sparse matrix
        self.C_gpu = cpsparse.csr_matrix(C, dtype=cp_dtype)
        # No need to store Ct_gpu explicitly, use C_gpu.T when needed

        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Initialization complete ({end_time - start_time:.2f}s).")


    def add_pi(self, new_pi: np.ndarray) -> bool:
        """
        Adds a new dual extreme point (pi vector), performs deduplication,
        and updates both CPU and GPU storage.

        Args:
            new_pi (np.ndarray): The new pi vector (size PI_DIM).

        Returns:
            bool: True if the pi was added successfully, False otherwise (e.g., duplicate or max capacity reached).
        """
        if self.num_pi >= self.MAX_PI:
            return False

        if new_pi.shape != (self.PI_DIM,):
            print(f"Error: new_pi has incorrect shape. Expected ({self.PI_DIM},), got {new_pi.shape}")
            return False

        new_pi = new_pi.astype(self.pi_cpu.dtype, copy=False) # Ensure consistent dtype (float32)

        try:
             pi_hash = hashlib.sha256(np.ascontiguousarray(new_pi).tobytes()).hexdigest()
             if pi_hash in self.pi_hashes:
                 return False
             self.pi_hashes.add(pi_hash)
        except Exception as e:
             print(f"Warning: Could not hash pi vector for deduplication. Adding anyway. Error: {e}")

        idx = self.num_pi
        self.pi_cpu[idx] = new_pi
        short_new_pi = new_pi[self.r_sparse_indices_cpu]
        self.short_pi_cpu[idx] = short_new_pi

        # Explicitly convert NumPy array to CuPy array before assignment
        # Keep these as float32 on GPU for memory/speed of main calculations
        self.pi_gpu[idx] = cp.asarray(new_pi, dtype=cp.float32)
        self.short_pi_gpu[idx] = cp.asarray(short_new_pi, dtype=cp.float32)

        self.num_pi += 1
        return True

    def add_scenarios(self, new_short_r_delta: np.ndarray) -> bool:
        """
        Adds new scenario data (packed r_delta values corresponding to r_sparse_indices).

        Args:
            new_short_r_delta (np.ndarray): Array of shape (R_SPARSE_LEN, num_new_scenarios)
                                             containing the packed r_delta values.

        Returns:
            bool: True if scenarios were added successfully (partially or fully), False otherwise.
        """
        if new_short_r_delta.ndim != 2 or new_short_r_delta.shape[0] != self.R_SPARSE_LEN:
             raise ValueError(f"new_short_r_delta shape mismatch. Expected ({self.R_SPARSE_LEN}, num_new_scenarios), "
                              f"got {new_short_r_delta.shape}")

        num_new_scenarios = new_short_r_delta.shape[1]
        available_slots = self.MAX_OMEGA - self.num_scenarios

        if num_new_scenarios <= 0:
             print("Warning: Received request to add zero scenarios.")
             return False

        if available_slots <= 0:
            print(f"Warning: Scenario capacity ({self.MAX_OMEGA}) reached. Cannot add more scenarios.")
            return False

        num_to_add = min(num_new_scenarios, available_slots)

        if num_to_add < num_new_scenarios:
             print(f"Warning: Exceeds MAX_OMEGA ({self.MAX_OMEGA}). Adding only {num_to_add} of {num_new_scenarios} new scenarios.")
             new_short_r_delta = new_short_r_delta[:, :num_to_add]

        start_col = self.num_scenarios
        end_col = self.num_scenarios + num_to_add

        # Ensure dtype matches GPU array before assignment (float32)
        new_short_r_delta_gpu = cp.asarray(new_short_r_delta, dtype=self.short_delta_r_gpu.dtype)
        self.short_delta_r_gpu[:, start_col:end_col] = new_short_r_delta_gpu

        self.num_scenarios += num_to_add
        return True

    def calculate_cut(self, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
        """
        Performs the core calculation for a given x: selects the best pi_k for
        each scenario (processed in batches) and computes the Benders cut
        coefficients alpha and beta. Uses float64 for reduction sums.

        Args:
            x (np.ndarray): The first-stage decision vector (size X_DIM).

        Returns:
            tuple[float, np.ndarray, np.ndarray] | None:
                - alpha (float): The constant part of the Benders cut (float64 accuracy).
                - beta (np.ndarray): The coefficient vector of the Benders cut (size X_DIM, float64 accuracy).
                - best_k_index (np.ndarray): Indices of the best pi_k for each scenario (size num_scenarios).
            Returns None if there are no pi vectors or no scenarios stored.
        """
        calc_start_time = time.time()
        if self.num_pi == 0:
            print("Error: No pi vectors stored. Cannot calculate cut.")
            return None
        if self.num_scenarios == 0:
            print("Error: No scenarios stored. Cannot calculate cut.")
            return None
        if x.shape != (self.X_DIM,):
             raise ValueError(f"x shape mismatch. Expected ({self.X_DIM},), got {x.shape}")

        # --- Prepare GPU data views (slices of the pre-allocated arrays) ---
        # Keep main data as float32 for memory/speed efficiency in batch calcs
        active_pi_gpu = self.pi_gpu[:self.num_pi]
        active_short_pi_gpu = self.short_pi_gpu[:self.num_pi]
        active_short_delta_r_gpu = self.short_delta_r_gpu[:, :self.num_scenarios]
        # Ensure x has the correct dtype for sparse matrix multiplication
        x_gpu = cp.asarray(x, dtype=self.C_gpu.dtype)

        # --- Precompute terms constant across batches ---
        Cx_gpu = self.C_gpu @ x_gpu # (PI_DIM,) - SpMV, result is float32

        # --- Initialize accumulators and result arrays ---
        # Use float64 for accumulators to maintain precision over many additions
        total_selected_pi_sum = cp.zeros(self.PI_DIM, dtype=cp.float64)
        total_s_sum = cp.zeros((), dtype=cp.float64) # Use 0-dim array for scalar sum on GPU
        all_best_k_indices = cp.zeros(self.num_scenarios, dtype=cp.int32) # Store all best indices

        # --- Process scenarios in batches ---
        num_batches = math.ceil(self.num_scenarios / self.scenario_batch_size)
        print(f"[{time.strftime('%H:%M:%S')}] Processing {self.num_scenarios} scenarios in {num_batches} batches of size {self.scenario_batch_size}...")

        mempool = cp.get_default_memory_pool()
        initial_used_bytes = mempool.used_bytes()

        for i in range(num_batches):
            # batch_start_time = time.time() # Removed timing comment
            start_idx = i * self.scenario_batch_size
            end_idx = min((i + 1) * self.scenario_batch_size, self.num_scenarios)
            current_batch_size = end_idx - start_idx

            # --- Get data for the current batch (float32) ---
            batch_short_delta_r = active_short_delta_r_gpu[:, start_idx:end_idx]

            # --- Step 1 (Batch): Compute h(omega_i, x) = r(omega_i) - Cx ---
            # Keep batch calculations in float32 for speed/memory
            delta_r_gpu_batch = cp.zeros((self.PI_DIM, current_batch_size), dtype=cp.float32)
            delta_r_gpu_batch[self.r_sparse_indices_gpu, :] = batch_short_delta_r

            # Ensure r_bar_gpu is broadcast correctly (it's float32)
            r_gpu_batch = self.r_bar_gpu[:, cp.newaxis] + delta_r_gpu_batch

            # Ensure Cx_gpu is broadcast correctly (it's float32)
            h_gpu_batch = r_gpu_batch - Cx_gpu[:, cp.newaxis]
            del delta_r_gpu_batch, r_gpu_batch

            # --- Step 2 (Batch): Select Best pi_k for each Scenario ---
            # Compute scores_batch = pi^T * h_batch (float32 @ float32 -> float32)
            scores_batch = active_pi_gpu @ h_gpu_batch
            del h_gpu_batch

            best_k_index_batch = cp.argmax(scores_batch, axis=0)
            del scores_batch

            all_best_k_indices[start_idx:end_idx] = best_k_index_batch

            # --- Step 3 (Batch): Compute Scenario-Specific Dot Products s_i ---
            # s_all_batch = short_pi^T * short_delta_r (float32 @ float32 -> float32)
            s_all_gpu_batch = active_short_pi_gpu @ batch_short_delta_r
            del batch_short_delta_r

            s_gpu_batch = s_all_gpu_batch[best_k_index_batch, cp.arange(current_batch_size)]
            del s_all_gpu_batch


            # --- Step 4 (Batch): Accumulate Results ---
            # Gather the full pi vectors for the batch (float32)
            selected_pi_batch = active_pi_gpu[best_k_index_batch]
            del best_k_index_batch

            # Accumulate sums using float64 for higher precision
            total_selected_pi_sum += cp.sum(selected_pi_batch, axis=0, dtype=cp.float64)
            total_s_sum += cp.sum(s_gpu_batch, dtype=cp.float64)
            del selected_pi_batch, s_gpu_batch

            # --- Clear intermediate batch arrays ---
            # Explicitly calling del helps Python's GC.
            # mempool.free_all_blocks() # Removed: Let memory pool manage reuse between batches for efficiency.

            # batch_end_time = time.time() # Removed timing comment


        # --- Calculate final averages (will be float64) ---
        if self.num_scenarios > 0:
            Avg_Pi = total_selected_pi_sum / self.num_scenarios
            Avg_S = total_s_sum / self.num_scenarios
        else:
            Avg_Pi = cp.zeros(self.PI_DIM, dtype=cp.float64)
            Avg_S = cp.zeros((), dtype=cp.float64)


        # --- Step 5: Final Coefficient Calculation (using float64 averages) ---
        # Ensure r_bar_gpu is compatible; cast it if needed for the dot product
        r_bar_gpu_fp64 = self.r_bar_gpu.astype(cp.float64, copy=False)
        # beta = -C^T * Avg_Pi (sparse float32.T @ float64 -> float64)
        beta_gpu = -self.C_gpu.T @ Avg_Pi
        # alpha = Avg_Pi^T * r_bar + Avg_S (float64 @ float64 + float64 -> float64)
        alpha_gpu = Avg_Pi @ r_bar_gpu_fp64 + Avg_S


        # --- Return results (transfer back to CPU) ---
        # Alpha and Beta will be float64
        alpha = alpha_gpu.item()
        beta = cp.asnumpy(beta_gpu)
        best_k_index = cp.asnumpy(all_best_k_indices) # int32

        calc_end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Cut calculation finished ({calc_end_time - calc_start_time:.4f}s)")
        print(f"    Final VRAM used: {mempool.used_bytes() / 1024**3:.2f} GB")


        return alpha, beta, best_k_index

# Example Usage (requires valid inputs)
if __name__ == '__main__':
    # --- Define Parameters ---
    _PI_DIM = 100
    _X_DIM = 10
    _MAX_PI = 10000
    _MAX_OMEGA = 1000000
    _SCENARIO_BATCH_SIZE = 250000
    _NUM_SPARSE = 15
    _C_DENSITY = 0.01
    _dtype = np.float32 # Main dtype for storage/batch calcs

    # --- Set GPU Device ---
    try:
        _TARGET_GPU_ID = 4
        print(f"Attempting to select GPU {_TARGET_GPU_ID}...")
        cp.cuda.Device(_TARGET_GPU_ID).use()
        print(f"Successfully selected GPU {_TARGET_GPU_ID}")
    except Exception as e:
        print(f"Could not select GPU {_TARGET_GPU_ID}. Using default or available GPUs. Error: {e}")
        # exit()


    print("--- Example Usage (Sparse C, Batched, FP64 Reduction) ---")

    # --- Generate Mock Data ---
    print("Generating mock data...")
    np.random.seed(42)
    _r_sparse_indices = np.random.choice(np.arange(_PI_DIM), _NUM_SPARSE, replace=False)
    _r_bar = np.random.rand(_PI_DIM).astype(_dtype)
    _C_sparse = scipy.sparse.random(_PI_DIM, _X_DIM, density=_C_DENSITY, format='csr', dtype=_dtype)
    print(f"Generated sparse C matrix with shape {_C_sparse.shape} and { _C_sparse.nnz} non-zero elements (dtype={_C_sparse.dtype}).")


    # --- Initialize Class ---
    try:
        calculator = ArgmaxOperation(
            PI_DIM=_PI_DIM,
            X_DIM=_X_DIM,
            MAX_PI=_MAX_PI,
            MAX_OMEGA=_MAX_OMEGA,
            r_sparse_indices=_r_sparse_indices,
            r_bar=_r_bar,
            C=_C_sparse,
            scenario_batch_size=_SCENARIO_BATCH_SIZE
        )

        # --- Add some Pi vectors ---
        num_pi_to_add = min(10000, _MAX_PI) # Use full capacity for test
        print(f"\nAdding {num_pi_to_add} mock pi vectors (Capacity: {_MAX_PI})...")
        added_pi_count = 0
        for i in range(num_pi_to_add):
             pi_vec = (np.random.rand(_PI_DIM) * 10 - 5).astype(_dtype)
             if calculator.add_pi(pi_vec):
                 added_pi_count += 1
        print(f"Added {added_pi_count} unique pi vectors.")


        # --- Add Scenarios ---
        num_scenarios_to_add = min(1000000, _MAX_OMEGA) # Use full capacity
        print(f"\nAdding {num_scenarios_to_add} mock scenarios (Capacity: {_MAX_OMEGA})...")
        _new_short_r_delta = (np.random.randn(_NUM_SPARSE, num_scenarios_to_add) * 0.1).astype(_dtype)
        if calculator.add_scenarios(_new_short_r_delta):
             print(f"Successfully added scenarios. Total scenarios: {calculator.num_scenarios}")
        else:
             print("Failed to add scenarios.")


        # --- Calculate Cut ---
        if calculator.num_pi > 0 and calculator.num_scenarios > 0:
             print("\nCalculating cut for a random x...")
             _x = np.random.rand(_X_DIM).astype(_dtype)
             result = calculator.calculate_cut(_x)

             if result:
                 alpha, beta, best_k_index = result
                 print(f"\n--- Results ---")
                 # Note: alpha/beta are now likely float64
                 print(f"Alpha: {alpha:.8f} (dtype: {type(alpha)})")
                 print(f"Beta shape: {beta.shape} (dtype: {beta.dtype})")
                 print(f"Best K Index shape: {best_k_index.shape} (dtype: {best_k_index.dtype})")
                 print("-" * 15)
             else:
                 print("Cut calculation failed.")
        else:
             print("\nSkipping cut calculation as no pi vectors or scenarios were added.")

    except ValueError as e:
        print(f"\nInitialization Error: {e}")
    except TypeError as e:
         print(f"\nType Error during Initialization (check sparse C?): {e}")
    except ImportError as e:
        print(f"\nImport Error: {e}. Please ensure SciPy and CuPy are installed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
