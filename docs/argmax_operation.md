# ArgmaxOperation Class Documentation

## Purpose

The `ArgmaxOperation` class is designed to efficiently calculate Benders decomposition optimality cut coefficients ($\alpha$ and $\beta$) on a GPU using CuPy. It operates under the assumption that a set of candidate dual extreme points ($\pi$ vectors) are pre-computed or added incrementally. The calculation involves finding the "best" $\pi$ vector for numerous scenarios ($\omega$) given a first-stage decision ($x$) and then averaging results to form the cut coefficients.

The class incorporates several optimizations:
* **GPU Acceleration:** Leverages CuPy for parallel computation on NVIDIA GPUs.
* **Scenario Batching:** Processes scenarios in batches to manage GPU memory usage, allowing for large numbers of scenarios.
* **Sparse Matrix Handling:** Efficiently handles a sparse transfer matrix `C` using CuPy's sparse formats (CSR).
* **Fixed Sparsity Pattern:** Exploits a fixed sparsity pattern in the stochastic part of the right-hand side ($r(\omega) = \bar{r} + \delta_r(\omega)$) by using packed "short" vectors for relevant calculations.
* **High-Precision Reduction:** Uses `float64` for accumulating sums across batches to improve numerical accuracy.
* **Deduplication:** Avoids storing duplicate $\pi$ vectors.

## Initialization (`__init__`)

* **Inputs:**
    * `PI_DIM`, `X_DIM`: Dimensions of the dual ($\pi$) and primal ($x$) spaces.
    * `MAX_PI`, `MAX_OMEGA`: Pre-allocated storage capacity for $\pi$ vectors and scenarios.
    * `r_sparse_indices`: A NumPy array containing the fixed indices where $\delta_r(\omega)$ is non-zero.
    * `r_bar`: A NumPy array for the constant part of the RHS vector.
    * `C`: The transfer matrix as a SciPy sparse matrix (e.g., CSR or CSC).
    * `scenario_batch_size`: The number of scenarios to process in each GPU batch (defaults to 250,000).
* **Actions:**
    * Validates inputs.
    * Allocates memory on both CPU (NumPy) and GPU (CuPy) for storing $\pi$ vectors (full and short versions) and scenario data (`short_delta_r_gpu`). Most data uses `float32` by default.
    * Stores `r_bar`, `r_sparse_indices` on the GPU.
    * Converts the input SciPy sparse matrix `C` to a CuPy CSR sparse matrix (`C_gpu`) on the GPU.
    * Initializes counters and a set for $\pi$ deduplication.

## Data Loading Methods

* **`add_pi(new_pi)`:**
    * Takes a NumPy array `new_pi`.
    * Checks for duplicates using hashing.
    * If new and capacity allows, stores the full `new_pi` and its corresponding "short" version (elements at `r_sparse_indices`) on both CPU and GPU. Increments `num_pi`.
* **`add_scenarios(new_short_r_delta)`:**
    * Takes a NumPy array `new_short_r_delta` where columns represent scenarios and rows correspond to `r_sparse_indices`.
    * Copies the data to the `short_delta_r_gpu` array on the GPU, handling capacity limits. Increments `num_scenarios`.

## Core Calculation (`calculate_cut`)

* **Input:** A NumPy array `x` representing the first-stage decision.
* **Goal:** Computes $\alpha$ and $\beta$ for the Benders cut $\eta \ge \alpha + \beta^T x$.
* **Process:**
    1.  **Preprocessing:** Gets active views of GPU data, computes the constant `Cx` term using sparse matrix-vector multiplication (SpMV).
    2.  **Initialization:** Initializes `float64` accumulators for sums (`total_selected_pi_sum`, `total_s_sum`) and an integer array for all best $\pi$ indices (`all_best_k_indices`).
    3.  **Scenario Batch Loop:** Iterates through all stored scenarios in batches of size `scenario_batch_size`.
        * **Get Batch Data:** Selects the `short_delta_r` data for the current batch.
        * **Compute `h`:** Reconstructs $r(\omega)$ for the batch using `r_bar` and `batch_short_delta_r`, then computes $h(\omega, x) = r(\omega) - Cx$ for the batch. (Intermediate calculations use `float32`).
        * **Compute Scores:** Calculates $\text{scores}_{k,i} = \pi_k^T h(\omega_i, x)$ for all stored $\pi_k$ and all scenarios $i$ in the batch using matrix multiplication (`float32`).
        * **Find Best Pi:** Determines the index $k^*$ (`best_k_index_batch`) of the $\pi_k$ vector yielding the maximum score for each scenario in the batch using `argmax`. Stores these indices.
        * **Compute `s`:** Calculates $s_i = (\text{short\_pi}_{k^*})^T (\text{short\_delta}_r(\omega_i))$ for each scenario $i$ in the batch. This uses a matrix multiplication between the pre-computed `short_pi` vectors and the `batch_short_delta_r` data, followed by selection based on `best_k_index_batch`. (Intermediate calculations use `float32`).
        * **Accumulate:** Gathers the selected full $\pi_{k^*}$ vectors for the batch. Adds the sum of these vectors and the sum of the $s_i$ values to the respective `float64` accumulators (`total_selected_pi_sum`, `total_s_sum`), specifying `dtype=cp.float64` in the `cp.sum` calls.
        * **Memory Cleanup:** Uses `del` to release large intermediate batch arrays, allowing the memory pool to reuse memory.
    4.  **Final Averages:** After the loop, calculates the final averages `Avg_Pi` and `Avg_S` by dividing the accumulated sums by the total number of scenarios. These averages will be `float64`.
    5.  **Compute Coefficients:**
        * Calculates $\beta = -C^T Avg_Pi$ using the sparse matrix transpose property (`C_gpu.T @ Avg_Pi`). The result is `float64`.
        * Calculates $\alpha = Avg_Pi^T \bar{r} + Avg_S$. The result is `float64`.
* **Output:** Returns `alpha` (Python float), `beta` (NumPy float64 array), and `best_k_index` (NumPy int32 array containing the index of the best $\pi$ vector for every scenario).

