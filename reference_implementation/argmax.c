#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- Problem Dimensions ---
// m_2: Full dimension of the second-stage dual vectors
// K: Number of stochastic elements (dimension of the sparse part)
const int M2 = 175;
const int K = 86;
const int NUM_REPLICATIONS = 10;

/**
 * @brief Generates random data for the experiment.
 *
 * @param N Number of scenarios.
 * @param M Number of dual vectors.
 * @param deltaOmega_ptr Pointer to the DeltaOmega matrix (N x K), will be allocated.
 * @param pi_ptr Pointer to the Pi matrix (M x m_2), will be allocated.
 * @param I_S_ptr Pointer to the sparsity index array (K), will be allocated.
 */
void generate_data(int N, int M, float*** deltaOmega_ptr, float*** pi_ptr, int** I_S_ptr) {
    // Allocate DeltaOmega (N x K)
    float** deltaOmega = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; ++i) {
        deltaOmega[i] = (float*)malloc(K * sizeof(float));
        for (int j = 0; j < K; ++j) {
            deltaOmega[i][j] = (float)rand() / RAND_MAX;
        }
    }

    // Allocate Pi (M x m_2)
    float** pi = (float**)malloc(M * sizeof(float*));
    for (int i = 0; i < M; ++i) {
        pi[i] = (float*)malloc(M2 * sizeof(float));
        for (int j = 0; j < M2; ++j) {
            pi[i][j] = (float)rand() / RAND_MAX;
        }
    }

    // Allocate and define sparsity pattern I_S
    int* I_S = (int*)malloc(K * sizeof(int));
    for (int i = 0; i < K; ++i) {
        I_S[i] = M2 - K + i;
    }

    *deltaOmega_ptr = deltaOmega;
    *pi_ptr = pi;
    *I_S_ptr = I_S;
}


/**
 * @brief The reference CPU implementation of the argmax procedure.
 */
void argmax_cpu_reference(int N, int M, float** deltaOmega, float** pi, const int* I_S, int* max_prod_index_all) {
    for (int n = 0; n < N; ++n) {
        float max_prod = -1.0f / 0.0f;
        int max_prod_index = -1;

        for (int m = 0; m < M; ++m) {
            float prod = 0.0f;
            for (int k = 0; k < K; ++k) {
                prod += deltaOmega[n][k] * pi[m][I_S[k]];
            }

            if (prod > max_prod) {
                max_prod = prod;
                max_prod_index = m;
            }
        }
        max_prod_index_all[n] = max_prod_index;
    }
}

/**
 * @brief Frees all dynamically allocated memory for one run.
 */
void free_data(int N, int M, float** deltaOmega, float** pi, int* I_S, int* max_prod_index_all) {
    for (int i = 0; i < N; ++i) {
        free(deltaOmega[i]);
    }
    free(deltaOmega);

    for (int i = 0; i < M; ++i) {
        free(pi[i]);
    }
    free(pi);

    free(I_S);
    free(max_prod_index_all);
}


int main() {
    // Seed the random number generator once for the entire suite of experiments
    srand(time(NULL));

    printf("Starting Experiment Suite...\n");
    printf("Reference C Implementation (m_2=%d, K=%d)\n", M2, K);
    printf("Averaging over %d runs for each instance size.\n\n", NUM_REPLICATIONS);

    // Print a header for the results table
    printf("+-----------------+-----------------+--------------------+\n");
    printf("| N (Scenarios)   | M (Dual Vecs)   | Avg Time (ms)      |\n");
    printf("+-----------------+-----------------+--------------------+\n");

    // Loop through the instance sizes, doubling N and M each time
    for (int N = 100, M = 10; N <= 10000; N *= 2, M *= 2) {
        double total_elapsed_time_ms = 0.0;

        // Run the experiment multiple times to get a stable average
        for (int i = 0; i < NUM_REPLICATIONS; ++i) {
            float** deltaOmega = NULL;
            float** pi = NULL;
            int* I_S = NULL;

            // Generate fresh data for this specific run
            generate_data(N, M, &deltaOmega, &pi, &I_S);
            int* max_prod_index_all = (int*)malloc(N * sizeof(int));

            // --- Start Timing ---
            clock_t start = clock();
            argmax_cpu_reference(N, M, deltaOmega, pi, I_S, max_prod_index_all);
            clock_t end = clock();
            // --- End Timing ---

            total_elapsed_time_ms += ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

            // Clean up memory from this run
            free_data(N, M, deltaOmega, pi, I_S, max_prod_index_all);
        }

        double avg_time_ms = total_elapsed_time_ms / NUM_REPLICATIONS;

        // Print the formatted results for this (N, M) pair
        printf("| %-15d | %-15d | %-18.2f |\n", N, M, avg_time_ms);
    }

    printf("+-----------------+-----------------+--------------------+\n");
    printf("\nExperiment suite complete.\n");

    return 0;
}