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


/**
 * @brief Single benchmark run for specified N and M values.
 */
void run_single_benchmark(int N, int M, int num_replications) {
    double total_elapsed_time_ms = 0.0;
    
    printf("Running C Reference Benchmark (N=%d, M=%d, replications=%d)\n", N, M, num_replications);
    printf("Problem dimensions: m_2=%d, K=%d\n", M2, K);
    
    // Run the experiment multiple times to get a stable average
    for (int i = 0; i < num_replications; ++i) {
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

        double elapsed_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
        total_elapsed_time_ms += elapsed_ms;
        
        if (i == 0) {
            printf("First run completed in %.2f ms\n", elapsed_ms);
        }

        // Clean up memory from this run
        free_data(N, M, deltaOmega, pi, I_S, max_prod_index_all);
    }

    double avg_time_ms = total_elapsed_time_ms / num_replications;
    double throughput_scenarios_per_sec = (N * num_replications * 1000.0) / total_elapsed_time_ms;
    
    printf("\nResults:\n");
    printf("  Average time: %.2f ms\n", avg_time_ms);
    printf("  Total time: %.2f ms\n", total_elapsed_time_ms);
    printf("  Throughput: %.0f scenarios/sec\n", throughput_scenarios_per_sec);
    printf("  Argmax operations: %d\n", N * num_replications);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc != 3 && argc != 4) {
        printf("Usage: %s <N_scenarios> <M_dual_vectors> [num_replications]\n", argv[0]);
        printf("  N_scenarios: Number of scenarios to process\n");
        printf("  M_dual_vectors: Number of dual vectors to compare\n");
        printf("  num_replications: Number of benchmark runs (default: %d)\n", NUM_REPLICATIONS);
        printf("\nExample: %s 1000 100\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int num_replications = (argc == 4) ? atoi(argv[3]) : NUM_REPLICATIONS;
    
    // Validate arguments
    if (N <= 0 || M <= 0 || num_replications <= 0) {
        printf("Error: All arguments must be positive integers\n");
        return 1;
    }
    
    if (N > 100000 || M > 10000) {
        printf("Warning: Large problem size may take significant time and memory\n");
    }
    
    // Seed the random number generator
    srand(time(NULL));
    
    // Run the single benchmark
    run_single_benchmark(N, M, num_replications);
    
    return 0;
}