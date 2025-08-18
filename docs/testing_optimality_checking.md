# Procedure to test optimality checking functionality.

The goal is to develop some tests on the optimality checking functionality of the ArgmaxOperations class.

Please read @docs/optimality_checking.md to see the background information about this procedure.

0. Load CEP instance. See @tests/test_argmax.py on how to load such instance.
1. Load the scenarios from cep_100scen_results. Load optimal x.
2. Use the loaded scenarios to initialize the ArgmaxOperation class, with optimality check enabled.
3. Solve all scenario subproblems with the x. Record second stage objective values. Take a look at @src/second_stage_worker.py on how to do so.
4. Add the FIRST 3 resulting dual solutions and basis information to the ArgmaxOperation instance. (We artificially hide the other 97 dual solutions, so only the first 3 are used in the argmax procedure.)
5. Run argmax procedure using find_best_k at the given x.
6. ASSERT (sanity check): Use get_best_k_results. the scores of the first three scenarios match the objective values in step 3. Note since single FP is used, you may need to adjust tolerance as needed.
7. Compare the scores of All scenarios with the objective values. This is the ground truth of `optimality` we will use later.
8. Run check_optimality, store the results.
9. ASSERT: the first 3 scenarios are optimal.
10. ASSERT: the remaining scenarios matches the ground truth of `optimality` in step 7. Note: you may need to adjust tolerance.