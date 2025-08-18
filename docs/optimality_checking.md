# GPU-Based Optimality Verification for Second-Stage LP Solutions (Batched LU Factorization Method)

This document specifies the procedure for verifying the optimality of a proposed basis for the second-stage LP problem. The strategy utilizes **LU factorization**, and the online verification is performed on **batches of scenarios**.

The core assumption remains: the proposed basis is **dual feasible**, so optimality is confirmed by verifying **primal feasibility**.

---

## 1. Problem Formulation

The second-stage problem for a given first-stage decision $x$ and scenario $\omega$ is defined as:

$$Q(x, \omega) = \min_{y} \quad d^T y$$

$$\text{s.t.} \quad Dy \ge h(\omega)$$

$$lb_y \le y \le ub_y$$

where $h(\omega) = r(\omega) - T x$. We convert this to standard form $Az = h(\omega)$ by introducing surplus variables $s \ge 0$, where $A = [D, -I]$ and $z = [y^T, s^T]^T$.

---

## 2. Dynamic Basis Factorization and Caching

As the primary optimization algorithm (e.g., a decomposition method) iteratively discovers new dual extreme points, each corresponding to a new candidate basis, the following **one-time computation** is performed for each new basis. The results are cached in **main system memory (RAM)** for future use.

Upon discovering a new basis `j`:

1.  **Partition**: Partition the columns of $A$ into the **basis matrix** $B_j \in \mathbb{R}^{m_2 \times m_2}$ and the non-basic matrix $N_j$.

2.  **Perform LU Factorization**: Compute the LU decomposition of the new basis matrix $B_j$. This yields a permutation matrix (or pivot vector) $P_j$, a unit lower triangular matrix $L_j$, and an upper triangular matrix $U_j$ such that:
    $$P_j B_j = L_j U_j$$
    Store these factors ($P_j, L_j, U_j$) in a cache in RAM, indexed by `j`.

3.  **Cache Non-basic Contribution**: Pre-compute and cache the constant vector $c_j = N_j z_{N_j}$, where $z_{N_j}$ are the non-basic variables fixed at their bounds.

4.  **Cache Basic Variable Bounds**: Cache the corresponding lower and upper bounds, $lb_{B_j}$ and $ub_{B_j}$, for the variables that are basic in this basis.

---

## 3. Batched Online Verification (per Batch of Scenarios on GPU)

The total set of scenarios is processed in sequential batches. For each batch $S_b$, the following steps are performed.

1.  **Propose Bases for Batch**: For every scenario $\omega_i \in S_b$, the argmax procedure returns the index $j_i = \pi^*(\omega_i)$ of a proposed optimal basis.

2.  **Identify and Transfer Unique Factors**:
    * Aggregate the set of **unique** basis indices required for the entire batch: `J_b = unique({j_i | ω_i ∈ S_b})`.
    * Perform a single, consolidated data transfer of the cached LU factors (`P_j, L_j, U_j`) for **every** $j \in J_b$ from main memory (RAM) to the GPU's VRAM.

3.  **Batched Calculation of Effective RHS**: For every scenario $\omega_i$ in the batch, compute the effective right-hand-side vector $v_i = h(\omega_i) - c_{j_i}$. This is executed as a large, parallel operation on the GPU.

4.  **Batched Solve for Basic Variables**: Solve the linear systems $B_{j_i} z_{B_{j_i}} = v_i$ for all scenarios in the batch simultaneously.
    * Each individual solve operation within the batch uses the appropriate LU factors (indexed by $j_i$) that are now resident in VRAM.
    * This is executed using highly efficient **batched triangular solvers**:
        * **Forward Substitution**: First, solve $L_{j_i} y_i = P_{j_i} v_i$ for all intermediate vectors $y_i$.
        * **Backward Substitution**: Then, solve $U_{j_i} z_{B_{j_i}} = y_i$ for all final basic solutions $z_{B_{j_i}}$.

5.  **Verify Primal Feasibility in Batch**: Check if each calculated solution $z_{B_{j_i}}$ respects its corresponding bounds $lb_{B_{j_i}}$ and $ub_{B_{j_i}}$ within a numerical tolerance $\epsilon$.

6.  **Determine Optimality**: If the feasibility check passes for a given scenario, its proposed basis is confirmed as **optimal**. The process then continues to the next batch of scenarios.