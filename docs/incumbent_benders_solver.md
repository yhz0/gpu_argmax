# Incumbent-Based Regularized Benders Decomposition

## Mathematical Summary of `run_incumbent_benders_solver.py`

### **Algorithm**: Incumbent-Based Regularized Benders Decomposition

This implements a **trust-region variant** of Benders decomposition with an **incumbent strategy** for solving two-stage stochastic linear programs.

### **Problem Structure**
- **Master Problem**: $\min_{x,\eta} \quad c^T x + \eta$ subject to Benders cuts
- **Subproblems**: $Q(x,\omega) = \min_y \quad d^T y$ subject to $Dy \leq r(\omega) - Cx$

### **Key Mathematical Components**

#### **1. Regularized Master Problem**
The master problem uses **quadratic regularization** around an incumbent solution:
$$\min_{x,\eta} \quad c^T x + \eta + \frac{\rho}{2}\|x - x_{\text{incumbent}}\|^2$$

- **Regularization parameter**: $\rho$ (adaptive)
- **Incumbent**: $x_{\text{incumbent}}$ (best feasible solution found)

#### **2. Trust-Region Mechanism**
Uses **filter-based acceptance criteria**:
- **Perceived decrease**: $\Delta_{\text{pred}} = f(x_{\text{incumbent}}) - f(x_{\text{candidate}})$
- **Actual decrease**: $\Delta_{\text{actual}} = \text{true objective change}$
- **Success criterion**: $\Delta_{\text{actual}} \geq \gamma \cdot \Delta_{\text{pred}}$ (where $\gamma \in (0,1)$)

**Adaptive $\rho$ Update**:
- If **successful step**: $x_{\text{incumbent}} \leftarrow x_{\text{candidate}}$, $\rho \leftarrow \max(\rho_{\min}, \rho \cdot \alpha_{\text{dec}})$
- If **unsuccessful step**: $\rho \leftarrow \rho \cdot \alpha_{\text{inc}}$

#### **3. Hybrid Subproblem Strategy**
**Two-phase approach per iteration**:

**Phase A - Selective LP Solving with Optimality Screening**:
- Select subset $S \subseteq \{1,\ldots,|\Omega|\}$ of scenarios (systematic rotation)
- For each scenario $\omega_i \in S$:
  1. **Top-k Selection**: Find $k$ best dual solutions: $\{k_1, k_2, \ldots, k_K\} = \text{top-k}\{\text{Score}(j,i)\}_{j=1}^{\text{stored}}$
  2. **Feasibility Check**: For each candidate $k_j$, solve $B_{k_j} z_B = r(\omega_i) - Cx - N_{k_j} z_N$ and verify $l_B \leq z_B \leq u_B$
  3. **Optimality Screening**: If feasible solution found, mark scenario as "optimal" and skip LP solve
  4. **Warmstart Basis**: Use basis from first feasible candidate for LP solve (if needed)
- Only solve LP subproblems for scenarios deemed non-optimal
- Add new dual solutions from LP solves to argmax database

**Phase B - Fast Cut Generation**:
- Run fast argmax over **all scenarios**: $\pi_{\text{indices}} = \arg\max_{k} \text{Score}(k,\omega_i) \quad \forall i$
- Generate Benders cut: $\eta \geq \bar{\alpha} + \bar{\beta}^T x$ where:
  $$\bar{\alpha} = \mathbb{E}[\pi_{k^*}^T \bar{r}] + \mathbb{E}[\pi_{k^*}^T \delta r_i] - \mathbb{E}[\lambda_{k^*}^T l] + \mathbb{E}[\mu_{k^*}^T u]$$
  $$\bar{\beta} = -C^T \mathbb{E}[\pi_{k^*}]$$

#### **4. GPU-Accelerated Argmax Operation**
**Core computation** for each $(k,\omega_i)$ pair:
$$\text{Score}(k,i) = \pi_k^T(r(\omega_i) - Cx) - \lambda_k^T l + \mu_k^T u$$

Where $\lambda_k = \max(0, RC_k)$ and $\mu_k = \max(0, -RC_k)$ from reduced costs.

#### **5. Optimality Screening via Feasibility Verification**
For each scenario in the selected subset, performs **top-k feasibility checking**:

1. **Rank candidates**: Sort stored duals by score $\text{Score}(k,\omega_i)$ 
2. **Sequential feasibility check**: For $j = 1, 2, \ldots, K$:
   - Solve: $B_{k_j} z_B = r(\omega_i) - Cx - N_{k_j} z_N$
   - Check: $l_B \leq z_B \leq u_B$ (within tolerance)
   - If feasible: declare scenario "optimal", use basis $B_{k_j}$ for warmstart (if LP needed)
   - If infeasible: try next candidate

This **optimality screening** reduces computational cost by avoiding LP solves for scenarios where a stored dual solution is already feasible (hence optimal).

### **Algorithmic Flow**

```
For each iteration k:
  1. Solve regularized master: min c^T x + η + (ρ/2)||x - x_incumbent||²
  2. Select scenario subset S ⊆ Ω (systematic rotation)  
  3. Optimality screening: 
     - For each ω_i ∈ S: find top-K dual candidates by score
     - Check feasibility: solve B_k z_B = r(ω_i) - Cx - N_k z_N
     - Mark feasible scenarios as "optimal" (skip LP solve)
  4. LP solve: Solve subproblems only for non-optimal scenarios with warmstart basis
  5. Update: Add new (π, RC, basis) from LP solves to argmax database  
  6. Fast cut: Run argmax over ALL scenarios, generate cut
  7. Trust region: Accept/reject x_candidate, update ρ and x_incumbent
```

### **Key Mathematical Innovations**

1. **Optimality Screening**: Top-k feasibility checking to avoid redundant LP solves
2. **Separation of Concerns**: Selective LP solving (subset) vs. cut generation (all scenarios)  
3. **Trust-Region Control**: Adaptive regularization with incumbent tracking
4. **Dual Solution Caching**: LRU-managed database of $(\pi, \lambda, \mu, \text{basis})$ tuples
5. **Scenario Rotation**: Systematic coverage ensuring all scenarios eventually processed
6. **Hybrid Precision**: Float32 for argmax, Float64 for cut coefficients

This approach **balances computational cost** (expensive LP solves on subsets) with **cut quality** (fast argmax over full scenario set) while maintaining **convergence** through the trust-region framework.

## Implementation Details

### **Configuration Parameters**

The algorithm is controlled by several key parameters:

```python
config = {
    # Trust-region parameters
    'initial_rho': 1.0,              # Initial regularization strength
    'rho_decrease_factor': 0.5,      # α_dec: decrease factor for successful steps
    'rho_increase_factor': 2.0,      # α_inc: increase factor for unsuccessful steps  
    'min_rho': 1e-6,                 # ρ_min: minimum regularization strength
    'gamma': 0.5,                    # Success ratio threshold
    
    # Scenario selection
    'num_lp_scenarios_per_iteration': 1000,     # |S|: scenarios to solve per iteration
    'lp_scenario_selection_strategy': 'systematic',  # 'systematic' or 'random'
    
    # Argmax operation
    'MAX_PI': 200000,                # Maximum stored dual solutions
    'MAX_OMEGA': 100000,             # Maximum scenarios
    'NUM_CANDIDATES': 8,             # Top-k candidates for feasibility checking
    
    # Dual management
    'num_duals_to_add_per_iteration': 10000,    # Max duals added per iteration
    'argmax_tol_cutoff': 1e-4,       # Coverage tolerance (unused in current version)
}
```

### **Convergence Properties**

The algorithm inherits convergence properties from trust-region methods:

1. **Global Convergence**: Under standard assumptions, the sequence $\{x_k\}$ converges to a stationary point
2. **Finite Cut Generation**: The number of active Benders cuts remains finite due to the argmax approximation
3. **Regularization Benefits**: The trust-region mechanism provides stability and faster practical convergence

### **Computational Complexity**

Per iteration complexity:
- **Master solve**: $O(\text{vars} \times \text{cuts})$ 
- **Scenario selection**: $O(|\Omega|)$ (systematic) or $O(|S|\log|\Omega|)$ (random)
- **Warmstart argmax**: $O(|S| \times \text{stored\_duals})$ with GPU acceleration
- **LP subproblem solve**: $O(|S| \times \text{LP\_complexity})$ 
- **Fast argmax**: $O(|\Omega| \times \text{stored\_duals})$ with GPU acceleration
- **Cut generation**: $O(\text{vars} + |\Omega|)$ 

The hybrid approach ensures that expensive LP solves scale with $|S| \ll |\Omega|$ while maintaining cut quality through the full-scenario argmax operation.

### **Memory Management**

- **GPU Memory**: Stores dual vectors and scenario data for fast argmax computation
- **CPU Memory**: Stores basis information and LU factorizations for warmstarting
- **LRU Cache**: Automatically evicts least-recently-used dual solutions when storage limits are reached
- **Incremental Updates**: New dual solutions are added progressively without full recomputation

### **Extensions and Variations**

The framework supports several algorithmic variations:

1. **Selection Strategies**: Systematic rotation, random sampling, or priority-based selection
2. **Regularization Schemes**: Quadratic (current), linear, or adaptive regularization
3. **Cut Aggregation**: Expected value cuts (current) or risk-aware formulations
4. **Parallelization**: Multi-GPU argmax computation and distributed LP solving

This makes the solver suitable for a wide range of two-stage stochastic programming applications while maintaining computational efficiency through GPU acceleration and intelligent dual solution management.