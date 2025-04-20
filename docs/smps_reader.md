# SMPSReader Class Documentation

## Purpose

The `SMPSReader` class reads and processes files for a two-stage stochastic linear program defined in the SMPS format. It aims to extract the deterministic problem structure and the stochastic information (limited to RHS randomness) into a usable format for optimization algorithms.

It specifically handles:
* Core problem files (`.cor` or `.mps`) using **Gurobi**.
* Time files (`.tim`) using the **implicit format** (where variable/constraint order and start markers define stages).
* Stochastic files (`.sto`) using the **INDEP DISCRETE** format for **RHS randomness only**.

## Key Functionality

The main workflow is orchestrated by the `load_and_extract()` method, which performs the following steps:

1.  **Reads Core File:** Uses `gurobipy` to read the `.cor` or `.mps` file into a `model` object.
2.  **Parses Time File:** Calls `_parse_time_file()` to find the start markers for stage 2 based on the implicit format rules.
3.  **Builds Mappings:** Creates dictionaries mapping variable/constraint names to their 0-based order index within the Gurobi model, and vice-versa. Stores these in `var_name_to_index`, `index_to_var_name`, `constr_name_to_index`, `index_to_constr_name`.
4.  **Parses Stochastic File:** Calls `_parse_sto_file()` to read RHS distributions from the `.sto` file. This uses the name-to-index mappings created previously. Stores distributions keyed by the *original Gurobi constraint index*.
5.  **Determines Stage Indices:** Using the start markers from the `.tim` file and the name/index mappings, it determines the ranges of *original Gurobi indices* belonging to stage 1 and stage 2 for both variables (`x_indices`, `y_indices`) and constraints (`row1_indices`, `row2_indices`). It also stores the corresponding names (`stageX_var_names`, `stageX_constr_names`).
6.  **Extracts Deterministic Coefficients:** Uses the determined index ranges to slice the full coefficient matrix, RHS vector, objective vector, bounds, and senses obtained from Gurobi, populating the standard template attributes (`A`, `b`, `c`, `C`, `D`, `d`, `r_bar`, `lb_x`, `ub_x`, `lb_y`, `ub_y`, `sense1`, `sense2`). Matrices `A`, `C`, `D` are stored as `scipy.sparse.csr_matrix`.
7.  **Calculates Dual/Bound Info:** Identifies stage 2 variables with non-trivial bounds (`y_bounded_indices_orig`) and calculates the full dimension of the second-stage dual vector (`pi_dim`).
8.  **Calculates Stochastic Relative Indices:** Determines the indices of the stochastic RHS constraints *relative* to the list of stage 2 constraints. These are stored in `stochastic_rows_relative_indices` and are crucial for interfacing with algorithms like `ArgmaxOperation`. It also extracts the corresponding deterministic RHS values into `short_r_bar`.

## Core Methods

* **`__init__(core_file, time_file, sto_file)`:** Constructor, stores file paths and initializes attributes.
* **`load_and_extract()`:** Orchestrates the entire reading, parsing, and extraction process. Must be called before accessing most data attributes.
* **`sample_stochastic_rhs()`:** Samples one realization for *only* the stochastic RHS components, returning a NumPy array ordered according to `stochastic_rows_relative_indices`. Relies on the global `np.random.seed()` for reproducibility.
* **`get_short_delta_r(sampled_stochastic_rhs)`:** Takes the output from `sample_stochastic_rhs()` and subtracts the pre-calculated `short_r_bar` to produce the $\delta_r$ vector suitable for `ArgmaxOperation`.
* **`get_template_dict()`:** Returns a dictionary containing most of the extracted deterministic and stochastic data, indices, and mappings.

## Private Parsing Methods

* **`_parse_time_file()`:** Implements the specific logic for parsing the implicit `.tim` format (finds `PERIODS`, reads the second data line for stage 2 start column/row names).
* **`_parse_sto_file()`:** Implements the logic for parsing the `INDEP DISCRETE` section of the `.sto` file, supporting only `RHS` randomness and validating probabilities.

## Important Attributes (Post `load_and_extract`)

* **Deterministic Data:** `A`, `b`, `c`, `C`, `D`, `d`, `r_bar`, `lb_x`, `ub_x`, `lb_y`, `ub_y`, `sense1`, `sense2`.
* **Stochastic Data:**
    * `rhs_distributions`: Dictionary mapping *original* constraint index to its distribution `[(value, prob), ...]`.
    * `short_r_bar`: Deterministic RHS values for *only* the stochastic rows (ordered by relative index).
    * `stochastic_rows_relative_indices`: Indices (0 to `len(row2_indices)-1`) of stochastic rows, needed for `ArgmaxOperation`.
* **Index/Name Info:**
    * `x_indices`, `y_indices`, `row1_indices`, `row2_indices`: NumPy arrays of *original* Gurobi indices for each stage.
    * `var_name_to_index`, `index_to_var_name`, etc.: Mappings based on Gurobi model order.
    * `stage1_var_names`, `stage2_var_names`, etc.: Lists of names for each stage.
* **Dual Info:**
    * `pi_dim`: Total dimension of the second-stage dual vector (constraints + bounds).
    * `y_bounded_indices_orig`: *Original* Gurobi indices of stage 2 variables with non-trivial bounds.

## Dependencies

* `gurobipy`: For reading `.cor`/`.mps` files.
* `numpy`: For numerical arrays.
* `scipy`: For sparse matrices (`scipy.sparse`).
