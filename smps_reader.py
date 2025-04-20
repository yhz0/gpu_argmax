import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import numpy as np
import re # For parsing time file
import os # For file path operations
from typing import List, Tuple, Dict, Optional, Any

class SMPSReader:
    """
    Reads the core problem structure and stochastic information (RHS only)
    of a two-stage stochastic linear program from SMPS format files
    (.cor/.mps, .tim, .sto). Handles implicit .tim format and INDEP DISCRETE .sto format.
    Prepares data structures needed for stochastic optimization algorithms like ArgmaxOperation.

    Core Problem Structure:
        min c'x + d'y
        s.t. Ax R1 b       (Stage 1 constraints)
             Cx + Dy R2 r_bar (Stage 2 constraints)
             lx <= x <= ux
             ly <= y <= uy

    Stochastic Information (RHS only):
        Stores discrete probability distributions for elements of the RHS vector
        that correspond to stage 2 constraints (r_bar).

    Provides methods for sampling stochastic RHS values and calculating RHS deviations (delta_r).
    Uses Gurobi to parse the .cor/.mps file. Assumes randomness affects only
    the RHS of stage 2 constraints identified via the .tim file.

    Attributes:
        core_file (str): Path to the .cor or .mps file.
        time_file (str): Path to the .tim file.
        sto_file (str): Path to the .sto file.
        model (gp.Model): The Gurobi model read from the core file.
        _start_col2_name (str): Name of the first column variable in stage 2.
        _start_row2_name (str): Name of the first constraint row in stage 2.
        var_name_to_index (Dict[str, int]): Variable name to 0-based order index.
        index_to_var_name (Dict[int, str]): Variable 0-based order index to name.
        constr_name_to_index (Dict[str, int]): Constraint name to 0-based order index.
        index_to_constr_name (Dict[int, str]): Constraint 0-based order index to name.
        x_indices (np.ndarray): Original Gurobi indices of stage 1 variables (x).
        y_indices (np.ndarray): Original Gurobi indices of stage 2 variables (y).
        row1_indices (np.ndarray): Original Gurobi indices of stage 1 constraints.
        row2_indices (np.ndarray): Original Gurobi indices of stage 2 constraints.
        A (sp.csr_matrix): Coefficient matrix for stage 1 constraints related to x.
        b (np.ndarray): RHS vector for stage 1 constraints.
        sense1 (np.ndarray): Senses (e.g., '<', '=', '>') for stage 1 constraints.
        c (np.ndarray): Objective coefficients for stage 1 variables (x).
        lb_x (np.ndarray): Lower bounds for stage 1 variables.
        ub_x (np.ndarray): Upper bounds for stage 1 variables.
        C (sp.csr_matrix): Coeff matrix for stage 2 constraints related to x.
        D (sp.csr_matrix): Coeff matrix for stage 2 constraints related to y.
        d (np.ndarray): Objective coefficients for stage 2 variables (y).
        r_bar (np.ndarray): Deterministic RHS vector for stage 2 constraints (full).
        short_r_bar (np.ndarray): Deterministic RHS values for *stochastic* stage 2 constraints only.
        sense2 (np.ndarray): Senses for stage 2 constraints.
        lb_y (np.ndarray): Lower bounds for stage 2 variables.
        ub_y (np.ndarray): Upper bounds for stage 2 variables.
        y_bounded_indices_orig (np.ndarray): Original Gurobi indices of stage 2 vars with non-trivial bounds.
        pi_dim (int): Dimension of the full 2nd-stage dual vector (rows + bounds).
        rhs_distributions (Dict[int, List[Tuple[float, float]]]):
             Maps original Gurobi constraint index to its discrete distribution.
        stochastic_rows_indices_orig (np.ndarray): Original Gurobi indices of constraints with stochastic RHS.
        stochastic_rows_relative_indices (np.ndarray):
             Indices of stochastic constraints *relative* to the stage 2 constraints (row2_indices).
             Needed for ArgmaxOperation's r_sparse_indices.
        stage1_var_names (list): List of variable names for stage 1 (x).
        stage2_var_names (list): List of variable names for stage 2 (y).
        stage1_constr_names (list): List of constraint names for stage 1.
        stage2_constr_names (list): List of constraint names for stage 2.
    """

    def __init__(self, core_file: str, time_file: str, sto_file: str):
        """
        Initializes the reader with file paths.

        Args:
            core_file (str): Path to the .cor or .mps file.
            time_file (str): Path to the .tim file (SMPS time format).
            sto_file (str): Path to the .sto file (SMPS stochastic format).
        """
        if not os.path.exists(core_file): raise FileNotFoundError(f"Core file not found: {core_file}")
        if not os.path.exists(time_file): raise FileNotFoundError(f"Time file not found: {time_file}")
        if not os.path.exists(sto_file): raise FileNotFoundError(f"Sto file not found: {sto_file}")

        self.core_file = core_file
        self.time_file = time_file
        self.sto_file = sto_file
        self.model: Optional[gp.Model] = None

        # Information extracted from .tim file
        self._start_col2_name: Optional[str] = None
        self._start_row2_name: Optional[str] = None

        # Name <-> Index Mappings
        self.var_name_to_index: Dict[str, int] = {}
        self.index_to_var_name: Dict[int, str] = {}
        self.constr_name_to_index: Dict[str, int] = {}
        self.index_to_constr_name: Dict[int, str] = {}

        # Indices corresponding to names in the Gurobi model
        self.x_indices = np.array([], dtype=int)
        self.y_indices = np.array([], dtype=int)
        self.row1_indices = np.array([], dtype=int)
        self.row2_indices = np.array([], dtype=int)

        # Extracted deterministic matrices and vectors
        self.A: Optional[sp.csr_matrix] = None
        self.b: Optional[np.ndarray] = None
        self.sense1: Optional[np.ndarray] = None
        self.c: Optional[np.ndarray] = None
        self.lb_x: Optional[np.ndarray] = None
        self.ub_x: Optional[np.ndarray] = None
        self.C: Optional[sp.csr_matrix] = None
        self.D: Optional[sp.csr_matrix] = None
        self.d: Optional[np.ndarray] = None
        self.r_bar: Optional[np.ndarray] = None
        self.short_r_bar: Optional[np.ndarray] = None
        self.sense2: Optional[np.ndarray] = None
        self.lb_y: Optional[np.ndarray] = None
        self.ub_y: Optional[np.ndarray] = None

        # Store names for reference
        self.stage1_var_names: List[str] = []
        self.stage2_var_names: List[str] = []
        self.stage1_constr_names: List[str] = []
        self.stage2_constr_names: List[str] = []

        # Dual / Bound Information
        self.y_bounded_indices_orig = np.array([], dtype=int)
        self.pi_dim: int = 0

        # Stochastic information (RHS distributions)
        self.rhs_distributions: Dict[int, List[Tuple[float, float]]] = {}
        self.stochastic_rows_indices_orig = np.array([], dtype=int)
        self.stochastic_rows_relative_indices = np.array([], dtype=int)


        print(f"Initialized SMPSReader for core='{core_file}', time='{time_file}', sto='{sto_file}'")

    def _parse_time_file(self):
        """
        Parses the .tim file assuming the implicit format where the *second*
        data line under the PERIODS section defines the start of stage 2.
        Extracts the starting column and row names for stage 2.
        """
        print(f"Parsing implicit time file: {self.time_file}...")
        self._start_col2_name = None
        self._start_row2_name = None
        in_periods_section = False
        period_data_lines_found = 0 # Count data lines *after* PERIODS line

        try:
            with open(self.time_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Check for comment line *before* stripping whitespace
                    if line.startswith('*'):
                        continue
                    # Now strip whitespace for content processing
                    line_content = line.strip()
                    if not line_content: continue # Skip empty lines

                    parts = line_content.split()
                    if not parts: continue
                    keyword = parts[0].upper()

                    if keyword == "TIME": continue
                    elif keyword == "PERIODS":
                        in_periods_section = True
                        print("  Found PERIODS section. Looking for second data line...")
                        continue
                    elif keyword == "ENDATA": break

                    # Process lines within the PERIODS section
                    if in_periods_section:
                        period_data_lines_found += 1
                        # We only care about the *second* data line
                        if period_data_lines_found == 2:
                            if len(parts) >= 2:
                                self._start_col2_name = parts[0]
                                self._start_row2_name = parts[1]
                                print(f"  Found stage 2 start markers: Col='{self._start_col2_name}', Row='{self._start_row2_name}'")
                                break # Found what we need
                            else:
                                raise ValueError(f"Malformed second data line in PERIODS section: {line.strip()}")

            # Check if markers were found after reading the whole file
            if not self._start_col2_name or not self._start_row2_name:
                 raise ValueError("Failed to parse stage 2 start column/row from .tim file (second data line under PERIODS not found or processed).")

        except FileNotFoundError:
            print(f"Error: Time file not found: {self.time_file}")
            raise
        except ValueError as e:
             print(f"Error parsing time file {self.time_file}: {e}")
             raise
        except Exception as e:
            print(f"An unexpected error occurred parsing time file {self.time_file}: {e}")
            raise

    def _parse_sto_file(self):
        """
        Parses the .sto file, extracting discrete RHS distributions.
        Supports INDEP DISCRETE format, only for RHS column.
        Requires self.constr_name_to_index to be populated first.
        Populates self.rhs_distributions and self.stochastic_rows_indices_orig.
        """
        print(f"Parsing stochastic file: {self.sto_file}...")
        if not self.constr_name_to_index:
             raise RuntimeError("_parse_sto_file called before constraint name mapping was created.")

        self.rhs_distributions = {}
        in_indep_discrete_section = False
        stochastic_rows_indices_set = set() # Use set for uniqueness

        try:
            with open(self.sto_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.startswith('*'): continue
                    line_content = line.strip()
                    if not line_content: continue

                    parts = line_content.split()
                    if not parts: continue
                    keyword = parts[0].upper()

                    # Section handling
                    if keyword == "STOCH": continue
                    elif keyword == "INDEP":
                        if len(parts) > 1 and parts[1].upper() == "DISCRETE":
                            in_indep_discrete_section = True
                            print("  Found INDEP DISCRETE section.")
                        else: # Stop processing if format changes
                            in_indep_discrete_section = False
                        continue
                    elif keyword == "ENDATA": break
                    # If other sections start, stop processing INDEP DISCRETE
                    elif keyword in ["BLOCKS", "SCENARIOS"]: in_indep_discrete_section = False

                    # Process data lines within INDEP DISCRETE section
                    if in_indep_discrete_section:
                        if len(parts) != 4:
                            print(f"  Warning: Skipping malformed line {line_num} in INDEP DISCRETE section: {line.strip()}")
                            continue

                        col_name, row_name, val_str, prob_str = parts

                        if col_name.upper() != 'RHS':
                            raise NotImplementedError(f"Sto parser only supports 'RHS' in column field, found '{col_name}' on line {line_num}.")

                        row_idx = self.constr_name_to_index.get(row_name)
                        if row_idx is None:
                            # This row might not exist or be named in the core model
                            # print(f"  Warning: Row name '{row_name}' from .sto (line {line_num}) not found in core model constraints. Skipping.")
                            continue

                        try:
                            value = float(val_str)
                            probability = float(prob_str)
                            # Basic check, full validation done later
                            if not (0.0 <= probability <= 1.0): print(f"  Warning: Probability {probability} for row '{row_name}' (line {line_num}) outside [0, 1].")
                        except ValueError:
                            print(f"  Warning: Could not convert value/probability on line {line_num}. Skipping.")
                            continue

                        # Store the distribution
                        if row_idx not in self.rhs_distributions: self.rhs_distributions[row_idx] = []
                        self.rhs_distributions[row_idx].append((value, probability))
                        stochastic_rows_indices_set.add(row_idx)

            # --- Store original indices and validate probabilities ---
            self.stochastic_rows_indices_orig = np.sort(np.array(list(stochastic_rows_indices_set), dtype=int))
            print("  Validating probabilities...")
            valid_indices = []
            for row_idx in self.stochastic_rows_indices_orig:
                row_name = self.index_to_constr_name.get(row_idx, f"INDEX_{row_idx}")
                total_prob = sum(p for v, p in self.rhs_distributions[row_idx])
                if not np.isclose(total_prob, 1.0, atol=1e-4):
                    print(f"  Warning: Probabilities for RHS of row '{row_name}' (index {row_idx}) sum to {total_prob:.6f}, not 1.0. Excluding.")
                else:
                    valid_indices.append(row_idx) # Keep only rows with valid distributions

            # Update to only include rows with valid distributions summing to 1
            if len(valid_indices) < len(self.stochastic_rows_indices_orig):
                 print(f"  Removed {len(self.stochastic_rows_indices_orig) - len(valid_indices)} rows due to invalid probability sums.")
                 self.stochastic_rows_indices_orig = np.sort(np.array(valid_indices, dtype=int))
                 # Also filter the distributions dictionary
                 self.rhs_distributions = {idx: dist for idx, dist in self.rhs_distributions.items() if idx in valid_indices}


            print(f"  Parsed valid distributions for {len(self.rhs_distributions)} RHS elements.")

        except FileNotFoundError:
            print(f"Error: Sto file not found: {self.sto_file}")
            raise
        except NotImplementedError as e:
             print(f"Error parsing sto file {self.sto_file}: {e}")
             raise
        except Exception as e:
            print(f"An unexpected error occurred parsing sto file {self.sto_file}: {e}")
            raise


    def load_and_extract(self):
        """
        Reads the .cor/.mps, .tim, and .sto files, determines stage indices,
        creates name/index mappings, parses stochastic info, and extracts coefficients.
        """
        print(f"Loading core file: {self.core_file}...")
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                self.model = gp.read(self.core_file, env=env)
            print(f"  Core file read successfully. Model '{self.model.ModelName}' has "
                  f"{self.model.NumVars} vars, {self.model.NumConstrs} constraints.")
        except gp.GurobiError as e:
            print(f"Error reading core file with Gurobi: {e}")
            if "License" in str(e): print("Hint: Check Gurobi license.")
            raise
        except FileNotFoundError:
             print(f"Error: Core file not found: {self.core_file}")
             raise
        except Exception as e:
            print(f"An unexpected error occurred reading core file: {e}")
            raise

        # Parse the time file to get stage 2 start markers
        self._parse_time_file() # Sets self._start_col2_name, self._start_row2_name

        # --- Get Ordered Variable and Constraint Names/Indices from Gurobi Model ---
        print("Getting ordered variable and constraint info from Gurobi model...")
        all_vars = self.model.getVars()
        all_constrs = self.model.getConstrs()

        # Create Name <-> Index Mappings based on Gurobi's order (0 to N-1)
        self.var_name_to_index = {}
        self.index_to_var_name = {}
        all_var_names_ordered = []
        for i, v in enumerate(all_vars):
             current_var_name = f"VAR_UNNAMED_{i}"
             if hasattr(v, 'VarName') and v.VarName: current_var_name = v.VarName
             elif hasattr(v, 'VarName'): print(f"  Warning: Variable at order index {i} has empty name. Assigning '{current_var_name}'.")
             else: print(f"  Warning: Variable at order index {i} has no name attribute. Assigning '{current_var_name}'.")
             if current_var_name in self.var_name_to_index: print(f"  Warning: Duplicate variable name '{current_var_name}' at index {i}.")
             self.var_name_to_index[current_var_name] = i
             self.index_to_var_name[i] = current_var_name
             all_var_names_ordered.append(current_var_name)

        self.constr_name_to_index = {}
        self.index_to_constr_name = {}
        all_constr_names_ordered = []
        for i, c in enumerate(all_constrs):
             current_constr_name = f"CONSTR_UNNAMED_{i}"
             if hasattr(c, 'ConstrName') and c.ConstrName: current_constr_name = c.ConstrName
             elif hasattr(c, 'ConstrName'): print(f"  Warning: Constraint at order index {i} has empty name. Assigning '{current_constr_name}'.")
             else: print(f"  Warning: Constraint at order index {i} has no name attribute. Assigning '{current_constr_name}'.")
             if current_constr_name in self.constr_name_to_index: print(f"  Warning: Duplicate constraint name '{current_constr_name}' at index {i}.")
             self.constr_name_to_index[current_constr_name] = i
             self.index_to_constr_name[i] = current_constr_name
             all_constr_names_ordered.append(current_constr_name)

        # --- Parse STO file (needs name->index mapping) ---
        self._parse_sto_file() # Populates self.rhs_distributions and self.stochastic_rows_indices_orig

        # --- Determine Index Ranges based on Period Starts ---
        try:
            start_col2_idx = self.var_name_to_index.get(self._start_col2_name)
            start_row2_idx = self.constr_name_to_index.get(self._start_row2_name)

            if start_col2_idx is None: raise ValueError(f"Stage 2 start col '{self._start_col2_name}' not found.")
            if start_row2_idx is None: raise ValueError(f"Stage 2 start row '{self._start_row2_name}' not found.")

            print(f"  Stage 2 starts at variable index {start_col2_idx} ('{self._start_col2_name}')")
            print(f"  Stage 2 starts at constraint index {start_row2_idx} ('{self._start_row2_name}')")

            self.x_indices = np.arange(0, start_col2_idx, dtype=int)
            self.y_indices = np.arange(start_col2_idx, self.model.NumVars, dtype=int)
            self.row1_indices = np.arange(0, start_row2_idx, dtype=int)
            self.row2_indices = np.arange(start_row2_idx, self.model.NumConstrs, dtype=int)

            self.stage1_var_names = [all_var_names_ordered[i] for i in self.x_indices]
            self.stage2_var_names = [all_var_names_ordered[i] for i in self.y_indices]
            self.stage1_constr_names = [all_constr_names_ordered[i] for i in self.row1_indices]
            self.stage2_constr_names = [all_constr_names_ordered[i] for i in self.row2_indices]

            print(f"  Determined {len(self.x_indices)} stage 1 vars, {len(self.y_indices)} stage 2 vars.")
            print(f"  Determined {len(self.row1_indices)} stage 1 constraints, {len(self.row2_indices)} stage 2 constraints.")

        except Exception as e:
             print(f"Error determining stage indices: {e}")
             raise

        # --- Extract Deterministic Coefficients using Indices ---
        print("Extracting deterministic coefficients based on stages...")
        try:
            M = self.model.getA()
            if self.model.NumObj > 0: obj_coeffs = np.array(self.model.getAttr("Obj", all_vars))
            else: obj_coeffs = np.zeros(self.model.NumVars)
            rhs_coeffs = np.array(self.model.getAttr("RHS", all_constrs))
            senses_char = np.array(self.model.getAttr("Sense", all_constrs))
            lb_full = np.array(self.model.getAttr("LB", all_vars))
            ub_full = np.array(self.model.getAttr("UB", all_vars))

            # --- Stage 1 Components ---
            self.A = M[self.row1_indices, :][:, self.x_indices] if len(self.row1_indices)>0 and len(self.x_indices)>0 else sp.csr_matrix((len(self.row1_indices), len(self.x_indices)), dtype=M.dtype)
            self.b = rhs_coeffs[self.row1_indices] if len(self.row1_indices)>0 else np.array([], dtype=rhs_coeffs.dtype)
            self.sense1 = senses_char[self.row1_indices] if len(self.row1_indices)>0 else np.array([], dtype=senses_char.dtype)
            self.c = obj_coeffs[self.x_indices] if len(self.x_indices)>0 else np.array([], dtype=obj_coeffs.dtype)
            self.lb_x = lb_full[self.x_indices] if len(self.x_indices)>0 else np.array([], dtype=lb_full.dtype)
            self.ub_x = ub_full[self.x_indices] if len(self.x_indices)>0 else np.array([], dtype=ub_full.dtype)

            # --- Stage 2 Components ---
            if len(self.row2_indices) > 0:
                 M_row2 = M[self.row2_indices, :]
                 self.C = M_row2[:, self.x_indices] if len(self.x_indices)>0 else sp.csr_matrix((len(self.row2_indices), 0), dtype=M.dtype)
                 self.D = M_row2[:, self.y_indices] if len(self.y_indices)>0 else sp.csr_matrix((len(self.row2_indices), 0), dtype=M.dtype)
                 self.r_bar = rhs_coeffs[self.row2_indices] # Deterministic part
                 self.sense2 = senses_char[self.row2_indices]
            else:
                 print("  Warning: No stage 2 constraints identified.")
                 self.C = sp.csr_matrix((0, len(self.x_indices)), dtype=M.dtype)
                 self.D = sp.csr_matrix((0, len(self.y_indices)), dtype=M.dtype)
                 self.r_bar = np.array([], dtype=rhs_coeffs.dtype)
                 self.sense2 = np.array([], dtype=senses_char.dtype)

            self.d = obj_coeffs[self.y_indices] if len(self.y_indices)>0 else np.array([], dtype=obj_coeffs.dtype)
            self.lb_y = lb_full[self.y_indices] if len(self.y_indices)>0 else np.array([], dtype=lb_full.dtype)
            self.ub_y = ub_full[self.y_indices] if len(self.y_indices)>0 else np.array([], dtype=ub_full.dtype)

            if len(self.row2_indices) > 0 and len(self.y_indices) == 0:
                 print("  Error: Stage 2 constraints found, but no stage 2 variables identified for D matrix!")

            # --- Calculate Dual/Bound Info ---
            print("Identifying bounded stage 2 variables...")
            bounded_idx_list = []
            if len(self.y_indices) > 0:
                 has_finite_ub = np.isfinite(self.ub_y)
                 has_finite_lb = np.isfinite(self.lb_y) & (self.lb_y != 0.0)
                 bounded_mask = has_finite_ub | has_finite_lb
                 self.y_bounded_indices_orig = self.y_indices[bounded_mask]
                 print(f"  Found {len(self.y_bounded_indices_orig)} stage 2 variables with potentially non-trivial bounds.")
            else:
                 self.y_bounded_indices_orig = np.array([], dtype=int)

            self.pi_dim = len(self.row2_indices) + len(self.y_bounded_indices_orig)
            print(f"  Calculated full second-stage dual dimension (pi_dim): {self.pi_dim}")

            # --- Calculate Relative Stochastic Indices & short_r_bar ---
            print("Calculating relative indices and short_r_bar for stochastic rows...")
            # Ensure stochastic rows are actually stage 2 rows
            row2_indices_set = set(self.row2_indices) # Faster lookup
            valid_stochastic_rows_orig = [idx for idx in self.stochastic_rows_indices_orig if idx in row2_indices_set]
            if len(valid_stochastic_rows_orig) != len(self.stochastic_rows_indices_orig):
                 print(f"  Warning: Some stochastic rows from .sto file are not stage 2 constraints. Using only valid ones.")
                 self.stochastic_rows_indices_orig = np.sort(np.array(valid_stochastic_rows_orig, dtype=int))
                 # Filter distributions to keep only valid ones
                 self.rhs_distributions = {idx: dist for idx, dist in self.rhs_distributions.items() if idx in valid_stochastic_rows_orig}

            # Create mapping from original row2 index to its relative position (0..N-1)
            row2_orig_to_rel_map = {orig_idx: rel_idx for rel_idx, orig_idx in enumerate(self.row2_indices)}
            rel_idx_list = []
            for orig_stoch_idx in self.stochastic_rows_indices_orig: # Use the validated list
                 rel_idx = row2_orig_to_rel_map.get(orig_stoch_idx)
                 if rel_idx is not None: rel_idx_list.append(rel_idx)

            self.stochastic_rows_relative_indices = np.sort(np.array(rel_idx_list, dtype=int))
            print(f"  Found {len(self.stochastic_rows_relative_indices)} stochastic rows relative to stage 2 constraints.")

            # Extract the deterministic r_bar values corresponding to these relative indices
            if len(self.r_bar) > 0 and len(self.stochastic_rows_relative_indices) > 0:
                 self.short_r_bar = self.r_bar[self.stochastic_rows_relative_indices]
            elif len(self.stochastic_rows_relative_indices) == 0:
                 self.short_r_bar = np.array([], dtype=self.r_bar.dtype)
            else:
                 self.short_r_bar = np.array([], dtype=rhs_coeffs.dtype)


            print("Coefficient extraction complete.")

        except IndexError as e:
             print(f"Error during coefficient extraction (check indices): {e}")
             raise
        except Exception as e:
             print(f"An unexpected error occurred during coefficient extraction: {e}")
             raise

    def sample_stochastic_rhs(self) -> np.ndarray:
        """
        Generates one realization of the stochastic components of the stage 2 RHS vector,
        ordered according to self.stochastic_rows_relative_indices.

        Reproducibility depends on the global NumPy random state. Call
        `np.random.seed()` before calling this method repeatedly for deterministic results.

        Returns:
            np.ndarray: A numpy array containing only the sampled values for the
                        stochastic RHS elements, in the order corresponding to
                        self.stochastic_rows_relative_indices.
                        Returns an empty array if no stochastic information found.
        """
        if not self.rhs_distributions or len(self.stochastic_rows_indices_orig) == 0:
            return np.array([], dtype=np.float64) # Return empty array consistent with short_r_bar

        num_stochastic = len(self.stochastic_rows_indices_orig)
        sampled_values = np.zeros(num_stochastic, dtype=np.float64) # Use float for values

        # Iterate through the *original* stochastic indices (which are sorted)
        # to ensure consistent output order corresponding to relative indices
        for i, orig_row_idx in enumerate(self.stochastic_rows_indices_orig):
            distribution = self.rhs_distributions.get(orig_row_idx)
            if distribution:
                values = [item[0] for item in distribution]
                probabilities = [item[1] for item in distribution]
                prob_sum = sum(probabilities)
                # Normalize probabilities if necessary for np.random.choice
                if not np.isclose(prob_sum, 1.0):
                    if prob_sum <= 0:
                         raise ValueError(f"Invalid probabilities (sum <=0) for row index {orig_row_idx}")
                    probabilities = np.array(probabilities, dtype=np.float64) / prob_sum
                try:
                    # Use NumPy's default global RNG state
                    sampled_values[i] = np.random.choice(values, p=probabilities)
                except ValueError as e:
                     # Error during choice usually means probabilities don't sum to 1 after potential normalization
                     print(f"Error sampling for row index {orig_row_idx}: {e}. Check probabilities: {probabilities}")
                     raise
            else:
                 # This indicates an inconsistency between stochastic_rows_indices_orig and rhs_distributions
                 raise KeyError(f"Distribution not found for stochastic row index {orig_row_idx} during sampling. Check .sto parsing and filtering.")


        # The order of sampled_values corresponds to the order of stochastic_rows_relative_indices
        return sampled_values

    def get_short_delta_r(self, sampled_stochastic_rhs: np.ndarray) -> np.ndarray:
        """
        Calculates the deviation delta_r = r(omega)_stochastic - r_bar_stochastic
        for a given sampled vector of *only* the stochastic RHS components.

        Args:
            sampled_stochastic_rhs (np.ndarray): A sampled vector containing only the
                                                 stochastic RHS components, ordered
                                                 according to stochastic_rows_relative_indices.
                                                 Typically the output of sample_stochastic_rhs().

        Returns:
            np.ndarray: A 1D array containing delta_r values for stochastic rows,
                        ordered by their relative indices. This is the format
                        needed for ArgmaxOperation.add_scenarios().
        """
        if self.short_r_bar is None:
             raise RuntimeError("Deterministic short_r_bar not calculated. Call load_and_extract() first.")
        if sampled_stochastic_rhs.shape != self.short_r_bar.shape:
             raise ValueError(f"Shape mismatch: sampled_stochastic_rhs shape {sampled_stochastic_rhs.shape} "
                              f"(len {len(sampled_stochastic_rhs)}) does not match short_r_bar shape {self.short_r_bar.shape} "
                              f"(len {len(self.short_r_bar)}). Ensure input is from sample_stochastic_rhs().")

        # Simple subtraction as both vectors correspond to the same stochastic elements in the same order
        short_delta_r = sampled_stochastic_rhs - self.short_r_bar
        return short_delta_r


    def get_template_dict(self) -> dict:
        """Returns the extracted components as a dictionary."""
        if self.A is None: # Check if extraction has run
             print("Warning: Extraction has not been performed. Call load_and_extract() first.")
             return {}
        # Add stochastic info and dual info
        return {
            # Stage 1 Deterministic
            'A': self.A, 'b': self.b, 'sense1': self.sense1,
            'c': self.c, 'lb_x': self.lb_x, 'ub_x': self.ub_x,
            # Stage 2 Deterministic
            'C': self.C, 'D': self.D, 'd': self.d,
            'r_bar': self.r_bar, 'sense2': self.sense2,
            'lb_y': self.lb_y, 'ub_y': self.ub_y,
            'short_r_bar': self.short_r_bar, # Added
            # Indices (Original Gurobi Order)
            'x_indices': self.x_indices, 'y_indices': self.y_indices,
            'row1_indices': self.row1_indices, 'row2_indices': self.row2_indices,
            # Name Mappings
            'var_name_to_index': self.var_name_to_index,
            'index_to_var_name': self.index_to_var_name,
            'constr_name_to_index': self.constr_name_to_index,
            'index_to_constr_name': self.index_to_constr_name,
            # Names by Stage
            'stage1_var_names': self.stage1_var_names,
            'stage2_var_names': self.stage2_var_names,
            'stage1_constr_names': self.stage1_constr_names,
            'stage2_constr_names': self.stage2_constr_names,
            # Dual / Bound Info
            'y_bounded_indices_orig': self.y_bounded_indices_orig,
            'pi_dim': self.pi_dim,
            # Stochastic Info
            'rhs_distributions': self.rhs_distributions,
            'stochastic_rows_indices_orig': self.stochastic_rows_indices_orig,
            'stochastic_rows_relative_indices': self.stochastic_rows_relative_indices
        }

# Example Usage:
if __name__ == '__main__':

    # --- Use the Reader with specified ssn files ---
    file_dir = os.path.join("smps_data", "ssn")
    core_filename = "ssn.mps"
    time_filename = "ssn.tim"
    sto_filename = "ssn.sto" # Added sto file
    core_filepath = os.path.join(file_dir, core_filename)
    time_filepath = os.path.join(file_dir, time_filename)
    sto_filepath = os.path.join(file_dir, sto_filename) # Assume .sto is in same folder

    print(f"Attempting to read SMPS problem from:")
    print(f"  Core file: {os.path.abspath(core_filepath)}")
    print(f"  Time file: {os.path.abspath(time_filepath)}")
    print(f"  Sto file: {os.path.abspath(sto_filepath)}")


    # Check if files exist before proceeding
    if not os.path.exists(core_filepath):
         print(f"ERROR: Core file not found at '{core_filepath}'")
    elif not os.path.exists(time_filepath):
         print(f"ERROR: Time file not found at '{time_filepath}'")
    elif not os.path.exists(sto_filepath):
         print(f"ERROR: Sto file not found at '{sto_filepath}'")
    else:
        try:
            # Instantiate and run extraction
            reader = SMPSReader(core_file=core_filepath,
                                time_file=time_filepath,
                                sto_file=sto_filepath) # Pass sto file path
            reader.load_and_extract()

            # --- Access extracted data ---
            print("\n--- Extracted Template Summary ---")
            template = reader.get_template_dict()

            if template: # Check if extraction was successful
                print(f"Stage 1 Variables (x): {len(reader.stage1_var_names)} (Indices: {len(template['x_indices'])})")
                print(f"Stage 1 Constraints (Ax R1 b): {len(reader.stage1_constr_names)} (Indices: {len(template['row1_indices'])})")
                print(f"Stage 2 Variables (y): {len(reader.stage2_var_names)} (Indices: {len(template['y_indices'])})")
                print(f"Stage 2 Constraints (Cx + Dy R2 r_bar): {len(reader.stage2_constr_names)} (Indices: {len(template['row2_indices'])})")
                print(f"Second Stage Dual Dimension (pi_dim): {template['pi_dim']}")
                print(f"Num Stage 2 Vars w/ Non-Trivial Bounds: {len(template['y_bounded_indices_orig'])}")


                # --- Stochastic Info Summary ---
                print("\n--- Stochastic Info (RHS Distributions) ---")
                num_stoch_rows = len(template.get('rhs_distributions', {}))
                print(f"Found distributions for {num_stoch_rows} RHS elements.")
                print(f"Indices relative to stage 2 constraints (for ArgmaxOp): {template['stochastic_rows_relative_indices'].shape}")
                print(f"Deterministic short_r_bar (for delta calc): {template['short_r_bar'].shape}")


                # --- Example Sampling ---
                if num_stoch_rows > 0:
                    print("\n--- Example Scenario Sampling ---")
                    # Set seed for reproducibility of this example run
                    np.random.seed(123)
                    print("Set np.random.seed(123) for sampling example.")

                    # Sample only the stochastic components
                    sampled_stochastic_rhs_vals = reader.sample_stochastic_rhs()
                    print(f"Sampled stochastic RHS component vector (shape {sampled_stochastic_rhs_vals.shape})")
                    print(f"  First 5 values: {sampled_stochastic_rhs_vals[:5]}")

                    # Calculate corresponding short delta_r needed by ArgmaxOperation
                    short_delta = reader.get_short_delta_r(sampled_stochastic_rhs_vals)
                    print(f"Corresponding short delta_r vector (shape {short_delta.shape})")
                    print(f"  First 5 values: {short_delta[:5]}")

                    # Verify shape matches stochastic_rows_relative_indices
                    assert len(template['stochastic_rows_relative_indices']) == len(short_delta)
                    assert template['short_r_bar'].shape == short_delta.shape


        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")
            import traceback
            traceback.print_exc()

