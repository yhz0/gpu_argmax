# smps_reader.py (Corrected Version with Batch Sampling Only)

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
    Prepares data structures needed for stochastic optimization algorithms.

    Core Problem Structure:
        min c'x + d'y
        s.t. Ax R1 b       (Stage 1 constraints)
             Cx + Dy R2 r_bar (Stage 2 constraints)
             lx <= x <= ux
             ly <= y <= uy

    Stochastic Information (RHS only):
        Stores discrete probability distributions for elements of the RHS vector
        that correspond to stage 2 constraints (r_bar).

    Provides methods for batch sampling stochastic RHS values and calculating
    RHS deviations (delta_r). Uses Gurobi to parse the .cor/.mps file.
    Assumes randomness affects only the RHS of stage 2 constraints
    identified via the .tim file.

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
        rhs_distributions (Dict[int, List[Tuple[float, float]]]):
             Maps original Gurobi constraint index to its *validated* discrete distribution.
        stochastic_rows_indices_orig (np.ndarray): *Final validated* original Gurobi indices of stage 2 constraints with stochastic RHS.
        stochastic_rows_relative_indices (np.ndarray):
             Indices of stochastic constraints *relative* to the stage 2 constraints (row2_indices).
        stage1_var_names (list): List of variable names for stage 1 (x).
        stage2_var_names (list): List of variable names for stage 2 (y).
        stage1_constr_names (list): List of constraint names for stage 1.
        stage2_constr_names (list): List of constraint names for stage 2.
        _stochastic_values (List[np.ndarray]): Pre-processed list of possible values for each stochastic RHS.
        _stochastic_probabilities (List[np.ndarray]): Pre-processed list of probabilities for each stochastic RHS.
        _sampling_data_prepared (bool): Flag to indicate if pre-processing is done.
    """

    def __init__(self, core_file: str, time_file: str, sto_file: str):
        """
        Initializes the reader with file paths.

        Args:
            core_file (str): Path to the .cor or .mps file.
            time_file (str): Path to the .tim file (SMPS time format).
            sto_file (str): Path to the .sto file (SMPS stochastic format).
        """
        # Input validation
        if not os.path.exists(core_file): raise FileNotFoundError(f"Core file not found: {core_file}")
        if not os.path.exists(time_file): raise FileNotFoundError(f"Time file not found: {time_file}")
        if not os.path.exists(sto_file): raise FileNotFoundError(f"Sto file not found: {sto_file}")

        # Store file paths
        self.core_file = core_file
        self.time_file = time_file
        self.sto_file = sto_file

        # Initialize core model attribute
        self.model: Optional[gp.Model] = None

        # Initialize attributes populated by _parse_time_file
        self._start_col2_name: Optional[str] = None
        self._start_row2_name: Optional[str] = None

        # Initialize name/index mappings
        self.var_name_to_index: Dict[str, int] = {}
        self.index_to_var_name: Dict[int, str] = {}
        self.constr_name_to_index: Dict[str, int] = {}
        self.index_to_constr_name: Dict[int, str] = {}

        # Initialize stage indices
        self.x_indices = np.array([], dtype=int)
        self.y_indices = np.array([], dtype=int)
        self.row1_indices = np.array([], dtype=int)
        self.row2_indices = np.array([], dtype=int)

        # Initialize deterministic problem data attributes
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

        # Initialize name lists
        self.stage1_var_names: List[str] = []
        self.stage2_var_names: List[str] = []
        self.stage1_constr_names: List[str] = []
        self.stage2_constr_names: List[str] = []

        # Initialize bound info
        self.y_bounded_indices_orig = np.array([], dtype=int)

        # Initialize stochastic information attributes
        self.rhs_distributions: Dict[int, List[Tuple[float, float]]] = {}
        self.stochastic_rows_indices_orig = np.array([], dtype=int)
        self.stochastic_rows_relative_indices = np.array([], dtype=int)

        # Initialize attributes for pre-processed sampling data
        self._stochastic_values: List[np.ndarray] = []
        self._stochastic_probabilities: List[np.ndarray] = []
        self._sampling_data_prepared: bool = False

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
                                raise ValueError(f"Malformed second data line in PERIODS section (line {line_num}): {line.strip()}")

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
        Populates self.rhs_distributions (raw) and self.stochastic_rows_indices_orig (initial set).
        Performs validation of probability sums.
        """
        print(f"Parsing stochastic file: {self.sto_file}...")
        if not self.constr_name_to_index:
             raise RuntimeError("_parse_sto_file called before constraint name mapping was created.")

        # Use temporary dict to store raw distributions before validation
        raw_rhs_distributions: Dict[int, List[Tuple[float, float]]] = {}
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
                            # Allow warning instead of error? For now, error.
                            raise NotImplementedError(f"Sto parser only supports 'RHS' in column field, found '{col_name}' on line {line_num}.")

                        row_idx = self.constr_name_to_index.get(row_name)
                        if row_idx is None:
                            # This row might not exist or be named in the core model
                            # print(f"  Warning: Row name '{row_name}' from .sto (line {line_num}) not found in core model constraints. Skipping.")
                            continue

                        try:
                            value = float(val_str)
                            probability = float(prob_str)
                            # Basic check for probability range
                            if not (0.0 <= probability <= 1.0):
                                print(f"  Warning: Probability {probability} for row '{row_name}' (line {line_num}) outside [0, 1]. Still processing.")
                        except ValueError:
                            print(f"  Warning: Could not convert value/probability on line {line_num}. Skipping.")
                            continue

                        # Store the distribution temporarily
                        if row_idx not in raw_rhs_distributions: raw_rhs_distributions[row_idx] = []
                        raw_rhs_distributions[row_idx].append((value, probability))
                        stochastic_rows_indices_set.add(row_idx) # Track rows mentioned

        except FileNotFoundError:
            print(f"Error: Sto file not found: {self.sto_file}")
            raise
        except NotImplementedError as e:
             print(f"Error parsing sto file {self.sto_file}: {e}")
             raise
        except Exception as e:
            print(f"An unexpected error occurred parsing sto file {self.sto_file}: {e}")
            raise

        # --- Validate probabilities and finalize self.rhs_distributions ---
        initial_stochastic_indices = np.sort(np.array(list(stochastic_rows_indices_set), dtype=int))
        print("  Validating probabilities...")
        valid_indices = []
        validated_distributions = {} # Store distributions that pass validation

        for row_idx in initial_stochastic_indices:
            row_name = self.index_to_constr_name.get(row_idx, f"INDEX_{row_idx}")
            distribution = raw_rhs_distributions.get(row_idx) # Get from raw data

            if not distribution: # Should not happen if row_idx came from the set
                print(f"  Internal Warning: Row index {row_idx} in set but not in raw distributions dict.")
                continue

            total_prob = sum(p for v, p in distribution)
            # Use a tolerance for floating point comparison
            if not np.isclose(total_prob, 1.0, atol=1e-4):
                print(f"  Warning: Probabilities for RHS of row '{row_name}' (index {row_idx}) sum to {total_prob:.6f}, not 1.0. Excluding.")
            else:
                valid_indices.append(row_idx) # Keep only rows with valid distributions
                validated_distributions[row_idx] = distribution # Store the valid distribution

        # Update attributes with validated data
        final_valid_indices = np.sort(np.array(valid_indices, dtype=int))
        if len(final_valid_indices) < len(initial_stochastic_indices):
            print(f"  Removed {len(initial_stochastic_indices) - len(final_valid_indices)} rows due to invalid probability sums.")

        self.stochastic_rows_indices_orig = final_valid_indices
        self.rhs_distributions = validated_distributions # Store only validated distributions

        print(f"  Parsed and validated distributions for {len(self.rhs_distributions)} RHS elements.")


    def _prepare_sampling_data(self):
        """
        Pre-processes validated distributions for efficient batch sampling.
        Populates _stochastic_values and _stochastic_probabilities.
        Should only be called after load_and_extract is complete or when sampling is first requested.
        """
        # Avoid redundant preparation
        if self._sampling_data_prepared:
            return

        print("  Pre-processing distributions for sampling...")
        self._stochastic_values = []
        self._stochastic_probabilities = []

        # Check if there are any validated stochastic rows
        if len(self.stochastic_rows_indices_orig) == 0:
             print("  No stochastic distributions to pre-process.")
             self._sampling_data_prepared = True
             return

        # Iterate through the *final, validated* stochastic indices
        for orig_row_idx in self.stochastic_rows_indices_orig:
            # Get distribution from the validated dictionary
            distribution = self.rhs_distributions.get(orig_row_idx)
            if not distribution:
                 # This indicates an inconsistency if index is in stochastic_rows_indices_orig
                 print(f"Internal Error: Distribution for validated index {orig_row_idx} not found during pre-processing.")
                 # Handle this case - maybe skip or raise error? Skipping for now.
                 continue

            # Extract values and probabilities
            values = np.array([item[0] for item in distribution], dtype=np.float64)
            probabilities = np.array([item[1] for item in distribution], dtype=np.float64)

            # --- Normalization and Clipping ---
            # Ensure probabilities sum exactly to 1 for np.random.choice, handling potential precision issues
            prob_sum = probabilities.sum()
            if not np.isclose(prob_sum, 1.0):
                 # If validation passed with tolerance, normalize precisely here
                 if prob_sum > 1e-9: # Avoid division by zero
                     print(f"    Normalizing probabilities for index {orig_row_idx} from sum {prob_sum:.8f}")
                     probabilities /= prob_sum
                 else: # Should not happen after validation
                      raise ValueError(f"Invalid zero probability sum for index {orig_row_idx} during pre-processing.")

            # Ensure no negative probabilities (can happen due to floating point issues)
            if np.any(probabilities < 0):
                print(f"    Clipping negative probabilities for index {orig_row_idx}")
                probabilities[probabilities < 0] = 0.0
                # Renormalize if negatives were clipped and sum is non-zero
                prob_sum = probabilities.sum()
                if prob_sum > 1e-9:
                     probabilities /= prob_sum
                else:
                     # Handle case where all probabilities became zero (unlikely)
                     print(f"    Warning: Probabilities for index {orig_row_idx} became zero after clipping negatives. Assigning equal probability.")
                     num_outcomes = len(probabilities)
                     probabilities = np.ones(num_outcomes) / num_outcomes if num_outcomes > 0 else np.array([])

            # Final check for sum after potential adjustments
            if not np.isclose(probabilities.sum(), 1.0):
                 print(f"    Warning: Final probability sum for index {orig_row_idx} is {probabilities.sum():.8f} after adjustments.")
                 # Optionally raise an error or try one last normalization

            # Append the processed arrays
            self._stochastic_values.append(values)
            self._stochastic_probabilities.append(probabilities)

        # Mark preparation as complete
        self._sampling_data_prepared = True
        print(f"  Sampling data pre-processing complete for {len(self._stochastic_values)} elements.")


    def load_and_extract(self):
        """
        Reads the .cor/.mps, .tim, and .sto files, determines stage indices,
        creates name/index mappings, parses stochastic info, validates,
        and extracts coefficients and other relevant data structures.
        """
        print(f"Loading core file: {self.core_file}...")
        try:
            # Use context manager for Gurobi environment
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0) # Suppress Gurobi console output
                env.start()
                # Read the model file
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
             # Handle potentially unnamed variables
             current_var_name = f"VAR_UNNAMED_{i}"
             try:
                 if v.VarName: current_var_name = v.VarName
                 elif hasattr(v, 'VarName'): print(f"  Warning: Variable at order index {i} has empty name. Assigning '{current_var_name}'.")
                 else: print(f"  Warning: Variable at order index {i} has no name attribute. Assigning '{current_var_name}'.")
             except AttributeError:
                  print(f"  Warning: Variable at order index {i} missing VarName attribute. Assigning '{current_var_name}'.")

             if current_var_name in self.var_name_to_index: print(f"  Warning: Duplicate variable name '{current_var_name}' at index {i}.")
             self.var_name_to_index[current_var_name] = i
             self.index_to_var_name[i] = current_var_name
             all_var_names_ordered.append(current_var_name)

        self.constr_name_to_index = {}
        self.index_to_constr_name = {}
        all_constr_names_ordered = []
        for i, c in enumerate(all_constrs):
             # Handle potentially unnamed constraints
             current_constr_name = f"CONSTR_UNNAMED_{i}"
             try:
                 if c.ConstrName: current_constr_name = c.ConstrName
                 elif hasattr(c, 'ConstrName'): print(f"  Warning: Constraint at order index {i} has empty name. Assigning '{current_constr_name}'.")
                 else: print(f"  Warning: Constraint at order index {i} has no name attribute. Assigning '{current_constr_name}'.")
             except AttributeError:
                  print(f"  Warning: Constraint at order index {i} missing ConstrName attribute. Assigning '{current_constr_name}'.")

             if current_constr_name in self.constr_name_to_index: print(f"  Warning: Duplicate constraint name '{current_constr_name}' at index {i}.")
             self.constr_name_to_index[current_constr_name] = i
             self.index_to_constr_name[i] = current_constr_name
             all_constr_names_ordered.append(current_constr_name)

        # --- Parse STO file (needs name->index mapping) ---
        # Populates self.rhs_distributions (validated) and self.stochastic_rows_indices_orig (validated)
        self._parse_sto_file()

        # --- Determine Index Ranges based on Period Starts ---
        print("Determining stage variable/constraint indices...")
        try:
            start_col2_idx = self.var_name_to_index.get(self._start_col2_name)
            start_row2_idx = self.constr_name_to_index.get(self._start_row2_name)

            # Validate that the start names were found in the model
            if start_col2_idx is None: raise ValueError(f"Stage 2 start col '{self._start_col2_name}' (from .tim) not found in core model variables.")
            if start_row2_idx is None: raise ValueError(f"Stage 2 start row '{self._start_row2_name}' (from .tim) not found in core model constraints.")

            print(f"  Stage 2 starts at variable index {start_col2_idx} ('{self._start_col2_name}')")
            print(f"  Stage 2 starts at constraint index {start_row2_idx} ('{self._start_row2_name}')")

            # Define index ranges based on start indices
            self.x_indices = np.arange(0, start_col2_idx, dtype=int)
            self.y_indices = np.arange(start_col2_idx, self.model.NumVars, dtype=int)
            self.row1_indices = np.arange(0, start_row2_idx, dtype=int)
            self.row2_indices = np.arange(start_row2_idx, self.model.NumConstrs, dtype=int)

            # Populate name lists based on determined indices
            self.stage1_var_names = [all_var_names_ordered[i] for i in self.x_indices] if len(self.x_indices) > 0 else []
            self.stage2_var_names = [all_var_names_ordered[i] for i in self.y_indices] if len(self.y_indices) > 0 else []
            self.stage1_constr_names = [all_constr_names_ordered[i] for i in self.row1_indices] if len(self.row1_indices) > 0 else []
            self.stage2_constr_names = [all_constr_names_ordered[i] for i in self.row2_indices] if len(self.row2_indices) > 0 else []

            print(f"  Determined {len(self.x_indices)} stage 1 vars, {len(self.y_indices)} stage 2 vars.")
            print(f"  Determined {len(self.row1_indices)} stage 1 constraints, {len(self.row2_indices)} stage 2 constraints.")

        except Exception as e:
             print(f"Error determining stage indices: {e}")
             raise

        # --- Extract Deterministic Coefficients using Indices ---
        print("Extracting deterministic coefficients based on stages...")
        try:
            # Get full model data from Gurobi
            M = self.model.getA() # Full constraint matrix as sparse matrix
            # Handle potentially missing objective
            if self.model.NumObj > 0:
                obj_coeffs = np.array(self.model.getAttr("Obj", all_vars))
            else:
                print("  Warning: Model has no objective function defined.")
                obj_coeffs = np.zeros(self.model.NumVars)
            # Get RHS, Senses, Bounds
            rhs_coeffs = np.array(self.model.getAttr("RHS", all_constrs))
            senses_char = np.array(self.model.getAttr("Sense", all_constrs))
            lb_full = np.array(self.model.getAttr("LB", all_vars))
            ub_full = np.array(self.model.getAttr("UB", all_vars))

            # --- Stage 1 Components ---
            num_x = len(self.x_indices)
            num_r1 = len(self.row1_indices)
            # Extract submatrix A using slicing; handle empty cases
            self.A = M[self.row1_indices, :][:, self.x_indices].tocsr() if num_r1 > 0 and num_x > 0 else sp.csr_matrix((num_r1, num_x), dtype=M.dtype)
            self.b = rhs_coeffs[self.row1_indices] if num_r1 > 0 else np.array([], dtype=rhs_coeffs.dtype)
            self.sense1 = senses_char[self.row1_indices] if num_r1 > 0 else np.array([], dtype=senses_char.dtype)
            self.c = obj_coeffs[self.x_indices] if num_x > 0 else np.array([], dtype=obj_coeffs.dtype)
            self.lb_x = lb_full[self.x_indices] if num_x > 0 else np.array([], dtype=lb_full.dtype)
            self.ub_x = ub_full[self.x_indices] if num_x > 0 else np.array([], dtype=ub_full.dtype)

            # --- Stage 2 Components ---
            num_y = len(self.y_indices)
            num_r2 = len(self.row2_indices)
            if num_r2 > 0:
                 # Extract rows corresponding to stage 2 constraints
                 M_row2 = M[self.row2_indices, :]
                 # Extract submatrix C (coupling x)
                 self.C = M_row2[:, self.x_indices].tocsr() if num_x > 0 else sp.csr_matrix((num_r2, 0), dtype=M.dtype)
                 # Extract submatrix D (technology matrix for y)
                 self.D = M_row2[:, self.y_indices].tocsr() if num_y > 0 else sp.csr_matrix((num_r2, 0), dtype=M.dtype)
                 # Extract deterministic RHS r_bar
                 self.r_bar = rhs_coeffs[self.row2_indices]
                 # Extract senses for stage 2
                 self.sense2 = senses_char[self.row2_indices]
            else: # Handle case with no stage 2 constraints
                 print("  Warning: No stage 2 constraints identified.")
                 self.C = sp.csr_matrix((0, num_x), dtype=M.dtype)
                 self.D = sp.csr_matrix((0, num_y), dtype=M.dtype)
                 self.r_bar = np.array([], dtype=rhs_coeffs.dtype)
                 self.sense2 = np.array([], dtype=senses_char.dtype)

            # Extract stage 2 objective and bounds
            self.d = obj_coeffs[self.y_indices] if num_y > 0 else np.array([], dtype=obj_coeffs.dtype)
            self.lb_y = lb_full[self.y_indices] if num_y > 0 else np.array([], dtype=lb_full.dtype)
            self.ub_y = ub_full[self.y_indices] if num_y > 0 else np.array([], dtype=ub_full.dtype)

            # Sanity check: Stage 2 constraints but no stage 2 variables
            if num_r2 > 0 and num_y == 0:
                 print("  Warning: Stage 2 constraints found, but no stage 2 variables identified for D matrix!")

        except IndexError as e:
             print(f"Error during coefficient extraction (check indices): {e}")
             raise
        except Exception as e:
             print(f"An unexpected error occurred during coefficient extraction: {e}")
             raise

        # --- Calculate Dual/Bound Info ---
        print("Identifying bounded stage 2 variables...")
        if num_y > 0:
             # Check for finite upper bounds OR non-zero finite lower bounds
             has_finite_ub = np.isfinite(self.ub_y)
             has_finite_lb = np.isfinite(self.lb_y) & (np.abs(self.lb_y) > 1e-9) # Check non-zero LB
             bounded_mask = has_finite_ub | has_finite_lb
             self.y_bounded_indices_orig = self.y_indices[bounded_mask]
             print(f"  Found {len(self.y_bounded_indices_orig)} stage 2 variables with potentially non-trivial bounds.")
        else:
             self.y_bounded_indices_orig = np.array([], dtype=int)


        # --- Calculate Relative Stochastic Indices & short_r_bar ---
        # This section ensures stochastic rows belong to stage 2 and calculates relative indices
        print("Calculating relative indices and short_r_bar for stochastic rows...")
        if len(self.stochastic_rows_indices_orig) > 0 and num_r2 > 0:
            row2_indices_set = set(self.row2_indices) # Use set for efficient lookup

            # Filter original stochastic indices to include only those present in stage 2 rows
            final_valid_stochastic_rows_orig = [idx for idx in self.stochastic_rows_indices_orig if idx in row2_indices_set]

            if len(final_valid_stochastic_rows_orig) != len(self.stochastic_rows_indices_orig):
                print(f"  Warning: {len(self.stochastic_rows_indices_orig) - len(final_valid_stochastic_rows_orig)} stochastic rows from .sto file are not stage 2 constraints. Using only valid ones.")
                # Update the primary list of stochastic indices
                self.stochastic_rows_indices_orig = np.sort(np.array(final_valid_stochastic_rows_orig, dtype=int))
                # Filter the distributions dictionary again based on the final valid indices
                self.rhs_distributions = {idx: dist for idx, dist in self.rhs_distributions.items() if idx in self.stochastic_rows_indices_orig}
                # Mark sampling data as unprepared since distributions might have changed
                self._sampling_data_prepared = False

            # Create mapping from original stage 2 index to its relative position (0..num_r2-1)
            row2_orig_to_rel_map = {orig_idx: rel_idx for rel_idx, orig_idx in enumerate(self.row2_indices)}

            # Calculate relative indices for the final set of stochastic rows
            rel_idx_list = []
            for orig_stoch_idx in self.stochastic_rows_indices_orig: # Use the final validated list
                 rel_idx = row2_orig_to_rel_map.get(orig_stoch_idx)
                 # This check should always pass if filtering above was correct
                 if rel_idx is not None:
                      rel_idx_list.append(rel_idx)
                 else:
                      print(f"Internal Warning: Validated stochastic index {orig_stoch_idx} not found in row2 map.")

            self.stochastic_rows_relative_indices = np.sort(np.array(rel_idx_list, dtype=int))
            num_stoch_final = len(self.stochastic_rows_relative_indices)
            print(f"  Found {num_stoch_final} stochastic rows relative to stage 2 constraints.")

            # Extract the deterministic r_bar values corresponding to these relative indices
            if num_stoch_final > 0 and self.r_bar is not None and len(self.r_bar) > 0:
                 # Ensure relative indices are within bounds of r_bar
                 if np.max(self.stochastic_rows_relative_indices) < len(self.r_bar):
                      self.short_r_bar = self.r_bar[self.stochastic_rows_relative_indices]
                 else:
                      print(f"Error: Max relative index {np.max(self.stochastic_rows_relative_indices)} out of bounds for r_bar (len {len(self.r_bar)}).")
                      self.short_r_bar = np.array([], dtype=self.r_bar.dtype)
            else:
                 # Handle cases: no stochastic rows, or r_bar is empty/None
                 dtype_ref = self.r_bar.dtype if self.r_bar is not None else rhs_coeffs.dtype
                 self.short_r_bar = np.array([], dtype=dtype_ref)

        else: # Handle cases: no initial stochastic rows, or no stage 2 constraints
            print("  No stochastic rows identified or no stage 2 constraints exist.")
            self.stochastic_rows_relative_indices = np.array([], dtype=int)
            dtype_ref = self.r_bar.dtype if self.r_bar is not None else rhs_coeffs.dtype
            self.short_r_bar = np.array([], dtype=dtype_ref)


        print("SMPSReader extraction complete.")
        # --- End of load_and_extract ---


    # --- BATCH SAMPLING METHOD ---
    def sample_stochastic_rhs_batch(self, num_samples: int) -> np.ndarray:
        """
        Generates multiple realizations (a batch) of the stochastic components
        of the stage 2 RHS vector efficiently.

        Args:
            num_samples (int): The number of scenario samples to generate.

        Returns:
            np.ndarray: A 2D numpy array of shape (num_samples, num_stochastic),
                        where num_stochastic is the number of stochastic RHS elements.
                        Each row is one scenario realization, ordered according to
                        the final validated self.stochastic_rows_indices_orig.
                        Returns an empty array shape (num_samples, 0) if no
                        stochastic information found.
        """
        # Ensure data is ready (calls _prepare_sampling_data if needed)
        if not self._sampling_data_prepared:
            self._prepare_sampling_data()

        # Check using pre-processed list length
        num_stochastic = len(self._stochastic_values)
        if num_stochastic == 0:
             # Return shape (num_samples, 0) to be consistent
             return np.zeros((num_samples, 0), dtype=np.float64)

        # Create the output array (num_samples rows, num_stochastic columns)
        batch_samples = np.zeros((num_samples, num_stochastic), dtype=np.float64)

        # Loop through each stochastic *element*, sampling num_samples at once
        for i in range(num_stochastic):
            values = self._stochastic_values[i]
            probabilities = self._stochastic_probabilities[i]
            try:
                # Use size=num_samples to get all samples for this element
                sampled_col = np.random.choice(values, size=num_samples, p=probabilities)
                batch_samples[:, i] = sampled_col # Assign to the i-th column
            except ValueError as e:
                # Error typically means probabilities don't sum to 1 precisely
                # Use the index 'i' which corresponds to the order in stochastic_rows_indices_orig
                orig_row_idx = self.stochastic_rows_indices_orig[i] # Get corresponding original index
                print(f"Error during batch sampling for element {i} (orig index {orig_row_idx}): {e}.")
                print(f"  Values: {values}")
                print(f"  Probabilities: {probabilities} (Sum: {np.sum(probabilities)})")
                # Consider re-normalizing probabilities here if needed, or raise
                # Example re-normalization attempt:
                # try:
                #    print("Attempting re-normalization...")
                #    probabilities /= probabilities.sum()
                #    sampled_col = np.random.choice(values, size=num_samples, p=probabilities)
                #    batch_samples[:, i] = sampled_col
                # except Exception as inner_e:
                #    print(f"Re-normalization failed: {inner_e}")
                #    raise e # Re-raise original error if re-normalization fails
                raise # Re-raise original error for now

        # The order of columns corresponds to the order of stochastic_rows_indices_orig
        return batch_samples

    # --- Single sample method removed ---

    def get_short_delta_r(self, sampled_stochastic_rhs_single_scenario: np.ndarray) -> np.ndarray:
        """
        Calculates the deviation delta_r = r(omega)_stochastic - r_bar_stochastic
        for a given sampled vector of *only* the stochastic RHS components FOR ONE SCENARIO.

        Args:
            sampled_stochastic_rhs_single_scenario (np.ndarray): A 1D sampled vector
                        containing only the stochastic RHS components for ONE scenario,
                        ordered corresponding to self.stochastic_rows_indices_orig.
                        Typically one row from sample_stochastic_rhs_batch().

        Returns:
            np.ndarray: A 1D array containing delta_r values for stochastic rows,
                        ordered corresponding to self.stochastic_rows_indices_orig.
                        (Note: The order matches short_r_bar).
        """
        if self.short_r_bar is None:
             raise RuntimeError("Deterministic short_r_bar not calculated. Call load_and_extract() first.")

        # Ensure input is 1D
        if sampled_stochastic_rhs_single_scenario.ndim != 1:
            raise ValueError(f"Input must be a 1D array representing a single scenario, but got shape {sampled_stochastic_rhs_single_scenario.shape}")

        # Check shape consistency (length must match num stochastic elements)
        num_stochastic_elements = len(self.stochastic_rows_indices_orig) # Use final validated count
        if sampled_stochastic_rhs_single_scenario.shape[0] != num_stochastic_elements:
             raise ValueError(f"Shape mismatch: input sample length {sampled_stochastic_rhs_single_scenario.shape[0]} "
                              f"does not match number of stochastic elements {num_stochastic_elements}. ")
        if self.short_r_bar.shape[0] != num_stochastic_elements:
             raise ValueError(f"Shape mismatch: short_r_bar length {self.short_r_bar.shape[0]} "
                              f"does not match number of stochastic elements {num_stochastic_elements}. ")


        # Simple subtraction as both vectors correspond to the same stochastic elements in the same order
        short_delta_r = sampled_stochastic_rhs_single_scenario - self.short_r_bar
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
            'short_r_bar': self.short_r_bar, # Deterministic part of stochastic RHS
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
            # Stochastic Info
            'rhs_distributions': self.rhs_distributions, # Validated distributions
            'stochastic_rows_indices_orig': self.stochastic_rows_indices_orig, # Final validated indices
            'stochastic_rows_relative_indices': self.stochastic_rows_relative_indices # Relative to stage 2
        }

# Example Usage:
if __name__ == '__main__':

    # Define file paths relative to script location or use absolute paths
    # Ensure the 'smps_data/ssn' directory structure exists relative to this script
    script_dir = os.path.dirname(__file__) # Get directory where script is located
    # Handle case where __file__ might not be defined (e.g., interactive session)
    if not script_dir:
        script_dir = os.getcwd() # Use current working directory as fallback
        print(f"Warning: __file__ not defined. Using current working directory: {script_dir}")
    file_dir = os.path.join(script_dir, "smps_data", "ssn")


    # Check if directory exists
    if not os.path.isdir(file_dir):
        print(f"ERROR: Directory not found: {file_dir}")
        print("Please ensure the 'smps_data/ssn' directory exists relative to the script or CWD.")
        exit()

    core_filename = "ssn.mps" # Prefer .mps if available
    time_filename = "ssn.tim"
    sto_filename = "ssn.sto"
    core_filepath = os.path.join(file_dir, core_filename)
    time_filepath = os.path.join(file_dir, time_filename)
    sto_filepath = os.path.join(file_dir, sto_filename)

    print(f"Attempting to read SMPS problem from:")
    print(f"  Core file: {os.path.abspath(core_filepath)}")
    print(f"  Time file: {os.path.abspath(time_filepath)}")
    print(f"  Sto file: {os.path.abspath(sto_filepath)}")


    # Check if individual files exist before proceeding
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
                                sto_file=sto_filepath)
            reader.load_and_extract() # This now implicitly handles _prepare_sampling_data when needed

            # --- Access extracted data ---
            print("\n--- Extracted Template Summary ---")
            template = reader.get_template_dict()

            if template: # Check if extraction was successful
                num_stoch_rows = len(template.get('stochastic_rows_indices_orig', [])) # Use final validated list length
                print(f"Stage 1 Variables (x): {len(template['x_indices'])}")
                print(f"Stage 1 Constraints: {len(template['row1_indices'])}")
                print(f"Stage 2 Variables (y): {len(template['y_indices'])}")
                print(f"Stage 2 Constraints: {len(template['row2_indices'])}")
                print(f"Num Stage 2 Vars w/ Non-Trivial Bounds: {len(template['y_bounded_indices_orig'])}")
                print(f"Validated Stochastic Rows (Original Indices): {num_stoch_rows}")
                if template['stochastic_rows_relative_indices'] is not None:
                     print(f"Stochastic Rows Relative Indices (Count): {len(template['stochastic_rows_relative_indices'])}")
                if template['short_r_bar'] is not None:
                     print(f"Deterministic short_r_bar (Shape): {template['short_r_bar'].shape}")


                # --- Example Sampling ---
                if num_stoch_rows > 0:
                    print("\n--- Example Scenario Sampling ---")
                    np.random.seed(123) # Set seed for reproducibility
                    print("Set np.random.seed(123) for sampling example.")

                    # --- Sample a batch (e.g., 5 scenarios) ---
                    print("\nBatch Sampling (5 scenarios):")
                    num_batch_samples = 5
                    batch_samples = reader.sample_stochastic_rhs_batch(num_batch_samples)
                    print(f"  Batch sample shape: {batch_samples.shape}") # Should be (5, num_stoch_rows)
                    if batch_samples.shape[1] > 0: # Check if there are stochastic elements
                        print(f"  First sample, first 5 values: {batch_samples[0, :min(5, batch_samples.shape[1])]}") # Avoid index error if < 5 elements

                    # --- Example: Calculate delta_r for the first sample from the batch ---
                    first_sample_from_batch = batch_samples[0, :] # Get the first row (1D array)

                    # Ensure short_r_bar exists before calculating delta
                    if template['short_r_bar'] is not None and first_sample_from_batch.shape[0] > 0:
                        print(f"\nCalculating short delta_r vector (for first batch sample):")
                        try:
                            # Pass the 1D array directly
                            short_delta = reader.get_short_delta_r(first_sample_from_batch)
                            print(f"  Shape: {short_delta.shape}") # Should be (num_stoch_rows,)
                            print(f"  First 5 values: {short_delta[:min(5, short_delta.shape[0])]}")

                            # Verify shape matches stochastic_rows_relative_indices and short_r_bar
                            assert len(template['stochastic_rows_relative_indices']) == len(short_delta)
                            assert template['short_r_bar'].shape == short_delta.shape
                            print("  Shape assertions passed.")
                        except ValueError as delta_err:
                            print(f"  Error calculating delta_r: {delta_err}")
                    elif first_sample_from_batch.shape[0] == 0:
                         print("\nSkipping delta_r calculation: No stochastic elements.")
                    else:
                         print("\nSkipping delta_r calculation: short_r_bar is None.")


        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
