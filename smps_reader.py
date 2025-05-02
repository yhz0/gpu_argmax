# smps_reader.py

import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import numpy as np
import re # Used for parsing time file (though implicitly handled now)
import os
import logging # Import logging module
from typing import List, Tuple, Dict, Optional, Any

# --- Setup Logger ---
# Configure logger for this module
logger = logging.getLogger(__name__)
# Set default level (can be overridden by application using the module)
# If no handler is configured by the application, add a default NullHandler
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())
    # Optional: Set a default level if you want the logger to be active
    # even if the main script doesn't configure logging (useful for libraries)
    # logger.setLevel(logging.WARNING) # Or INFO, DEBUG etc.


class SMPSReader:
    """
    Reads a two-stage stochastic linear program from SMPS format files.

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
            core_file: Path to the .cor or .mps file.
            time_file: Path to the .tim file (SMPS time format).
            sto_file: Path to the .sto file (SMPS stochastic format).
        """
        if not os.path.exists(core_file): raise FileNotFoundError(f"Core file not found: {core_file}")
        if not os.path.exists(time_file): raise FileNotFoundError(f"Time file not found: {time_file}")
        if not os.path.exists(sto_file): raise FileNotFoundError(f"Sto file not found: {sto_file}")

        self.core_file = core_file
        self.time_file = time_file
        self.sto_file = sto_file

        self.model: Optional[gp.Model] = None
        self._start_col2_name: Optional[str] = None
        self._start_row2_name: Optional[str] = None

        # Name <-> Index mappings (based on Gurobi's internal order)
        self.var_name_to_index: Dict[str, int] = {}
        self.index_to_var_name: Dict[int, str] = {}
        self.constr_name_to_index: Dict[str, int] = {}
        self.index_to_constr_name: Dict[int, str] = {}

        # Original Gurobi Indices for each stage
        self.x_indices = np.array([], dtype=int)
        self.y_indices = np.array([], dtype=int)
        self.row1_indices = np.array([], dtype=int)
        self.row2_indices = np.array([], dtype=int)

        # --- Deterministic Data Structures ---
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
        self.sense2: Optional[np.ndarray] = None
        self.lb_y: Optional[np.ndarray] = None
        self.ub_y: Optional[np.ndarray] = None
        self.stage1_var_names: List[str] = []
        self.stage2_var_names: List[str] = []
        self.stage1_constr_names: List[str] = []
        self.stage2_constr_names: List[str] = []

        # --- Stochastic Data Structures ---
        self.rhs_distributions: Dict[int, List[Tuple[float, float]]] = {}
        self.stochastic_rows_indices_orig = np.array([], dtype=int)
        self.stochastic_rows_relative_indices = np.array([], dtype=int)
        self.short_r_bar: Optional[np.ndarray] = None

        # Internal state
        self._stochastic_values: List[np.ndarray] = []
        self._stochastic_probabilities: List[np.ndarray] = []
        self._sampling_data_prepared: bool = False
        self._data_loaded: bool = False

        logger.info(f"Initialized SMPSReader for core='{os.path.basename(core_file)}', "
                    f"time='{os.path.basename(time_file)}', sto='{os.path.basename(sto_file)}'")

    def _parse_time_file(self):
        """Parses the .tim file (implicit format) to find stage 2 start markers."""
        logger.info(f"Parsing implicit time file: {self.time_file}...")
        self._start_col2_name = None
        self._start_row2_name = None
        in_periods_section = False
        period_data_lines_found = 0

        try:
            with open(self.time_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.startswith('*'): continue
                    line_content = line.strip()
                    if not line_content: continue

                    parts = line_content.split()
                    if not parts: continue
                    keyword = parts[0].upper()

                    if keyword == "PERIODS":
                        in_periods_section = True
                        logger.debug("Found PERIODS section. Looking for second data line...")
                        continue
                    elif keyword == "ENDATA": break

                    if in_periods_section:
                        period_data_lines_found += 1
                        if period_data_lines_found == 2: # Implicit format assumption
                            if len(parts) >= 2:
                                self._start_col2_name = parts[0]
                                self._start_row2_name = parts[1]
                                logger.info(f"Found stage 2 start markers: Col='{self._start_col2_name}', Row='{self._start_row2_name}'")
                                break
                            else:
                                raise ValueError(f"Malformed second data line in PERIODS section (line {line_num}): {line.strip()}")

            if not self._start_col2_name or not self._start_row2_name:
                raise ValueError("Failed to parse stage 2 start column/row from .tim file "
                                 "(second data line under PERIODS not found or processed).")
        except Exception as e:
            logger.error(f"Error parsing time file {self.time_file}: {e}")
            raise

    def _parse_sto_file(self):
        """Parses the .sto file (INDEP DISCRETE RHS) and validates probabilities."""
        logger.info(f"Parsing stochastic file: {self.sto_file}...")
        if not self.constr_name_to_index:
            logger.error("_parse_sto_file called before constraint name mapping was created.")
            raise RuntimeError("_parse_sto_file called before constraint name mapping was created.")

        raw_rhs_distributions: Dict[int, List[Tuple[float, float]]] = {}
        in_indep_discrete_section = False
        stochastic_rows_indices_set = set()

        try:
            with open(self.sto_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.startswith('*'): continue
                    line_content = line.strip()
                    if not line_content: continue

                    parts = line_content.split()
                    if not parts: continue
                    keyword = parts[0].upper()

                    if keyword == "INDEP":
                        if len(parts) > 1 and parts[1].upper() == "DISCRETE":
                            in_indep_discrete_section = True
                            logger.debug("Found INDEP DISCRETE section.")
                        else:
                            in_indep_discrete_section = False
                        continue
                    elif keyword in ["STOCH", "ENDATA", "BLOCKS", "SCENARIOS"]:
                        if keyword == "ENDATA": break
                        in_indep_discrete_section = False

                    if in_indep_discrete_section:
                        if len(parts) != 4:
                            logger.warning(f"Skipping malformed line {line_num} in INDEP DISCRETE section: {line.strip()}")
                            continue

                        col_name, row_name, val_str, prob_str = parts

                        if col_name.upper() != 'RHS':
                            raise NotImplementedError(f"Sto parser only supports 'RHS' in column field, found '{col_name}' on line {line_num}.")

                        row_idx = self.constr_name_to_index.get(row_name)
                        if row_idx is None:
                            logger.debug(f"Row name '{row_name}' from .sto (line {line_num}) not found in core model. Skipping.")
                            continue

                        try:
                            value = float(val_str)
                            probability = float(prob_str)
                            if not (0.0 <= probability <= 1.0):
                                logger.warning(f"Probability {probability} for row '{row_name}' (line {line_num}) outside [0, 1]. Still processing.")
                        except ValueError:
                            logger.warning(f"Could not convert value/probability on line {line_num}. Skipping.")
                            continue

                        if row_idx not in raw_rhs_distributions: raw_rhs_distributions[row_idx] = []
                        raw_rhs_distributions[row_idx].append((value, probability))
                        stochastic_rows_indices_set.add(row_idx)

        except Exception as e:
            logger.error(f"An error occurred parsing sto file {self.sto_file}: {e}")
            raise

        # --- Validate probabilities ---
        initial_stochastic_indices = np.sort(np.array(list(stochastic_rows_indices_set), dtype=int))
        logger.debug("Validating probability sums...")
        valid_indices = []
        validated_distributions = {}
        prob_tolerance = 1e-4

        for row_idx in initial_stochastic_indices:
            row_name = self.index_to_constr_name.get(row_idx, f"INDEX_{row_idx}")
            distribution = raw_rhs_distributions.get(row_idx)
            if not distribution: continue

            total_prob = sum(p for v, p in distribution)
            if not np.isclose(total_prob, 1.0, atol=prob_tolerance):
                logger.warning(f"Probabilities for RHS of row '{row_name}' (index {row_idx}) sum to {total_prob:.6f}, not 1.0 (tol={prob_tolerance}). Excluding.")
            else:
                valid_indices.append(row_idx)
                validated_distributions[row_idx] = distribution

        final_valid_indices = np.sort(np.array(valid_indices, dtype=int))
        removed_count = len(initial_stochastic_indices) - len(final_valid_indices)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows due to invalid probability sums.")

        self.stochastic_rows_indices_orig = final_valid_indices
        self.rhs_distributions = validated_distributions

        logger.info(f"Parsed and validated distributions for {len(self.rhs_distributions)} RHS elements.")

    def _prepare_sampling_data(self):
        """Pre-processes validated distributions for efficient batch sampling."""
        if self._sampling_data_prepared: return

        logger.debug("Pre-processing distributions for sampling...")
        self._stochastic_values = []
        self._stochastic_probabilities = []

        if len(self.stochastic_rows_indices_orig) == 0:
            logger.debug("No stochastic distributions to pre-process.")
            self._sampling_data_prepared = True
            return

        for orig_row_idx in self.stochastic_rows_indices_orig:
            distribution = self.rhs_distributions.get(orig_row_idx)
            if not distribution:
                logger.error(f"Internal Error: Distribution for validated index {orig_row_idx} not found during pre-processing.")
                continue

            values = np.array([item[0] for item in distribution], dtype=np.float64)
            probabilities = np.array([item[1] for item in distribution], dtype=np.float64)

            # --- Normalization and Clipping for np.random.choice ---
            prob_sum = probabilities.sum()
            if not np.isclose(prob_sum, 1.0):
                if prob_sum > 1e-9:
                    logger.debug(f"Normalizing probabilities for index {orig_row_idx} from sum {prob_sum:.8f}")
                    probabilities /= prob_sum
                else:
                     raise ValueError(f"Invalid zero probability sum for index {orig_row_idx} during pre-processing.")

            if np.any(probabilities < 0):
                logger.debug(f"Clipping negative probabilities for index {orig_row_idx}")
                probabilities[probabilities < 0] = 0.0
                prob_sum = probabilities.sum()
                if prob_sum > 1e-9: probabilities /= prob_sum
                else:
                    logger.warning(f"Probabilities for index {orig_row_idx} became zero after clipping. Assigning equal probability.")
                    num_outcomes = len(probabilities)
                    probabilities = np.ones(num_outcomes) / num_outcomes if num_outcomes > 0 else np.array([])

            if not np.isclose(probabilities.sum(), 1.0):
                 logger.warning(f"Final probability sum for index {orig_row_idx} is {probabilities.sum():.8f} after adjustments.")

            self._stochastic_values.append(values)
            self._stochastic_probabilities.append(probabilities)

        self._sampling_data_prepared = True
        logger.debug(f"Sampling data pre-processing complete for {len(self._stochastic_values)} elements.")

    def load_and_extract(self):
        """
        Reads all SMPS files, determines stages, extracts coefficients,
        parses and validates stochastic RHS data.
        """
        logger.info(f"Loading core file: {self.core_file}...")
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                self.model = gp.read(self.core_file, env=env)
            logger.info(f"Core file read successfully. Model '{self.model.ModelName}' has "
                        f"{self.model.NumVars} vars, {self.model.NumConstrs} constraints.")
        except gp.GurobiError as e:
            logger.error(f"Error reading core file with Gurobi: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred reading core file: {e}")
            raise

        self._parse_time_file()

        # --- Get Ordered Variable and Constraint Names/Indices ---
        logger.debug("Getting variable/constraint names and indices from Gurobi...")
        all_vars = self.model.getVars()
        all_constrs = self.model.getConstrs()
        self.var_name_to_index = {v.VarName if v.VarName else f"VAR_UNNAMED_{i}": i for i, v in enumerate(all_vars)}
        self.index_to_var_name = {i: name for name, i in self.var_name_to_index.items()}
        self.constr_name_to_index = {c.ConstrName if c.ConstrName else f"CONSTR_UNNAMED_{i}": i for i, c in enumerate(all_constrs)}
        self.index_to_constr_name = {i: name for name, i in self.constr_name_to_index.items()}
        if len(self.var_name_to_index) != self.model.NumVars: logger.warning("Duplicate or unnamed variable names detected.")
        if len(self.constr_name_to_index) != self.model.NumConstrs: logger.warning("Duplicate or unnamed constraint names detected.")

        self._parse_sto_file() # Parses and validates stochastic RHS info

        # --- Determine Stage Indices ---
        logger.info("Determining stage variable/constraint indices...")
        try:
            start_col2_idx = self.var_name_to_index.get(self._start_col2_name)
            start_row2_idx = self.constr_name_to_index.get(self._start_row2_name)
            if start_col2_idx is None: raise ValueError(f"Stage 2 start col '{self._start_col2_name}' not found.")
            if start_row2_idx is None: raise ValueError(f"Stage 2 start row '{self._start_row2_name}' not found.")

            logger.info(f"Stage 2 starts at var index {start_col2_idx}, constraint index {start_row2_idx}.")
            self.x_indices = np.arange(0, start_col2_idx, dtype=int)
            self.y_indices = np.arange(start_col2_idx, self.model.NumVars, dtype=int)
            self.row1_indices = np.arange(0, start_row2_idx, dtype=int)
            self.row2_indices = np.arange(start_row2_idx, self.model.NumConstrs, dtype=int)

            # Populate name lists
            all_var_names_ordered = [self.index_to_var_name[i] for i in range(self.model.NumVars)]
            all_constr_names_ordered = [self.index_to_constr_name[i] for i in range(self.model.NumConstrs)]
            self.stage1_var_names = [all_var_names_ordered[i] for i in self.x_indices]
            self.stage2_var_names = [all_var_names_ordered[i] for i in self.y_indices]
            self.stage1_constr_names = [all_constr_names_ordered[i] for i in self.row1_indices]
            self.stage2_constr_names = [all_constr_names_ordered[i] for i in self.row2_indices]

            logger.info(f"Stage 1: {len(self.x_indices)} vars, {len(self.row1_indices)} constraints.")
            logger.info(f"Stage 2: {len(self.y_indices)} vars, {len(self.row2_indices)} constraints.")
        except Exception as e:
            logger.error(f"Error determining stage indices: {e}")
            raise

        # --- Extract Deterministic Coefficients ---
        logger.info("Extracting deterministic coefficients...")
        try:
            M = self.model.getA() # Full constraint matrix
            obj_coeffs = np.array(self.model.getAttr("Obj", all_vars)) if self.model.NumObj > 0 else np.zeros(self.model.NumVars)
            rhs_coeffs = np.array(self.model.getAttr("RHS", all_constrs))
            senses_char = np.array(self.model.getAttr("Sense", all_constrs))
            lb_full = np.array(self.model.getAttr("LB", all_vars))
            ub_full = np.array(self.model.getAttr("UB", all_vars))

            num_x, num_r1 = len(self.x_indices), len(self.row1_indices)
            num_y, num_r2 = len(self.y_indices), len(self.row2_indices)

            self.A = M[self.row1_indices, :][:, self.x_indices].tocsr() if num_r1 > 0 and num_x > 0 else sp.csr_matrix((num_r1, num_x))
            self.b = rhs_coeffs[self.row1_indices] if num_r1 > 0 else np.array([])
            self.sense1 = senses_char[self.row1_indices] if num_r1 > 0 else np.array([])
            self.c = obj_coeffs[self.x_indices] if num_x > 0 else np.array([])
            self.lb_x = lb_full[self.x_indices] if num_x > 0 else np.array([])
            self.ub_x = ub_full[self.x_indices] if num_x > 0 else np.array([])

            if num_r2 > 0:
                M_row2 = M[self.row2_indices, :]
                self.C = M_row2[:, self.x_indices].tocsr() if num_x > 0 else sp.csr_matrix((num_r2, 0))
                self.D = M_row2[:, self.y_indices].tocsr() if num_y > 0 else sp.csr_matrix((num_r2, 0))
                self.r_bar = rhs_coeffs[self.row2_indices]
                self.sense2 = senses_char[self.row2_indices]
            else:
                self.C, self.D = sp.csr_matrix((0, num_x)), sp.csr_matrix((0, num_y))
                self.r_bar, self.sense2 = np.array([]), np.array([])

            self.d = obj_coeffs[self.y_indices] if num_y > 0 else np.array([])
            self.lb_y = lb_full[self.y_indices] if num_y > 0 else np.array([])
            self.ub_y = ub_full[self.y_indices] if num_y > 0 else np.array([])

            if num_r2 > 0 and num_y == 0: logger.warning("Stage 2 constraints found, but no stage 2 variables!")

        except Exception as e:
            logger.error(f"An unexpected error occurred during coefficient extraction: {e}")
            raise

        # --- Finalize Stochastic Relative Indices & short_r_bar ---
        logger.debug("Calculating relative indices and short_r_bar for stochastic rows...")
        if len(self.stochastic_rows_indices_orig) > 0 and num_r2 > 0:
            row2_indices_set = set(self.row2_indices)
            final_valid_stochastic_rows_orig = [idx for idx in self.stochastic_rows_indices_orig if idx in row2_indices_set]

            removed_count_stage2 = len(self.stochastic_rows_indices_orig) - len(final_valid_stochastic_rows_orig)
            if removed_count_stage2 > 0:
                logger.warning(f"{removed_count_stage2} stochastic rows from .sto are not stage 2. Using only valid ones.")
                self.stochastic_rows_indices_orig = np.sort(np.array(final_valid_stochastic_rows_orig, dtype=int))
                self.rhs_distributions = {idx: dist for idx, dist in self.rhs_distributions.items() if idx in self.stochastic_rows_indices_orig}
                self._sampling_data_prepared = False # Mark for re-preparation

            row2_orig_to_rel_map = {orig_idx: rel_idx for rel_idx, orig_idx in enumerate(self.row2_indices)}
            rel_idx_list = [row2_orig_to_rel_map[orig_idx] for orig_idx in self.stochastic_rows_indices_orig if orig_idx in row2_orig_to_rel_map]
            self.stochastic_rows_relative_indices = np.sort(np.array(rel_idx_list, dtype=int))
            num_stoch_final = len(self.stochastic_rows_relative_indices)
            logger.info(f"Found {num_stoch_final} stochastic rows relative to stage 2 constraints.")

            if num_stoch_final > 0 and self.r_bar is not None and len(self.r_bar) > 0:
                 if np.max(self.stochastic_rows_relative_indices) < len(self.r_bar):
                     self.short_r_bar = self.r_bar[self.stochastic_rows_relative_indices]
                 else:
                     logger.error(f"Max relative index {np.max(self.stochastic_rows_relative_indices)} out of bounds for r_bar (len {len(self.r_bar)}).")
                     self.short_r_bar = np.array([], dtype=self.r_bar.dtype)
            else: self.short_r_bar = np.array([], dtype=rhs_coeffs.dtype) # Use fallback dtype
        else:
            logger.info("No stochastic rows identified or no stage 2 constraints exist.")
            self.stochastic_rows_relative_indices = np.array([], dtype=int)
            self.short_r_bar = np.array([], dtype=rhs_coeffs.dtype) # Use fallback dtype

        self._data_loaded = True # Mark data as loaded
        logger.info("SMPSReader extraction complete.")

    def sample_stochastic_rhs_batch(self, num_samples: int) -> np.ndarray:
        """
        Generates a batch of realizations of the stochastic RHS components.

        Args:
            num_samples: The number of scenario samples to generate.

        Returns:
            2D numpy array (num_samples, num_stochastic), ordered according
            to self.stochastic_rows_indices_orig. Returns shape (num_samples, 0)
            if no stochastic elements exist.
        """
        if not self._data_loaded:
             raise RuntimeError("Data not loaded. Call load_and_extract() first.")
        if not self._sampling_data_prepared:
            self._prepare_sampling_data()

        num_stochastic = len(self._stochastic_values)
        if num_stochastic == 0:
            return np.zeros((num_samples, 0), dtype=np.float64)

        batch_samples = np.zeros((num_samples, num_stochastic), dtype=np.float64)

        for i in range(num_stochastic):
            values = self._stochastic_values[i]
            probabilities = self._stochastic_probabilities[i]
            try:
                sampled_col = np.random.choice(values, size=num_samples, p=probabilities)
                batch_samples[:, i] = sampled_col
            except ValueError as e:
                orig_row_idx = self.stochastic_rows_indices_orig[i]
                logger.error(f"Error sampling element {i} (orig index {orig_row_idx}): {e}.")
                logger.error(f"  Probabilities: {probabilities} (Sum: {np.sum(probabilities)})")
                raise # Re-raise after logging info

        return batch_samples

    def get_short_delta_r(self, sampled_stochastic_rhs_batch: np.ndarray) -> np.ndarray:
        """
        Calculates deviation delta_r = r(omega)_stochastic - r_bar_stochastic
        for a BATCH of sampled scenarios' stochastic RHS components using broadcasting.

        Args:
            sampled_stochastic_rhs_batch: 2D numpy array where each row represents
                a sampled scenario for stochastic RHS elements (ordered like
                stochastic_rows_indices_orig).
                Shape: (num_scenarios, num_stochastic_elements).

        Returns:
            2D numpy array where each row contains the delta_r values for the
            corresponding input scenario.
            Shape: (num_scenarios, num_stochastic_elements).
        """
        if not self._data_loaded or self.short_r_bar is None:
            raise RuntimeError("Data not loaded or short_r_bar not calculated. Call load_and_extract() first.")

        # --- Input Validation ---
        if sampled_stochastic_rhs_batch.ndim != 2:
            raise ValueError(f"Input must be a 2D array (batch of scenarios), got shape {sampled_stochastic_rhs_batch.shape}")

        num_stochastic_elements = len(self.stochastic_rows_indices_orig)
        # Check the second dimension (features/elements per scenario) of the input batch
        if sampled_stochastic_rhs_batch.shape[1] != num_stochastic_elements:
            raise ValueError(f"Input batch's second dimension (features/elements: {sampled_stochastic_rhs_batch.shape[1]}) != "
                             f"expected num stochastic elements ({num_stochastic_elements}).")

        delta_r_batch = sampled_stochastic_rhs_batch - self.short_r_bar

        return delta_r_batch


    def print_summary(self):
        """Prints a summary of the loaded problem structure."""
        print("\n--- SMPS Problem Summary ---")
        if not self._data_loaded:
            print("WARNING: Data has not been loaded. Call load_and_extract() first.")
            return

        print(f"Core File: {self.core_file}")
        print(f"Time File: {self.time_file}")
        print(f"Sto File : {self.sto_file}")
        print("-" * 30)
        print(f"Stage 1 Variables (x): {len(self.x_indices)}")
        print(f"Stage 1 Constraints:   {len(self.row1_indices)}")
        print(f"Stage 2 Variables (y): {len(self.y_indices)}")
        print(f"Stage 2 Constraints:   {len(self.row2_indices)}")
        print("-" * 30)
        print(f"Stochastic RHS Rows: {len(self.stochastic_rows_indices_orig)}")
        print("-" * 30)
        if self.A is not None:
            print(f"Matrix A shape: {self.A.shape}, nnz: {self.A.nnz}")
        else:
            print("Matrix A: Not loaded")
        if self.C is not None:
            print(f"Matrix C shape: {self.C.shape}, nnz: {self.C.nnz}")
        else:
            print("Matrix C: Not loaded")
        if self.D is not None:
            print(f"Matrix D shape: {self.D.shape}, nnz: {self.D.nnz}")
        else:
            print("Matrix D: Not loaded")
        print("-" * 30)


# Example Usage:
if __name__ == '__main__':

    logging.basicConfig(level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_dir = os.path.join("smps_data", "ssn")

    core_filename = "ssn.mps"
    time_filename = "ssn.tim"
    sto_filename = "ssn.sto"
    core_filepath = os.path.join(file_dir, core_filename)
    time_filepath = os.path.join(file_dir, time_filename)
    sto_filepath = os.path.join(file_dir, sto_filename)

    # --- Instantiate and Load ---
    reader = SMPSReader(core_filepath, time_filepath, sto_filepath)
    reader.load_and_extract()

    # --- Print Summary (uses logger) ---
    reader.print_summary()
