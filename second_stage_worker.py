import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import typing as t
# import warnings # No longer needed unless adding optional checks

from typing import Tuple, Optional, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from smps_reader import SMPSReader

class SecondStageWorker:
    """
    Represents and solves the second-stage problem of a two-stage stochastic LP.

    Manages an independent Gurobi model for the subproblem:
        min d'y
        s.t. Dy R2 h
             ly <= y <= uy
    where h = r_bar - Cx + delta_r (omega).

    Updates the Gurobi model's RHS attribute directly in `set_x` and
    `set_scenario`. Relies on `model.optimize()` processing these pending
    changes before solving. Maintains its own Gurobi environment.
    """

    def __init__(self,
                 d: np.ndarray,
                 D: sp.csr_matrix,
                 sense2: np.ndarray,
                 lb_y: np.ndarray,
                 ub_y: np.ndarray,
                 r_bar: np.ndarray,
                 C: sp.csr_matrix,
                 stage2_var_names: t.List[str],
                 stage2_constr_names: t.List[str],
                 stochastic_rows_relative_indices: np.ndarray):
        """
        Initializes the SecondStageWorker and its Gurobi model structure.

        Args:
            d: Objective coefficients for stage 2 variables (y).
            D: Coefficient matrix for stage 2 constraints related to y.
            sense2: Senses ('<', '=', '>') for stage 2 constraints.
            lb_y: Lower bounds for stage 2 variables.
            ub_y: Upper bounds for stage 2 variables.
            r_bar: Deterministic RHS vector for stage 2 constraints.
            C: Coeff matrix coupling stage 1 (x) to stage 2 constraints.
            stage2_var_names: List of variable names for stage 2 (y).
            stage2_constr_names: List of constraint names for stage 2.
            stochastic_rows_relative_indices: Indices (relative to stage 2
                constraints) of constraints with stochastic RHS.
        """
        # --- Store essential parameters ---
        self.d = d.copy()
        self.D = D.copy()
        self.sense2 = sense2.copy()
        self.lb_y = lb_y.copy()
        self.ub_y = ub_y.copy()
        self.r_bar = r_bar.copy()
        self.C = C.copy()
        self.stage2_var_names = stage2_var_names[:]
        self.stage2_constr_names = stage2_constr_names[:]
        self.stochastic_rows_relative_indices = stochastic_rows_relative_indices.copy()

        # --- Internal state for calculations ---
        # Store calculated base RHS (r_bar - Cx) from set_x for use in set_scenario
        self._h_bar: t.Optional[np.ndarray] = None
        # Store the relevant stochastic parts of _h_bar for faster calculation
        self._h_bar_stochastic_part: t.Optional[np.ndarray] = None

        # --- Gurobi Setup (try-except for robust initialization) ---
        try:
            self.env = gp.Env(empty=True)
            self.env.start() # Start the independent environment

            self.model = gp.Model(name="SecondStageSubproblem", env=self.env)
            self.model.setParam('OutputFlag', 0) # Suppress solver console output

            # --- Define Gurobi Model Structure ---
            num_y_vars = len(self.d)
            num_stage2_constrs = len(self.r_bar)
            senses_char = np.array([s if isinstance(s, str) else chr(s) for s in self.sense2])

            self.y_vars = self.model.addMVar(shape=num_y_vars, lb=self.lb_y, ub=self.ub_y, obj=self.d, name=self.stage2_var_names)
            # Initialize Gurobi RHS attribute; will be overwritten by set_x/set_scenario
            self.constraints = self.model.addMConstr(A=self.D, x=self.y_vars, sense=senses_char, b=np.zeros(num_stage2_constrs), name=self.stage2_constr_names)
            self.model.ModelSense = gp.GRB.MINIMIZE
            # No model.update() needed after initial addMVar/addMConstr build
        except gp.GurobiError as e:
            print(f"FATAL: Error initializing Gurobi model: {e.code} - {e}")
            if hasattr(self, 'env'): self.env.dispose() # Attempt cleanup
            raise
        except Exception as e:
            print(f"FATAL: An unexpected error occurred during initialization: {e}")
            if hasattr(self, 'env'): self.env.dispose() # Attempt cleanup
            raise

    @classmethod
    def from_smps_reader(cls, reader: 'SMPSReader') -> 'SecondStageWorker':
        """
        Factory method to create a SecondStageWorker instance from a loaded SMPSReader.

        Args:
            reader: An instance of the SMPSReader class, assumed to have
                    successfully parsed the SMPS files.

        Returns:
            A new instance of SecondStageWorker.

        Raises:
            AttributeError: If the reader instance is missing required attributes,
                           suggesting it wasn't loaded properly.
            ValueError: If essential data in the reader appears invalid (e.g., None).
        """
        # Check if reader appears loaded (has essential attributes)
        required_attrs = [
            'd', 'D', 'sense2', 'lb_y', 'ub_y', 'r_bar', 'C',
            'stage2_var_names', 'stage2_constr_names',
            'stochastic_rows_relative_indices', 'model' # Check 'model' as a proxy for successful parsing
        ]
        for attr in required_attrs:
            if not hasattr(reader, attr):
                raise AttributeError(f"SMPSReader instance is missing required attribute '{attr}'. Was it loaded correctly?")

        return cls(
            d=reader.d,
            D=reader.D,
            sense2=reader.sense2,
            lb_y=reader.lb_y,
            ub_y=reader.ub_y,
            r_bar=reader.r_bar, # Use the reader's stage 2 specific r_bar
            C=reader.C,
            stage2_var_names=reader.stage2_var_names,
            stage2_constr_names=reader.stage2_constr_names,
            stochastic_rows_relative_indices=reader.stochastic_rows_relative_indices
        )

    def set_x(self, x: np.ndarray):
        """
        Calculates the base RHS h_bar = r_bar - Cx based on first-stage
        decision 'x'. Updates the Gurobi model's RHS attribute directly. Stores
        h_bar internally for use in set_scenario.
        """
        if x.shape[0] != self.C.shape[1]:
             raise ValueError(f"Dimension mismatch: x ({x.shape[0]}) vs C columns ({self.C.shape[1]})")

        x = x.flatten()
        cx_contribution = self.C @ x
        self._h_bar = self.r_bar - cx_contribution # Store base RHS
        self._h_bar_stochastic_part = self._h_bar[self.stochastic_rows_relative_indices]

        # Directly update the Gurobi model's RHS attribute state
        self.constraints.RHS = self._h_bar

    def set_scenario(self, short_delta_r: np.ndarray):
        """
        Applies scenario deviations (delta_r) to the base RHS h_bar = r_bar - Cx.

        Updates the Gurobi model's RHS attribute directly with the final
        scenario-specific RHS:
            rhs[stochastic] = (r_bar - Cx)[stochastic] + short_delta_r
            rhs[non-stochastic] = (r_bar - Cx)[non-stochastic]

        Requires `set_x` to have been called previously to calculate h_bar.

        Args:
            short_delta_r: Numpy array of stochastic deviations *relative* to r_bar
                           for the stochastic constraints only, ordered according to
                           `stochastic_rows_relative_indices`.
        """
        if self._h_bar is None or self._h_bar_stochastic_part is None:
             raise RuntimeError("`set_x` must be called before `set_scenario` to calculate base RHS.")

        # Check dimensions of the input deviation vector
        if len(short_delta_r) != len(self.stochastic_rows_relative_indices):
            raise ValueError(
                f"Dimension mismatch: short_delta_r ({len(short_delta_r)}) vs "
                f"stochastic rows ({len(self.stochastic_rows_relative_indices)})"
            )

        final_rhs = self._h_bar.copy() # Start from base RHS (r_bar - Cx)
        final_rhs[self.stochastic_rows_relative_indices] = self._h_bar_stochastic_part + short_delta_r
        self.constraints.RHS = final_rhs

    def solve(self, use_dual_simplex: bool = True, single_thread: bool = True) -> t.Optional[t.Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Solves the second-stage subproblem.

        Sets solver parameters and calls optimize(), which processes pending
        model modifications (like the latest RHS value assigned).

        Args:
            use_dual_simplex: If True, forces the dual simplex algorithm.
            single_thread: If True, restricts Gurobi to a single thread.

        Returns:
            A tuple (objective_value, y_solution, dual_solution_pi, reduced_costs)
            if optimal, otherwise None.
        """
        # --- Set Gurobi parameters for this specific solve ---
        self.model.setParam(gp.GRB.Param.Threads, 1 if single_thread else 0) # 0 lets Gurobi decide
        self.model.setParam(gp.GRB.Param.Method, 1 if use_dual_simplex else -1) # 1=Dual, -1=Auto

        # --- Solve the model ---
        # optimize() processes pending changes (e.g., RHS update from set_x/set_scenario)
        self.model.optimize()

        # --- Check status and extract results ---
        status = self.model.Status
        if status == gp.GRB.OPTIMAL:
            obj_val = self.model.ObjVal
            y_sol = self.y_vars.X
            pi_sol = self.constraints.Pi # Dual values
            rc_sol = self.y_vars.RC      # Reduced costs
            return obj_val, y_sol, pi_sol, rc_sol
        else:
            # You might want to log the status for debugging non-optimal solves
            # print(f"Worker solve finished with non-optimal status: {status}")
            return None # Indicate non-optimal solution

    def get_basis(self) -> t.Optional[t.Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieves the current basis (VBasis, CBasis) after an optimal solve.

        Returns:
            Tuple (vbasis, cbasis) as np.int8 arrays, or None if unavailable.
        """
        if self.model.Status == gp.GRB.OPTIMAL and self.model.SolCount > 0:
            # GurobiError propagates if basis attributes unavailable
            vbasis = np.array(self.y_vars.VBasis, dtype=np.int8)
            cbasis = np.array(self.constraints.CBasis, dtype=np.int8)
            return vbasis, cbasis
        else:
            return None # Basis not available/meaningful

    def set_basis(self, vbasis: np.ndarray, cbasis: np.ndarray):
        """
        Sets the VBasis and CBasis attributes for warm starting the next solve.
        Calls model.update() to ensure the change is registered by Gurobi.
        """
        num_y_vars = len(self.d)
        num_stage2_constrs = len(self.r_bar)
        if len(vbasis) != num_y_vars: raise ValueError(f"vbasis length mismatch")
        if len(cbasis) != num_stage2_constrs: raise ValueError(f"cbasis length mismatch")

        # Set Gurobi attributes directly
        self.y_vars.VBasis = vbasis.astype(int, copy=False)
        self.constraints.CBasis = cbasis.astype(int, copy=False)

        # Update is recommended after setting basis before the next optimize call
        self.model.update()

    def close(self):
        """
        Disposes of the Gurobi environment associated with this worker.
        Call explicitly when the worker is no longer needed to free resources.
        """
        if hasattr(self, 'env'):
            try:
                self.env.dispose()
            except gp.GurobiError as e:
                print(f"Warning: Error disposing Gurobi environment: {e.code} - {e}")
        # Help Python GC by removing references
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'env'): del self.env

    def __del__(self):
        """
        Fallback cleanup attempt when the object is garbage collected.
        Explicitly calling close() is safer.
        """
        self.close()