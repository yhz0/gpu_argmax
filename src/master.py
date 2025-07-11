import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import typing as t
from abc import ABC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from smps_reader import SMPSReader


class AbstractMasterProblem(ABC):

    """
    Base class for representing first-stage master problems in decomposition algorithms.

    Manages its own Gurobi environment and provides methods to create the
    core deterministic problem:
        min c'x
        s.t. Ax R1 b  (sense1 respected)
             lb_x <= x <= ub_x

    Includes a concrete `solve` method. Subclasses are expected to modify
    `self.model` before calling `solve`.
    """

    def __init__(self,
                 c: np.ndarray,
                 A: sp.csr_matrix,
                 b: np.ndarray,
                 sense1: np.ndarray,
                 lb_x: np.ndarray,
                 ub_x: np.ndarray,
                 var_names: t.Optional[t.List[str]] = None,
                 constr_names: t.Optional[t.List[str]] = None):
        """
        Initializes the master problem holder with core problem data.
        (Args description omitted for brevity - see previous version)
        """
        # --- Store Core Problem Data ---
        self.c = c.copy()
        self.A = A.copy() if A is not None else None
        self.b = b.copy() if b is not None else None
        self.sense1 = sense1.copy() if sense1 is not None else None
        self.lb_x = lb_x.copy()
        self.ub_x = ub_x.copy()
        self.num_x = len(c)
        self.num_stage1_constrs = len(b) if b is not None else 0
        self.var_names = var_names[:] if var_names else [f"x_{i}" for i in range(self.num_x)]
        self.constr_names = constr_names[:] if constr_names else [f"stage1_cons_{i}" for i in range(self.num_stage1_constrs)]

        # --- Gurobi Objects ---
        self.env: gp.Env = None
        self.model: gp.Model = None
        self.x_vars: gp.MVar = None
        self.stage1_constraints: gp.MConstr = None
        self._model_created: bool = False

        self._setup_gurobi_environment()

    @classmethod
    def from_smps_reader(cls, reader: 'SMPSReader') -> 'AbstractMasterProblem':
        """
        Factory method to create an instance from a loaded SMPSReader object.

        Extracts the required first-stage problem data (c, A, b, sense1,
        bounds, names) directly from the reader's attributes. Assumes the
        reader object has the necessary attributes populated.

        Args:
            reader: An instance of SMPSReader (or compatible structure).

        Returns:
            An instance of the class this method is called on (e.g.,
            AbstractMasterProblem or a subclass).

        Raises:
            AttributeError: If the reader is missing required attributes.
            ValueError: If essential reader attributes are unexpectedly None.
        """
        # Check if reader appears loaded (essential attributes exist)
        required_attrs = [
            'c', 'A', 'b', 'sense1', 'lb_x', 'ub_x',
            'stage1_var_names', 'stage1_constr_names', 'model' # Include 'model' as load check
        ]
        for attr in required_attrs:
            if not hasattr(reader, attr):
                # More specific error if fundamental attributes like 'c' are missing
                if attr == 'c' or attr == 'model':
                     raise AttributeError(f"Input 'reader' ({type(reader)}) is missing fundamental attribute '{attr}'. Is it a valid SMPSReader?")
                # For potentially optional attributes, raise if missing (could refine logic if needed)
                raise AttributeError(f"SMPSReader instance is missing expected attribute '{attr}'.")

            value = getattr(reader, attr)
            # Check essential attributes are not None (allowing for optional A, b, etc. if no constraints)
            is_optional_constr_data = attr in ['A', 'b', 'sense1', 'stage1_constr_names']
            num_cons1 = len(getattr(reader, 'b', [])) if hasattr(reader, 'b') and getattr(reader, 'b') is not None else 0

            if value is None and not (is_optional_constr_data and num_cons1 == 0):
                 raise ValueError(f"SMPSReader attribute '{attr}' is None but expected non-None value based on problem structure.")

        # Create an instance using extracted data
        return cls(
            c=reader.c,
            A=reader.A,
            b=reader.b,
            sense1=reader.sense1,
            lb_x=reader.lb_x,
            ub_x=reader.ub_x,
            var_names=reader.stage1_var_names,
            constr_names=reader.stage1_constr_names
        )

    def _setup_gurobi_environment(self):
        """Sets up the independent Gurobi environment. Raises error on failure."""
        try:
            self.env = gp.Env(empty=True)
            self.env.start()
        except gp.GurobiError as e:
            print(f"FATAL: Error creating Gurobi environment: {e.code} - {e}")
            raise
        except Exception as e:
             print(f"FATAL: An unexpected error occurred during Gurobi environment setup: {e}")
             raise

    def create_core_problem(self, model_name: str = "MasterCoreProblem"):
        """
        Builds or rebuilds the internal Gurobi model containing only the core
        deterministic problem structure (min c'x s.t. Ax R1 b, bounds).
        """
        try:
            if self.model is not None:
                self.model.dispose()
                self.x_vars = None; self.stage1_constraints = None; self._model_created = False

            self.model = gp.Model(name=model_name, env=self.env)
            self.model.setParam('OutputFlag', 0)

            self.x_vars = self.model.addMVar(
                shape=self.num_x, lb=self.lb_x, ub=self.ub_x, obj=self.c, name=self.var_names
            )

            if self.num_stage1_constrs > 0 and self.A is not None and self.sense1 is not None:
                senses_char = np.array([s if isinstance(s, str) else chr(s) for s in self.sense1])
                self.stage1_constraints = self.model.addMConstr(
                    A=self.A, x=self.x_vars, sense=senses_char, b=self.b, name=self.constr_names
                )
            else:
                self.stage1_constraints = None

            self.model.ModelSense = gp.GRB.MINIMIZE
            self.model.update()
            self._model_created = True
        except gp.GurobiError as e:
            print(f"FATAL: Gurobi error creating core model '{model_name}': {e.code} - {e}")
            self._model_created = False; raise
        except Exception as e:
            print(f"FATAL: Unexpected error creating core model '{model_name}': {e}")
            self._model_created = False; raise


    def solve(self, set_output_flag: t.Optional[bool] = None, **gurobi_params) -> t.Tuple[t.Optional[np.ndarray], t.Optional[float], int]:
        """
        Solves the current Gurobi master problem (`self.model`) using Gurobi.

        This method optimizes the model in its current state. It's expected
        that subclasses will have modified `self.model` appropriately (e.g.,
        by adding cuts or columns via methods like `add_optimality_cut`)
        before this method is invoked.

        It handles temporary Gurobi parameter settings specified via arguments
        for the duration of this specific solve call and restores the original
        settings afterwards.

        Args:
            set_output_flag (bool | None): Temporarily overrides the model's
                OutputFlag setting for this solve. If True, turns Gurobi console
                output on; if False, turns it off. If None (default), the model's
                current OutputFlag setting is used.
            **gurobi_params: Accepts additional Gurobi parameters as keyword
                             arguments (e.g., `Method=1`, `Threads=4`, `Presolve=0`).
                             These parameters are set only for this specific
                             solve call and are restored to their previous values
                             after the solve attempt completes or fails.

        Returns:
            tuple[np.ndarray | None, float | None, int]: A tuple containing:
                - x_solution (np.ndarray | None): The optimal values of the core 'x'
                  variables (from `self.x_vars`) if the solve status is Optimal
                  (e.g., gp.GRB.OPTIMAL). Returns None otherwise.
                - objective_value (float | None): The optimal objective function
                  value of the solved model if the solve status is Optimal.
                  Returns None otherwise.
                - status_code (int): The final Gurobi status code from the
                  `model.optimize()` call (e.g., gp.GRB.OPTIMAL,
                  gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED). Returns -999 in case of
                  uncaught Python exceptions during the solve process, or the
                  Gurobi error code if a GurobiError occurs during optimize.

        Raises:
            RuntimeError: If the internal Gurobi model (`self.model`) has not
                          been created yet (i.e., `create_core_problem` or a
                          subclass equivalent like `create_benders_problem`
                          was not called successfully).
            # Note: Internal GurobiErrors during the optimize call itself or
            # during temporary parameter setting/restoration are caught within
            # the method and typically reflected in the returned status code.
            # Errors during model creation (e.g., invalid parameters passed
            # via gurobi_params) might still raise GurobiError directly.
        """
        if not self._model_created or self.model is None:
            raise RuntimeError("Core model not created. Call create_core_problem() before solve().")

        original_params = {}; original_output_flag = None
        try:
            if set_output_flag is not None:
                 original_output_flag = self.model.Params.OutputFlag
                 self.model.setParam('OutputFlag', 1 if set_output_flag else 0)
            for key, value in gurobi_params.items():
                original_params[key] = self.model.getParamInfo(key)[2]
                self.model.setParam(key, value)

            self.model.optimize()
            status = self.model.Status

            x_solution = None; objective_value = None
            if status == gp.GRB.OPTIMAL:
                if self.x_vars is not None: x_solution = self.x_vars.X
                else: print("Warning: Model optimal but self.x_vars is None.")
                objective_value = self.model.ObjVal
            return x_solution, objective_value, status
        except gp.GurobiError as e:
            print(f"Gurobi error during solve(): {e.code} - {e}")
            return None, None, getattr(e, 'errno', -999)
        except Exception as e:
            print(f"Unexpected error during solve(): {e}")
            return None, None, -999
        finally:
            if original_output_flag is not None:
                try: self.model.setParam('OutputFlag', original_output_flag)
                except (gp.GurobiError, AttributeError): pass
            for key, value in original_params.items():
                 try: self.model.setParam(key, value)
                 except (gp.GurobiError, AttributeError): pass


    @property
    def is_model_created(self) -> bool:
        """Check if the core Gurobi model has been successfully created."""
        return self._model_created


    def dump_model(self, filename: str = "debug_master_problem.lp"):
        """
        Writes the current state of the internal Gurobi model to a file.

        This is useful for debugging the master problem structure at any point,
        including any modifications like added cuts or variables made by
        subclass methods or external logic. The file format is determined
        by Gurobi based on the filename extension (e.g., .lp, .mps, .rew).

        Args:
            filename: The path (including extension) where the model file
                      should be written. Defaults to "debug_master_problem.lp".

        Raises:
            RuntimeError: If the internal Gurobi model (`self.model`) has not
                          been created yet (e.g., if `create_core_problem`
                          or a subclass equivalent hasn't been called).
            # GurobiError can also be raised by model.write() on I/O or Gurobi errors.
        """
        # Check if the model object exists and has been initialized
        if not self.is_model_created or self.model is None:
            raise RuntimeError(
                "Cannot dump model: The Gurobi model has not been created yet. "
                "Call create_core_problem() or a subclass equivalent first."
            )

        try:
            print(f"INFO: Writing current master problem model to '{filename}'...")
            self.model.write(filename)
            print(f"INFO: Model successfully written to '{filename}'.")

        except gp.GurobiError as e:
            print(f"ERROR: Gurobi failed to write model to file '{filename}':")
            print(f"  Error code {e.code}: {e}")
            raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while writing model to '{filename}': {e}")


    def close(self):
        """
        Disposes of the Gurobi model and environment to release resources.
        """
        if hasattr(self, 'model') and self.model is not None:
            try: self.model.dispose()
            except gp.GurobiError as e: print(f"Warning: Error disposing Gurobi model: {e.code} - {e}")
            finally: self.model = None; self.x_vars = None; self.stage1_constraints = None; self._model_created = False
        if hasattr(self, 'env') and self.env is not None:
            try: self.env.dispose()
            except gp.GurobiError as e: print(f"Warning: Error disposing Gurobi environment: {e.code} - {e}")
            finally: self.env = None

    def __del__(self):
        """Attempt fallback cleanup."""
        self.close()
