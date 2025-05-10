import warnings
import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import typing as t

from master import AbstractMasterProblem

class BendersMasterProblem(AbstractMasterProblem):
    """
    Manages the Benders decomposition master problem (single-cut formulation).

    Inherits core problem structure (min c'x, Ax R1 b, bounds) from
    AbstractMasterProblem and adds an epigraph variable 'eta' and methods
    to add optimality cuts.

    The mathematical formulation solved is:
        min  c'x + eta
        s.t. Ax R1 b                  (Core constraints)
             lb_x <= x <= ub_x        (Core variable bounds)
             eta >= eta_lower_bound   (Epigraph variable bound)
             eta - beta_k^T * x >= alpha_k  (Optimality cuts added iteratively)
    """

    def __init__(self,
                 # Core problem arguments (pass to base class)
                 c: np.ndarray,
                 A: sp.csr_matrix,
                 b: np.ndarray,
                 sense1: np.ndarray,
                 lb_x: np.ndarray,
                 ub_x: np.ndarray,
                 var_names: t.Optional[t.List[str]] = None,
                 constr_names: t.Optional[t.List[str]] = None,
                 # Benders-specific arguments
                 eta_lower_bound: float = -gp.GRB.INFINITY,
                 eta_name: str = "eta_expected_cost"):
        """
        Initializes the Benders master problem holder.

        Args:
            c, A, b, sense1, lb_x, ub_x, var_names, constr_names: Data for the
                core first-stage problem (passed to AbstractMasterProblem).
            eta_lower_bound: Lower bound for the epigraph variable eta. Defaults
                to negative infinity (no lower bound).
            eta_name: Name assigned to the eta variable in the Gurobi model.
        """
        # Initialize base class with core deterministic data
        super().__init__(c, A, b, sense1, lb_x, ub_x, var_names, constr_names)

        # --- Benders Specific Attributes ---
        self.eta_name = eta_name
        self.eta_var: gp.Var = None # Will hold the reference to the Gurobi eta variable
        self.optimality_cuts: t.List[gp.Constr] = [] # Stores references to added cut constraints
        self._benders_components_added: bool = False

        self.eta_lower_bound = eta_lower_bound
        self._eta_bound_explicitly_set: bool = (eta_lower_bound != -gp.GRB.INFINITY)


    def set_eta_lower_bound(self, new_lower_bound: float):
        """
        Sets or updates the lower bound for the epigraph variable 'eta'.

        If the Gurobi model and 'eta' variable have already been created,
        this method will update the variable's lower bound directly in the model.

        Args:
            new_lower_bound: The new lower bound for eta.
        """
        self.eta_lower_bound = new_lower_bound
        self._eta_bound_explicitly_set = True # Mark that it was set by the user

        if self.model is not None and self.eta_var is not None:
            self.eta_var.lb = new_lower_bound
            self.model.update()


    def create_benders_problem(self, model_name: str = "BendersMaster"):
        """
        Creates the core problem structure via the base class, then adds the
        Benders epigraph variable 'eta' and ensures the objective is c'x + eta.

        This method should be called after initialization and before adding cuts
        or solving. If called again, it rebuilds the model.

        Args:
            model_name: Name for the Gurobi model.
        """

        # --- Warning for eta_lower_bound ---
        if not self._eta_bound_explicitly_set and self.eta_lower_bound == -gp.GRB.INFINITY:
            warnings.warn(
                f"The lower bound for eta ('{self.eta_name}') is currently -GRB.INFINITY. "
                "Cuts should be added prior to solving. The model may be unbounded below.",
                UserWarning
            )
            # suppress further warnings
            self._eta_bound_explicitly_set = True

        # 1. Build the core model (min c'x, Ax R1 b, bounds)
        super().create_core_problem(model_name=model_name)

        if not self.is_model_created or self.model is None:
             raise RuntimeError("Core problem creation failed; cannot add Benders components.")

        try:
            # 2. Add the epigraph variable 'eta'
            # Its objective coefficient (1.0) contributes to the overall objective
            self.eta_var = self.model.addVar(
                lb=self.eta_lower_bound,
                ub=gp.GRB.INFINITY,
                obj=1.0, # Makes objective c'x + 1.0*eta
                vtype=gp.GRB.CONTINUOUS,
                name=self.eta_name,
            )

            # 3. Update model (integrates the new variable)
            self.model.update()

            # Reset internal state specific to Benders structure
            self.optimality_cuts = []
            self._benders_components_added = True

        except gp.GurobiError as e:
            print(f"FATAL: Gurobi error adding Benders components to '{model_name}': {e.code} - {e}")
            self._benders_components_added = False
            raise
        except Exception as e:
             print(f"FATAL: Unexpected error adding Benders components to '{model_name}': {e}")
             self._benders_components_added = False
             raise

    def add_optimality_cut(self, beta_k: np.ndarray, alpha_k: float, cut_name_prefix: str = "opt_cut"):
         """
         Adds a Benders optimality cut to the master problem model.

         The cut is added in the form:
             eta - beta_k^T * x >= alpha_k

         Assumes beta_k and alpha_k have been correctly derived from subproblem
         duals (or extreme rays for feasibility cuts, if implemented separately).

         Args:
             beta_k: Vector of coefficients for the 'x' variables in the cut
                     (numpy array, size num_x).
             alpha_k: Constant term (scalar) on the right-hand side of the cut.
             cut_name_prefix: Base name for the constraint added to the Gurobi
                              model. A unique index is appended.

         Raises:
            RuntimeError: If `create_benders_problem` was not successfully called first.
            ValueError: If `beta_k` has dimensions inconsistent with `num_x`.
            gp.GurobiError: If Gurobi fails to add the constraint.
         """
         # Verify that the necessary Benders structure exists
         if not self._benders_components_added or not self.is_model_created or self.model is None:
             raise RuntimeError("Benders problem structure not created via create_benders_problem().")
         if self.eta_var is None or self.x_vars is None:
              raise RuntimeError("Internal state error: eta_var or x_vars not initialized.")

         # Validate input shape
         if not isinstance(beta_k, np.ndarray) or beta_k.shape != (self.num_x,):
             raise ValueError(
                 f"Cut vector beta_k must be a numpy array of shape ({self.num_x},), "
                 f"received shape {beta_k.shape}"
             )

         try:
             # Construct the linear expression: eta - sum(beta_k[i] * x_i)
             cut_lhs_expression = self.eta_var - beta_k @ self.x_vars

             # Create a unique name for the cut
             cut_index = len(self.optimality_cuts)
             cut_name = f"{cut_name_prefix}_{cut_index}"

             # Add the constraint to the Gurobi model
             constraint_object = self.model.addConstr(cut_lhs_expression >= alpha_k, name=cut_name)

             # Store reference to the added cut
             self.optimality_cuts.append(constraint_object)

         except gp.GurobiError as e:
              print(f"ERROR: Gurobi failed to add optimality cut '{cut_name}': {e.code} - {e}")
              raise
         except Exception as e:
              print(f"ERROR: An unexpected error occurred while adding optimality cut '{cut_name}': {e}")
              raise

    # Note: solve() method is inherited from AbstractMasterProblem.
    # Note: close() and __del__ methods are inherited from AbstractMasterProblem.