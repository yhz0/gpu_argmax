import gurobipy as gp
import numpy as np
import typing as t
import logging

from benders import BendersCutData, BendersMasterProblem # Assuming benders.py is in the same directory or accessible

class RegularizedBendersMasterProblem(BendersMasterProblem):
    """
    Manages a regularized Benders decomposition master problem.

    This class extends BendersMasterProblem by adding a quadratic regularization
    term to the objective function: rho/2 * ||x - x_bar||^2.
    The problem is transformed by optimizing over d = x - x_bar.

    The mathematical formulation solved internally (in terms of d) is:
        min  c'd + eta + rho/2 * ||d||^2
        s.t. A(d + x_bar) R1 b  => Ad R1 b - A*x_bar
             lb_x - x_bar <= d <= ub_x - x_bar
             eta >= eta_lower_bound
             eta - beta_k^T * (d + x_bar) >= alpha_k_orig
               => eta - beta_k^T * d >= alpha_k_orig + beta_k^T * x_bar
                                       (Optimality cuts)

    The true solution x_true and its objective are recovered from the solution d_sol.
    x_true = d_sol + x_bar
    obj_true = (c'*d_sol + eta_sol + rho/2 * ||d_sol||^2) + c'*x_bar

    Attributes:
        rho (float): The regularization strength (penalty coefficient).
        x_bar (np.ndarray): The regularization center vector.
        c (np.ndarray): Original cost vector for 'x' variables.
    """

    def __init__(self,
                 c: np.ndarray,
                 A: t.Optional[np.ndarray], # From AbstractMasterProblem
                 b: t.Optional[np.ndarray], # From AbstractMasterProblem
                 sense1: t.Optional[np.ndarray], # From AbstractMasterProblem
                 lb_x: np.ndarray, # From AbstractMasterProblem
                 ub_x: np.ndarray, # From AbstractMasterProblem
                 var_names: t.Optional[t.List[str]] = None,
                 constr_names: t.Optional[t.List[str]] = None,
                 eta_lower_bound: float = -gp.GRB.INFINITY,
                 eta_name: str = "eta_expected_cost",
                 rho_initial: float = 0.0,
                 x_bar_initial: t.Optional[np.ndarray] = None):
        """
        Initializes the RegularizedBendersMasterProblem.

        Args:
            c, A, b, sense1, lb_x, ub_x, var_names, constr_names:
                Core problem data (passed to BendersMasterProblem).
            eta_lower_bound: Lower bound for the epigraph variable 'eta'.
            eta_name: Name for the 'eta' variable.
            rho_initial (float): Initial regularization strength. Defaults to 0.0.
            x_bar_initial (np.ndarray, optional): Initial regularization center.
                Defaults to a zero vector of appropriate size if None.
        """
        super().__init__(c, A, b, sense1, lb_x, ub_x, var_names, constr_names,
                         eta_lower_bound, eta_name)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.rho: float = rho_initial
        if x_bar_initial is None:
            self.x_bar: np.ndarray = np.zeros(self.num_x)
        else:
            if not isinstance(x_bar_initial, np.ndarray) or x_bar_initial.shape != (self.num_x,):
                raise ValueError(f"x_bar_initial must be a numpy array of shape ({self.num_x},), "
                                 f"received shape {x_bar_initial.shape if isinstance(x_bar_initial, np.ndarray) else type(x_bar_initial)}")
            self.x_bar: np.ndarray = x_bar_initial.copy()
        
        # self.c is inherited and stores the original cost vector for x.

    def set_regularization_strength(self, new_rho: float):
        """
        Sets the regularization strength (rho).

        If the Gurobi model exists, the objective function will be updated.

        Args:
            new_rho (float): The new regularization strength.
        """
        if new_rho < 0:
            self.logger.warning(f"Regularization strength rho should be non-negative. Received: {new_rho}")
        self.rho = new_rho
        if self.model and self._benders_components_added: # Check if model is ready
            self.logger.info(f"Updating regularization strength rho to {self.rho} in Gurobi model.")
            self._update_gurobi_objective()
            self.model.update()

    def get_regularization_strength(self) -> float:
        """Returns the current regularization strength (rho)."""
        return self.rho

    def set_regularization_center(self, new_x_bar: np.ndarray):
        """
        Sets the regularization center (x_bar).

        If the Gurobi model exists, variable bounds, core constraint RHS,
        and optimality cut RHS values will be updated in the model.

        Args:
            new_x_bar (np.ndarray): The new regularization center vector.
                                    Must have shape (self.num_x,).
        Raises:
            ValueError: If new_x_bar has an incorrect shape.
        """
        if not isinstance(new_x_bar, np.ndarray) or new_x_bar.shape != (self.num_x,):
            raise ValueError(f"new_x_bar must be a numpy array of shape ({self.num_x},), "
                             f"received shape {new_x_bar.shape if isinstance(new_x_bar, np.ndarray) else type(new_x_bar)}")
        
        self.x_bar = new_x_bar.copy()
        if self.model and self._benders_components_added: # Check if model is ready
            self.logger.info("Updating regularization center x_bar. Gurobi model components will be adjusted.")
            self._update_gurobi_model_for_x_bar_change()
            self.model.update()


    def get_regularization_center(self) -> np.ndarray:
        """Returns a copy of the current regularization center (x_bar)."""
        return self.x_bar.copy()

    def _update_gurobi_objective(self):
        """
        (Private) Updates the Gurobi model's objective function to:
        c'd + eta + rho/2 * ||d||^2
        where 'd' variables are represented by self.x_vars.
        """
        if not self.model or self.x_vars is None or self.eta_var is None:
            self.logger.warning("Cannot update Gurobi objective: model or key variables not initialized.")
            return

        try:
            # Linear part: c'd + 1.0*eta
            linear_expr = self.c @ self.x_vars + self.eta_var
            
            # Quadratic part: 0.5 * rho * d'I d
            # Gurobi's MVar @ MVar performs dot product for 1D MVars.
            if self.rho != 0:
                quadratic_expr = 0.5 * self.rho * (self.x_vars @ self.x_vars)
                objective_expr = linear_expr + quadratic_expr
            else:
                objective_expr = linear_expr
                
            self.model.setObjective(objective_expr, gp.GRB.MINIMIZE)
            # self.model.update() # Usually called by the calling method
            self.logger.debug("Gurobi objective updated with regularization term.")
        except gp.GurobiError as e:
            self.logger.error(f"Gurobi error updating objective: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error updating objective: {e}", exc_info=True)
            raise

    def _update_gurobi_model_for_x_bar_change(self):
        """
        (Private) Updates Gurobi model components (variable bounds, constraint RHS)
        when x_bar changes.
        Assumes self.model, self.x_vars, and self.eta_var are initialized.
        """
        if not self.model or self.x_vars is None:
            self.logger.warning("Cannot update model for x_bar change: model or d-variables not initialized.")
            return

        try:
            # 1. Update d-variable bounds: lb_x - x_bar <= d <= ub_x - x_bar
            # self.lb_x and self.ub_x are original bounds for x
            self.x_vars.lb = self.lb_x - self.x_bar
            self.x_vars.ub = self.ub_x - self.x_bar
            self.logger.debug("D-variable bounds updated for new x_bar.")

            # 2. Update core constraint RHS: Ad R1 b_orig - A*x_bar
            if self.stage1_constraints is not None and self.A is not None and self.b is not None:
                # self.b stores original b_orig
                # self.A stores original A
                new_rhs_core = self.b - (self.A @ self.x_bar)
                self.stage1_constraints.rhs = new_rhs_core
                self.logger.debug("Core constraint RHS updated for new x_bar.")

            # 3. Update RHS of existing optimality cuts in the Gurobi model
            # eta - beta_k^T * d >= alpha_k_orig + beta_k^T * x_bar
            for cut_id, cut_data in self.stored_cuts.items():
                if cut_data['gurobi_constr'] is not None: # If the cut is active in the model
                    beta_k = cut_data['beta_k']
                    alpha_k_orig = cut_data['alpha_k'] # Original alpha_k
                    new_rhs_cut = alpha_k_orig + (beta_k @ self.x_bar)
                    
                    # Gurobi constraints are typically expr sense rhs. We modify the RHS.
                    # For cut: eta - beta_k^T * d >= new_rhs_cut
                    # Gurobi stores constraints as Ax=b, Ax<=b, Ax>=b.
                    # If constraint is LinExpr >= RHS, then RHS is on the right.
                    # Constraint object's RHS attribute can be set.
                    cut_data['gurobi_constr'].rhs = new_rhs_cut
            self.logger.debug("Optimality cut RHS values in Gurobi model updated for new x_bar.")
            
            # self.model.update() # Usually called by the calling method
        except gp.GurobiError as e:
            self.logger.error(f"Gurobi error updating model for x_bar change: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error updating model for x_bar change: {e}", exc_info=True)
            raise

    def create_benders_problem(self, model_name: str = "RegularizedBendersMaster"):
        """
        Creates or rebuilds the regularized Benders master problem in Gurobi.

        This method builds the Gurobi model in terms of variables d = x - x_bar.
        It sets up the transformed objective, variable bounds, core constraints,
        and re-adds any persistently stored optimality cuts with transformed RHS.

        Args:
            model_name (str): Name to assign to the Gurobi model.
        """
        self.logger.info(f"Creating/rebuilding regularized Benders master problem: '{model_name}'")
        if not self._eta_bound_explicitly_set and self.eta_lower_bound == -gp.GRB.INFINITY:
            self.logger.warning(
                f"The lower bound for eta ('{self.eta_name}') is currently -GRB.INFINITY. "
                "If no cuts are added, the model may be unbounded."
            ) # Warning from parent is fine, this is just a reminder.

        # Preserve data of existing cuts (coefficients, name, original ID)
        existing_cuts_to_rebuild_info: t.List[t.Dict[str, t.Any]] = []
        if hasattr(self, 'stored_cuts') and self.stored_cuts:
            self.logger.info(f"Preserving data for {len(self.stored_cuts)} existing cuts before model rebuild.")
            for cut_id, data in self.stored_cuts.items():
                existing_cuts_to_rebuild_info.append({
                    'id': cut_id, 'beta_k': data['beta_k'].copy(),
                    'alpha_k': data['alpha_k'], 'name': data['name']
                })

        # Dispose of old model if it exists
        if self.model is not None:
            try:
                self.model.dispose()
            except gp.GurobiError as e:
                self.logger.warning(f"Error disposing previous Gurobi model: {e}", exc_info=True)
            finally:
                self.model = None
                self.x_vars = None # Represents d-variables
                self.eta_var = None
                self.stage1_constraints = None
                self._model_created = False
                self._benders_components_added = False


        try:
            # Create new Gurobi model
            self.model = gp.Model(name=model_name, env=self.env)
            self.model.setParam('OutputFlag', 0) # Default Gurobi console output off

            # Define d-variables (x - x_bar), store them in self.x_vars for consistency
            # Bounds are lb_x - x_bar <= d <= ub_x - x_bar
            d_lb = self.lb_x - self.x_bar
            d_ub = self.ub_x - self.x_bar
            self.x_vars = self.model.addMVar(
                shape=self.num_x, lb=d_lb, ub=d_ub, name=self.var_names # No obj here yet
            )
            self.logger.debug("d-variables (x - x_bar) created.")

            # Add eta variable
            self.eta_var = self.model.addVar(
                lb=self.eta_lower_bound, ub=gp.GRB.INFINITY, name=self.eta_name # No obj here yet
            )
            self.logger.debug("Eta variable created.")
            
            self.model.update() # Integrate variables before setting objective

            # Set the full objective: c'd + eta + rho/2 * ||d||^2
            self._update_gurobi_objective() # This sets the full objective

            # Add core constraints: Ad R1 (b_orig - A*x_bar)
            if self.num_stage1_constrs > 0 and self.A is not None and self.b is not None and self.sense1 is not None:
                rhs_core_transformed = self.b - (self.A @ self.x_bar)
                senses_char = np.array([s if isinstance(s, str) else chr(s) for s in self.sense1])
                self.stage1_constraints = self.model.addMConstr(
                    A=self.A, x=self.x_vars, sense=senses_char, b=rhs_core_transformed, name=self.constr_names
                )
                self.logger.debug("Core constraints added with transformed RHS.")
            else:
                self.stage1_constraints = None

            # Re-add stored optimality cuts with transformed RHS
            # self.stored_cuts holds original alpha_k
            newly_stored_cuts_after_rebuild: t.Dict[int, BendersCutData] = {}
            if existing_cuts_to_rebuild_info:
                self.logger.info(f"Re-adding {len(existing_cuts_to_rebuild_info)} stored Benders cuts to the new model.")
                for cut_info in existing_cuts_to_rebuild_info:
                    cut_id_readd = cut_info['id']
                    beta_k = cut_info['beta_k']
                    alpha_k_orig = cut_info['alpha_k'] # Original alpha
                    cut_name_readd = cut_info['name']
                    
                    rhs_cut_transformed = alpha_k_orig + (beta_k @ self.x_bar)
                    cut_lhs_expr = self.eta_var - (beta_k @ self.x_vars) # d-variables are self.x_vars
                    
                    gurobi_constr_readded: t.Optional[gp.Constr] = None
                    try:
                        gurobi_constr_readded = self.model.addConstr(
                            cut_lhs_expr >= rhs_cut_transformed, name=cut_name_readd
                        )
                    except gp.GurobiError as e_readd:
                        self.logger.error(f"Gurobi failed to re-add cut '{cut_name_readd}' (ID: {cut_id_readd}): {e_readd}", exc_info=True)
                    except Exception as e_readd_generic:
                         self.logger.error(f"Unexpected error re-adding cut '{cut_name_readd}' (ID: {cut_id_readd}): {e_readd_generic}", exc_info=True)

                    newly_stored_cuts_after_rebuild[cut_id_readd] = {
                        'beta_k': beta_k, # Already a copy from preservation step
                        'alpha_k': alpha_k_orig,
                        'gurobi_constr': gurobi_constr_readded,
                        'name': cut_name_readd
                    }
                self.stored_cuts = newly_stored_cuts_after_rebuild
            
            self.model.ModelSense = gp.GRB.MINIMIZE
            self.model.update()
            self._model_created = True
            self._benders_components_added = True # Mark Benders-specific (eta, cuts) and regularized setup as complete
            self.logger.info(f"Regularized Benders master problem '{model_name}' successfully created/rebuilt.")

        except gp.GurobiError as e:
            self.logger.critical(f"Gurobi error creating regularized master model '{model_name}': {e}", exc_info=True)
            self._model_created = False; self._benders_components_added = False; raise
        except Exception as e:
            self.logger.critical(f"Unexpected error creating regularized master model '{model_name}': {e}", exc_info=True)
            self._model_created = False; self._benders_components_added = False; raise

    def add_optimality_cut(self, beta_k: np.ndarray, alpha_k_orig: float, cut_name_prefix: str = "opt_cut") -> int:
        """
        Adds a Benders optimality cut using original alpha_k (before x_bar transformation).
        The cut stored is eta - beta_k^T * x >= alpha_k_orig.
        The cut added to Gurobi is eta - beta_k^T * d >= alpha_k_orig + beta_k^T * x_bar.

        Args:
            beta_k (np.ndarray): Coefficients for x variables (and thus for d variables).
            alpha_k_orig (float): Original constant term for the cut related to x.
            cut_name_prefix (str): Prefix for the cut name.

        Returns:
            int: The ID of the added cut.
        
        Raises:
            RuntimeError, ValueError, gp.GurobiError as in parent,
            plus errors from transformation.
        """
        if not self._benders_components_added or not self.is_model_created or self.model is None:
            raise RuntimeError("Regularized Benders problem structure not created. Call create_benders_problem() first.")
        if self.eta_var is None or self.x_vars is None: # self.x_vars are d-variables
            raise RuntimeError("Internal state error: eta_var or d-variables (self.x_vars) not initialized.")
        if not isinstance(beta_k, np.ndarray) or beta_k.shape != (self.num_x,):
            raise ValueError(f"Cut vector beta_k must be a numpy array of shape ({self.num_x},), received {beta_k.shape}")

        cut_id = self._next_cut_id # Get next available ID from parent
        cut_name = f"{cut_name_prefix}_{cut_id}"

        try:
            # RHS for Gurobi model: alpha_k_orig + beta_k^T * x_bar
            rhs_for_gurobi_model = alpha_k_orig + (beta_k @ self.x_bar)
            
            # LHS: eta - beta_k^T * d (where d are self.x_vars)
            cut_lhs_expression = self.eta_var - (beta_k @ self.x_vars)
            
            constraint_object: gp.Constr = self.model.addConstr(
                cut_lhs_expression >= rhs_for_gurobi_model, name=cut_name
            )
            
            # Store with original alpha_k
            cut_data_entry: BendersCutData = {
                'beta_k': beta_k.copy(),
                'alpha_k': alpha_k_orig, # Store original alpha
                'gurobi_constr': constraint_object,
                'name': cut_name
            }
            self.stored_cuts[cut_id] = cut_data_entry
            self._next_cut_id += 1 # Increment for the next cut
            
            # self.model.update() # Optional: batch updates might be better
            self.logger.info(f"Added optimality cut ID {cut_id}: '{cut_name}' with transformed RHS for x_bar.")
            return cut_id
        except gp.GurobiError as e:
            self.logger.error(f"Gurobi failed to add regularized optimality cut '{cut_name}': {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error adding regularized optimality cut '{cut_name}': {e}", exc_info=True)
            raise

    def solve(self, set_output_flag: t.Optional[bool] = None, **gurobi_params) \
            -> t.Tuple[t.Optional[np.ndarray], t.Optional[float], int]:
        """
        Solves the regularized Benders master problem (which is in terms of d-variables).
        Transforms the solution (d_sol) and objective back to the original x-space.

        Args:
            set_output_flag (bool | None): Temporarily overrides Gurobi's OutputFlag.
            **gurobi_params: Additional Gurobi parameters for this solve.

        Returns:
            tuple[np.ndarray | None, float | None, int]:
                - x_true_solution (np.ndarray | None): Solution for original 'x' variables.
                - true_objective_value (float | None): Objective value for original problem.
                - status_code (int): Gurobi status code.
        """
        if not self._model_created or self.model is None:
            raise RuntimeError("Model not created. Call create_benders_problem() before solve().")

        # The parent solve() method optimizes self.model, which is in terms of d.
        # It returns d_solution (as self.x_vars.X), internal_obj_val, status
        d_solution_values, internal_objective_val, status_code = \
            super(BendersMasterProblem, self).solve(set_output_flag=set_output_flag, **gurobi_params)
            # We call BendersMasterProblem.solve, not AbstractMasterProblem.solve directly,
            # in case BendersMasterProblem.solve has specific logic we want to retain
            # (though currently it just calls AbstractMasterProblem.solve).

        x_true_solution: t.Optional[np.ndarray] = None
        true_objective_value: t.Optional[float] = None

        if status_code == gp.GRB.OPTIMAL:
            if d_solution_values is not None:
                # Transform d solution back to x solution: x_true = d_sol + x_bar
                x_true_solution = d_solution_values + self.x_bar
                
                # Transform objective: obj_true = (c'd + eta + rho/2 ||d||^2) + c'*x_bar
                # internal_objective_val is (c'd + eta + rho/2 ||d||^2)
                if internal_objective_val is not None:
                    true_objective_value = internal_objective_val + (self.c @ self.x_bar)
                else: # Should not happen if optimal and d_solution_values is not None
                    self.logger.warning("Optimal solution for d, but internal_objective_val is None.")
                    
                self.logger.info(f"Solved regularized problem. Optimal d found. "
                                 f"True x objective: {true_objective_value if true_objective_value is not None else 'N/A'}")
            else: # Should not happen if optimal
                self.logger.warning("Optimal status, but d_solution_values is None.")
        else:
            # If not optimal, the internal_objective_val might still be relevant (e.g., bound if infeasible/unbounded)
            # For simplicity, we only calculate true_objective if optimal.
            # If needed, one could try to compute a "true" bound if internal_objective_val is a bound.
            true_objective_value = internal_objective_val # Or None, depending on desired behavior
            self.logger.info(f"Regularized problem solved with status: {status_code}. "
                             f"Internal objective: {internal_objective_val if internal_objective_val is not None else 'N/A'}")


        return x_true_solution, true_objective_value, status_code
