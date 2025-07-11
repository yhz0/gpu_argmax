import warnings
import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import typing as t # Main alias for typing
from typing import TypedDict, Optional, Dict, List, Any # Explicit imports for clarity
from abc import ABC

import logging

from .master import AbstractMasterProblem # Assuming AbstractMasterProblem is in master.py

# --- Definition of the TypedDict for Benders Cut Data ---
class BendersCutData(TypedDict):
    """
    A TypedDict defining the structure for storing data related to a single Benders cut.

    Attributes:
        beta_k (np.ndarray): The coefficients of the 'x' variables in the cut.
        alpha_k (float): The constant term (right-hand side) of the cut.
        gurobi_constr (t.Optional[gp.Constr]): The Gurobi constraint object
            representing this cut in the current Gurobi model. It can be None if
            the cut is stored but not currently in a Gurobi model (e.g., after
            a model rebuild failure or before being added).
        name (str): The unique name assigned to this cut in the Gurobi model.
    """
    beta_k: np.ndarray
    alpha_k: float
    gurobi_constr: t.Optional[gp.Constr]
    name: str

class BendersMasterProblem(AbstractMasterProblem):
    """
    Manages the Benders decomposition master problem, including an epigraph
    variable 'eta' and functionality for adding, removing, and persistently
    storing optimality cuts.

    Inherits core problem structure (min c'x, Ax R1 b, bounds) from
    AbstractMasterProblem.

    The mathematical formulation solved is:
        min  c'x + eta
        s.t. Ax R1 b             (Core constraints from base class)
             lb_x <= x <= ub_x   (Core variable bounds from base class)
             eta >= eta_lower_bound (Epigraph variable bound)
             eta - beta_k^T * x >= alpha_k (Optimality cuts added iteratively)

    Key Features:
    - Manages an epigraph variable 'eta'.
    - Stores optimality cuts with unique integer IDs. Cut coefficients and
      metadata are preserved even if the Gurobi model is rebuilt.
    - Allows removal of specific cuts by their IDs.
    - Allows clearing all stored cuts.
    - Uses Python's `logging` module for operational messages.
    - Uses `BendersCutData` (a `TypedDict`) for structured storage of cut information.
    """

    def __init__(self,
                 c: np.ndarray,
                 A: sp.csr_matrix,
                 b: np.ndarray,
                 sense1: np.ndarray,
                 lb_x: np.ndarray,
                 ub_x: np.ndarray,
                 var_names: t.Optional[t.List[str]] = None,
                 constr_names: t.Optional[t.List[str]] = None,
                 eta_lower_bound: float = -gp.GRB.INFINITY,
                 eta_name: str = "eta_expected_cost"):
        """
        Initializes the Benders master problem holder.

        Args:
            c: Cost vector for the first-stage variables 'x'.
            A: Constraint matrix for first-stage constraints Ax R1 b.
            b: Right-hand side vector for first-stage constraints.
            sense1: Sense array for first-stage constraints (e.g., '<', '=', '>').
            lb_x: Lower bounds for 'x' variables.
            ub_x: Upper bounds for 'x' variables.
            var_names: Optional list of names for 'x' variables.
            constr_names: Optional list of names for first-stage constraints.
            eta_lower_bound: Lower bound for the epigraph variable 'eta'.
                Defaults to negative infinity.
            eta_name: Name for the 'eta' variable in the Gurobi model.
        """
        super().__init__(c, A, b, sense1, lb_x, ub_x, var_names, constr_names)

        # Instance-specific logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.eta_name: str = eta_name
        self.eta_var: t.Optional[gp.Var] = None
        self.eta_lower_bound: float = eta_lower_bound
        self._eta_bound_explicitly_set: bool = (eta_lower_bound != -gp.GRB.INFINITY)
        self._benders_components_added: bool = False # True if eta_var and base model are set up

        # Stores Benders cuts: cut_id -> BendersCutData
        self.stored_cuts: t.Dict[int, BendersCutData] = {}
        self._next_cut_id: int = 0 # Counter for generating unique cut IDs

    def set_eta_lower_bound(self, new_lower_bound: float):
        """
        Sets or updates the lower bound for the epigraph variable 'eta'.

        If the Gurobi model and 'eta' variable have already been created,
        this method updates the variable's lower bound directly in the model.

        Args:
            new_lower_bound: The new lower bound for eta.
        """
        self.eta_lower_bound = new_lower_bound
        self._eta_bound_explicitly_set = True # Mark that user has set it
        if self.model is not None and self.eta_var is not None:
            try:
                self.eta_var.lb = new_lower_bound
                self.model.update()
            except gp.GurobiError as e:
                self.logger.warning(f"Gurobi error updating eta lower bound: {e}", exc_info=True)

    def create_benders_problem(self, model_name: str = "BendersMaster"):
        """
        Creates or rebuilds the Benders master problem in Gurobi.

        This involves:
        1. Calling the base class's `create_core_problem` to set up the
           deterministic part (min c'x, Ax R1 b, bounds).
        2. Adding the epigraph variable 'eta' to the model with objective coefficient 1.0.
        3. Re-adding any previously stored optimality cuts (from `self.stored_cuts`)
           to the new Gurobi model instance.

        If called again, it disposes of the old Gurobi model and rebuilds it,
        ensuring that stored cuts are re-applied.

        Args:
            model_name: Name to assign to the Gurobi model.
        """
        if not self._eta_bound_explicitly_set and self.eta_lower_bound == -gp.GRB.INFINITY:
            self.logger.warning(
                f"The lower bound for eta ('{self.eta_name}') is currently -GRB.INFINITY. "
                "If no cuts are added, the model may be unbounded. Consider using set_eta_lower_bound()."
            )
            self._eta_bound_explicitly_set = True # Suppress future identical warnings

        # Preserve data of existing cuts (coefficients, name, original ID) before model is rebuilt
        existing_cuts_to_rebuild_info: t.List[t.Dict[str, t.Any]] = []
        if hasattr(self, 'stored_cuts') and self.stored_cuts:
            self.logger.info(f"Preserving data for {len(self.stored_cuts)} existing cuts before model rebuild.")
            for cut_id, data in self.stored_cuts.items():
                existing_cuts_to_rebuild_info.append({
                    'id': cut_id,
                    'beta_k': data['beta_k'].copy(), # Ensure a fresh copy of numpy array
                    'alpha_k': data['alpha_k'],
                    'name': data['name']
                })

        # 1. Build/Rebuild the core model (min c'x, Ax R1 b, bounds)
        super().create_core_problem(model_name=model_name)

        if not self.is_model_created or self.model is None:
            # Base class logs critical error and raises, so no need to log here again
            raise RuntimeError("Core problem creation failed; cannot add Benders components.")

        try:
            # 2. Add the epigraph variable 'eta' to the new model
            self.eta_var = self.model.addVar(
                lb=self.eta_lower_bound,
                ub=gp.GRB.INFINITY,
                obj=1.0, # Objective becomes c'x + 1.0*eta
                vtype=gp.GRB.CONTINUOUS,
                name=self.eta_name,
            )
            self.model.update() # Integrate eta_var

            # 3. Re-initialize cut storage for the new model and re-add cuts
            newly_stored_cuts: t.Dict[int, BendersCutData] = {}
            if existing_cuts_to_rebuild_info:
                self.logger.info(f"Re-adding {len(existing_cuts_to_rebuild_info)} stored Benders cuts to the new model.")
                for cut_info in existing_cuts_to_rebuild_info:
                    cut_id_readd = cut_info['id']
                    beta_k_readd = cut_info['beta_k']
                    alpha_k_readd = cut_info['alpha_k']
                    cut_name_readd = cut_info['name']
                    gurobi_constr_readded: t.Optional[gp.Constr] = None
                    try:
                        if self.x_vars is None or self.eta_var is None: # Should be set by now
                            raise RuntimeError("x_vars or eta_var not available in the new model for re-adding cuts.")
                        
                        cut_lhs_expression = self.eta_var - beta_k_readd @ self.x_vars
                        gurobi_constr_readded = self.model.addConstr(
                            cut_lhs_expression >= alpha_k_readd,
                            name=cut_name_readd
                        )
                    except gp.GurobiError as e_readd:
                        self.logger.error(f"Gurobi failed to re-add cut '{cut_name_readd}' (ID: {cut_id_readd}): {e_readd.code} - {e_readd}", exc_info=True)
                    except Exception as e_readd_generic:
                        self.logger.error(f"Unexpected error re-adding cut '{cut_name_readd}' (ID: {cut_id_readd}): {e_readd_generic}", exc_info=True)
                    
                    # Create a BendersCutData compliant dictionary
                    newly_stored_cuts[cut_id_readd] = {
                        'beta_k': beta_k_readd,
                        'alpha_k': alpha_k_readd,
                        'gurobi_constr': gurobi_constr_readded, # Store new Gurobi constr or None
                        'name': cut_name_readd
                    }
                if self.model: self.model.update() # Update model after re-adding all cuts

            self.stored_cuts = newly_stored_cuts # Replace with the re-added cuts
            self._benders_components_added = True

        except gp.GurobiError as e:
            self.logger.critical(f"Gurobi error adding Benders components to '{model_name}': {e.code} - {e}", exc_info=True)
            self._benders_components_added = False; self.eta_var = None; raise
        except Exception as e:
            self.logger.critical(f"Unexpected error adding Benders components to '{model_name}': {e}", exc_info=True)
            self._benders_components_added = False; self.eta_var = None; raise


    def add_optimality_cut(self, beta_k: np.ndarray, alpha_k: float, cut_name_prefix: str = "opt_cut") -> int:
        """
        Adds a Benders optimality cut to the master problem model and stores its data.

        The cut is of the form:
            eta - beta_k^T * x >= alpha_k

        A unique integer ID is assigned to the cut, and its coefficients
        (`beta_k`, `alpha_k`), Gurobi constraint object, and generated name
        are stored internally in `self.stored_cuts` using the `BendersCutData` format.

        Args:
            beta_k: Numpy array of coefficients for the 'x' variables in the cut (size num_x).
            alpha_k: Constant term (scalar) on the right-hand side of the cut.
            cut_name_prefix: Base name for the Gurobi constraint. A unique
                             suffix based on the cut ID will be appended.

        Returns:
            int: The unique ID assigned to the newly added cut.

        Raises:
            RuntimeError: If `create_benders_problem` was not successfully called first,
                          or if internal Gurobi variables (eta_var, x_vars) are not initialized.
            ValueError: If `beta_k` has dimensions inconsistent with `self.num_x`.
            gp.GurobiError: If Gurobi fails to add the constraint.
        """
        if not self._benders_components_added or not self.is_model_created or self.model is None:
            raise RuntimeError("Benders problem structure not created. Call create_benders_problem() first.")
        if self.eta_var is None or self.x_vars is None:
            raise RuntimeError("Internal state error: eta_var or x_vars not initialized in Benders problem.")
        if not isinstance(beta_k, np.ndarray) or beta_k.shape != (self.num_x,):
            raise ValueError(f"Cut vector beta_k must be a numpy array of shape ({self.num_x},), received {beta_k.shape}")

        cut_id = self._next_cut_id
        cut_name = f"{cut_name_prefix}_{cut_id}"
        try:
            cut_lhs_expression = self.eta_var - beta_k @ self.x_vars
            constraint_object: gp.Constr = self.model.addConstr(cut_lhs_expression >= alpha_k, name=cut_name)
            
            cut_data_entry: BendersCutData = {
                'beta_k': beta_k.copy(), # Store a copy of coefficients
                'alpha_k': alpha_k,
                'gurobi_constr': constraint_object,
                'name': cut_name
            }
            self.stored_cuts[cut_id] = cut_data_entry
            self._next_cut_id += 1 # Increment for the next cut
            # self.model.update() # Consider if update is needed here or batched later
            return cut_id
        except gp.GurobiError as e:
            self.logger.error(f"Gurobi failed to add optimality cut '{cut_name}': {e.code} - {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while adding optimality cut '{cut_name}': {e}", exc_info=True)
            raise

    def remove_cuts(self, cut_ids_to_remove: t.List[int]):
        """
        Removes one or more Benders optimality cuts from the Gurobi model
        and internal storage, specified by their unique IDs.

        Args:
            cut_ids_to_remove: A list of integer cut IDs to be removed.

        Raises:
            KeyError: If any of the provided `cut_id`s are not found in the
                      `self.stored_cuts` dictionary.
            RuntimeError: If the Benders problem structure (model, eta_var) is not
                          properly initialized (though basic removal from storage
                          might proceed if model is unavailable).
            gp.GurobiError: If Gurobi encounters an error while removing constraints
                            from the model.
        """
        if not self.model or not self._benders_components_added:
            # Model isn't ready or available. Try to remove from storage only.
            for cut_id in cut_ids_to_remove:
                if cut_id in self.stored_cuts:
                    del self.stored_cuts[cut_id]
                    self.logger.info(f"Cut ID {cut_id} removed from internal storage (model not available/ready).")
                else:
                    self.logger.error(f"Attempted to remove non-existent cut ID: {cut_id} (model not available/ready).")
                    raise KeyError(f"Cut ID {cut_id} not found in stored cuts.")
            return

        gurobi_constrs_to_delete: t.List[gp.Constr] = []
        ids_actually_removed_from_storage: t.List[int] = []
        for cut_id in cut_ids_to_remove:
            if cut_id in self.stored_cuts:
                cut_data = self.stored_cuts[cut_id]
                if cut_data['gurobi_constr'] is not None: # Check if it's an active Gurobi constraint
                    gurobi_constrs_to_delete.append(cut_data['gurobi_constr'])
                del self.stored_cuts[cut_id] # Remove from internal storage
                ids_actually_removed_from_storage.append(cut_id)
            else:
                self.logger.error(f"Attempted to remove non-existent cut ID: {cut_id}")
                raise KeyError(f"Cut ID {cut_id} not found in stored cuts.")

        if gurobi_constrs_to_delete:
            try:
                self.model.remove(gurobi_constrs_to_delete)
                self.model.update()
                self.logger.info(f"Successfully removed {len(gurobi_constrs_to_delete)} Gurobi constraints for cut IDs: {ids_actually_removed_from_storage}.")
            except gp.GurobiError as e:
                self.logger.warning(f"Gurobi error removing constraints: {e}. Constraints might be partially removed or already gone.", exc_info=True)
        elif ids_actually_removed_from_storage: # If only removed from storage but not model
            self.logger.info(f"Removed {len(ids_actually_removed_from_storage)} cuts from internal storage (no corresponding Gurobi constraints found/removed).")

    def clear_all_cuts(self):
        """
        Removes all Benders optimality cuts from the Gurobi model and clears
        the internal `self.stored_cuts` dictionary.

        The `_next_cut_id` counter is NOT reset, ensuring that future cuts
        will continue to receive unique IDs throughout the instance's lifetime.
        """
        num_stored_cuts_before_clear = len(self.stored_cuts)
        if not self.stored_cuts and (not self.model or not self._benders_components_added):
            self.logger.info("No cuts to clear from storage (and/or model not available/ready).")
            return # _next_cut_id is not reset

        if self.model and self._benders_components_added:
            gurobi_constrs_to_clear: t.List[gp.Constr] = []
            for data in self.stored_cuts.values():
                if data['gurobi_constr'] is not None:
                    gurobi_constrs_to_clear.append(data['gurobi_constr'])
            
            if gurobi_constrs_to_clear:
                try:
                    self.model.remove(gurobi_constrs_to_clear)
                    self.model.update()
                    self.logger.info(f"Removed {len(gurobi_constrs_to_clear)} Gurobi cut constraints from the model.")
                except gp.GurobiError as e:
                    self.logger.warning(f"Gurobi error during batch removal of all cuts: {e}. Model state might be inconsistent.", exc_info=True)
            else:
                self.logger.info("No active Gurobi cut constraints found in the model to clear.")
        
        self.stored_cuts.clear()
        # self._next_cut_id is NOT reset.
        self.logger.info(f"Cleared {num_stored_cuts_before_clear} cuts from internal storage. Next cut ID counter remains at {self._next_cut_id}.")


    def get_cut_data(self, cut_id: int) -> t.Optional[BendersCutData]:
        """
        Retrieves the stored data for a specific Benders cut by its ID.

        Args:
            cut_id: The unique integer ID of the cut.

        Returns:
            A `BendersCutData` dictionary containing the cut's information
            (beta_k, alpha_k, gurobi_constr, name) if the ID is found,
            otherwise `None`.
        """
        return self.stored_cuts.get(cut_id)

    def get_all_cut_ids(self) -> t.List[int]:
        """
        Returns a list of all unique IDs of the currently stored Benders cuts.

        Returns:
            A list of integer cut IDs.
        """
        return list(self.stored_cuts.keys())


    def calculate_epigraph_value(self, x_candidate: np.ndarray) -> float:
        """
        Calculates the implied epigraph value (eta) for a given candidate solution x_candidate.

        The value is determined by:
        eta = max(self.eta_lower_bound, max_k(alpha_k + beta_k^T * x_candidate))
        This represents the minimum eta value that satisfies all current optimality cuts
        and the explicit lower bound on eta, for the given x_candidate.

        Args:
            x_candidate (np.ndarray): A candidate solution vector for 'x' variables.
                                      Must have shape (self.num_x,).

        Returns:
            float: The calculated epigraph value. Can be -np.inf if eta is
                   effectively unbounded below by current constraints.

        Raises:
            ValueError: If x_candidate has an incorrect shape.
        """
        if not isinstance(x_candidate, np.ndarray) or x_candidate.shape != (self.num_x,):
            raise ValueError(
                f"x_candidate must be a numpy array of shape ({self.num_x},), "
                f"received shape {x_candidate.shape if isinstance(x_candidate, np.ndarray) else type(x_candidate)}"
            )

        # Initialize with eta's own lower bound.
        current_max_rhs = self.eta_lower_bound

        # Iterate through all stored cuts to find the most restrictive one for x_candidate
        for cut_data in self.stored_cuts.values():
            # Calculate RHS of the cut: alpha_k + beta_k^T * x_candidate
            cut_rhs_at_x_candidate = cut_data['alpha_k'] + (cut_data['beta_k'] @ x_candidate)
            current_max_rhs = max(current_max_rhs, cut_rhs_at_x_candidate)

        return current_max_rhs

    def close(self):
        """
        Disposes of the Gurobi model and environment, and performs cleanup
        specific to Benders components.

        Specifically, it sets the `gurobi_constr` field in all `stored_cuts`
        to `None` as the Gurobi model they belonged to is being disposed.
        The cut coefficient data (`beta_k`, `alpha_k`, `name`) remains in
        `stored_cuts` for potential re-use if the problem is recreated.
        """
        if hasattr(self, 'stored_cuts'): # Check if attribute exists
            for cut_id in list(self.stored_cuts.keys()): # Iterate over keys if modifying dict
                # Update the gurobi_constr to None as the model is being disposed
                # This ensures the BendersCutData remains valid.
                self.stored_cuts[cut_id]['gurobi_constr'] = None
        
        if hasattr(self, 'eta_var') and self.eta_var is not None:
            self.eta_var = None # Clear reference to Gurobi variable

        super().close() # Call base class close to dispose model and env
        
        self._benders_components_added = False # Mark Benders components as not added
