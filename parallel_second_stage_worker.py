import numpy as np
import scipy.sparse as sp
import multiprocessing
import os # For debugging process IDs, can be removed
from typing import List, Tuple, Optional, TYPE_CHECKING

from smps_reader import SMPSReader
from second_stage_worker import SecondStageWorker

# These global variables will exist in each spawned worker process.
_g_worker_instance: Optional[SecondStageWorker] = None
_g_current_x_for_worker_process: Optional[np.ndarray] = None
_g_worker_init_args_for_process: Optional[tuple] = None

def init_worker_process(worker_constructor_args_tuple: tuple, initial_x_for_worker: np.ndarray):
    """
    Initializer function for each worker process in the multiprocessing.Pool.
    It creates a SecondStageWorker instance and sets its initial x state.
    """
    global _g_worker_instance, _g_current_x_for_worker_process, _g_worker_init_args_for_process

    if _g_worker_instance is None:  # Create worker only if it doesn't exist in this process
        # print(f"Process {os.getpid()}: init_worker_process creating new SecondStageWorker.") # Debug
        _g_worker_init_args_for_process = worker_constructor_args_tuple
        _g_worker_instance = SecondStageWorker(*_g_worker_init_args_for_process)
    # else:
        # print(f"Process {os.getpid()}: init_worker_process reusing existing SecondStageWorker.") # Debug

    # This function is called when processes are created as part of the pool.
    # Set the initial x for the worker.
    # print(f"Process {os.getpid()}: init_worker_process setting initial x for worker.") # Debug
    _g_worker_instance.set_x(initial_x_for_worker)
    _g_current_x_for_worker_process = initial_x_for_worker.copy()

def solve_scenario_task_glob_worker(task_details: tuple) -> tuple:
    """
    Task function executed by worker processes in the pool.
    It uses the process-global SecondStageWorker to solve a single scenario.
    """
    global _g_worker_instance, _g_current_x_for_worker_process
    scenario_idx, short_delta_r, vbasis, cbasis, x_from_main_process = task_details

    if _g_worker_instance is None:
        # This should not happen if init_worker_process ran correctly.
        raise RuntimeError(f"Worker not initialized in process {os.getpid()}. This is unexpected.")

    # Sanity check: Ensure the worker's x matches the x intended for this batch.
    # This should be true if the pool is managed correctly (recreated on x change).
    if not np.array_equal(_g_current_x_for_worker_process, x_from_main_process):
        # This indicates a potential issue in pool management or stale state.
        # For robustness, we can re-apply set_x, but it might hide underlying problems.
        # print(f"Warning in process {os.getpid()}: Worker's x differs from task's x. Re-setting x.") # Debug
        _g_worker_instance.set_x(x_from_main_process)
        _g_current_x_for_worker_process = x_from_main_process.copy()

    _g_worker_instance.set_scenario(short_delta_r)

    if vbasis is not None and cbasis is not None:
        _g_worker_instance.set_basis(vbasis, cbasis)

    result = _g_worker_instance.solve() # Uses default params (e.g., dual simplex, single thread)

    if result:
        obj_val, y_sol, pi_sol, rc_sol = result
        return scenario_idx, obj_val, y_sol, pi_sol, rc_sol
    else:
        # Handle cases where the subproblem solve fails (e.g., infeasible)
        # Return NaNs or appropriate placeholders. Dimensions are from the worker's setup.
        num_y = len(_g_worker_instance.d)
        num_constr = len(_g_worker_instance.r_bar)
        return (scenario_idx, np.nan,
                np.full(num_y, np.nan, dtype=float),
                np.full(num_constr, np.nan, dtype=float),
                np.full(num_y, np.nan, dtype=float))

def update_worker_x_task(new_x_value: np.ndarray) -> int: # Return PID for confirmation
    """
    Task function executed by a worker process to update the 'x' vector
    in its global SecondStageWorker instance.
    """
    global _g_worker_instance, _g_current_x_for_worker_process
    if _g_worker_instance is None:
        # This should not happen if the pool was initialized correctly.
        raise RuntimeError(f"Worker not initialized in process {os.getpid()} during update_worker_x_task.")

    # print(f"Process {os.getpid()}: update_worker_x_task received new x. Updating worker's x.") # Debug
    _g_worker_instance.set_x(new_x_value)
    _g_current_x_for_worker_process = new_x_value.copy() # Update the record in the worker process
    return os.getpid()



class ParallelSecondStageWorker:
    """
    A parallel version of SecondStageWorker that manages multiple SecondStageWorker
    instances across different processes to solve batches of second-stage problems.
    """

    def __init__(self, num_workers: int,
                 d: np.ndarray, D: sp.csr_matrix, sense2: np.ndarray,
                 lb_y: np.ndarray, ub_y: np.ndarray, r_bar: np.ndarray,
                 C: sp.csr_matrix, stage2_var_names: List[str],
                 stage2_constr_names: List[str], stochastic_rows_relative_indices: np.ndarray):
        """
        Initializes the ParallelSecondStageWorker.

        Args:
            num_workers: The number of parallel worker processes to use.
            d, D, sense2, lb_y, ub_y, r_bar, C, stage2_var_names,
            stage2_constr_names, stochastic_rows_relative_indices:
                Parameters required to initialize individual SecondStageWorker instances.
                See SecondStageWorker.__init__ for details.
        """
        if num_workers <= 0:
            raise ValueError("Number of workers must be positive.")
        self.num_workers = num_workers

        # Store arguments needed to construct SecondStageWorker instances in child processes
        self._worker_constructor_args = (
            d, D, sense2, lb_y, ub_y, r_bar, C,
            stage2_var_names, stage2_constr_names, stochastic_rows_relative_indices
        )

        self._pool: Optional[multiprocessing.pool.Pool] = None
        self._current_x_for_pool: Optional[np.ndarray] = None # Tracks x used for current pool
        self._closed = False

        # Store dimensions for creating placeholder results for failed solves
        self._num_y_vars = len(d)
        self._num_stage2_constrs = len(r_bar)

    @classmethod
    def from_smps_reader(cls, reader: 'SMPSReader', num_workers: int) -> 'ParallelSecondStageWorker':
        """
        Factory method to create a ParallelSecondStageWorker from an SMPSReader instance.

        Args:
            reader: An initialized and loaded SMPSReader instance.
            num_workers: The number of parallel worker processes.

        Returns:
            A new instance of ParallelSecondStageWorker.
        """
        # Validate that the reader has all necessary attributes
        required_attrs = [
            'd', 'D', 'sense2', 'lb_y', 'ub_y', 'r_bar', 'C',
            'stage2_var_names', 'stage2_constr_names',
            'stochastic_rows_relative_indices', 'model' # 'model' as proxy for loaded
        ]
        for attr in required_attrs:
            if not hasattr(reader, attr):
                raise AttributeError(f"SMPSReader instance is missing attribute '{attr}'.")
            if getattr(reader, attr) is None and attr != 'model': # model can be None before load
                 # Re-check if 'model' is a critical data source or just a load indicator
                 if reader.model is None and attr not in ['d','D']: # example, d, D must exist
                    raise ValueError(f"SMPSReader attribute '{attr}' is None.")


        return cls(
            num_workers=num_workers,
            d=reader.d,
            D=reader.D,
            sense2=reader.sense2,
            lb_y=reader.lb_y,
            ub_y=reader.ub_y,
            r_bar=reader.r_bar,
            C=reader.C,
            stage2_var_names=reader.stage2_var_names,
            stage2_constr_names=reader.stage2_constr_names,
            stochastic_rows_relative_indices=reader.stochastic_rows_relative_indices
        )

    def _ensure_pool_initialized(self, x: np.ndarray):
        """
        Ensures the multiprocessing pool is initialized.
        If x has changed since the last initialization, the pool is recreated
        so that worker processes are initialized with the new x.
        """
        if self._closed:
            raise RuntimeError("Cannot operate on a closed ParallelSecondStageWorker.")

        x_has_changed = (self._current_x_for_pool is None or
                         not np.array_equal(self._current_x_for_pool, x))

        if self._pool is not None and not x_has_changed:
            return  # Pool exists and is configured for the current x

        if self._pool is not None and x_has_changed:
            # print(f"Main process: X has changed. Recreating worker pool.") # Debug
            self._close_pool_resources() # Gracefully close existing pool

        # print(f"Main process: Initializing worker pool with {self.num_workers} workers for x: {x.shape if x is not None else 'None'}.") # Debug
        # Use 'spawn' context for safety, especially with external libraries like Gurobi.
        # 'spawn' is default on macOS and Windows for Python 3.8+.
        ctx = multiprocessing.get_context("spawn")
        self._pool = ctx.Pool(
            processes=self.num_workers,
            initializer=init_worker_process,
            initargs=(self._worker_constructor_args, x) # Pass args for SSW and current x
        )
        self._current_x_for_pool = x.copy()


    def _synchronize_x_with_workers(self, x: np.ndarray):
        """
        Ensures the pool is initialized and all workers are synchronized
        with the given first-stage decision vector 'x'.
        If the pool doesn't exist, it's created.
        If 'x' has changed, 'set_x' is called on all persistent workers.
        """
        if self._closed:
            raise RuntimeError("Cannot operate on a closed ParallelSecondStageWorker.")

        # Step 1: Create the pool if it doesn't exist.
        if self._pool is None:
            # print(f"Main process: Pool is None. Creating a new pool with initial x.") # Debug
            ctx = multiprocessing.get_context("spawn")
            # init_worker_process will create the worker and call set_x with this x.
            self._pool = ctx.Pool(
                processes=self.num_workers,
                initializer=init_worker_process,
                initargs=(self._worker_constructor_args, x.copy()) # Pass current x for initial setup
            )
            self._current_x_for_pool = x.copy()
            return # Pool is now initialized, and workers have x set.

        # Step 2: Pool exists. Check if x has changed.
        # (Ensure _current_x_for_pool is not None if pool exists, which should be true after Step 1)
        if self._current_x_for_pool is not None and np.array_equal(self._current_x_for_pool, x):
            # print(f"Main process: X has not changed. Workers are already synchronized.") # Debug
            return # x is the same, workers are already configured.

        # Step 3: Pool exists, but x has changed. Send update_worker_x_task to all workers.
        # print(f"Main process: X has changed. Sending update_worker_x_task to all workers.") # Debug
        update_tasks_args = [x.copy() for _ in range(self.num_workers)] # One task for each worker
        try:
            # This pool.map call is blocking. It ensures all workers have
            # processed set_x before this method returns.
            results = self._pool.map(update_worker_x_task, update_tasks_args)
            # print(f"Main process: update_worker_x_task completed by PIDs: {results}") # Debug
            self._current_x_for_pool = x.copy() # Record that pool workers are now updated with this x
        except Exception as e:
            # A critical error occurred while trying to update x across workers.
            # The pool might be in an inconsistent state.
            # Safest action is to close the current pool and recreate it.
            print(f"Main process: Error during x update for workers: {e}. Recreating pool for safety.") # Log this
            self._close_pool_resources() # Close the potentially faulty pool

            # Recreate the pool (similar to Step 1)
            ctx = multiprocessing.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.num_workers,
                initializer=init_worker_process,
                initargs=(self._worker_constructor_args, x.copy())
            )
            self._current_x_for_pool = x.copy()
            # print(f"Main process: Pool recreated after x update failure.") # Debug

    # solve_batch method will now call _synchronize_x_with_workers:
    def solve_batch(self, x: np.ndarray, short_delta_r_batch: np.ndarray,
                    vbasis_batch: Optional[np.ndarray] = None,
                    cbasis_batch: Optional[np.ndarray] = None) \
                    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("ParallelSecondStageWorker has been closed.")

        num_scenarios = short_delta_r_batch.shape[0]
        if num_scenarios == 0:
            return (np.array([]),
                    np.empty((0, self._num_y_vars)),
                    np.empty((0, self._num_stage2_constrs)),
                    np.empty((0, self._num_y_vars)))

        # Synchronize x with all workers (creates pool if needed, updates x if changed)
        self._synchronize_x_with_workers(x)
        if self._pool is None: # Should be initialized by _synchronize_x_with_workers
            raise RuntimeError("Worker pool was not initialized, even after synchronization attempt.")

        # ... rest of the solve_batch method (task preparation, pool.map, result processing)
        # remains largely the same as in the previous version.
        # The x.copy() in tasks.append(...) is still useful for the sanity check
        # within solve_scenario_task_glob_worker.

        tasks = []
        for i in range(num_scenarios):
            vb_i = vbasis_batch[i] if vbasis_batch is not None and len(vbasis_batch) == num_scenarios else None
            cb_i = cbasis_batch[i] if cbasis_batch is not None and len(cbasis_batch) == num_scenarios else None
            tasks.append((i, short_delta_r_batch[i], vb_i, cb_i, x.copy())) # x.copy() for verification

        results_from_pool = self._pool.map(solve_scenario_task_glob_worker, tasks)

        obj_values_all = np.empty(num_scenarios, dtype=float)
        y_solutions_all = np.empty((num_scenarios, self._num_y_vars), dtype=float)
        pi_solutions_all = np.empty((num_scenarios, self._num_stage2_constrs), dtype=float)
        rc_solutions_all = np.empty((num_scenarios, self._num_y_vars), dtype=float)

        for scenario_idx, obj_val, y_sol, pi_sol, rc_sol in results_from_pool:
            obj_values_all[scenario_idx] = obj_val
            y_solutions_all[scenario_idx, :] = y_sol
            pi_solutions_all[scenario_idx, :] = pi_sol
            rc_solutions_all[scenario_idx, :] = rc_sol

        return obj_values_all, y_solutions_all, pi_solutions_all, rc_solutions_all

    def _close_pool_resources(self):
        """Helper to close and join the current pool."""
        if self._pool is not None:
            # print(f"Main process: Closing and joining worker pool...") # Debug
            try:
                self._pool.close()  # Prevents new tasks from being submitted
                self._pool.join()   # Waits for worker processes to complete current tasks and exit
            except Exception as e:
                print(f"Error closing pool: {e}") # Should log this appropriately
            finally:
                self._pool = None
                # print(f"Main process: Worker pool resources released.") # Debug


    def close(self):
        """
        Closes all worker processes and releases resources.
        This method should be called when the ParallelSecondStageWorker is no longer needed.
        """
        if self._closed:
            return

        # print(f"Main process: ParallelSecondStageWorker close() called.") # Debug
        self._close_pool_resources()
        self._current_x_for_pool = None # Clear tracked x
        self._closed = True
        # The global SecondStageWorker instances in child processes will have their __del__
        # methods called upon process termination, which should call their own close() methods.

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object, ensuring cleanup."""
        self.close()

    def __del__(self):
        """
        Destructor to attempt cleanup if close() was not explicitly called.
        Note: Behavior of __del__ can be unreliable in some Python exit scenarios.
        Explicitly calling close() or using a context manager is preferred.
        """
        if not self._closed:
            # print(f"Main process: ParallelSecondStageWorker __del__ ensuring close.") # Debug
            self.close()