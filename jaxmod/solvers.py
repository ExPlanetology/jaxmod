#
# Copyright 2025 Dan J. Bower
#
# This file is part of Jaxmod.
#
# Jaxmod is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Jaxmod is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Jaxmod. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Solvers"""

from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
import optimistix as optx
from equinox._enum import EnumerationItem
from jax import lax, random
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, PyTree


class MultiTrySolution(optx.Solution):
    """A solution wrapper for handling multiple solver attempts per problem

    This class extends :class:`optimistix.Solution` to manage problems where each entry in the batch may
    require multiple attempts to converge. The `attempts` field tracks the number of tries made for
    each solution.
    """

    attempts: Integer[Array, "..."]


def check_convergence(
    objective_function: Callable,
    solution: Float[Array, "batch solution"],
    parameters: PyTree,
    tolerance: float,
) -> Bool[Array, " batch"]:
    """Checks convergence of a batched solution

    Evaluates the objective function for each solution in a batch and determines whether each model
    has converged within a specified tolerance. Convergence is defined by the norm of the objective
    function value being less than ``tolerance``.

    Args:
        objective_function: A callable taking ``solution`` and ``parameters`` that return the
            objective residuals for each model in the batch
        solution: Batched array of candidate solutions
        parameters: Parameters passed to the objective function
        tolerance: Convergence threshold; smaller values imply stricter convergence

    Returns:
        An array indicating which solutions have converged within the specified tolerance
    """
    return jnp.linalg.norm(objective_function(solution, parameters), axis=1) < tolerance


# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def batch_retry_solver(
    solver_fn: Callable,
    initial_guess: Float[Array, "batch solution"],
    parameters: PyTree,
    perturb_scale: ArrayLike,
    max_attempts: int,
    key: PRNGKeyArray,
) -> MultiTrySolution:
    """Batched solver with retry and perturbation for failed cases

    Runs a batched solver function on a set of initial guesses. If some entries fail to converge,
    the function perturbs only the failed solutions and retries, up to ``max_attempts``.
    Successfully converged solutions are kept fixed throughout.

    This approach is useful when solving large batches of nonlinear systems where certain initial
    guesses may fail. Perturbations help the solver escape poor local minima or flat regions of the
    objective function.

    Args:
        solver_fn: Callable that performs a single solve and returns an ``optx.Solution`` object
        initial_guess: Batched array of initial guesses for the solver
        parameters: Model parameters passed to the solver
        perturb_scale: Array or scalar that scales the random perturbation applied to failed
            solutions
        max_attempts: Maximum number of solver retries per batch entry
        key: JAX PRNG key for reproducible random perturbations

    Returns:
        :class:`MultiTrySolution` instance
    """

    def body_fn(state: tuple[Array, Array, Array, optx.RESULTS, Array, Array]) -> tuple:
        """Performs one retry iteration for failed solutions.

        This function executes a single iteration of the solver retry loop. It perturbs only the
        solutions that previously failed, reruns the solver, and updates the batch state
        accordingly. Successfully converged entries remain unchanged.

        Args:
            state: Tuple containing:
                i: Current attempt index
                key: Random key for perturbation generation
                solution: Current batch of solution estimates
                result: Current result of the solver for each entry
                steps: Number of solver steps recorded for each entry
                attempt: Attempt index when each entry first succeeded

        Returns:
            Updated state tuple with the same structure as in the input
        """
        i, key, solution, result, steps, attempt = state
        # jax.debug.print("Iteration: {out}", out=i)

        failed_mask: Bool[Array, " batch"] = result != optx.RESULTS.successful
        key, subkey = random.split(key)

        # Perturb the original solution for cases that failed. Something more sophisticated could
        # be implemented, such as a regressor or neural network to inform failed cases based on
        # successful solves.
        perturb_shape: tuple[int, int] = (solution.shape[0], solution.shape[1])
        raw_perturb: Float[Array, "batch solution"] = random.uniform(
            subkey, shape=perturb_shape, minval=-1.0, maxval=1.0
        )
        perturbations: Float[Array, "batch solution"] = jnp.where(
            failed_mask[:, None], perturb_scale * raw_perturb, jnp.zeros_like(solution)
        )
        new_initial_solution: Float[Array, "batch solution"] = solution + perturbations
        # jax.debug.print("new_initial_solution = {out}", out=new_initial_solution)

        new_sol: optx.Solution = solver_fn(new_initial_solution, parameters)

        new_solution: Float[Array, "batch solution"] = new_sol.value
        new_result: optx.RESULTS = new_sol.result
        new_successful: Bool[Array, " batch"] = new_sol.result == optx.RESULTS.successful
        new_steps: Integer[Array, " batch"] = new_sol.stats["num_steps"]

        # Determine which entries to update: previously failed, now succeeded
        update_mask: Bool[Array, " batch"] = failed_mask & new_successful
        updated_solution: Float[Array, "batch solution"] = cast(
            Array, jnp.where(update_mask[:, None], new_solution, solution)
        )
        updated_result_value: Integer[Array, " batch"] = jnp.where(
            update_mask, new_result._value, result._value
        )
        # jax.debug.print("updated_result_value = {out}", out=updated_result_value)
        updated_result: optx.RESULTS = cast(
            optx.RESULTS,
            EnumerationItem(updated_result_value, optx.RESULTS),  # pyright: ignore
        )
        updated_steps: Integer[Array, " batch"] = cast(
            Array, jnp.where(update_mask, new_steps, steps)
        )
        updated_attempt: Array = jnp.where(update_mask, i, attempt)  # pyright: ignore

        return (i, key, updated_solution, updated_result, updated_steps, updated_attempt)

    def cond_fn(state: tuple[Array, ...]) -> Bool[Array, "..."]:
        """Determines whether additional solver retries are needed.

        This condition function controls the ``lax.while_loop``. The retry loop continues as long
        as at least one batch entry has not converged and the maximum number of attempts has not
        been reached.

        Args:
            state: Tuple containing:
                i: Current attempt index
                _: Unused (PRNG key)
                _: Unused (current bacth solution)
                result: Current result of the solver for each entry
                _: Unused (step count)
                _: Unused (success attempt index)

        Returns:
            ``True`` if any entry has failed and the number of attempts is less than
                ``max_attempts``; otherwise ``False``.
        """
        i, _, _, result, _, _ = state

        # For debugging to force the loop to run to the maximum allowable value
        # return jnp.logical_and(i < max_attempts, True)

        return jnp.logical_and(i < max_attempts, jnp.any(result != optx.RESULTS.successful))

    # Try first solution
    first_sol: optx.Solution = solver_fn(initial_guess, parameters)

    first_solution: Float[Array, "batch solution"] = first_sol.value
    # jax.debug.print("first_solution = {out}", out=first_solution)
    first_solve_result: optx.RESULTS = first_sol.result
    # jax.debug.print("first_solve_result = {out}", out=first_solve_result)
    first_solve_successful: Bool[Array, " batch"] = first_solve_result == optx.RESULTS.successful
    # jax.debug.print("first_solve_successful = {out}", out=first_solve_successful)
    first_solve_steps: Integer[Array, " batch"] = first_sol.stats["num_steps"]
    # jax.debug.print("first_solve_steps = {out}", out=first_solve_steps)

    # Failback solution to initial guess, which should be a better starting guess to perturb
    solution = cast(
        Array,
        jnp.where(first_solve_successful[:, None], first_solution, initial_guess),
    )
    # jax.debug.print("solution = {out}", out=solution)

    initial_state: tuple = (
        jnp.array(2),  # First attempt of the repeat_solver is the second overall attempt
        key,
        solution,
        first_solve_result,
        first_solve_steps,
        first_solve_successful.astype(int),  # 1 for solved, otherwise 0
    )

    _, _, final_solution, final_result, final_steps, final_attempt = lax.while_loop(
        cond_fn, body_fn, initial_state
    )

    # Bundle the final solution into a single object
    multi_sol: MultiTrySolution = MultiTrySolution(
        final_solution, final_result, None, {"num_steps": final_steps}, None, final_attempt
    )

    return multi_sol
