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
from typing import Any, Literal, cast

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from equinox._enum import EnumerationItem
from jax import lax, random
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, PyTree
from lineax import AbstractLinearSolver
from optimistix import RESULTS, Solution

from jaxmod.type_aliases import NpFloat, OptxSolver

POSTCHECK_TOLERANCE: float = 1.0e-6
"""Default tolerance for the objective-based convergence validation performed after each solve
attempt"""


class RootFindParameters(eqx.Module):
    """Parameters for Optimistix root finding

    Args:
        solver: Solver. Defaults to :class:`optimistix.Newton`.
        atol: Absolute tolerance. Defaults to ``1.0e-6``.
        rtol: Relative tolerance. Defaults to ``1.0e-6``.
        linear_solver: Linear solver. Defaults to ``AutoLinearSolver(well_posed=False)``.
        norm: Norm. Defaults to :func:`optimistix.max_norm`.
        throw: How to report any failures. Defaults to ``False``.
        max_steps: The maximum number of steps the solver can take. Defaults to ``256``.
        jac: Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian.
            Can be either ``fwd`` or ``bwd``. Defaults to ``fwd``.
    """

    solver: type[OptxSolver] = optx.Newton
    """Solver"""
    atol: float = 1.0e-6
    """Absolute tolerance"""
    rtol: float = 1.0e-6
    """Relative tolerance"""
    linear_solver: AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    """Linear solver (see https://docs.kidger.site/lineax/api/solvers/)"""
    norm: Callable = optx.max_norm
    """Norm"""
    throw: bool = False
    """How to report any failures"""
    max_steps: int = 256
    """Maximum number of steps the solver can take"""
    jac: Literal["fwd", "bwd"] = "fwd"
    """Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian"""

    def get_solver_instance(self) -> OptxSolver:
        """Instantiates the solver"""
        return self.solver(
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
            linear_solver=self.linear_solver,  # type: ignore because there is a parameter
            # For debugging LM solver. Not valid for all solvers (e.g. Newton)
            # verbose=frozenset({"step_size", "y", "loss", "accepted"}),
        )


class MultiAttemptSolution(eqx.Module):  # pragma: no cover
    """A solution wrapper for handling multiple solver attempts per problem

    This class standardises solver outputs from multi-attempt strategies. Some attributes
    (e.g. ``converged``, ``solver_success``, ``num_steps``) are broadcast to the batch dimension
    to ensure consistent shapes across all outputs, whether the underlying solver returns scalar
    or per-attempt values.

    Args:
        solution: Optimistix solution
        _attempts: Number of attempts required for each batch element to converge (``0`` indicates
            no successful attempt). Defaults to ``1``.
    """

    solution: optx.Solution
    _attempts: ArrayLike = 1

    @property
    def attempts(self) -> Integer[Array, " batch"]:
        return jnp.broadcast_to(self._attempts, self.batch_size)

    @property
    def aux(self):
        return self.solution.aux

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return self.solution.value.shape[0]

    @property
    def converged(self) -> Bool[Array, " batch"]:
        """Boolean mask indicating objective-based convergence"""
        return jnp.broadcast_to(self.attempts > 0, self.batch_size)

    @property
    def num_steps(self) -> Integer[Array, " batch"]:
        """Number of steps"""
        return jnp.broadcast_to(self.stats["num_steps"], self.batch_size)

    @property
    def result(self) -> RESULTS:
        return self.solution.result

    @property
    def value(self) -> Float[Array, "batch solution"]:
        return self.solution.value

    @property
    def solver_success(self) -> Bool[Array, " batch"]:
        """Whether the underlying solver claims success"""
        return jnp.broadcast_to(self.solution.result == RESULTS.successful, self.batch_size)

    @property
    def state(self) -> Any:
        return self.solution.state

    @property
    def stats(self) -> dict[str, PyTree[ArrayLike]]:
        return self.solution.stats

    def asdict(self) -> dict[Any, NpFloat]:
        """Converts pertinent solution statistics to a dictionary"""
        return {
            "status": np.asarray(self.solver_success),
            "steps": np.asarray(self.num_steps),
            "attempts": np.asarray(self.attempts),
            "converged": np.asarray(self.converged),
        }


def max_norm(
    objective_function: Callable, solution: Float[Array, "batch solution"], parameters: PyTree
) -> Float[Array, " batch"]:  # pragma: no cover
    """Computes the L-infinity norm of batched objective residuals.

    Evaluates the objective function for each model in the batch and returns the maximum absolute
    residual across all components of each system. This is a vectorised variant of
    :func:`optimistix.max_norm`, producing one scalar L-infinity norm per system in the batch.

    See: https://docs.kidger.site/optimistix/api/norms/

    Args:
        objective_function: A callable taking ``solution`` and ``parameters`` that returns the
            objective residuals for each model in the batch
        solution: Batched array of candidate solutions
        parameters: Parameters passed to the objective function

    Returns:
        An array of the L-infinity norm
    """
    return jnp.linalg.norm(objective_function(solution, parameters), ord=jnp.inf, axis=1)


def make_batch_retry_solver(solver_function: Callable, objective_function: Callable) -> Callable:
    """Makes a batch retry solver.

    ``solver_function`` and ``objective_function`` must be pure JAX-callable functions compatible
    with :func:`equinox.filter_jit``. They must not close over non-JAX state or produce Python side
    effects.

    Args:
        solver_function: Callable that performs a single solve and returns an
            :class:`~optimistix.Solution` object. Must accept arguments of an initial guess and a
            pytree of parameters.
        objective_function: Callable for the objective function

    Returns:
        Callable
    """

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def batch_retry_solver(
        initial_guess: Float[Array, "batch solution"],
        parameters: PyTree,
        key: PRNGKeyArray,
        perturb_scale: ArrayLike,
        max_attempts: int,
        tolerance: float = POSTCHECK_TOLERANCE,
    ) -> MultiAttemptSolution:
        """Batched solver with retry and perturbation for failed cases

        Runs a batched solver function on a set of initial guesses. If some entries fail to
        converge, the function perturbs only the failed solutions and retries, up to
        ``max_attempts``. Successfully converged solutions are kept fixed throughout.

        This approach is useful when solving large batches of nonlinear systems where certain
        initial guesses may fail. Perturbations help the solver escape poor local minima or flat
        regions of the objective function.

        Note:
            ``solver_function`` may return a solver result indicating success even when the
            objective residual remains above tolerance. Convergence is therefore validated
            independently and the result of that validation is tracked in
            :meth:`MultiAttemptSolution.attempts`.
                - ``solution.result``: solver's internal convergence classification
                - ``attempts``: first iteration satisfying objective-based check
                - ``attempts == 0``: did not converge within ``max_attempts``

        Args:
            initial_guess: Batched array of initial guesses for the solver
            parameters: Model parameters passed to the solver
            key: JAX PRNG key for reproducible random perturbations
            perturb_scale: Array or scalar that scales the random perturbation applied to failed
                solutions
            max_attempts: Maximum number of solver retries per batch entry
            tolerance: Tolerance for the objective-based convergence validation performed after
                each solve attempt. Defaults to :obj:`POSTCHECK_TOLERANCE`.

        Returns:
            :class:`MultiAttemptSolution` instance
        """

        batch_size: int = initial_guess.shape[0]

        def body_fn(state: tuple[Array, Array, Array, Array, Array, Array]) -> tuple:
            """Performs one retry iteration for failed solutions.

            This function executes a single iteration of the solver retry loop. It perturbs only
            the solutions that previously failed, reruns the solver, and updates the batch state
            accordingly. Successfully converged entries remain unchanged.

            Args:
                tuple:
                    i: Current attempt index
                    key: Random key for perturbation generation
                    solution: Current batch of solution estimates
                    result_value: Current result value of the solver for each entry
                    steps: Number of solver steps recorded for each entry
                    attempt: Attempt index when each entry first succeeded or 0 if it did not
                         converge at all.

            Returns:
                Updated state tuple with the same structure as in the input
            """
            i, key, solution, result_value, steps, attempt = state
            # jax.debug.print("Iteration: {out}", out=i)

            failed_mask: Bool[Array, " batch"] = result_value != 0  # Failed have non-zero result
            key, subkey = random.split(key)

            # Perturb the original solution for cases that failed. Something more sophisticated
            # could be implemented, such as a regressor or neural network to inform failed cases
            # based on successful solves.
            perturb_shape: tuple[int, int] = (solution.shape[0], solution.shape[1])
            raw_perturb: Float[Array, "batch solution"] = random.uniform(
                subkey, shape=perturb_shape, minval=-1.0, maxval=1.0
            )
            perturbations: Float[Array, "batch solution"] = jnp.where(
                failed_mask[:, None], perturb_scale * raw_perturb, jnp.zeros_like(solution)
            )
            new_initial_solution: Float[Array, "batch solution"] = solution + perturbations
            # jax.debug.print("new_initial_solution = {out}", out=new_initial_solution)

            new_sol: optx.Solution = solver_function(new_initial_solution, parameters)
            new_solution: Float[Array, "batch solution"] = new_sol.value
            # jax.debug.print("new_solution = {out}", out=new_solution)
            new_result_value: Integer[Array, " batch"] = jnp.broadcast_to(
                new_sol.result._value, batch_size
            )
            # jax.debug.print("new_result_value = {out}", out=new_result_value)

            # If the solver result is broadcast from a scalar we can't use it to decide which
            # individual models failed. Instead we must perform a per-system check.
            new_successful: Bool[Array, " batch"] = (
                max_norm(objective_function, new_solution, parameters) < tolerance
            )
            # jax.debug.print("new_successful = {out}", out=new_successful)

            new_num_steps: Integer[Array, " batch"] = jnp.broadcast_to(
                new_sol.stats["num_steps"], batch_size
            )
            # jax.debug.print("new_num_steps = {out}", out=new_num_steps)

            # Determine which entries to update: previously failed, now succeeded
            update_mask: Bool[Array, " batch"] = jnp.logical_and(failed_mask, new_successful)
            # jax.debug.print("update_mask = {out}", out=update_mask)
            updated_solution: Float[Array, "batch solution"] = cast(
                Array, jnp.where(update_mask[:, None], new_solution, solution)
            )
            updated_result_value: Integer[Array, " batch"] = jnp.where(
                update_mask, new_result_value, result_value
            )
            # jax.debug.print("updated_result_value = {out}", out=updated_result_value)
            updated_num_steps: Integer[Array, " batch"] = cast(
                Array, jnp.where(update_mask, new_num_steps, steps)
            )
            # jax.debug.print("updated_num_steps = {out}", out=updated_num_steps)
            updated_attempt: Array = jnp.where(update_mask, i, attempt)  # pyright: ignore
            # jax.debug.print("updated_attempt = {out}", out=updated_attempt)

            return (
                i + 1,
                key,
                updated_solution,
                updated_result_value,
                updated_num_steps,
                updated_attempt,
            )

        def cond_fn(
            state: tuple[Array, Array, Array, Array, Array, Array],
        ) -> Bool[Array, "..."]:
            """Determines whether additional solver retries are needed.

            This condition function controls the ``lax.while_loop``. The retry loop continues as
            long as at least one batch entry has not converged and the maximum number of attempts
            has not been reached.

            Args:
                tuple:
                    i: Current attempt index
                    _: Unused (PRNG key)
                    _: Unused (current batch solution)
                    _: Unused (result value)
                    _: Unused (number of steps)
                    attempts: Unused (success attempt index)

            Returns:
                ``True`` if any entry has failed and the number of attempts is less than
                    ``max_attempts``; otherwise ``False``.
            """
            i, _, _, _, _, attempt = state

            # For debugging to force the loop to run to the maximum allowable value
            # return jnp.logical_and(i < max_attempts, True)

            # Convergence is determined by `check_convergence`, which enforces the objective
            # tolerance on each batch entry individually. We track the first successful attempt
            # index in `attempts`. An entry is considered converged if attempts > 0, ensuring
            # consistency with the convergence mask used elsewhere in the code.
            continue_loop: Bool[Array, "..."] = jnp.logical_and(
                jnp.any(attempt == 0), i < max_attempts
            )

            return continue_loop

        # Try first solution
        # jax.debug.print("Iteration: 1")
        first_sol: optx.Solution = solver_function(initial_guess, parameters)
        first_solution: Float[Array, "batch solution"] = first_sol.value
        # jax.debug.print("first_solution = {out}", out=first_solution)

        first_result_value: Integer[Array, " batch"] = jnp.broadcast_to(
            first_sol.result._value, batch_size
        )
        # jax.debug.print("first_result_value = {out}", out=first_result_value)

        # If the solver result is broadcast from a scalar we can't use it to decide which
        # individual models failed. Instead we must perform a per-system check.
        first_converged: Bool[Array, " batch"] = (
            max_norm(objective_function, first_solution, parameters) < tolerance
        )
        first_num_steps: Integer[Array, " batch"] = jnp.broadcast_to(
            first_sol.stats["num_steps"], batch_size
        )
        # jax.debug.print("first_num_steps = {out}", out=first_num_steps)

        # Failback solution to initial guess for failed models
        solution: Float[Array, "batch solution"] = cast(
            Array, jnp.where(first_converged[:, None], first_solution, initial_guess)
        )
        # jax.debug.print("solution = {out}", out=solution)
        # jax.debug.print("Completed iteration: 1")

        initial_state: tuple = (
            jnp.array(2),  # Second overall attempt
            key,
            solution,
            first_result_value,
            first_num_steps,
            first_converged.astype(int),  # 1 for solved, otherwise 0
        )

        _, _, final_solution, final_result_value, final_num_steps, final_attempt = lax.while_loop(
            cond_fn, body_fn, initial_state
        )
        # jax.debug.print("After lax.while_loop")

        # jax.debug.print("final_solution = {out}", out=final_solution)
        # jax.debug.print("final_result_value = {out}", out=final_result_value)
        # jax.debug.print("final_num_steps = {out}", out=final_num_steps)
        # jax.debug.print("final_attempt = {out}", out=final_attempt)

        # Bundle the final outputs into a single optimistix Solution object
        final_result: optx.RESULTS = cast(
            optx.RESULTS,
            EnumerationItem(final_result_value, optx.RESULTS),  # pyright: ignore
        )

        # NOTE: This solution instance does not return all the information from the solves, but it
        # encapsulates the most important (final) quantities
        sol: optx.Solution = Solution(
            final_solution, final_result, None, {"num_steps": final_num_steps}, None
        )
        multi_sol: MultiAttemptSolution = MultiAttemptSolution(sol, final_attempt)

        return multi_sol

    return batch_retry_solver
