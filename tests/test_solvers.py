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
"""Tests for solvers"""

import logging
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import random
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, PyTree
from numpy.testing import assert_allclose

from jaxmod import debug_logger
from jaxmod.solvers import MultiTrySolution, make_batch_retry_solver
from jaxmod.type_aliases import OptxSolver

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

RTOL: float = 1.0e-6
"""Relative tolerance for the solver"""
ATOL: float = 1.0e-6
"""Absolute tolerance for the solver"""
THROW: bool = True
"""Throw an error if the solver fails"""
MAX_STEPS: int = 16
"""Maximum steps for the solver"""


@eqx.filter_jit
def simple_objective(solution: Float[Array, "..."], parameters: PyTree) -> Float[Array, "..."]:
    """A simple objective function for root finding

    This function is shape-agnostic and supports both single-solution and batched inputs
    transparently. The returned residual will broadcast to match the shape of the ``solution``
    input.

    Args:
        solution: Solution
        parameters: Parameters

    Returns:
        Residual
    """
    a: ArrayLike = parameters["a"]
    residual: Float[Array, "batch residual"] = jnp.square(solution) - a

    return residual


@eqx.filter_jit
def simple_solver(initial_guess: Float[Array, "..."], parameters: PyTree) -> optx.Solution:
    """A simple Newton-based solver for root finding

    This solver is shape-agnostic and works seamlessly with both single solution inputs and batched
    inputs when combined with ``vmap``. The returned solution will broadcast to match the shape of
    ``initial_guess``.

    Args:
        initial_guess: Initial guess of the solution
        parameters: Parameters

    Returns:
       Optimistic solution
    """
    solver: OptxSolver = optx.Newton(rtol=RTOL, atol=ATOL)

    sol: optx.Solution = optx.root_find(
        simple_objective, solver, initial_guess, args=parameters, throw=THROW, max_steps=MAX_STEPS
    )

    return sol


# Batch retry solver
simple_batch_retry_solver: Callable = make_batch_retry_solver(simple_solver, simple_objective)

# Vectorise to test that a vectorised solver also works with the batch_retry_solver
simple_solver_vmapped: Callable = eqx.filter_vmap(simple_solver, in_axes=(0, None))
simple_objective_vmapped: Callable = eqx.filter_vmap(simple_objective, in_axes=(0, None))
simple_batch_retry_solver_vmapped: Callable = make_batch_retry_solver(
    simple_solver_vmapped, simple_objective_vmapped
)


def test_single_solve() -> None:
    """Tests a single solve"""

    parameters: PyTree = {"a": jnp.array(4.0)}  # root = 2

    # As per the design model, the initial guess must be batched even for a single solve
    initial_guess: Float[Array, "batch solution"] = jnp.array([[1.0]])

    multi_sol: optx.Solution = simple_solver(initial_guess, parameters)

    # Confirm all array shapes and types
    # 2-D float array
    assert_allclose(multi_sol.value, np.array([[2]], dtype=float), strict=True)
    # Integer
    assert_allclose(multi_sol.stats["num_steps"], 5, strict=True)
    # Integer 32
    assert_allclose(multi_sol.result._value, np.array(0, dtype=np.int32), strict=True)


def test_batch_solve() -> None:
    """Tests a batch solve"""

    parameters: PyTree = {"a": jnp.array([[4.0], [9.0], [16.0]])}  # root = 2

    initial_guess = jnp.array([[1.0], [10.0], [1.0]])

    multi_sol: optx.Solution = simple_solver(initial_guess, parameters)

    # Confirm all array shapes and types
    # 2-D float array
    assert_allclose(multi_sol.value, np.array([[2], [3], [4]], dtype=float), strict=True)
    # Integer
    assert_allclose(multi_sol.stats["num_steps"], 7, strict=True)
    # Integer 32
    assert_allclose(multi_sol.result._value, np.array(0, dtype=np.int32), strict=True)


def test_batch_retry_solver_single():
    """Tests the batch retry solver for a single case"""

    parameters: PyTree = {"a": jnp.array(4.0)}  # root = 2

    # As per the design model, the initial guess must be batched even for a single solve
    initial_guess: Float[Array, "batch solution"] = jnp.array([[1.0]])

    key: PRNGKeyArray = random.PRNGKey(0)

    multi_sol: MultiTrySolution = simple_batch_retry_solver(initial_guess, parameters, key, 1, 10)

    print("solution returned = ", multi_sol.value)

    print("num_steps returned = ", multi_sol.stats["num_steps"])

    print("result returned = ", multi_sol.result._value)

    # Confirm all array shapes and types
    # 2-D float array
    # assert_allclose(multi_sol.value, np.array([[2]], dtype=float), strict=True)
    # Integer
    # assert_allclose(multi_sol.stats["num_steps"], 5, strict=True)
    # Integer 32
    # assert_allclose(multi_sol.result._value, np.array(0, dtype=np.int32), strict=True)
    # 1-D integer array
    # assert_allclose(multi_sol.attempts, np.array([1]), strict=True)


def test_batch_retry_solver_batch():
    """Tests the batch retry solver for a single case"""

    parameters: PyTree = {"a": jnp.array([[4.0], [9.0]])}  # root = 2

    # As per the design model, the initial guess must be batched even for a single solve
    initial_guess: Float[Array, "batch solution"] = jnp.array([[1.0], [2.0]])

    key: PRNGKeyArray = random.PRNGKey(0)

    multi_sol: MultiTrySolution = simple_batch_retry_solver(initial_guess, parameters, key, 1, 10)

    print("solution returned = ", multi_sol.value)

    print("num_steps returned = ", multi_sol.stats["num_steps"])

    print("result returned = ", multi_sol.result._value)


def test_batch_retry_solver_batch_vmap() -> None:
    """TODO"""

    parameters: PyTree = {"a": 4.0}  # root = 2

    # As per the design model, the initial guess must be batched even for a single solve
    initial_guess: Float[Array, "batch solution"] = jnp.array([[1.0], [2.0]])

    key: PRNGKeyArray = random.PRNGKey(0)

    multi_sol: MultiTrySolution = simple_batch_retry_solver_vmapped(
        initial_guess, parameters, key, 1, 10
    )

    print("solution returned = ", multi_sol.value)

    print("num_steps returned = ", multi_sol.stats["num_steps"])

    print("result returned = ", multi_sol.result._value)
