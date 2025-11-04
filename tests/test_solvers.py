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
"""Tests for solvers

These tests establish the expected design model for all solver in Jaxmod:

    - Inputs and outputs must consistently follow the shape convention (batch_size, dimension)
    - The batch dimension must always be present, even for single solves
    - This ensures shape stability under JAX JIT and prevents unnecessary recompilation
"""

import logging

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import random
from jaxtyping import Array, ArrayLike, Float, Integer, PRNGKeyArray, PyTree
from numpy.testing import assert_allclose

from jaxmod import debug_logger
from jaxmod.solvers import MultiTrySolution
from jaxmod.type_aliases import OptxSolver

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

RTOL: float = 1.0e-6
"""Relative tolerance for solver"""
ATOL: float = 1.0e-6
"""Absolute tolerance for solver"""
THROW: bool = True
"""Throw an error if the solver fails"""
MAX_STEPS: int = 16
"""Maximum steps for the solver"""


@eqx.filter_jit
def simple_objective_function(
    solution: Float[Array, "batch solution"], parameters: PyTree
) -> Float[Array, "batch residual"]:
    """Simple objective function for root finding

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
def simple_solver(
    initial_guess: Float[Array, "batch solution"], parameters: PyTree, key: PRNGKeyArray
) -> MultiTrySolution:
    """A simple solver for root finding

    Args:
        initial_guess: Initial guess of the solution
        parameters: Parameters
        key: A random key

    Returns:
        MultiTrySolution
    """
    del key

    solver: OptxSolver = optx.Newton(rtol=RTOL, atol=ATOL)

    sol: optx.Solution = optx.root_find(
        simple_objective_function,
        solver,
        initial_guess,
        args=parameters,
        throw=THROW,
        max_steps=MAX_STEPS,
    )

    # Set attempts to unity and match the batch size
    attempts: Integer[Array, " batch"] = jnp.ones(sol.value.shape[0], dtype=int)

    multi_sol: MultiTrySolution = MultiTrySolution(sol, attempts)

    return multi_sol


def test_single_solve() -> None:
    """Tests a single solve"""

    parameters: PyTree = {"a": jnp.array(4.0)}  # root = 2

    # As per the design model, the initial guess must be batched even for a single solve
    initial_guess: Float[Array, "batch solution"] = jnp.array([[1]])

    key: PRNGKeyArray = random.PRNGKey(0)

    multi_sol: MultiTrySolution = simple_solver(initial_guess, parameters, key)

    # Confirm all array shapes and types
    # 2-D float array
    assert_allclose(multi_sol.value, np.array([[2]], dtype=float), strict=True)
    # Integer
    assert_allclose(multi_sol.stats["num_steps"], 5, strict=True)
    # Integer 32
    assert_allclose(multi_sol.result._value, np.array(0, dtype=np.int32), strict=True)
    # 1-D integer array
    assert_allclose(multi_sol.attempts, np.array([1]), strict=True)


def test_batch_solve() -> None:
    """Tests a batch solve"""

    parameters: PyTree = {"a": jnp.array([[4.0], [9.0], [16.0]])}  # root = 2

    initial_guess = jnp.array([[1.0], [10.0], [1.0]])

    key: PRNGKeyArray = random.PRNGKey(0)

    multi_sol: MultiTrySolution = simple_solver(initial_guess, parameters, key)

    # Confirm all array shapes and types
    # 2-D float array
    assert_allclose(multi_sol.value, np.array([[2], [3], [4]], dtype=float), strict=True)
    # Integer
    assert_allclose(multi_sol.stats["num_steps"], 7, strict=True)
    # Integer 32
    assert_allclose(multi_sol.result._value, np.array(0, dtype=np.int32), strict=True)
    # 1-D integer array
    assert_allclose(multi_sol.attempts, np.array([1, 1, 1]), strict=True)


# def test_batch_retry_solver_single():
#     """Tests the batch retry solver for a single case"""

#     params_single: PyTree = {"a": jnp.array([4.0])}  # root = 2

#     initial_single: Float[Array, "batch solution"] = jnp.array([[1.0]])  # shape (batch, solution)

#     key = random.PRNGKey(0)
#     perturb_scale = 0.5
#     max_attempts = 5

#     single_result = batch_retry_solver(
#         simple_solver,
#         initial_single,
#         params_single,
#         perturb_scale,
#         max_attempts,
#         key,
#     )
