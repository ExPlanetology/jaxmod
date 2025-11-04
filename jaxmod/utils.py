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
"""Utils"""

import logging
from collections.abc import Callable, Iterable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.tree_util import tree_map
from jaxtyping import Array, ArrayLike, Float, PyTree

from jaxmod import MAX_EXP_INPUT
from jaxmod.type_aliases import NpArray, NpInt

logger: logging.Logger = logging.getLogger(__name__)


def as_j64(x: ArrayLike | tuple) -> Float[Array, "..."]:  # pragma: no cover
    """Converts input to a :class:`jax.Array` of dtype :obj:`jax.numpy.float64`.

    This ensures that the array has a fixed dtype, preventing JAX from recompiling functions due to
    type changes between calls.

    Args:
        x: Input to convert

    Returns:
        :class:`jax.Array` of dtype :obj:`jax.numpy.float64`
    """
    return jnp.asarray(x, dtype=jnp.float64)


def get_batch_axis(x: Any) -> Literal[0, None]:
    """Determines the batch axis for a JAX array.

    Determines whether an object should be treated as batched along axis ``0`` for
    :func:`jax.vmap`.

    This function only considers JAX arrays for batching. While :func:`equinox.is_array` regards
    both JAX and NumPy arrays as arrays for tracing, NumPy arrays are treated here as static
    constants and are never batched. This allows fixed matrices to remain inside pytrees without
    being inadvertently vectorised.

    Rules:
        - 1-D JAX arrays: Batched along axis 0
        - 2-D JAX arrays: Batched along axis 0 if ``shape[0]``>1
        - 0-D (scalar) JAX arrays: Not batched
        - NumPy arrays or other objects: Not batched

    Args:
        x: Object to check for batching

    Returns:
        ``0`` if batched along axis ``0``, otherwise ``None``
    """
    if is_jax_array(x):
        # Vectorise over any 1-D array
        if x.ndim == 1:
            return 0
        # Any 2-D array should be vectorised over the first dimension if it is not unity
        elif x.ndim == 2 and x.shape[0] > 1:
            return 0
    return None  # explicit fallback


def get_batch_size(x: PyTree) -> int:
    """Determines the maximum batch size (i.e., length along axis ``0``) amongst all array-like
    leaves.

    This inspects every leaf in the pytree and checks whether it is an array. Scalars contribute a
    size of ``1``, while arrays contribute the length of their leading dimension (``shape[0]``).
    The result is the largest such size found.

    Note:
        Unlike :func:`get_batch_axis`, which only considers JAX arrays for batching, this function
        counts both JAX and NumPy arrays as array-like leaves when computing the maximum batch
        size.

    Args:
        x: Pytree of nested containers that may include arrays or scalars

    Returns:
        The maximum leading dimension size across all array-like leaves
    """
    max_size: int = 1
    for leaf in jax.tree_util.tree_leaves(x):
        if eqx.is_array(leaf):
            max_size = max(max_size, leaf.shape[0] if leaf.ndim else 1)

    return max_size


def is_hashable(x: Any) -> None:  # pragma: no cover
    """Checks whether an object is hashable and prints the result.

    Args:
        x: Object to check
    """
    try:
        hash(x)
        print("%s is hashable" % x.__class__.__name__)
    except TypeError:
        print("%s is not hashable" % x.__class__.__name__)


def is_jax_array(element: Any) -> bool:  # pragma: no cover
    """Checks if ``element`` is a JAX array.

    Note:
        NumPy arrays are not considered JAX arrays

    Args:
        element: Object to check

    Returns:
        ``True`` if ``element`` is a JAX array, otherwise ``False``
    """
    return isinstance(element, jax.Array)


def partial_rref(matrix: NpArray) -> NpArray:
    """Computes a partial reduced row echelon form (RREF) to determine linear components.

    This function performs the computation using NumPy in-place operations and is therefore not
    compatible with JAX transformations. The returned matrix represents the linear components of
    the input, extracted from the augmented RREF procedure.

    Args:
        matrix: A 2-D NumPy array of shape (nrows, ncols).

    Returns:
        A :class:`numpy.ndarray` containing the linear components.
    """
    nrows, ncols = matrix.shape

    augmented_matrix: NpArray = np.hstack((matrix, np.eye(nrows)))
    logger.debug("augmented_matrix = \n%s", augmented_matrix)
    # Permutation matrix
    # P: NpArray = np.eye(nrows)

    # Forward elimination with partial pivoting
    for i in range(min(nrows, ncols)):
        # Pivot selection with check
        nonzero: NpInt = np.flatnonzero(augmented_matrix[i:, i])
        # logger.debug("nonzero = %s", nonzero)
        if nonzero.size == 0:
            # logger.debug("i: %d. No pivot in this column.", i)
            continue  # no pivot in this column
        # Absolute row index of first non-zero index
        pivot_row: np.int_ = nonzero[0] + i
        # Swap if pivot row is not already in place
        if pivot_row != i:
            augmented_matrix[[i, pivot_row], :] = augmented_matrix[[pivot_row, i], :]
            # P[[i, nonzero_row], :] = P[[nonzero_row, i], :]

        # Perform row operations to eliminate values below the pivot.
        pivot_value: np.float64 = augmented_matrix[i, i]
        if i + 1 < nrows:
            factors = augmented_matrix[i + 1 :, i : i + 1] / pivot_value  # shape (nrows-i-1, 1)
            augmented_matrix[i + 1 :] -= factors * augmented_matrix[i]

    # logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

    # Backward substitution
    for i in range(min(nrows, ncols) - 1, -1, -1):
        pivot_value = augmented_matrix[i, i]
        if pivot_value == 0:
            # logger.debug("i: %d. Pivot is zero, skipping backward elimination.", i)
            continue  # skip columns with no pivot
        # Normalize the pivot row.
        augmented_matrix[i] /= augmented_matrix[i, i]

        # Eliminate entries above the pivot
        if i > 0:
            factors = augmented_matrix[:i, i : i + 1] / pivot_value  # shape (i, 1)
            augmented_matrix[:i] -= factors * augmented_matrix[i]

    # logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

    # reduced_matrix: NpArray = augmented_matrix[:, :ncols]
    component_matrix: NpArray = augmented_matrix[min(ncols, nrows) :, ncols:]
    # logger.debug("reduced_matrix = \n%s", reduced_matrix)
    # logger.debug("component_matrix = \n%s", component_matrix)
    # logger.debug("permutation_matrix = \n%s", P)

    return component_matrix


def power_law(
    values: ArrayLike, constant: ArrayLike, exponent: ArrayLike
) -> Array:  # pragma no cover
    """Power law

    Args:
        values: Values
        constant: Constant for the power law
        exponent: Exponent for the power law

    Returns:
        Evaluated power law
    """
    return jnp.power(values, exponent) * constant


def safe_exp(x: ArrayLike) -> Array:  # pragma: no cover
    """Computes the elementwise exponential of ``x`` with input clipping to prevent overflow.

    This function clips the input ``x`` to a maximum value defined by
    :const:`~jaxmod.MAX_EXP_INPUT` before applying :func:`jax.numpy.exp`, ensuring numerical
    stability for large values.

    Args:
        x: Array-like input

    Returns:
        Array of the same shape as ``x``, where each element is the exponential of the clipped
        input
    """
    return jnp.exp(jnp.clip(x, max=MAX_EXP_INPUT))


def to_hashable(x: Callable) -> Callable:  # pragma: no cover
    """Wraps a callable to make it hashable for JAX transformations.

    This wrapper is useful when passing bound methods of Equinox PyTrees (with JAX arrays as
    attributes) to transformations like :func:`jax.jit`, :func:`jax.vmap`, or :func:`lax.scan`. It
    wraps the callable in a lambda to forward all arguments while avoiding JAX trying to trace the
    method itself. See discussion: https://github.com/patrick-kidger/equinox/issues/1011

    Args:
        x: A callable to wrap

    Returns:
        A hashable lambda forwarding all arguments to the original callable.
    """
    return lambda *args, **kwargs: x(*args, **kwargs)


def to_native_floats(value: Any) -> Any:
    """Recursively converts any structure to nested tuples of native floats.

    Args:
        value: A scalar, list/tuple/array of floats, or nested thereof

    Returns:
        A float or nested tuple of floats
    """
    # Scalars (covers Python, NumPy, JAX scalars)
    if jnp.isscalar(value):
        return float(value)

    # Pandas DataFrame: convert to list of rows (as tuples)
    if isinstance(value, pd.DataFrame):
        iterable: Iterable = value.itertuples(index=False, name=None)
        return tuple(to_native_floats(row) for row in iterable)

    # Array-like (NumPy, JAX)
    if hasattr(value, "ndim"):
        return tuple(to_native_floats(sub) for sub in value.tolist())

    # Generic iterables (lists, tuples, etc.)
    try:
        iterable = list(value)
    except Exception:
        raise TypeError(f"Cannot convert to float or iterate over type {type(value)}")

    return tuple(to_native_floats(item) for item in iterable)


def vmap_axes_spec(x: PyTree) -> PyTree[Literal[0, None]]:
    """Recursively generate ``in_axes`` for :func:`jax.vmap` over a pytree.

    Only JAX arrays are considered for batching. NumPy arrays and other objects are treated as
    static constants (not batched).

    Args:
        x: A pytree potentially containing JAX arrays, NumPy arrays, or scalars

    Returns:
        A pytree with the same structure as ``x``. Each leaf is ``0`` if batched, or ``None``
        if not.
    """
    return tree_map(get_batch_axis, x)
