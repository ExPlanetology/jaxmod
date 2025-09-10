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

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from jaxmod import MAX_EXP_INPUT


def safe_exp(x: ArrayLike) -> Array:
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
