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
"""Package level variables"""

import jax
import numpy as np

__version__: str = "0.1.0"

jax.config.update("jax_enable_x64", True)


# Maximum x for which exp(x) is finite in 64-bit precision (to prevent overflow)
MAX_EXP_INPUT: float = np.log(np.finfo(np.float64).max)
# Minimum x for which exp(x) is non-zero in 64-bit precision
MIN_EXP_INPUT: float = np.log(np.finfo(np.float64).tiny)
