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
"""Common type aliases"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

NpArray: TypeAlias = npt.NDArray
"""NumPy array"""
NpBool: TypeAlias = npt.NDArray[np.bool_]
"""NumPy :obj:`numpy.bool_` array"""
NpFloat: TypeAlias = npt.NDArray[np.float64]
"""NumPy :obj:`numpy.float64` array"""
NpInt: TypeAlias = npt.NDArray[np.int_]
"""NumPy :obj:`numpy.int_` array"""
Scalar: TypeAlias = int | float
"""Scalar"""
