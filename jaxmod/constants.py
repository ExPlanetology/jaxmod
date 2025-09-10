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
"""Physical, chemical, and mathematical constants for scientific modelling."""

from molmass import Formula
from scipy import constants

AVOGADRO: float = constants.Avogadro
r"""Avogadro constant in :math:`\mathrm{mol}^{-1}`"""

GAS_CONSTANT: float = constants.gas_constant
r"""Gas constant in :math:`\mathrm{J}\ \mathrm{K}^{-1}\ \mathrm{mol}^{-1}`"""

GAS_CONSTANT_BAR: float = GAS_CONSTANT * 1.0e-5
r"""Gas constant in :math:`\mathrm{m}^3\ \mathrm{bar}^{-1}\ \mathrm{K}^{-1}\ \mathrm{mol}^{-1}`"""

GRAVITATIONAL_CONSTANT: float = constants.gravitational_constant
r"""Gravitational constant in :math:`\mathrm{m}^3\ \mathrm{kg}^{-1}\ \mathrm{s}^{-2}`"""

ATMOSPHERE: float = constants.atmosphere / constants.bar
"""Atmospheres in 1 bar"""

BOLTZMANN_CONSTANT: float = constants.Boltzmann
r"""Boltzmann constant in :math:`\mathrm{J}\ \mathrm{K}^{-1}`"""

BOLTZMANN_CONSTANT_BAR: float = BOLTZMANN_CONSTANT * 1e-5
r"""Boltzmann constant in :math:`\mathrm{bar}\ \mathrm{m}^3\ \mathrm{K}^{-1}`"""

OCEAN_MOLES: float = 7.68894973907177e22
r"""Moles of :math:`\mathrm{H}_2` or :math:`\mathrm{H}_2\mathrm{O}` in present-day Earth's ocean"""

OCEAN_MASS_H2: float = OCEAN_MOLES * Formula("H2").mass / 1e3
r"""Mass of :math:`\mathrm{H}_2` in one present-day Earth ocean in kg"""

OCEAN_MASS_H2O: float = OCEAN_MOLES * Formula("H2O").mass / 1e3
r"""Mass of :math:`\mathrm{H}_2\mathrm{O}` in one present-day Earth ocean in kg"""
