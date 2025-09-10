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
"""Unit conversion factors for scientific calculations"""

import equinox as eqx
from scipy.constants import atmosphere, bar, kilo, mega


class UnitConversion(eqx.Module):
    """Unit conversions"""

    # Pressure
    atmosphere_to_bar: float = atmosphere / bar
    bar_to_Pa: float = 1.0e5
    bar_to_MPa: float = 1.0e-1
    bar_to_GPa: float = 1.0e-4
    Pa_to_bar: float = 1.0e-5
    MPa_to_bar: float = 1.0e1
    GPa_to_bar: float = 1.0e4

    # Concentration / fraction

    fraction_to_ppm: float = mega
    ppm_to_fraction: float = 1 / mega
    ppm_to_percent: float = 100 / mega
    percent_to_ppm: float = 1.0e4

    # Mass / volume
    g_to_kg: float = 1 / kilo
    cm3_to_m3: float = 1.0e-6
    m3_to_cm3: float = 1.0e6
    litre_to_m3: float = 1.0e-3

    # Energy / work
    m3_bar_to_J: float = 1.0e5
    J_to_m3_bar: float = 1.0e-5


# Single instance for convenient access
unit_conversion: UnitConversion = UnitConversion()
