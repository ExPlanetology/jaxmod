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

import logging

import jax
import numpy as np

__version__: str = "0.2.1"

jax.config.update("jax_enable_x64", True)


MAX_EXP_INPUT: float = np.log(np.finfo(np.float64).max)
"""Maximum x for which exp(x) is finite in 64-bit precision to prevent overflow"""
MIN_EXP_INPUT: float = np.log(np.finfo(np.float64).tiny)
"""Minimum x for which exp(x) is non-zero in 64-bit precision to prevent underflow"""

# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def simple_formatter() -> logging.Formatter:
    """Simple formatter for logging

    Returns:
        Formatter for logging
    """
    fmt: str = "[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def debug_logger() -> logging.Logger:
    """Sets up debug logging to the console.

    Returns:
        A logger
    """
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return package_logger
