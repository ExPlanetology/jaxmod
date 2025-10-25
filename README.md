# Jaxmod

[![Release 0.2.3](https://img.shields.io/badge/Release-0.2.3-blue.svg)](https://github.com/ExPlanetology/jaxmod/releases/tag/v0.9.3)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/ExPlanetology/jaxmod/actions/workflows/ci.yml/badge.svg)](https://github.com/ExPlanetology/jaxmod/actions/workflows/ci.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Test coverage](https://img.shields.io/badge/Coverage-87%25-brightgreen)](https://github.com/ExPlanetology/jaxmod)

## About
Jaxmod is a Python package that provides lightweight utility functions for JAX arrays, batching, and pytrees. It mostly builds on top of the amazing [Equinox](https://docs.kidger.site/equinox/) package, whilst notably incorporating structural conventions and helper functions that make JAX-based scientific programming more convenient and consistent.

Although generally useful for numerical and scientific computing, *Jaxmod* is somewhat biased toward applications in chemistry, geochemistry, and planetary science, where tasks like handling stoichiometric matrices, managing physical constants, and ensuring numerical stability are common.

## Documentation

The documentation is available online, with options to download it in EPUB or PDF format:

[https://jaxmod.readthedocs.io/en/latest/](https://jaxmod.readthedocs.io/en/latest/)

## Quick install

Jaxmod is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux). It is recommended to install Jaxmod in a dedicated Python environment. Before installation, create and activate the environment, then run:

```pip install jaxmod```