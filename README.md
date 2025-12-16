# Jaxmod

[![Release 0.4.0](https://img.shields.io/badge/Release-0.4.0-blue.svg)](https://github.com/ExPlanetology/jaxmod/releases/tag/v0.4.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/ExPlanetology/jaxmod/actions/workflows/ci.yml/badge.svg)](https://github.com/ExPlanetology/jaxmod/actions/workflows/ci.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Test coverage](https://img.shields.io/badge/Coverage-93%25-brightgreen)](https://github.com/ExPlanetology/jaxmod)

## About
Jaxmod&mdash;short for "JAX for modelling"&mdash;is a lightweight "extension pack" for JAX-based scientific computing. It provides convenience utilities, wrappers, and conventions on top of the excellent [Equinox](https://docs.kidger.site/equinox/) and [Optimistix](https://docs.kidger.site/optimistix/) libraries, making them even easier to use for real scientific workflows.

The library was originally created to avoid code duplication in thermochemistry applications for planetary science. Many such problems share similar structural patterns&mdash;stoichiometric systems, batched equilibrium calculations, constraints, and differentiable solvers. Jaxmod consolidates this boilerplate into reusable components and establishes consistent conventions, whether for vectorising models efficiently or extending solver behaviour in a principled way.

## Documentation

The documentation is available online, with options to download it in EPUB or PDF format:

[https://jaxmod.readthedocs.io/en/latest/](https://jaxmod.readthedocs.io/en/latest/)

## Quick install

Jaxmod is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux). It is recommended to install Jaxmod in a dedicated Python environment. Before installation, create and activate the environment, then run:

```pip install jaxmod```