About
=====

*Jaxmod*---short for "JAX for modelling"---is a lightweight "extension pack" for JAX-based scientific computing. It provides convenience utilities, wrappers, and conventions on top of the excellent `Equinox <https://docs.kidger.site/equinox/>`_ and `Optimistix <https://docs.kidger.site/optimistix>`_ libraries, making them even easier to use for real scientific workflows.

The library was originally created to avoid code duplication in thermochemistry applications for planetary science. Many such problems share similar structural patterns---stoichiometric systems, batched equilibrium calculations, constraints, and differentiable solvers. *Jaxmod* consolidates this boilerplate into reusable components and establishes consistent conventions, whether for vectorising models efficiently or extending solver behaviour in a principled way.

*Jaxmod* is released under `The GNU General Public License v3.0 or later <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.

The main author is Dan J. Bower (ETH Zurich).