About
=====

*Jaxmod* is a Python package that provides lightweight utility functions for JAX arrays, batching, and pytrees. It mostly builds on top of the amazing `Equinox <https://docs.kidger.site/equinox/>`_ package, whilst notably incorporating structural conventions and helper functions that make JAX-based scientific programming more convenient and consistent.

The package is designed to simplify common tasks like array type handling, batch axis inference, and pytree manipulation, making it easier to write robust, performant, and composable numerical code.

Although generally useful for numerical and scientific computing, *Jaxmod* is somewhat biased toward applications in chemistry, geochemistry, and planetary science, where tasks like handling stoichiometric matrices, managing physical constants, and ensuring numerical stability are common.

*Jaxmod* is released under `The GNU General Public License v3.0 or later <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.

The main author is Dan J. Bower (ETH Zurich).