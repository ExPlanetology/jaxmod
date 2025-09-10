Overview
========

Whilst `Equinox <https://docs.kidger.site/equinox/>`_ and `JAX <https://docs.jax.dev/en/latest/>`_ provide great flexibility for building software, establishing a few conventions can help reduce boilerplate, ensure consistent behaviour, and make scientific computing code more robust. For example:

1. **Trace all arrays consistently.** Both NumPy and JAX arrays should be recognised in pytrees for tracing. *Note: This functionality is already provided by Equinox.*
2. **Use a consistent dtype.** Convert arrays to JAX arrays with a known dtype (typically ``float64``) to prevent recompilation due to changing types. This also allows user-provided NumPy arrays to be safely converted internally, enabling them to participate in batching when appropriate.
3. **Separate tracing from batching.** Treat JAX arrays as batchable over their leading dimension, whilst holding NumPy arrays fixed as constants. This distinction clarifies which data participates in ``vmap`` or other vectorised operations.

In addition:

4. **Hashable callables.** Some JAX transformations (e.g., ``jax.jit``, ``jax.vmap``) require that all static arguments, including function references, are hashable.
5. **Functions that preserve numerical precision.** Operations such as summing logarithms or taking differences can suffer from loss of precision or catastrophic cancellation, especially when combined with automatic differentiation. *Jaxmod* provides helpers that mitigate these issues for more reliable computations.
6. **Generally useful scientific functions.** Many common operations---such as power laws, scaling relations, or standard mathematical utilities---are ubiquitous in scientific modeling. *Jaxmod* includes convenient implementations to reduce boilerplate and ensure consistency.

Hence *Jaxmod* provides a set of helper functions that encode these conventions and offer a collection of convenience utilities, making it easier to write JAX-based scientific code.