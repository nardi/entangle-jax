"""A generic ordering primitive for JAX, opaque to XLA's algebraic simplifier.

`entangle(payload, *witnesses)` returns `payload` unchanged, but ordered under `jit`
after every witness's producer. Any XLA custom call is opaque to the algebraic
simplifier by construction, so this holds without needing `has_side_effect=True`.
It is more reliable than `jax.lax.optimization_barrier` which needs a non-default flag on
CPU.

Intended for sequencing native side-effecting custom calls that touch the same
resource but share no ordinary data dependency.
"""

from entangle_jax._primitive import entangle

__all__ = ["entangle"]
