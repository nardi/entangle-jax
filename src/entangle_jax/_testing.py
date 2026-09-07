"""Private test-only stand-ins for a mutable native resource.

Used only by this package's own hazard regression test, to reproduce the
sibling-calls reordering hazard entangle exists to fix, without needing a real
consumer package. Not part of the public API. Do not import from outside this
package's own test suite.
"""

import jax
import jax.numpy as jnp

from entangle_jax import _ffi  # noqa: F401


def reset_slots():
    _ffi.testing_reset_slots()


def write(slot_id, value):
    """Mutate slot `slot_id` to `value`, in place. Has side effect, mirrors refactor."""
    return jax.ffi.ffi_call(
        "entangle_jax_testing_write",
        jax.ShapeDtypeStruct((), jnp.int32),
        has_side_effect=True,
    )(jnp.asarray(slot_id, jnp.int32), jnp.asarray(value, jnp.float32))


def read(slot_id):
    """Read slot `slot_id`. No side effect declared, mirrors solve."""
    return jax.ffi.ffi_call(
        "entangle_jax_testing_read",
        jax.ShapeDtypeStruct((), jnp.float32),
    )(jnp.asarray(slot_id, jnp.int32))
