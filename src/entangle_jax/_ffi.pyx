# distutils: language = c++
# cython: language_level=3
"""Registers the compiled entangle FFI handlers as JAX custom call targets.

The handler addresses come from _entangle_ffi.cc and _testing_ffi.cc, both compiled
directly into this extension. Each address is wrapped in a PyCapsule, the format JAX's
FFI registration expects, and handed to jax.ffi.register_ffi_target. The _testing_*
handlers back entangle_jax._testing, a private module used only by this package's own
hazard regression test and not part of the public API.
"""

from cpython.pycapsule cimport PyCapsule_New


cdef extern from "_entangle_ffi.h":
    void* entangle_handler_address()

cdef extern from "_testing_ffi.h":
    void* write_handler_address()
    void* read_handler_address()
    void reset_slots()


cdef object _capsule(void* address):
    return PyCapsule_New(address, NULL, NULL)


def _register_targets():
    import jax

    # JAX canonicalizes only "cpu" and "gpu"; every other name is filed as given. The GPU
    # plugins pick up targets filed under "CUDA" and "ROCM", so lowercase names would stay
    # queued forever and never reach them.
    for platform in ("cpu", "CUDA", "ROCM", "tpu"):
        jax.ffi.register_ffi_target(
            "entangle_jax", _capsule(entangle_handler_address()), platform=platform
        )
        jax.ffi.register_ffi_target(
            "entangle_jax_testing_write", _capsule(write_handler_address()), platform=platform
        )
        jax.ffi.register_ffi_target(
            "entangle_jax_testing_read", _capsule(read_handler_address()), platform=platform
        )


_register_targets()


def testing_reset_slots():
    reset_slots()
