"""Primitive definition for the entangle ordering operation."""

from typing import TypeVar

import jax.custom_batching
import jax.extend.core
import jax.interpreters.ad as ad
import jax.interpreters.batching
import jax.interpreters.mlir as mlir
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from entangle_jax import _ffi  # noqa: F401

_Payload = TypeVar("_Payload", bound=PyTree[Array])
"""The pytree of arrays passed as `entangle`'s payload, returned unchanged so a caller
keeps the exact type it passed in."""

entangle_p = jax.extend.core.Primitive("entangle_jax")


@entangle_p.def_impl
def _entangle_impl(payload: Array, witness: Array) -> Array:
    return jax.ffi.ffi_call(
        "entangle_jax",
        jax.ShapeDtypeStruct(payload.shape, payload.dtype),
        input_output_aliases={0: 0},
    )(payload, witness)


@entangle_p.def_abstract_eval
def _entangle_abstract_eval(payload, witness):
    del witness
    return payload


mlir.register_lowering(entangle_p, mlir.lower_fun(_entangle_impl, multiple_results=False))


def _entangle_p_vmap(vector_arg_values, batch_axes):
    # entangle is a cheap passthrough, so a sequential map over the batch keeps
    # it correct without a bespoke batching rule.
    out = jax.vmap(jax.custom_batching.sequential_vmap(entangle_p.bind), in_axes=batch_axes)(
        *vector_arg_values
    )
    return out, 0


jax.interpreters.batching.primitive_batchers[entangle_p] = _entangle_p_vmap


def _entangle_jvp(primals, tangents):
    payload, witness = primals
    t_payload, t_witness = tangents
    if not isinstance(t_witness, ad.Zero):
        raise NotImplementedError(
            "entangle: cannot differentiate w.r.t. a witness argument. Witnesses only "
            "establish an execution-order dependency. You should stop-gradient anything passed as "
            "a witness if it originates from a differentiated computation."
        )
    out = entangle_p.bind(payload, witness)
    if isinstance(t_payload, ad.Zero):
        return out, ad.Zero(out.aval.to_tangent_aval())
    return out, entangle_p.bind(t_payload, witness)


ad.primitive_jvps[entangle_p] = _entangle_jvp


def _entangle_transpose(cts_out, payload, witness):
    if ad.is_undefined_primal(witness):
        raise NotImplementedError("entangle: cannot differentiate w.r.t. a witness")
    if ad.is_undefined_primal(payload):
        return cts_out, None
    return None, None


ad.primitive_transposes[entangle_p] = _entangle_transpose


def entangle(payload: _Payload, *witnesses: PyTree[Array]) -> _Payload:
    """Return `payload` unchanged, ordered under `jit` after every witness's producer."""
    payload_leaves, treedef = jtu.tree_flatten(payload)
    witness_leaves = [leaf for w in witnesses for leaf in jtu.tree_leaves(w)]
    if not witness_leaves:
        return payload
    signal, *rest = witness_leaves
    for w in rest:
        signal = entangle_p.bind(signal, w)
    entangled_leaves = [entangle_p.bind(leaf, signal) for leaf in payload_leaves]
    return jtu.tree_unflatten(treedef, entangled_leaves)
