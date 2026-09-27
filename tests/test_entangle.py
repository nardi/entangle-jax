"""Regression tests for entangle_jax, covering the reordering hazard it fixes, its
JVP/transpose rules, and N-ary witness folding under jit."""

import re
import typing
from collections import Counter

import jax
import jax.numpy as jnp

from entangle_jax import entangle
from entangle_jax._testing import read, reset_slots, write


def test_hazard_reordering_is_fixed_by_entangle():
    def unordered(slot, v1, v2):
        id1 = write(slot, v1)
        x = read(id1)
        _id2 = write(slot, v2)
        return x

    def ordered(slot, v1, v2):
        id1 = write(slot, v1)
        x = read(id1)
        ordered_id1 = entangle(id1, x)
        _id2 = write(ordered_id1, v2)
        return x

    slot = jnp.asarray(3, jnp.int32)
    n_trials = 200

    def run(fn):
        f = jax.jit(fn)
        wrong = 0
        for _ in range(n_trials):
            reset_slots()
            x = f(slot, 1.0, 2.0)
            if not jnp.allclose(x, 1.0):
                wrong += 1
        return wrong

    assert run(unordered) > 0
    assert run(ordered) == 0


def test_grad_matches_plain_computation_with_stop_gradiented_witness():
    def f(x, w):
        y = entangle(x, jax.lax.stop_gradient(w))
        return jnp.sum(y**2)

    x0 = jnp.asarray([1.0, 2.0, 3.0])
    w0 = jnp.asarray(5.0)

    g_entangled = jax.jit(jax.grad(f))(x0, w0)
    g_plain = jax.jit(jax.grad(lambda x, w: jnp.sum(x**2)))(x0, w0)
    assert jnp.allclose(g_entangled, g_plain)


def test_grad_wrt_witness_raises():
    import pytest

    x0 = jnp.asarray([1.0, 2.0, 3.0])
    w0 = jnp.asarray(5.0)

    def f(x, w):
        return jnp.sum(entangle(x, w) ** 2)

    with pytest.raises(NotImplementedError):
        jax.grad(f, argnums=1)(x0, w0)


def test_jvp_matches_plain_computation():
    def f(x, w):
        y = entangle(x, jax.lax.stop_gradient(w))
        return jnp.sum(y**2)

    x0 = jnp.asarray([1.0, 2.0, 3.0])
    w0 = jnp.asarray(5.0)

    _, tangent = jax.jvp(f, (x0, w0), (jnp.ones_like(x0), jnp.zeros_like(w0)))
    _, tangent_plain = jax.jvp(
        lambda x, w: jnp.sum(x**2), (x0, w0), (jnp.ones_like(x0), jnp.zeros_like(w0))
    )
    assert jnp.allclose(tangent, tangent_plain)


def test_nary_witness_folding_produces_one_call_per_witness():
    def f(payload, w1, w2, w3):
        return entangle(payload, w1, w2, w3)

    payload = jnp.asarray(7, jnp.int32)
    w1, w2, w3 = jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(3.0)

    hlo = jax.jit(f).lower(payload, w1, w2, w3).compile().as_text()
    hlo_str = typing.cast(str, hlo)
    targets = Counter(re.findall(r'custom_call_target="([^"]+)"', hlo_str))
    assert targets["entangle_jax"] == 3

    out = jax.jit(f)(payload, w1, w2, w3)
    assert bool(out == payload)

    out_nan = jax.jit(f)(payload, jnp.nan, w2, w3)
    assert bool(out_nan == payload)


def test_no_witnesses_is_identity():
    payload = jnp.asarray([1, 2, 3])
    assert entangle(payload) is payload


def test_vmap_over_payload_and_witness_matches_per_example_entangle():
    """`vmap(entangle)` must apply the same per-example entangling as calling `entangle`
    once per example directly, not just pass batched arrays through unchanged.

    A regression test for a broken `primitive_batchers` stub that used to accept
    any number of positional arguments and echo them straight back with a
    hardcoded output batch dimension, without invoking `entangle_p` at all. This
    is silently correct only by coincidence for a bare identity payload, but
    breaks any real caller (like `jax.vmap` over a stateful solve) with a
    `TypeError`, since the "batched args" tuple JAX passes in is not itself a
    valid operand.
    """
    payload = jnp.asarray([1.0, 2.0, 3.0])
    witness = jnp.asarray([10.0, 20.0, 30.0])

    batched = jax.vmap(entangle)(payload, witness)
    per_example = jnp.stack([entangle(p, w) for p, w in zip(payload, witness, strict=True)])
    assert jnp.array_equal(batched, per_example)
    assert jnp.array_equal(batched, payload)

    # Also under jit, and with one argument batched and the other not.
    jitted = jax.jit(jax.vmap(entangle, in_axes=(0, None)))(payload, witness[0])
    assert jnp.array_equal(jitted, payload)


def test_gpu_targets_are_queued_under_the_plugin_platform_names():
    from jaxlib import xla_client

    def queued_under(platform):
        return {entry[0] for entry in xla_client._custom_callback.get(platform, [])}

    for wrong in ("cuda", "rocm"):
        assert not any(name.startswith("entangle_jax") for name in queued_under(wrong))
