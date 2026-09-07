# entangle-jax

A generic JAX primitive that is opaque to XLA reordering optimizations.

`entangle(payload, *witnesses)` returns `payload` unchanged, but ordered under
`jit` after every witness's producer. Useful for sequencing native
side-effecting calls that XLA would otherwise be free to reorder, since they
share no ordinary data dependency.

Install with `uv add entangle-jax` or `pip install entangle-jax`.

## Why

When interacting with external native libraries, it is sometimes necessary to
interact with objects they manage through opaque pointers. This conflicts with
the XLA memory model where everything is a static-sized buffer that can be
arbitrarily copied or reused when needed. One way to achieve this interaction is
by creating a token that XLA can pass around that references this object, but
XLA is not aware that e.g. multiple copies of this token in fact reference the
same memory, and so will perform optimizations that are at worst unsafe and at
best inefficient.

Unfortunately, there is no way to tell XLA that a buffer will be modified
in-place and that it should maintain consistent order of operations involving
this buffer. Marking a primitive as side-effecting will stop XLA from removing
it as dead code, but it will still assume that the input and output buffers are
distinct, independent objects. At most, it is possible to create an in-place
modifying primitive by using [input-output aliasing](https://openxla.org/xla/aliasing), in which case the input and
output buffers are the same, but then XLA will simply copy the input buffer if
it is used by another function to ensure both calls are safe.

To work around this, we can create an artificial data dependency with
`entangle`. The contents of a custom call are not visible to the XLA optimizer, so
it has to treat the output `payload` as a new distinct variable from the input
`payload` that depends on all `witness` values. A standard-library alternative
would be `jax.lax.optimization_barrier`, but this is currently 
[unreliable on CPU without a non-default XLA flag](https://github.com/openxla/xla/issues/20440).

The actual runtime behavior of the primitive is a no-op: it uses aliasing to tell XLA that the input and output `payload` should be the same buffer, so it does literally nothing. This also means it is supported on every platform. The usual pattern is to overwrite the `payload` variable so that its buffer can always be donated, but if it cannot be (for example because it is owned by a caller outside of the JIT context) XLA will insert a copy of the `payload` buffer.

## Example

```python
import jax.numpy as jnp

token1 = jnp.asarray(0, jnp.int32)
x = some_native_read(token1)
token2 = modify_token(token1)
y = some_native_read(token2)
```

Here `token*` is a value that has some library object associated with it, and `modify_token` changes something about this backing object. Even if `modify_token` is marked as side-effecting, `x = some_native_read(token1)` and `token2 = modify_token(token1)` read the same token variable. This means XLA could choose to reorder them, which would lead to the following order:

```python
# Equivalent reordering:
token1 = jnp.asarray(0, jnp.int32)
token2 = modify_token(token1)
x = some_native_read(token1)
y = some_native_read(token2)
```

In this case, both `x` and `y` read the same value. To enforce ordering, we can entangle `x` and the `token` used in its computation, before modifying it.

```python
from entangle_jax import entangle

token1 = jnp.asarray(0, jnp.int32)
x = some_native_read(token1)
token2 = entangle(token1, x)
token3 = modify_token(token2)
y = some_native_read(token3)
```

This means that `modify_token` cannot be reordered to be before the line that determines `x`, since it uses a value that depends on `x`.

```python
# Invalid reordering:
token1 = jnp.asarray(0, jnp.int32)
token2 = entangle(token1, x)  # x doesn't exist yet!
token3 = modify_token(token2)
x = some_native_read(token1)
y = some_native_read(token3)
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

The compiled extension rebuilds automatically on import while developing, so the
default editable install (`uv sync`) is what you want day to day.
