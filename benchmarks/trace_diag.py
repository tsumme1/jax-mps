"""Focused timing of unroll=100 (2 iterations) to see per-iteration breakdown."""

import os

import jax
import jax.numpy as jnp
from jax import lax, random

os.environ.setdefault("JAX_PLATFORMS", "mps")

key = random.PRNGKey(42)
keys = random.split(key, 5)
Wg = random.normal(keys[0], (32, 64)) * 0.01
Ug = random.normal(keys[1], (64, 64)) * 0.01
Wc = random.normal(keys[2], (32, 64)) * 0.01
Uc = random.normal(keys[3], (64, 64)) * 0.01
xs = random.normal(keys[4], (200, 32))


# carry only
def scan_fn_carry(params, xs):
    Wg, Ug, Wc, Uc = params

    def body(h, x):
        gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
        cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
        return (1 - gate) * h + gate * cand, None

    return lax.scan(body, jnp.zeros(64), xs, unroll=100)[0]


# with outputs
def scan_fn_out(params, xs):
    Wg, Ug, Wc, Uc = params

    def body(h, x):
        gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
        cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
        return (1 - gate) * h + gate * cand, h

    return lax.scan(body, jnp.zeros(64), xs, unroll=100)


print("=== unroll=100, carry only (expect 2 iters) ===", flush=True)
fn = jax.jit(scan_fn_carry)
r = fn((Wg, Ug, Wc, Uc), xs)
r.block_until_ready()
print("first call done", flush=True)
r = fn((Wg, Ug, Wc, Uc), xs)
r.block_until_ready()
print("second call done", flush=True)

print("\n=== unroll=100, with outputs (expect 2 iters) ===", flush=True)
fn2 = jax.jit(scan_fn_out)
r2 = fn2((Wg, Ug, Wc, Uc), xs)
jax.tree.map(lambda x: x.block_until_ready(), r2)
print("first call done", flush=True)
r2 = fn2((Wg, Ug, Wc, Uc), xs)
jax.tree.map(lambda x: x.block_until_ready(), r2)
print("second call done", flush=True)
