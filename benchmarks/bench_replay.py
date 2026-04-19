"""Measure compile + replay separately, and compare single-eval vs per-step-eval."""

import os
import time

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

params = (Wg, Ug, Wc, Uc)


def body(h, x):
    Wg, Ug, Wc, Uc = params
    gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
    cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
    return (1 - gate) * h + gate * cand, None


for unroll in [1, 10, 50, 100, 200]:
    xs = random.normal(keys[4], (200, 32))
    fn = jax.jit(lambda x: lax.scan(body, jnp.zeros(64), x, unroll=unroll)[0])

    # First call (compile + run)
    t0 = time.perf_counter()
    r = fn(xs)
    r.block_until_ready()
    first_ms = (time.perf_counter() - t0) * 1000

    # Warmup cached path
    for _ in range(10):
        xs_w = random.normal(random.PRNGKey(unroll), (200, 32))
        r = fn(xs_w)
        r.block_until_ready()

    # Measure cached replay
    times = []
    for i in range(100):
        xs_i = random.normal(random.PRNGKey(i + 100), (200, 32))
        t0 = time.perf_counter()
        r = fn(xs_i)
        r.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    print(
        f"unroll={unroll:3d}: first={first_ms:7.1f}ms  replay(med)={times[len(times) // 2]:6.2f}ms"
    )
