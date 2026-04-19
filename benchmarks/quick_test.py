"""Quick test: verify compile is being used and count ops."""

import os
import time

import jax
import jax.numpy as jnp
from jax import lax, random

os.environ["JAX_PLATFORMS"] = "mps"

key = random.PRNGKey(42)
keys = random.split(key, 5)
Wg = random.normal(keys[0], (32, 64)) * 0.01
Ug = random.normal(keys[1], (64, 64)) * 0.01
Wc = random.normal(keys[2], (32, 64)) * 0.01
Uc = random.normal(keys[3], (64, 64)) * 0.01
xs = random.normal(keys[4], (200, 32))

params = (Wg, Ug, Wc, Uc)


def body(h, x):
    Wg, Ug, Wc, Uc = params
    gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
    cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
    return (1 - gate) * h + gate * cand, None


for unroll in [1, 10, 100, 200]:
    fn = jax.jit(lambda x: lax.scan(body, jnp.zeros(64), x, unroll=unroll)[0])

    # First call (compile)
    t0 = time.perf_counter()
    r = fn(xs)
    r.block_until_ready()
    compile_ms = (time.perf_counter() - t0) * 1000

    # Subsequent calls (cached)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        r = fn(xs)
        r.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    print(
        f"unroll={unroll:3d}: compile={compile_ms:8.1f}ms  run={times[len(times) // 2]:6.2f}ms"
    )
