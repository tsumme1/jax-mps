"""Benchmark a single GRU body step (no loop) to measure raw op cost."""

import os
import time

import jax
import jax.numpy as jnp
from jax import random

os.environ.setdefault("JAX_PLATFORMS", "mps")

key = random.PRNGKey(42)
keys = random.split(key, 5)
Wg = random.normal(keys[0], (32, 64)) * 0.01
Ug = random.normal(keys[1], (64, 64)) * 0.01
Wc = random.normal(keys[2], (32, 64)) * 0.01
Uc = random.normal(keys[3], (64, 64)) * 0.01
h = random.normal(keys[4], (64,))
x = random.normal(keys[4], (32,))


@jax.jit
def gru_step(h, x, Wg, Ug, Wc, Uc):
    gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
    cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
    return (1 - gate) * h + gate * cand


# Warmup
for _ in range(20):
    r = gru_step(h, x, Wg, Ug, Wc, Uc)
    r.block_until_ready()

# Measure
times = []
for _ in range(200):
    t0 = time.perf_counter()
    r = gru_step(h, x, Wg, Ug, Wc, Uc)
    r.block_until_ready()
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
med = times[len(times) // 2]
p10 = times[len(times) // 10]
p90 = times[9 * len(times) // 10]
print(f"jax-mps single GRU step: {med:.3f} ms (p10={p10:.3f}, p90={p90:.3f})")
