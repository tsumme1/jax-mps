"""A/B comparison: jax-mps compiled vs direct (lazy eval), vs C++ baselines."""

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
xs = random.normal(keys[4], (200, 32))

C_DIRECT = {1: 48.80, 5: 16.53, 10: 10.49, 50: 8.07, 100: 7.27, 200: 7.26}
C_COMPILED = {1: 37.05, 5: 11.92, 10: 8.41, 50: 4.94, 100: 4.37, 200: 4.07}
C_COMPILED_B = {
    1: 42.37,
    5: 13.55,
    10: 8.09,
    50: 5.42,
    100: 5.02,
    200: 4.82,
}  # JAX-like structure


def bench_scan(params, xs, unroll, warmup=5, repeats=30):
    Wg, Ug, Wc, Uc = params

    def body(h, x):
        gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
        cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
        return (1 - gate) * h + gate * cand, None

    fn = jax.jit(lambda p, x: lax.scan(body, jnp.zeros(64), x, unroll=unroll)[0])
    # Warmup
    for _ in range(warmup):
        r = fn(params, xs)
        r.block_until_ready()
    # Measure
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = fn(params, xs)
        r.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


params = (Wg, Ug, Wc, Uc)
unrolls = [1, 5, 10, 50, 100, 200]

print(
    f"  {'unroll':>8}  {'jax-mps':>12}  {'C++ direct':>12}  {'C++ compiled':>14}  {'vs direct':>10}  {'vs compiled':>12}"
)
print(
    f"  {'------':>8}  {'-------':>12}  {'----------':>12}  {'------------':>14}  {'---------':>10}  {'-----------':>12}"
)

for u in unrolls:
    t = bench_scan(params, xs, u)
    cd = C_DIRECT.get(u, 0)
    cc = C_COMPILED_B.get(u, 0)
    ratio_d = t / cd if cd else 0
    ratio_c = t / cc if cc else 0
    print(
        f"  {u:>8}  {t:>10.2f} ms  {cd:>10.2f} ms  {cc:>12.2f} ms  {ratio_d:>9.2f}x  {ratio_c:>11.2f}x"
    )
