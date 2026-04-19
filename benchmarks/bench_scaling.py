"""Scaling test: jax-mps GRU scan vs C++ compiled baseline.

Weights are passed as explicit arguments (not closures) so the JIT-compiled
MLIR module's outer ops are minimal — matching the C++ benchmark structure
where weights are passed directly to the compiled body function.
"""

import os
import time

import jax
import jax.numpy as jnp
from jax import lax, random

os.environ.setdefault("JAX_PLATFORMS", "mps")


def bench_scan(Wg, Ug, Wc, Uc, xs, hidden, unroll, warmup=5, repeats=20):
    """Benchmark a GRU scan with weights as explicit args."""

    @jax.jit
    def run(Wg, Ug, Wc, Uc, xs):
        def body(h, x):
            gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
            cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
            return (1 - gate) * h + gate * cand, None

        return lax.scan(body, jnp.zeros(hidden), xs, unroll=unroll)[0]

    for _ in range(warmup):
        r = run(Wg, Ug, Wc, Uc, xs)
        r.block_until_ready()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = run(Wg, Ug, Wc, Uc, xs)
        r.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# Model configurations: (input_dim, hidden_dim, seq_len, label)
configs = [
    (32, 64, 200, "tiny   (32→64,    T=200)"),
    (256, 512, 200, "small  (256→512,  T=200)"),
    (512, 1024, 200, "medium (512→1024, T=200)"),
    (1024, 2048, 100, "large  (1024→2048, T=100)"),
    (2048, 4096, 50, "xlarge (2048→4096, T=50)"),
]

unrolls = [1, 10, 50]

key = random.PRNGKey(42)

print("=" * 72)
print("  jax-mps GRU scan scaling benchmark")
print("  (weights as explicit args — matches C++ benchmark structure)")
print("=" * 72)

for input_dim, hidden, seq_len, label in configs:
    keys = random.split(key, 5)
    Wg = random.normal(keys[0], (input_dim, hidden)) * 0.01
    Ug = random.normal(keys[1], (hidden, hidden)) * 0.01
    Wc = random.normal(keys[2], (input_dim, hidden)) * 0.01
    Uc = random.normal(keys[3], (hidden, hidden)) * 0.01
    xs = random.normal(keys[4], (seq_len, input_dim))

    print(f"\n  {label}")
    print(f"  {'unroll':>8}  {'jax-mps':>12}  {'ms/step':>10}")
    print(f"  {'------':>8}  {'-------':>12}  {'-------':>10}")

    for u in unrolls:
        if seq_len % u != 0:
            continue
        t = bench_scan(Wg, Ug, Wc, Uc, xs, hidden, u, warmup=3, repeats=15)
        ms_per_step = t / seq_len
        print(f"  {u:>8}  {t:>10.2f} ms  {ms_per_step:>8.3f} ms")

print()
