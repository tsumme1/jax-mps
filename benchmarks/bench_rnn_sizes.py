"""Multi-size GRU RNN benchmark with full unrolling.
Tests small → xlarge configs with unroll=1 (WhileLoopPrimitive per-step)
through full unroll (entire loop in one compiled graph).
"""

import os
import time

import jax
import jax.numpy as jnp
from jax import lax, random

os.environ.setdefault("JAX_PLATFORMS", "mps")


def bench(fn, warmup=5, repeats=15):
    for _ in range(warmup):
        r = fn()
        r.block_until_ready()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = fn()
        r.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


configs = [
    ("small", 32, 64, 200),
    ("medium", 128, 256, 200),
    ("large", 512, 1024, 200),
    ("xlarge", 2048, 4096, 200),
]

key = random.PRNGKey(42)

for name, input_dim, hidden, seq_len in configs:
    keys = random.split(key, 5)
    Wg = random.normal(keys[0], (input_dim, hidden)) * 0.01
    Ug = random.normal(keys[1], (hidden, hidden)) * 0.01
    Wc = random.normal(keys[2], (input_dim, hidden)) * 0.01
    Uc = random.normal(keys[3], (hidden, hidden)) * 0.01
    xs = random.normal(keys[4], (seq_len, input_dim))

    # full unroll = seq_len (entire loop in one graph, no WhileLoopPrimitive)
    unrolls = [1, 10, 50, seq_len]

    print("=" * 75)
    print(f"  {name} ({input_dim}→{hidden}, T={seq_len})")
    print("=" * 75)
    print(f"  {'unroll':<12}  {'time':>10}  {'ms/step':>10}  {'note'}")
    print(f"  {'------':<12}  {'----':>10}  {'-------':>10}  {'----'}")

    for unroll in unrolls:
        label = f"{unroll}" if unroll < seq_len else f"{unroll} (full)"

        @jax.jit
        def run(Wg=Wg, Ug=Ug, Wc=Wc, Uc=Uc, xs=xs, _u=unroll):
            def body(h, x):
                gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
                cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
                return (1 - gate) * h + gate * cand, None

            return lax.scan(body, jnp.zeros(Wg.shape[1]), xs, unroll=_u)[0]

        try:
            med = bench(run)
            note = "WhileLoopPrimitive" if unroll < seq_len else "single compiled graph"
            print(f"  {label:<12}  {med:8.2f} ms  {med / seq_len:8.3f} ms  {note}")
        except Exception as e:
            print(f"  {label:<12}  {'FAILED':>10}  {'':>10}  {str(e)[:50]}")

    print()

print("Done. Compare with C++ MLX baseline: ./build/benchmarks/bench_scaling")
