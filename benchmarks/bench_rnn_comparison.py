"""GRU RNN benchmark for comparing jax-mps branches.

Usage:
  # On the PR branch:
  python benchmarks/bench_rnn_comparison.py --save pr_results.json

  # On the main branch:
  python benchmarks/bench_rnn_comparison.py --save main_results.json

  # Compare:
  python benchmarks/bench_rnn_comparison.py --compare main_results.json pr_results.json
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
from jax import lax, random

os.environ.setdefault("JAX_PLATFORMS", "mps")


def bench(fn, warmup=5, repeats=15):
    """Run fn, return median wall time in ms."""
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


CONFIGS = [
    ("small", 32, 64, 200),
    ("medium", 128, 256, 200),
    ("large", 512, 1024, 200),
    ("xlarge", 2048, 4096, 200),
]

UNROLLS = [1, 10, 50]  # 200 (full) added dynamically


def run_benchmark():
    """Run GRU scan benchmark across configs and unroll factors."""
    key = random.PRNGKey(42)
    results = {}

    for name, input_dim, hidden, seq_len in CONFIGS:
        keys = random.split(key, 5)
        Wg = random.normal(keys[0], (input_dim, hidden)) * 0.01
        Ug = random.normal(keys[1], (hidden, hidden)) * 0.01
        Wc = random.normal(keys[2], (input_dim, hidden)) * 0.01
        Uc = random.normal(keys[3], (hidden, hidden)) * 0.01
        xs = random.normal(keys[4], (seq_len, input_dim))

        unrolls = UNROLLS + [seq_len]  # add full unroll

        print(f"\n{'=' * 70}")
        print(f"  {name} ({input_dim}→{hidden}, T={seq_len})")
        print(f"{'=' * 70}")
        print(f"  {'unroll':<12}  {'time':>10}  {'ms/step':>10}")
        print(f"  {'------':<12}  {'----':>10}  {'-------':>10}")

        config_results = {}
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
                config_results[str(unroll)] = round(med, 2)
                print(f"  {label:<12}  {med:8.2f} ms  {med / seq_len:8.3f} ms")
            except Exception as e:
                config_results[str(unroll)] = None
                print(f"  {label:<12}  {'FAILED':>10}  {str(e)[:40]}")

        results[name] = config_results

    return results


def compare(baseline_path, current_path):
    """Compare two saved results files and print speedup table."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    print(f"\nComparison: {baseline_path} (baseline) vs {current_path} (current)")
    print(f"{'=' * 80}")
    print(
        f"  {'config':<10} {'unroll':>8}  {'baseline':>10}  {'current':>10}  {'speedup':>10}"
    )
    print(
        f"  {'------':<10} {'------':>8}  {'--------':>10}  {'-------':>10}  {'-------':>10}"
    )

    for config in baseline:
        if config not in current:
            continue
        for unroll in baseline[config]:
            b = baseline[config].get(unroll)
            c = current[config].get(unroll)
            if b is None or c is None:
                print(f"  {config:<10} {unroll:>8}  {'N/A':>10}  {'N/A':>10}")
                continue
            speedup = b / c
            marker = " ✓" if speedup > 1.05 else (" ✗" if speedup < 0.95 else "")
            print(
                f"  {config:<10} {unroll:>8}  {b:8.2f} ms  {c:8.2f} ms  {speedup:8.2f}x{marker}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRU RNN benchmark for jax-mps")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CURRENT"),
        help="Compare two saved result files",
    )
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
    else:
        results = run_benchmark()
        if args.save:
            with open(args.save, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.save}")
        print("\nBenchmark complete.")
