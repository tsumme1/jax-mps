"""Post-revert benchmark: validate WhileLoopPrimitive + basic op correctness/perf."""

import time

import jax
import jax.numpy as jnp
from jax import lax, random

# Force MPS backend
jax.config.update("jax_platforms", "mps")
print(f"Backend: {jax.devices()[0].platform}")

results = {}


def bench(name, fn, *args, warmup=3, iters=10):
    """Benchmark a JIT'd function."""
    jitted = jax.jit(fn)
    # Warmup
    for _ in range(warmup):
        out = jitted(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = jitted(*args)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median = sorted(times)[len(times) // 2]
    results[name] = median
    print(f"  {name}: {median:.2f} ms (median of {iters})")
    return out


print("\n=== 1. Basic Ops (verify arithmetic/shape/reduction reverts) ===")

key = random.PRNGKey(42)
x = random.normal(key, (1000,))
y = random.normal(random.split(key)[0], (1000,))

bench("add_1k", jnp.add, x, y)
bench("exp_1k", jnp.exp, x)
bench("sum_1k", jnp.sum, x)

# Integer division (restored handler)
a_int = jnp.array([7, -7, 7, -7, 0, 100], dtype=jnp.int32)
b_int = jnp.array([2, 2, -2, -2, 3, 7], dtype=jnp.int32)
int_div_result = jax.jit(lax.div)(a_int, b_int)
jax.block_until_ready(int_div_result)
expected = jnp.array([3, -3, -3, 3, 0, 14], dtype=jnp.int32)
assert jnp.array_equal(int_div_result, expected), (
    f"Integer div FAILED: got {int_div_result}, expected {expected}"
)
print(f"  integer_div: PASS (got {int_div_result})")

# Sign NaN propagation (restored handler)
sign_input = jnp.array([1.0, -1.0, 0.0, float("nan")])
sign_result = jax.jit(jnp.sign)(sign_input)
jax.block_until_ready(sign_result)
assert jnp.isnan(sign_result[3]), (
    f"sign(NaN) FAILED: expected NaN, got {sign_result[3]}"
)
print("  sign_nan: PASS (sign(NaN)=NaN)")

print("\n=== 2. Matmul (verify linalg revert doesn't regress) ===")
A = random.normal(key, (128, 256))
B = random.normal(random.split(key)[0], (256, 64))
bench("matmul_128x256x64", jnp.matmul, A, B)

A_big = random.normal(key, (1024, 1024))
B_big = random.normal(random.split(key)[0], (1024, 1024))
bench("matmul_1024x1024", jnp.matmul, A_big, B_big)

print("\n=== 3. Reduction (verify reduction revert) ===")
x_large = random.normal(key, (100_000,))
bench("sum_100k", jnp.sum, x_large)
bench(
    "softmax_10x1000",
    lambda x: jax.nn.softmax(x, axis=-1),
    random.normal(key, (10, 1000)),
)

print("\n=== 4. GRU Scan (WhileLoopPrimitive — MAIN BENCHMARK) ===")
hidden = 64
input_dim = 32
seq_len = 200

Wg = random.normal(random.split(key)[0], (input_dim, hidden)) * 0.01
Ug = random.normal(random.split(key)[0], (hidden, hidden)) * 0.01
Wc = random.normal(random.split(key)[0], (input_dim, hidden)) * 0.01
Uc = random.normal(random.split(key)[0], (hidden, hidden)) * 0.01
xs = random.normal(key, (seq_len, input_dim))


def gru_scan_fwd(params, xs):
    Wg, Ug, Wc, Uc = params

    def body(h, x):
        gate = jax.nn.sigmoid(x @ Wg + h @ Ug)
        cand = jnp.tanh(x @ Wc + gate * (h @ Uc))
        h_new = (1 - gate) * h + gate * cand
        return h_new, h_new

    h0 = jnp.zeros(Wg.shape[1])
    return jax.lax.scan(body, h0, xs)[1].sum()


bench("scan_gru_fwd_h64_t200", gru_scan_fwd, (Wg, Ug, Wc, Uc), xs)


# Grad (backward through scan)
def gru_scan_grad(params, xs):
    return jax.grad(gru_scan_fwd)(params, xs)


bench("scan_gru_grad_h64_t200", gru_scan_grad, (Wg, Ug, Wc, Uc), xs, warmup=2, iters=5)

# Larger hidden
hidden_lg = 128
Wg2 = random.normal(random.split(key)[0], (input_dim, hidden_lg)) * 0.01
Ug2 = random.normal(random.split(key)[0], (hidden_lg, hidden_lg)) * 0.01
Wc2 = random.normal(random.split(key)[0], (input_dim, hidden_lg)) * 0.01
Uc2 = random.normal(random.split(key)[0], (hidden_lg, hidden_lg)) * 0.01

bench("scan_gru_fwd_h128_t200", gru_scan_fwd, (Wg2, Ug2, Wc2, Uc2), xs)
bench(
    "scan_gru_grad_h128_t200",
    gru_scan_grad,
    (Wg2, Ug2, Wc2, Uc2),
    xs,
    warmup=2,
    iters=5,
)

print("\n=== 5. Sort/TopK (verify sort_fft_complex revert) ===")
sort_data = random.normal(key, (10000,))
bench("sort_10k", jnp.sort, sort_data)

topk_data = random.normal(key, (1000,))
bench("top_k_10", lambda x: lax.top_k(x, 10), topk_data)

print("\n=== Summary ===")
for name, ms in results.items():
    print(f"  {name}: {ms:.2f} ms")
print("\nAll correctness checks PASSED ✓")
