"""Empirical test for nested while-loops in the eager compile-probe path.

Exercises the exact scenario Copilot flagged in PR #8 comment #6:
a nested stablehlo.while inside an outer while-loop's body, where the
outer while takes the eager compile-probe path (lines 522-565 of
control_flow.cc). The inner while will see ctx.inside_compile=true and
create a WhileLoopPrimitive.

If the removed allow_while_primitive guard was needed, the inner while's
external values would be frozen to their first-call values, causing
incorrect results on subsequent outer loop iterations.
"""

import os
os.environ["JAX_PLATFORMS"] = "mps,cpu"

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

CPU = jax.devices("cpu")[0]
MPS = jax.devices("mps")[0]

passed = 0
failed = 0


def check(name, cpu_val, mps_val, rtol=1e-5, atol=1e-6):
    global passed, failed
    cpu_np = np.asarray(cpu_val)
    mps_np = np.asarray(mps_val)
    if np.allclose(cpu_np, mps_np, rtol=rtol, atol=atol):
        print(f"  PASS  {name}: cpu={cpu_np.flat[:4]}  mps={mps_np.flat[:4]}")
        passed += 1
    else:
        maxdiff = np.max(np.abs(cpu_np - mps_np))
        print(f"  FAIL  {name}: maxdiff={maxdiff}")
        print(f"        cpu={cpu_np}")
        print(f"        mps={mps_np}")
        failed += 1


def run_both(fn, *args, **kwargs):
    """Run fn on CPU and MPS via jit, return (cpu_result, mps_result)."""
    cpu_args = [jax.device_put(a, CPU) for a in args]
    mps_args = [jax.device_put(a, MPS) for a in args]
    cpu_out = jax.jit(fn, device=CPU)(*cpu_args)
    mps_out = jax.jit(fn, device=MPS)(*mps_args)
    return cpu_out, mps_out


def main():
    print("=== Nested While-Loop Probe Path Tests ===\n")

    # ---------------------------------------------------------------
    # Test 1: fori_loop (scan-based counted while) nested inside while
    # The outer while iterates 3 times. Each iteration, the inner
    # fori_loop runs s[0] iterations (1, 2, 3). If the inner while
    # freezes the trip count to its first-call value (1), the result
    # will be wrong.
    # ---------------------------------------------------------------
    print("--- Test 1: fori_loop inside while (trip count depends on outer) ---")

    def nested_fori_in_while(x):
        def body(state):
            counter, val = state
            # Inner loop runs 'counter' iterations, each adding 1.0
            val = lax.fori_loop(0, counter, lambda i, v: v + 1.0, val)
            return (counter + 1, val)

        _, result = lax.while_loop(
            lambda s: s[0] < 4,  # 3 outer iterations: counter=1,2,3
            body,
            (jnp.int32(1), x),
        )
        return result

    # Expected: x + 1 + 2 + 3 = x + 6
    cpu_out, mps_out = run_both(nested_fori_in_while, jnp.float32(0.0))
    check("fori_in_while (x=0, expect 6.0)", cpu_out, mps_out)

    cpu_out, mps_out = run_both(nested_fori_in_while, jnp.float32(10.0))
    check("fori_in_while (x=10, expect 16.0)", cpu_out, mps_out)

    # ---------------------------------------------------------------
    # Test 2: while_loop nested inside while_loop (dynamic condition)
    # The outer while runs 3 iterations. Each iteration, the inner
    # while accumulates until val >= threshold, where threshold
    # depends on the outer counter.
    # ---------------------------------------------------------------
    print("\n--- Test 2: while_loop inside while_loop (dynamic threshold) ---")

    def nested_while_in_while(x):
        def outer_body(state):
            counter, val = state
            # Inner while: add 0.5 until val >= counter (threshold changes per outer iter)
            threshold = counter.astype(jnp.float32)

            def inner_body(inner_state):
                return inner_state + 0.5

            def inner_cond(inner_state):
                return inner_state < threshold

            val = lax.while_loop(inner_cond, inner_body, val)
            return (counter + 1, val)

        _, result = lax.while_loop(
            lambda s: s[0] < 4,
            outer_body,
            (jnp.int32(1), x),
        )
        return result

    # Start at 0.0:
    # iter 1 (counter=1): add 0.5 until >= 1.0 → val=1.0
    # iter 2 (counter=2): add 0.5 until >= 2.0 → val=2.0
    # iter 3 (counter=3): add 0.5 until >= 3.0 → val=3.0
    cpu_out, mps_out = run_both(nested_while_in_while, jnp.float32(0.0))
    check("while_in_while (x=0, expect 3.0)", cpu_out, mps_out)

    cpu_out, mps_out = run_both(nested_while_in_while, jnp.float32(5.0))
    check("while_in_while (x=5, expect 5.0 — already above all thresholds)", cpu_out, mps_out)

    # ---------------------------------------------------------------
    # Test 3: cond inside while (already tested, but verify multiple
    # calls with different inputs produce correct results — stale
    # capture would freeze branch selection)
    # ---------------------------------------------------------------
    print("\n--- Test 3: cond inside while (varying inputs, multiple jit calls) ---")

    @jax.jit
    def cond_in_while_mps(x):
        def body(state):
            i, val = state
            val = lax.cond(
                i % 2 == 0,
                lambda v: v + 1.0,
                lambda v: v * 2.0,
                val,
            )
            return (i + 1, val)

        return lax.while_loop(lambda s: s[0] < 5, body, (jnp.int32(0), x))[1]

    @jax.jit
    def cond_in_while_cpu(x):
        def body(state):
            i, val = state
            val = lax.cond(
                i % 2 == 0,
                lambda v: v + 1.0,
                lambda v: v * 2.0,
                val,
            )
            return (i + 1, val)

        return lax.while_loop(lambda s: s[0] < 5, body, (jnp.int32(0), x))[1]

    # Call multiple times with different inputs to test replay
    for val in [0.0, 1.0, 5.0, -2.0]:
        x_mps = jax.device_put(jnp.float32(val), MPS)
        x_cpu = jax.device_put(jnp.float32(val), CPU)
        mps_r = cond_in_while_mps(x_mps)
        cpu_r = cond_in_while_cpu(x_cpu)
        check(f"cond_in_while replay (x={val})", cpu_r, mps_r)

    # ---------------------------------------------------------------
    # Test 4: Doubly-nested while (while inside while inside jit)
    # with the inner loop's iteration count derived from an outer
    # loop variable. This is the most sensitive scenario for stale
    # captures.
    # ---------------------------------------------------------------
    print("\n--- Test 4: Double-nested while (inner count = outer counter) ---")

    def double_nested(x):
        def outer_body(outer_state):
            outer_i, val = outer_state

            def inner_body(inner_state):
                inner_i, inner_val = inner_state
                return (inner_i + 1, inner_val + outer_i.astype(jnp.float32))

            _, val = lax.while_loop(
                lambda s: s[0] < outer_i,
                inner_body,
                (jnp.int32(0), val),
            )
            return (outer_i + 1, val)

        return lax.while_loop(
            lambda s: s[0] < 4,
            outer_body,
            (jnp.int32(1), x),
        )[1]

    # iter 1 (outer_i=1): inner runs 1 time, adds 1.0 → val = x+1
    # iter 2 (outer_i=2): inner runs 2 times, adds 2.0 each → val = x+1+4
    # iter 3 (outer_i=3): inner runs 3 times, adds 3.0 each → val = x+1+4+9
    # total = x + 14
    cpu_out, mps_out = run_both(double_nested, jnp.float32(0.0))
    check("double_nested (x=0, expect 14.0)", cpu_out, mps_out)

    cpu_out, mps_out = run_both(double_nested, jnp.float32(100.0))
    check("double_nested (x=100, expect 114.0)", cpu_out, mps_out)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print("SOME TESTS FAILED ✗")
    else:
        print("ALL TESTS PASSED ✓")
    return 1 if failed else 0


if __name__ == "__main__":
    exit(main())
