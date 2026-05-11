"""Standalone test for CasePrimitive (lax.cond / lax.switch under jax.jit).

Exercises the CasePrimitive path by running lax.cond and lax.switch inside
jax.jit, which triggers mx::compile and the CasePrimitive handler.
"""

import os
os.environ["JAX_PLATFORMS"] = "mps,cpu"

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

CPU = jax.devices("cpu")[0]
MPS = jax.devices("mps")[0]


def run_test(name, fn, *args, rtol=1e-5, atol=1e-6):
    """Run fn on both CPU and MPS, compare results."""
    cpu_args = [jax.device_put(a, CPU) for a in args]
    mps_args = [jax.device_put(a, MPS) for a in args]

    cpu_result = jax.jit(fn, device=CPU)(*cpu_args)
    mps_result = jax.jit(fn, device=MPS)(*mps_args)

    cpu_val = np.asarray(cpu_result)
    mps_val = np.asarray(mps_result)

    if np.allclose(cpu_val, mps_val, rtol=rtol, atol=atol):
        print(f"  PASS  {name}: cpu={cpu_val.flat[:4]}  mps={mps_val.flat[:4]}")
    else:
        maxdiff = np.max(np.abs(cpu_val - mps_val))
        print(f"  FAIL  {name}: maxdiff={maxdiff}")
        print(f"        cpu={cpu_val}")
        print(f"        mps={mps_val}")
        return False
    return True


def main():
    all_passed = True
    print("=== CasePrimitive Tests (lax.cond / lax.switch under jax.jit) ===\n")

    # --- lax.cond tests ---
    print("--- lax.cond ---")

    # True branch (scalar)
    all_passed &= run_test(
        "cond.true_scalar",
        lambda pred, x, y: lax.cond(pred, lambda: x + 1, lambda: y * 2),
        np.bool_(True), np.float32(3.0), np.float32(4.0),
    )

    # False branch (scalar)
    all_passed &= run_test(
        "cond.false_scalar",
        lambda pred, x, y: lax.cond(pred, lambda: x + 1, lambda: y * 2),
        np.bool_(False), np.float32(3.0), np.float32(4.0),
    )

    # Array cond with operands
    key = jax.random.PRNGKey(42)
    x_arr = jax.random.normal(key, (4,))
    all_passed &= run_test(
        "cond.array",
        lambda pred, x: lax.cond(pred, lambda a: a + jnp.flip(a), lambda a: a * 2, x),
        np.bool_(True), np.asarray(x_arr),
    )
    all_passed &= run_test(
        "cond.array_false",
        lambda pred, x: lax.cond(pred, lambda a: a + jnp.flip(a), lambda a: a * 2, x),
        np.bool_(False), np.asarray(x_arr),
    )

    # --- lax.switch tests ---
    print("\n--- lax.switch ---")

    x_switch = jax.random.normal(key, (4,))
    for idx in [0, 1, 2]:
        all_passed &= run_test(
            f"switch.branch{idx}",
            lambda sel, x: lax.switch(sel, [
                lambda y: y + 1,
                lambda y: y * 2,
                lambda y: y - 3,
            ], x),
            np.int32(idx), np.asarray(x_switch),
        )

    # Out-of-bounds (negative → clamp to last)
    all_passed &= run_test(
        "switch.oob_negative",
        lambda sel, x: lax.switch(sel, [
            lambda y: y + 1, lambda y: y * 2, lambda y: y - 3,
        ], x),
        np.int32(-1), np.float32(5.0),
    )

    # Out-of-bounds (too large → clamp to last)
    all_passed &= run_test(
        "switch.oob_large",
        lambda sel, x: lax.switch(sel, [
            lambda y: y + 1, lambda y: y * 2, lambda y: y - 3,
        ], x),
        np.int32(100), np.float32(5.0),
    )

    # Multiple return values
    all_passed &= run_test(
        "switch.multi_return",
        lambda sel, x: lax.switch(sel, [
            lambda y: (y + 1, y * 2),
            lambda y: (y - 1, y / 2),
            lambda y: (y * 3, y + 4),
        ], x)[0],
        np.int32(1), np.float32(5.0),
    )

    # Many branches (5)
    all_passed &= run_test(
        "switch.many_branches",
        lambda sel, x: lax.switch(sel, [
            lambda y: y + 1,
            lambda y: y * 2,
            lambda y: y - 3,
            lambda y: y / 2,
            lambda y: y ** 2,
        ], x),
        np.int32(3), np.float32(4.0),
    )

    # 2D array switch
    x_2d = np.asarray(jax.random.normal(key, (4, 4)))
    all_passed &= run_test(
        "switch.2d_array",
        lambda sel, x: lax.switch(sel, [
            lambda y: y + jnp.flip(y, axis=0),
            lambda y: y + jnp.flip(y, axis=1),
            lambda y: y + jnp.swapaxes(y, 0, 1),
        ], x),
        np.int32(2), x_2d,
    )

    # --- Nested: switch inside switch ---
    print("\n--- Nested ---")

    all_passed &= run_test(
        "switch.nested",
        lambda outer, inner, x: lax.switch(outer, [
            lambda y: lax.switch(inner, [lambda z: z + 1, lambda z: z + 2], y),
            lambda y: lax.switch(inner, [lambda z: z * 2, lambda z: z * 3], y),
        ], x),
        np.int32(0), np.int32(1), np.float32(5.0),
    )

    # --- cond inside while (already tested by upstream, but verify CasePrimitive path) ---
    print("\n--- Mixed: cond inside while ---")

    all_passed &= run_test(
        "while_with_cond",
        lambda x: lax.while_loop(
            lambda state: state[0] < 4,
            lambda state: (
                state[0] + 1,
                lax.cond(
                    state[0] % 2 == 0,
                    lambda v: v + 1,
                    lambda v: v * 2,
                    state[1],
                ),
            ),
            (np.int32(0), x),
        )[1],
        np.float32(1.0),
    )

    all_passed &= run_test(
        "while_with_switch",
        lambda x: lax.while_loop(
            lambda state: state[0] < 3,
            lambda state: (
                state[0] + 1,
                lax.switch(
                    state[0] % 3,
                    [lambda v: v + 1, lambda v: v * 2, lambda v: v - 0.5],
                    state[1],
                ),
            ),
            (np.int32(0), x),
        )[1],
        np.float32(1.0),
    )

    print()
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
