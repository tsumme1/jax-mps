"""Test float64 operations on the MPS backend via CPU stream."""
import os
os.environ["JAX_PLATFORMS"] = "mps"

import jax
import jax.numpy as jnp
import numpy as np

# Enable float64
jax.config.update("jax_enable_x64", True)

def test_basic_f64():
    """Test basic float64 array creation and arithmetic."""
    print("=== Test 1: Basic float64 creation ===")
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    print(f"  dtype: {x.dtype}, values: {x}")
    assert x.dtype == jnp.float64, f"Expected float64, got {x.dtype}"
    print("  PASSED")

def test_f64_arithmetic():
    """Test float64 arithmetic operations."""
    print("\n=== Test 2: Float64 arithmetic ===")
    a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    b = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float64)
    
    c = a + b
    assert c.dtype == jnp.float64, f"add: expected f64, got {c.dtype}"
    np.testing.assert_allclose(c, [5.0, 7.0, 9.0])
    print(f"  add: {c}")
    
    d = a * b
    assert d.dtype == jnp.float64
    np.testing.assert_allclose(d, [4.0, 10.0, 18.0])
    print(f"  mul: {d}")
    
    e = a / b
    assert e.dtype == jnp.float64
    print(f"  div: {e}")
    
    f = a - b
    assert f.dtype == jnp.float64
    np.testing.assert_allclose(f, [-3.0, -3.0, -3.0])
    print(f"  sub: {f}")
    print("  PASSED")

def test_f64_precision():
    """Test that we actually get 64-bit precision, not silently downcast to 32."""
    print("\n=== Test 3: Float64 precision verification ===")
    # This value is representable in f64 but NOT in f32
    # f32 has ~7 decimal digits of precision, f64 has ~15
    x = jnp.array(1.0, dtype=jnp.float64)
    eps = jnp.array(1e-12, dtype=jnp.float64)
    result = x + eps
    
    # In float32, 1.0 + 1e-12 == 1.0 (the eps is below f32 ULP)
    # In float64, 1.0 + 1e-12 > 1.0
    diff = float(result - x)
    print(f"  1.0 + 1e-12 - 1.0 = {diff}")
    assert diff > 0, f"Lost f64 precision! diff={diff} (would be 0 in f32)"
    assert abs(diff - 1e-12) < 1e-15, f"Unexpected precision loss: diff={diff}"
    print("  PASSED — genuine 64-bit precision confirmed")

def test_f64_unary_ops():
    """Test unary operations on float64."""
    print("\n=== Test 4: Float64 unary ops ===")
    x = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
    
    ops = [
        ("exp", jnp.exp),
        ("log", jnp.log),
        ("sqrt", jnp.sqrt),
        ("sin", jnp.sin),
        ("cos", jnp.cos),
        ("tanh", jnp.tanh),
        ("abs", jnp.abs),
        ("negative", jnp.negative),
    ]
    
    for name, fn in ops:
        result = fn(x)
        assert result.dtype == jnp.float64, f"{name}: expected f64, got {result.dtype}"
        # Compare against numpy reference
        expected = fn(np.array([0.5, 1.0, 2.0], dtype=np.float64))
        np.testing.assert_allclose(result, expected, rtol=1e-14,
                                   err_msg=f"{name} mismatch")
        print(f"  {name}: OK")
    print("  PASSED")

def test_f64_reduction():
    """Test reduction operations on float64."""
    print("\n=== Test 5: Float64 reductions ===")
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    
    s = jnp.sum(x)
    assert s.dtype == jnp.float64
    assert float(s) == 15.0
    print(f"  sum: {s}")
    
    m = jnp.mean(x)
    assert m.dtype == jnp.float64
    assert float(m) == 3.0
    print(f"  mean: {m}")
    
    mx = jnp.max(x)
    assert mx.dtype == jnp.float64
    assert float(mx) == 5.0
    print(f"  max: {mx}")
    
    mn = jnp.min(x)
    assert mn.dtype == jnp.float64
    assert float(mn) == 1.0
    print(f"  min: {mn}")
    print("  PASSED")

def test_f64_matmul():
    """Test matrix multiplication in float64."""
    print("\n=== Test 6: Float64 matmul ===")
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float64)
    c = a @ b
    assert c.dtype == jnp.float64
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_allclose(c, expected)
    print(f"  result:\n{c}")
    print("  PASSED")

def test_f64_jit():
    """Test JIT compilation with float64."""
    print("\n=== Test 7: Float64 under JIT ===")
    
    @jax.jit
    def f(x, y):
        return jnp.sum(x * y) + jnp.mean(x)
    
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float64)
    result = f(x, y)
    assert result.dtype == jnp.float64
    expected = np.sum(np.array([1,2,3.]) * np.array([4,5,6.])) + np.mean([1,2,3.])
    np.testing.assert_allclose(float(result), expected)
    print(f"  JIT result: {result}")
    print("  PASSED")

def test_f64_grad():
    """Test gradient computation with float64."""
    print("\n=== Test 8: Float64 gradient ===")
    
    def loss(x):
        return jnp.sum(x ** 2)
    
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    g = jax.grad(loss)(x)
    assert g.dtype == jnp.float64
    np.testing.assert_allclose(g, [2.0, 4.0, 6.0])
    print(f"  grad(sum(x^2)) at [1,2,3] = {g}")
    print("  PASSED")

def test_mixed_f64_f32():
    """Test mixed float64/float32 operations."""
    print("\n=== Test 9: Mixed f64/f32 ===")
    x64 = jnp.array([1.0, 2.0], dtype=jnp.float64)
    x32 = jnp.array([3.0, 4.0], dtype=jnp.float32)
    
    # JAX should promote f32 to f64
    result = x64 + x32
    print(f"  f64 + f32 -> dtype={result.dtype}, values={result}")
    # With x64 mode enabled, this should promote to f64
    assert result.dtype == jnp.float64, f"Expected f64 promotion, got {result.dtype}"
    print("  PASSED")

if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()
    
    tests = [
        test_basic_f64,
        test_f64_arithmetic,
        test_f64_precision,
        test_f64_unary_ops,
        test_f64_reduction,
        test_f64_matmul,
        test_f64_jit,
        test_f64_grad,
        test_mixed_f64_f32,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests passed! ✅")
    else:
        print(f"{failed} tests failed ❌")
