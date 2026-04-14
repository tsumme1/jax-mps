# jax-mps Developer Build & Run Guide

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python ≥ 3.11** (project requires it in `pyproject.toml`)
- **Xcode Command Line Tools** (provides Clang and Metal SDK)
- **CMake** (installed via Homebrew: `brew install cmake`)

## Setup

### 1. Create the venv

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

### 2. Install dependencies + build the plugin (editable)

```bash
.venv/bin/python -m pip install -e .
```

This uses `scikit-build-core` to invoke CMake, compile the C++ PJRT plugin against the MLX headers from the installed `mlx` package, and install it as an editable package.

### 3. Fix the runtime library search path

The editable install's `_find_library()` searches for `libpjrt_plugin_mps.dylib` using this priority:

1. `src/jax_plugins/mps/libpjrt_plugin_mps.dylib` — package directory  
2. `src/jax_plugins/mps/lib/libpjrt_plugin_mps.dylib` — wheel-style  
3. `build/*/lib/libpjrt_plugin_mps.dylib` — cmake build directory (glob)  
4. System paths (`/usr/local/lib`, `/opt/homebrew/lib`)

The glob at step 3 is what typically matches. **However**, the built dylib has `@rpath/libmlx.dylib` as a dependency with `@loader_path` as the only rpath. This means `libmlx.dylib` must be co-located with the plugin dylib.

**After building**, create a symlink:

```bash
# Find the actual build directory (it encodes the Python version)
BUILD_DIR=$(ls -d build/cp3*-macosx_*/lib 2>/dev/null | head -1)

# Symlink libmlx.dylib from the MLX package
ln -sf $(python3 -c "import mlx; print(mlx.__path__[0])")/lib/libmlx.dylib "$BUILD_DIR/libmlx.dylib"
```

Or using the venv directly:

```bash
ln -sf $(pwd)/.venv/lib/python3.13/site-packages/mlx/lib/libmlx.dylib \
       build/cp313-cp313-macosx_26_0_arm64/lib/libmlx.dylib
```

### 4. Remove stale build directories

If you previously built with a different Python version, **remove the old build directory**. The glob `build/*/lib/` matches alphabetically, so an old `cp311` build will shadow a newer `cp313` build:

```bash
# Check which library JAX is actually loading
.venv/bin/python -c "
import os; os.environ['JAX_PLATFORMS'] = 'mps'
import jax_plugins.mps as m
print('Loading:', m._find_library())
"

# Remove any stale build dirs that don't match your Python version
rm -rf build/cp311-*  # example: remove Python 3.11 build
```

## Building After Code Changes

After modifying C++ source files under `src/pjrt_plugin/`:

```bash
# Rebuild (incremental — CMake only recompiles changed files)
.venv/bin/python -m pip install -e .

# Re-create the libmlx symlink (pip install may remove it)
ln -sf $(pwd)/.venv/lib/python3.13/site-packages/mlx/lib/libmlx.dylib \
       build/cp313-cp313-macosx_26_0_arm64/lib/libmlx.dylib
```

**Tip:** Wrap this in a script for convenience:

```bash
#!/bin/bash
set -e
.venv/bin/python -m pip install -e .
BUILD_DIR=$(ls -d build/cp3*-macosx_*/lib 2>/dev/null | head -1)
ln -sf $(pwd)/.venv/lib/python3.13/site-packages/mlx/lib/libmlx.dylib "$BUILD_DIR/libmlx.dylib"
echo "Build complete. Library at: $BUILD_DIR/libpjrt_plugin_mps.dylib"
```

## Running

Always use the venv's Python and set `JAX_PLATFORMS=mps`:

```bash
# Run a script
JAX_PLATFORMS=mps .venv/bin/python my_script.py

# Or set it inside the script:
# import os; os.environ["JAX_PLATFORMS"] = "mps"
```

### Enable float64 support

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Float64 operations are automatically routed to the MLX CPU stream. No additional configuration needed.

### Debugging

```bash
# Enable verbose debug logging
MPS_DEBUG=1 JAX_PLATFORMS=mps .venv/bin/python my_script.py

# Disable mx::compile() (fall back to direct dispatch, useful for debugging)
MPS_NO_COMPILE=1 JAX_PLATFORMS=mps .venv/bin/python my_script.py

# Enable performance profiling
MPS_PROFILE=1 JAX_PLATFORMS=mps .venv/bin/python my_script.py
```

## Running Tests

```bash
# Float64 test suite
JAX_PLATFORMS=mps .venv/bin/python test_float64.py

# Existing test suite (if available)
JAX_PLATFORMS=mps .venv/bin/python -m pytest tests/
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Library not loaded: @rpath/libmlx.dylib` | Missing symlink in build dir | Create the libmlx symlink (see step 3) |
| Code changes not taking effect | Stale build dir from old Python | Remove old `build/cp3XX-*` dirs (see step 4) |
| `requires a different Python: 3.9.20` | System Python picked up instead of venv | Use `.venv/bin/python -m pip` explicitly |
| `float64 is not supported on the GPU` | Missing DefaultDeviceGuard or float64 handler | Check `HasFloat64()` detection and `DefaultDeviceGuard` in `mlx_executable.cc` |

## Current Versions (tested)

| Component | Version |
|-----------|---------|
| Python | 3.13.12 |
| JAX | 0.9.2 |
| MLX | 0.31.1 |
| jax-mps | 0.9.14 |
