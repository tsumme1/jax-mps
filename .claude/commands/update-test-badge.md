# Update Test Badge

Run the JAX upstream test suite against the MPS backend and update the pass-rate badge in README.md.

## Step 1: Run the JAX Test Suite

Run the test suite script with `--keep` to reuse any existing clone:

```
uv run python scripts/run_jax_tests.py --keep
```

This runs all upstream JAX tests (except float64-only tests) against the MPS backend. It takes a long time — wait for it to complete.

## Step 2: Extract the Results

From the summary output, note:
- **Passed**: number of passing tests
- **Available**: collected - skipped - xfailed (the denominator)
- **Pass rate**: passed / available as a percentage

## Step 3: Update the README Badge

The badge is on line 3 of `README.md` and looks like:

```
![JAX tests](https://img.shields.io/badge/JAX_tests-XX.X%25_passing-COLOR)
```

Update the percentage to match the new pass rate, rounded to one decimal place. Do not change the badge color.

Only edit the badge if the percentage has actually changed. If it already matches, report that no update is needed.

## Step 4: Report

Provide a summary to the user with:
- Total collected, skipped, available, passed, failed, errors
- The new pass rate
- Whether the badge was updated or already correct
