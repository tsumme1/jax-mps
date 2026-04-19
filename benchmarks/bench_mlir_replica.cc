// Benchmark: replicate the exact MLX graph structure that jax-mps produces
// from the STABLEHLO IR for lax.scan with unroll=200.
//
// This mimics:
// - Decomposed sigmoid (negate → exp → broadcast(1.0) → add → broadcast(1.0) → divide)
// - slice + reshape for input extraction (instead of pre-sliced or mx::take)
// - Constants via broadcast_in_dim from scalar
//
// Build: cmake --build build --target bench_mlir_replica
// Run:   ./build/benchmarks/bench_mlir_replica

#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/compile_impl.h"

namespace mx = mlx::core;

struct BenchResult {
    double median_ms;
};

BenchResult bench(std::function<void()> fn, int warmup = 5, int repeats = 30) {
    for (int i = 0; i < warmup; ++i)
        fn();
    std::vector<double> times(repeats);
    for (int i = 0; i < repeats; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    return {times[repeats / 2]};
}

// GRU step using the SAME op decomposition as STABLEHLO IR:
// - sigmoid is decomposed into: negate → exp → add(broadcast(1.0)) → divide(broadcast(1.0))
// - all scalar constants are broadcast_in_dim'd
mx::array gru_step_mlir_style(const mx::array& h, const mx::array& x, const mx::array& Wg,
                                const mx::array& Ug, const mx::array& Wc, const mx::array& Uc) {
    // sigmoid(x @ Wg + h @ Ug)  -- decomposed
    auto xWg = mx::matmul(x, Wg);
    auto hUg = mx::matmul(h, Ug);
    auto pre_sigmoid = mx::add(xWg, hUg);
    auto neg = mx::negative(pre_sigmoid);
    auto exp_val = mx::exp(neg);
    auto one_scalar = mx::array(1.0f);
    auto one_bcast1 = mx::broadcast_to(one_scalar, exp_val.shape());
    auto denominator = mx::add(one_bcast1, exp_val);
    auto one_bcast2 = mx::broadcast_to(mx::array(1.0f), denominator.shape());
    auto gate = mx::divide(one_bcast2, denominator);

    // tanh(x @ Wc + gate * (h @ Uc))
    auto xWc = mx::matmul(x, Wc);
    auto hUc = mx::matmul(h, Uc);
    auto gated = mx::multiply(gate, hUc);
    auto pre_tanh = mx::add(xWc, gated);
    auto cand = mx::tanh(pre_tanh);

    // (1 - gate) * h + gate * cand
    auto one_bcast3 = mx::broadcast_to(mx::array(1.0f), gate.shape());
    auto one_minus_gate = mx::subtract(one_bcast3, gate);
    auto term1 = mx::multiply(one_minus_gate, h);
    auto term2 = mx::multiply(gate, cand);
    return mx::add(term1, term2);
}

// GRU step using native MLX ops (what C++ baseline uses)
mx::array gru_step_native(const mx::array& h, const mx::array& x, const mx::array& Wg,
                           const mx::array& Ug, const mx::array& Wc, const mx::array& Uc) {
    auto gate = mx::sigmoid(mx::add(mx::matmul(x, Wg), mx::matmul(h, Ug)));
    auto cand = mx::tanh(mx::add(mx::matmul(x, Wc), mx::multiply(gate, mx::matmul(h, Uc))));
    return mx::add(mx::multiply(mx::subtract(mx::array(1.0f), gate), h),
                   mx::multiply(gate, cand));
}

int main() {
    const int TOTAL_STEPS = 200;
    const int HIDDEN = 64;
    const int INPUT_DIM = 32;

    auto Wg = mx::multiply(mx::random::normal({INPUT_DIM, HIDDEN}), mx::array(0.01f));
    auto Ug = mx::multiply(mx::random::normal({HIDDEN, HIDDEN}), mx::array(0.01f));
    auto Wc = mx::multiply(mx::random::normal({INPUT_DIM, HIDDEN}), mx::array(0.01f));
    auto Uc = mx::multiply(mx::random::normal({HIDDEN, HIDDEN}), mx::array(0.01f));
    auto xs = mx::random::normal({TOTAL_STEPS, INPUT_DIM});
    auto h0 = mx::zeros({HIDDEN});
    mx::eval({Wg, Ug, Wc, Uc, xs, h0});

    // ===================================================================
    // MODE 1: Native C++ (baseline — what bench_while_loop Mode A does)
    // ===================================================================
    {
        auto body_fn = [TOTAL_STEPS, INPUT_DIM](const std::vector<mx::array>& inputs)
            -> std::vector<mx::array> {
            auto h = inputs[0];
            auto& xs_full = inputs[1];
            auto& Wg = inputs[2]; auto& Ug = inputs[3];
            auto& Wc = inputs[4]; auto& Uc = inputs[5];
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                h = gru_step_native(h, x, Wg, Ug, Wc, Uc);
            }
            return {h};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&body_fn);
        auto compiled = mx::detail::compile(body_fn, id, false, {});
        mx::detail::compile_erase(id);

        auto run = [&]() {
            auto r = compiled({h0, xs, Wg, Ug, Wc, Uc});
            mx::eval(r);
        };
        auto result = bench(run);
        std::printf("Native C++ (single compile, slice):      %8.2f ms\n", result.median_ms);
    }

    // ===================================================================
    // MODE 2: MLIR-style decomposed (mimics jax-mps graph exactly)
    // ===================================================================
    {
        auto body_fn = [TOTAL_STEPS, INPUT_DIM](const std::vector<mx::array>& inputs)
            -> std::vector<mx::array> {
            auto h = inputs[0];
            auto& xs_full = inputs[1];
            auto& Wg = inputs[2]; auto& Ug = inputs[3];
            auto& Wc = inputs[4]; auto& Uc = inputs[5];
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                h = gru_step_mlir_style(h, x, Wg, Ug, Wc, Uc);
            }
            return {h};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&body_fn);
        auto compiled = mx::detail::compile(body_fn, id, false, {});
        mx::detail::compile_erase(id);

        auto run = [&]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = compiled({h0, xs, Wg, Ug, Wc, Uc});
            auto t1 = std::chrono::high_resolution_clock::now();
            mx::eval(r);
            auto t2 = std::chrono::high_resolution_clock::now();
            static int count = 0;
            if (count == 10) {
                std::printf("  [breakdown] compile_replace: %.3f ms, eval: %.3f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
            count++;
        };
        auto result = bench(run);
        std::printf("MLIR-style (decomposed sigmoid, slice):  %8.2f ms\n", result.median_ms);
    }

    // ===================================================================
    // MODE 3: MLIR-style but with func.call simulation
    //         (each GRU step is a separate function, like MLIR's call)
    // ===================================================================
    {
        // Simulate what jax-mps does: the body function calls a "closed_call"
        // sub-function for each step. This creates a different trace because
        // the sub-function has its own argument arrays.
        auto body_fn = [TOTAL_STEPS, INPUT_DIM](const std::vector<mx::array>& inputs)
            -> std::vector<mx::array> {
            auto h = inputs[0];
            auto& xs_full = inputs[1];
            auto& Wg = inputs[2]; auto& Ug = inputs[3];
            auto& Wc = inputs[4]; auto& Uc = inputs[5];

            // Simulate the main function's structure:
            // For each step: slice → reshape → call closed_call
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto slice_result = mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM});
                auto x = mx::reshape(slice_result, {INPUT_DIM});
                // This is exactly closed_call's body — inlined since we can't
                // actually call a sub-function in MLX compile
                h = gru_step_mlir_style(h, x, Wg, Ug, Wc, Uc);
            }
            return {h};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&body_fn);
        auto compiled = mx::detail::compile(body_fn, id, false, {});
        mx::detail::compile_erase(id);

        auto run = [&]() {
            auto r = compiled({h0, xs, Wg, Ug, Wc, Uc});
            mx::eval(r);
        };
        auto result = bench(run);
        std::printf("MLIR-style + call sim (full replica):    %8.2f ms\n", result.median_ms);
    }

    // ===================================================================
    // MODE 4: Native + per-step eval (simulates what happens with
    //         multiple Execute() calls)
    // ===================================================================
    {
        auto body_fn = [INPUT_DIM](const std::vector<mx::array>& inputs)
            -> std::vector<mx::array> {
            auto h = inputs[0];
            auto& x = inputs[1];
            auto& Wg = inputs[2]; auto& Ug = inputs[3];
            auto& Wc = inputs[4]; auto& Uc = inputs[5];
            return {gru_step_native(h, x, Wg, Ug, Wc, Uc)};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&body_fn);
        auto compiled = mx::detail::compile(body_fn, id, false, {});
        mx::detail::compile_erase(id);

        auto run = [&]() {
            auto h = h0;
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                mx::eval(x);
                auto r = compiled({h, x, Wg, Ug, Wc, Uc});
                h = r[0];
                mx::eval(h);
            }
        };
        auto result = bench(run);
        std::printf("Native C++ (per-step eval, 200 evals):   %8.2f ms\n", result.median_ms);
    }

    std::printf("\njax-mps target: ~9 ms | C++ ceiling: ~5 ms\n");
    std::printf("If Mode 2 ≈ jax-mps, the gap is purely from MLIR op decomposition.\n");
    std::printf("If Mode 2 ≈ Mode 1, the jax-mps issue is in graph tracing, not op structure.\n");

    return 0;
}
