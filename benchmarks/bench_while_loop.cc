// Benchmark: scan/while loop performance at different unroll factors.
//
// Measures the per-iteration overhead of eval() in HandleWhile using a
// GRU-style recurrence. Use this to detect performance regressions in
// the while-loop execution path.
//
// Build: cmake --build build --target bench_while_loop
// Run:   ./build/benchmarks/bench_while_loop

#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

struct BenchResult {
    double median_ms;
    double min_ms;
    double max_ms;
};

BenchResult bench(std::function<void()> fn, int warmup = 3, int repeats = 20) {
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
    return {times[repeats / 2], times[0], times[repeats - 1]};
}

// GRU-style step: gate=sigmoid(x@Wg+h@Ug), cand=tanh(x@Wc+gate*(h@Uc)),
// h_new=(1-gate)*h + gate*cand
mx::array gru_step(const mx::array& h, const mx::array& x, const mx::array& Wg,
                   const mx::array& Ug, const mx::array& Wc, const mx::array& Uc) {
    auto gate = mx::sigmoid(mx::add(mx::matmul(x, Wg), mx::matmul(h, Ug)));
    auto cand =
        mx::tanh(mx::add(mx::matmul(x, Wc), mx::multiply(gate, mx::matmul(h, Uc))));
    return mx::add(mx::multiply(mx::subtract(mx::array(1.0f), gate), h),
                   mx::multiply(gate, cand));
}

int main() {
    const int TOTAL_STEPS = 200;
    const int HIDDEN = 64;
    const int INPUT_DIM = 32;
    const int UNROLL_VALUES[] = {1, 5, 10, 50, 100, 200};

    // GRU parameters
    auto Wg = mx::multiply(mx::random::normal({INPUT_DIM, HIDDEN}), mx::array(0.01f));
    auto Ug = mx::multiply(mx::random::normal({HIDDEN, HIDDEN}), mx::array(0.01f));
    auto Wc = mx::multiply(mx::random::normal({INPUT_DIM, HIDDEN}), mx::array(0.01f));
    auto Uc = mx::multiply(mx::random::normal({HIDDEN, HIDDEN}), mx::array(0.01f));
    auto xs = mx::random::normal({TOTAL_STEPS, INPUT_DIM});
    auto h0 = mx::zeros({HIDDEN});
    mx::eval({Wg, Ug, Wc, Uc, xs, h0});

    std::printf("=== GRU scan benchmark (%d steps, h=%d, input=%d) ===\n", TOTAL_STEPS, HIDDEN,
                INPUT_DIM);
    std::printf("    Models jax.lax.scan(gru_body, ..., length=%d, unroll=K)\n\n", TOTAL_STEPS);
    std::printf("  %-8s  %8s  %12s  %10s\n", "unroll", "evals", "median (ms)", "ms/step");
    std::printf("  %-8s  %8s  %12s  %10s\n", "------", "-----", "-----------", "-------");

    for (int unroll : UNROLL_VALUES) {
        if (TOTAL_STEPS % unroll != 0)
            continue;
        int n_iters = TOTAL_STEPS / unroll;

        // Simulate HandleWhile: while-loop with K unrolled steps + eval per iteration.
        // Each iteration builds a K-step lazy graph, then eval() materializes it.
        auto run = [&]() {
            auto h = h0;
            for (int i = 0; i < n_iters; ++i) {
                for (int k = 0; k < unroll; ++k) {
                    int idx = i * unroll + k;
                    auto x = mx::reshape(
                        mx::slice(xs, {idx, 0}, {idx + 1, INPUT_DIM}), {INPUT_DIM});
                    h = gru_step(h, x, Wg, Ug, Wc, Uc);
                }
                mx::eval(h);
            }
        };

        auto r = bench(run);
        std::printf("  %-8d  %8d  %10.2f ms  %8.3f ms\n", unroll, n_iters, r.median_ms,
                    r.median_ms / TOTAL_STEPS);
    }

    std::printf("\n  Higher unroll → fewer eval() calls → better amortization.\n");
    std::printf("  ms/step approaching a constant means eval overhead is fully amortized.\n");

    return 0;
}
