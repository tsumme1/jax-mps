// Benchmark: isolate WHERE the jax-mps graph becomes slower.
// 
// This creates the MLX graph through SEPARATE sub-functions (like jax-mps does),
// where each sub-function creates its own scope of intermediate arrays.
// This tests whether the graph DAG structure differs when created via
// function-scoped intermediates vs single-scope.
//
// Build: cmake --build build --target bench_graph_trace
// Run:   ./build/benchmarks/bench_graph_trace

#include <chrono>
#include <cstdio>
#include <functional>
#include <unordered_map>
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

// Simulate the jax-mps ExecuteFunction pattern:
// Each "closed_call" creates a FRESH scope of arrays.
// Input arrays are COPIED into the scope (like ValueMap emplace from inputs).
// Output arrays are returned from the function (like func.return).
mx::array closed_call_scoped(const mx::array& Wg, const mx::array& Ug,
                              const mx::array& Wc, const mx::array& Uc,
                              const mx::array& h, const mx::array& x) {
    // Create a scope simulating ExecuteFunction's ValueMap
    // All intermediate results are local to this scope
    
    // sigmoid(x @ Wg + h @ Ug)  -- decomposed
    auto v0 = mx::matmul(x, Wg);          // dot_general
    auto v1 = mx::matmul(h, Ug);          // dot_general
    auto v2 = mx::add(v0, v1);            // add
    auto v3 = mx::negative(v2);           // negate
    auto v4 = mx::exp(v3);               // exp
    auto cst0 = mx::array(1.0f);          // constant
    auto v5 = mx::broadcast_to(cst0, v4.shape()); // broadcast_in_dim
    auto v6 = mx::add(v5, v4);           // add
    auto cst1 = mx::array(1.0f);          // constant (separate!)
    auto v7 = mx::broadcast_to(cst1, v6.shape()); // broadcast_in_dim
    auto v8 = mx::divide(v7, v6);        // divide (gate)

    // tanh(x @ Wc + gate * (h @ Uc))
    auto v9 = mx::matmul(x, Wc);          // dot_general
    auto v10 = mx::matmul(h, Uc);         // dot_general
    auto v11 = mx::multiply(v8, v10);     // multiply
    auto v12 = mx::add(v9, v11);          // add
    auto v13 = mx::tanh(v12);             // tanh (cand)

    // (1 - gate) * h + gate * cand
    auto cst2 = mx::array(1.0f);          // constant (separate!)
    auto v14 = mx::broadcast_to(cst2, v8.shape()); // broadcast_in_dim
    auto v15 = mx::subtract(v14, v8);     // subtract
    auto v16 = mx::multiply(v15, h);      // multiply
    auto v17 = mx::multiply(v8, v13);     // multiply
    auto v18 = mx::add(v16, v17);         // add
    return v18;
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
    // MODE A: Direct loop (what we know works at ~4ms)
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
                h = closed_call_scoped(Wg, Ug, Wc, Uc, h, x);
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
                std::printf("  [breakdown] replace: %.3f ms, eval: %.3f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
            count++;
        };
        auto result = bench(run);
        std::printf("MODE A: Direct call (6 compile inputs): %8.2f ms\n", result.median_ms);
    }

    // ===================================================================
    // MODE B: Weights as captured constants (like jax-mps where weights
    //         are MLIR constants, not compile inputs)
    // ===================================================================
    {
        // jax-mps calls compiled_fn_({xs_array}), weights are inside the graph
        auto body_fn = [TOTAL_STEPS, INPUT_DIM, &Wg, &Ug, &Wc, &Uc, &h0](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto& xs_full = inputs[0];
            auto h = h0;  // h0 is captured, not an input
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                h = closed_call_scoped(Wg, Ug, Wc, Uc, h, x);
            }
            return {h};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&body_fn);
        auto compiled = mx::detail::compile(body_fn, id, false, {});
        mx::detail::compile_erase(id);

        auto run = [&]() {
            auto r = compiled({xs});
            mx::eval(r);
        };

        // Measure tape size
        auto wrap_fn = [&body_fn](const std::vector<mx::array>& ins)
            -> mx::detail::ArraysAndExtra {
            return {body_fn(ins), nullptr};
        };
        auto [ti, to, ex] = mx::detail::compile_trace(wrap_fn, {xs}, false);
        auto [tape, parents] = mx::detail::compile_dfs(ti, to, {xs});
        std::printf("  [tape_size=%zu, inputs=%zu, outputs=%zu]\n",
                    tape.size(), ti.size(), to.size());

        auto result = bench(run);
        std::printf("MODE B: Captured weights (1 compile input): %8.2f ms\n", result.median_ms);
    }

    std::printf("\njax-mps timing: replace ~0.8ms, eval ~8.4ms, total ~9.2ms\n");
    std::printf("If MODE B >> MODE A, the issue is captured-vs-input weight handling.\n");

    return 0;
}
