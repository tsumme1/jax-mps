// Benchmark: Test if creating SEPARATE copies of weight arrays per step
// causes the same slowdown as jax-mps.
//
// Theory: jax-mps creates new mx::array nodes for weight block arguments
// on each @closed_call invocation. Even though the arrays reference the
// same data, they're different graph nodes, preventing MLX from sharing them.

#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/compile_impl.h"

namespace mx = mlx::core;

struct BenchResult { double median_ms; };

BenchResult bench(std::function<void()> fn, int warmup = 5, int repeats = 30) {
    for (int i = 0; i < warmup; ++i) fn();
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

mx::array gru_step(const mx::array& h, const mx::array& x,
                    const mx::array& Wg, const mx::array& Ug,
                    const mx::array& Wc, const mx::array& Uc) {
    auto v0 = mx::matmul(x, Wg);
    auto v1 = mx::matmul(h, Ug);
    auto v2 = mx::add(v0, v1);
    auto v3 = mx::negative(v2);
    auto v4 = mx::exp(v3);
    auto v5 = mx::broadcast_to(mx::array(1.0f), v4.shape());
    auto v6 = mx::add(v5, v4);
    auto v7 = mx::broadcast_to(mx::array(1.0f), v6.shape());
    auto v8 = mx::divide(v7, v6);  // gate
    auto v9 = mx::matmul(x, Wc);
    auto v10 = mx::matmul(h, Uc);
    auto v11 = mx::multiply(v8, v10);
    auto v12 = mx::add(v9, v11);
    auto v13 = mx::tanh(v12);
    auto v14 = mx::broadcast_to(mx::array(1.0f), v8.shape());
    auto v15 = mx::subtract(v14, v8);
    auto v16 = mx::multiply(v15, h);
    auto v17 = mx::multiply(v8, v13);
    return mx::add(v16, v17);
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

    // Test: Does mx::array copy preserve graph node identity?
    {
        auto a = mx::array(1.0f);
        auto b = a;  // copy construction
        auto c = mx::array(a);  // explicit copy
        auto d = mx::array(1.0f);  // new construction
        std::printf("Array ID test: a=%lu, b=%lu, c=%lu, d=%lu\n", a.id(), b.id(), c.id(), d.id());
        std::printf("Same? a==b: %d, a==c: %d, a==d: %d\n",
                    a.id() == b.id(), a.id() == c.id(), a.id() == d.id());
    }


    // MODE 1: Shared weights (baseline)
    {
        auto fn = [TOTAL_STEPS, INPUT_DIM, &Wg, &Ug, &Wc, &Uc, &h0](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto& xs_full = inputs[0];
            auto h = h0;
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                h = gru_step(h, x, Wg, Ug, Wc, Uc);
            }
            return {h};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&fn);
        auto compiled = mx::detail::compile(fn, id, false, {});
        mx::detail::compile_erase(id);

        auto run = [&]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = compiled({xs});
            auto t1 = std::chrono::high_resolution_clock::now();
            mx::eval(r);
            auto t2 = std::chrono::high_resolution_clock::now();
            static int c = 0;
            if (c == 10) printf("  [breakdown] replace: %.3f ms, eval: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count());
            c++;
        };
        auto result = bench(run);
        std::printf("Shared weights:     %8.2f ms\n", result.median_ms);
    }

    // MODE 2: Per-step weight COPIES (simulating jax-mps func.call)
    // Each iteration creates mx::array copies of the weights
    {
        auto fn = [TOTAL_STEPS, INPUT_DIM, &Wg, &Ug, &Wc, &Uc, &h0](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto& xs_full = inputs[0];
            auto h = h0;
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                // Copy weights like jax-mps does in ExecuteFunction's ValueMap
                mx::array w1(Wg), w2(Ug), w3(Wc), w4(Uc);
                h = gru_step(h, x, w1, w2, w3, w4);
            }
            return {h};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&fn);
        auto compiled = mx::detail::compile(fn, id, false, {});
        mx::detail::compile_erase(id);

        auto run = [&]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = compiled({xs});
            auto t1 = std::chrono::high_resolution_clock::now();
            mx::eval(r);
            auto t2 = std::chrono::high_resolution_clock::now();
            static int c = 0;
            if (c == 10) printf("  [breakdown] replace: %.3f ms, eval: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count());
            c++;
        };
        auto result = bench(run);
        std::printf("Per-step copies:    %8.2f ms\n", result.median_ms);
    }

    // MODE 3: Per-step weight IDENTITY OPS (forces new graph nodes)
    // Create genuinely different graph nodes by adding identity ops
    {
        auto fn = [TOTAL_STEPS, INPUT_DIM, &Wg, &Ug, &Wc, &Uc, &h0](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto& xs_full = inputs[0];
            auto h = h0;
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                // Add zero to force new graph nodes (like jax-mps block arg copies)
                auto w1 = mx::add(Wg, mx::array(0.0f));
                auto w2 = mx::add(Ug, mx::array(0.0f));
                auto w3 = mx::add(Wc, mx::array(0.0f));
                auto w4 = mx::add(Uc, mx::array(0.0f));
                h = gru_step(h, x, w1, w2, w3, w4);
            }
            return {h};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&fn);
        auto compiled = mx::detail::compile(fn, id, false, {});
        mx::detail::compile_erase(id);

        auto run = [&]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = compiled({xs});
            auto t1 = std::chrono::high_resolution_clock::now();
            mx::eval(r);
            auto t2 = std::chrono::high_resolution_clock::now();
            static int c = 0;
            if (c == 10) printf("  [breakdown] replace: %.3f ms, eval: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count());
            c++;
        };
        auto result = bench(run);
        std::printf("Per-step identity:  %8.2f ms\n", result.median_ms);
    }

    std::printf("\njax-mps: ~9.2ms. If 'Per-step copies' ~= jax-mps, root cause confirmed.\n");

    return 0;
}
