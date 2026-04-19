// Benchmark: Test how tape size affects eval time
// Creates graphs of varying sizes to establish the curve

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
    auto v8 = mx::divide(v7, v6);
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

// Add extra no-op nodes to bloat the graph (identity ops)
mx::array add_bloat(const mx::array& h, int extra_ops) {
    auto result = h;
    for (int i = 0; i < extra_ops; ++i) {
        result = mx::add(result, mx::array(0.0f));
    }
    return result;
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

    // Test with different amounts of graph bloat
    for (int extra_per_step : {0, 5, 13, 20}) {
        auto body_fn = [TOTAL_STEPS, INPUT_DIM, extra_per_step, &Wg, &Ug, &Wc, &Uc, &h0](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto& xs_full = inputs[0];
            auto h = h0;
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                h = gru_step(h, x, Wg, Ug, Wc, Uc);
                h = add_bloat(h, extra_per_step);
            }
            return {h};
        };
        auto id = reinterpret_cast<std::uintptr_t>(&body_fn);
        auto compiled = mx::detail::compile(body_fn, id, false, {});
        mx::detail::compile_erase(id);

        // Measure tape
        auto wrap_fn = [&body_fn](const std::vector<mx::array>& ins)
            -> mx::detail::ArraysAndExtra {
            return {body_fn(ins), nullptr};
        };
        auto [ti, to, ex] = mx::detail::compile_trace(wrap_fn, {xs}, false);
        auto [tape, parents] = mx::detail::compile_dfs(ti, to, {xs});
        auto tape_before = tape.size();
        mx::detail::compile_simplify(tape, parents, to, 4);
        auto tape_after = tape.size();

        auto run = [&]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = compiled({xs});
            auto t1 = std::chrono::high_resolution_clock::now();
            mx::eval(r);
            auto t2 = std::chrono::high_resolution_clock::now();
            static int c = 0;
            if (c == 10) printf("  replace: %.3f ms, eval: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count());
            c++;
        };
        auto result = bench(run);
        std::printf("extra=%2d: tape=%5zu→%5zu, total=%6.2f ms\n", extra_per_step, tape_before, tape_after, result.median_ms);
    }

    std::printf("\njax-mps: tape=9008, total=~9.2ms\n");
    std::printf("Find which tape size matches ~9ms to understand the scaling.\n");

    return 0;
}
