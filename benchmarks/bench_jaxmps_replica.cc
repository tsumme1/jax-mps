// Replicate EXACTLY what jax-mps produces: per-step constant copies
// to verify whether the graph topology is the performance bottleneck.
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/compile.h"
#include "mlx/compile_impl.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

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

    // Mode A: Shared weights (C++ baseline - captures by reference)
    auto fn_shared = [TOTAL_STEPS, INPUT_DIM, &Wg, &Ug, &Wc, &Uc, &h0](
        const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto& xs_full = inputs[0];
        auto h = h0;
        for (int i = 0; i < TOTAL_STEPS; ++i) {
            auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
            auto v0 = mx::matmul(x, Wg);
            auto v1 = mx::matmul(h, Ug);
            auto gate = mx::sigmoid(mx::add(v0, v1));
            auto v3 = mx::matmul(x, Wc);
            auto v4 = mx::matmul(h, Uc);
            auto cand = mx::tanh(mx::add(v3, mx::multiply(gate, v4)));
            auto one = mx::broadcast_to(mx::array(1.0f), gate.shape());
            h = mx::add(mx::multiply(mx::subtract(one, gate), h),
                        mx::multiply(gate, cand));
        }
        return {h};
    };

    // Mode B: Per-step weight copies (simulating jax-mps func.call pattern)
    // Each step creates NEW array copies of Wg, Ug, Wc, Uc
    auto fn_copies = [TOTAL_STEPS, INPUT_DIM, &Wg, &Ug, &Wc, &Uc, &h0](
        const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto& xs_full = inputs[0];
        auto h = h0;
        for (int i = 0; i < TOTAL_STEPS; ++i) {
            // Simulate: each func.call creates fresh weight arrays from constant data
            auto wg_copy = mx::array(Wg.data<float>(), Wg.shape(), mx::float32);
            auto ug_copy = mx::array(Ug.data<float>(), Ug.shape(), mx::float32);
            auto wc_copy = mx::array(Wc.data<float>(), Wc.shape(), mx::float32);
            auto uc_copy = mx::array(Uc.data<float>(), Uc.shape(), mx::float32);

            auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
            auto v0 = mx::matmul(x, wg_copy);
            auto v1 = mx::matmul(h, ug_copy);
            auto gate = mx::sigmoid(mx::add(v0, v1));
            auto v3 = mx::matmul(x, wc_copy);
            auto v4 = mx::matmul(h, uc_copy);
            auto cand = mx::tanh(mx::add(v3, mx::multiply(gate, v4)));
            auto one = mx::broadcast_to(mx::array(1.0f), gate.shape());
            h = mx::add(mx::multiply(mx::subtract(one, gate), h),
                        mx::multiply(gate, cand));
        }
        return {h};
    };

    // Compile both
    auto compiled_shared = mx::compile(fn_shared);
    auto compiled_copies = mx::compile(fn_copies);

    // Warmup both
    for (int i = 0; i < 10; ++i) {
        auto r = compiled_shared({xs}); mx::eval(r);
    }
    for (int i = 0; i < 10; ++i) {
        auto r = compiled_copies({xs}); mx::eval(r);
    }

    // Time shared
    int N = 30;
    double shared_replace = 0, shared_eval = 0;
    for (int i = 0; i < N; ++i) {
        auto t0 = Clock::now();
        auto r = compiled_shared({xs});
        auto t1 = Clock::now();
        mx::eval(r);
        auto t2 = Clock::now();
        shared_replace += Duration(t1 - t0).count();
        shared_eval += Duration(t2 - t1).count();
    }

    // Time copies
    double copies_replace = 0, copies_eval = 0;
    for (int i = 0; i < N; ++i) {
        auto t0 = Clock::now();
        auto r = compiled_copies({xs});
        auto t1 = Clock::now();
        mx::eval(r);
        auto t2 = Clock::now();
        copies_replace += Duration(t1 - t0).count();
        copies_eval += Duration(t2 - t1).count();
    }

    // Dump tape info for both
    auto wrap_shared = [&fn_shared](const std::vector<mx::array>& ins)
        -> mx::detail::ArraysAndExtra { return {fn_shared(ins), nullptr}; };
    auto [ti_s, to_s, _es] = mx::detail::compile_trace(wrap_shared, {xs}, false);
    auto [tape_s, parents_s] = mx::detail::compile_dfs(ti_s, to_s, {xs});
    auto raw_s = tape_s.size();
    mx::detail::compile_simplify(tape_s, parents_s, to_s, 4);

    auto wrap_copies = [&fn_copies](const std::vector<mx::array>& ins)
        -> mx::detail::ArraysAndExtra { return {fn_copies(ins), nullptr}; };
    auto [ti_c, to_c, _ec] = mx::detail::compile_trace(wrap_copies, {xs}, false);
    auto [tape_c, parents_c] = mx::detail::compile_dfs(ti_c, to_c, {xs});
    auto raw_c = tape_c.size();
    mx::detail::compile_simplify(tape_c, parents_c, to_c, 4);

    printf("Mode A (shared weights): replace=%.3f ms, eval=%.3f ms, total=%.3f ms  tape=%zu→%zu\n",
           shared_replace/N, shared_eval/N, (shared_replace+shared_eval)/N, raw_s, tape_s.size());
    printf("Mode B (per-step copies): replace=%.3f ms, eval=%.3f ms, total=%.3f ms  tape=%zu→%zu\n",
           copies_replace/N, copies_eval/N, (copies_replace+copies_eval)/N, raw_c, tape_c.size());

    return 0;
}
