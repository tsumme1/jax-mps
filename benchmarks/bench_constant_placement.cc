// Benchmark: Compare eval time when constants are created INSIDE vs OUTSIDE trace
// This tests whether constant placement affects Metal kernel scheduling

#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/compile_impl.h"

namespace mx = mlx::core;

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

    // MODE 1: Weights OUTSIDE trace (captured by reference) - mimics C++ baseline
    {
        auto fn = [TOTAL_STEPS, INPUT_DIM, &Wg, &Ug, &Wc, &Uc, &h0](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto& xs_full = inputs[0];
            auto h = h0;
            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                auto v0 = mx::matmul(x, Wg);
                auto v1 = mx::matmul(h, Ug);
                auto v2 = mx::add(v0, v1);
                auto gate = mx::sigmoid(v2);
                auto v3 = mx::matmul(x, Wc);
                auto v4 = mx::matmul(h, Uc);
                auto v5 = mx::multiply(gate, v4);
                auto v6 = mx::add(v3, v5);
                auto cand = mx::tanh(v6);
                auto one = mx::broadcast_to(mx::array(1.0f), gate.shape());
                auto v7 = mx::subtract(one, gate);
                h = mx::add(mx::multiply(v7, h), mx::multiply(gate, cand));
            }
            return {h};
        };
        auto id1 = reinterpret_cast<std::uintptr_t>(&fn);
        auto compiled = mx::detail::compile(fn, id1, false, {});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto r = compiled({xs});
            mx::eval(r);
        }

        // Measure
        double total_replace = 0, total_eval = 0;
        int N = 30;
        for (int i = 0; i < N; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = compiled({xs});
            auto t1 = std::chrono::high_resolution_clock::now();
            mx::eval(r);
            auto t2 = std::chrono::high_resolution_clock::now();
            total_replace += std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_eval += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        printf("Mode 1 (weights OUTSIDE): replace=%.3f ms, eval=%.3f ms, total=%.3f ms\n",
               total_replace/N, total_eval/N, (total_replace+total_eval)/N);
    }

    // MODE 2: Weights INSIDE trace (created from raw data) - mimics jax-mps HandleConstant
    {
        // Get raw data buffers from evaluated weights
        auto Wg_data = std::vector<float>(Wg.data<float>(), Wg.data<float>() + Wg.size());
        auto Ug_data = std::vector<float>(Ug.data<float>(), Ug.data<float>() + Ug.size());
        auto Wc_data = std::vector<float>(Wc.data<float>(), Wc.data<float>() + Wc.size());
        auto Uc_data = std::vector<float>(Uc.data<float>(), Uc.data<float>() + Uc.size());

        auto fn = [TOTAL_STEPS, INPUT_DIM, HIDDEN,
                   &Wg_data, &Ug_data, &Wc_data, &Uc_data](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto& xs_full = inputs[0];
            auto h = mx::zeros({HIDDEN});

            // Create weights from raw data INSIDE trace (like HandleConstant does)
            auto Wg_local = mx::array(Wg_data.data(), {INPUT_DIM, HIDDEN}, mx::float32);
            auto Ug_local = mx::array(Ug_data.data(), {HIDDEN, HIDDEN}, mx::float32);
            auto Wc_local = mx::array(Wc_data.data(), {INPUT_DIM, HIDDEN}, mx::float32);
            auto Uc_local = mx::array(Uc_data.data(), {HIDDEN, HIDDEN}, mx::float32);

            for (int i = 0; i < TOTAL_STEPS; ++i) {
                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                auto v0 = mx::matmul(x, Wg_local);
                auto v1 = mx::matmul(h, Ug_local);
                auto v2 = mx::add(v0, v1);
                auto gate = mx::sigmoid(v2);
                auto v3 = mx::matmul(x, Wc_local);
                auto v4 = mx::matmul(h, Uc_local);
                auto v5 = mx::multiply(gate, v4);
                auto v6 = mx::add(v3, v5);
                auto cand = mx::tanh(v6);
                auto one = mx::broadcast_to(mx::array(1.0f), gate.shape());
                auto v7 = mx::subtract(one, gate);
                h = mx::add(mx::multiply(v7, h), mx::multiply(gate, cand));
            }
            return {h};
        };
        auto id2 = reinterpret_cast<std::uintptr_t>(&fn);
        auto compiled = mx::detail::compile(fn, id2, false, {});

        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto r = compiled({xs});
            mx::eval(r);
        }

        // Measure
        double total_replace = 0, total_eval = 0;
        int N = 30;
        for (int i = 0; i < N; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = compiled({xs});
            auto t1 = std::chrono::high_resolution_clock::now();
            mx::eval(r);
            auto t2 = std::chrono::high_resolution_clock::now();
            total_replace += std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_eval += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        printf("Mode 2 (weights INSIDE):  replace=%.3f ms, eval=%.3f ms, total=%.3f ms\n",
               total_replace/N, total_eval/N, (total_replace+total_eval)/N);
    }

    // MODE 3: Constants created per-step INSIDE trace - mimics jax-mps per-call constants
    {
        auto Wg_data = std::vector<float>(Wg.data<float>(), Wg.data<float>() + Wg.size());
        auto Ug_data = std::vector<float>(Ug.data<float>(), Ug.data<float>() + Ug.size());
        auto Wc_data = std::vector<float>(Wc.data<float>(), Wc.data<float>() + Wc.size());
        auto Uc_data = std::vector<float>(Uc.data<float>(), Uc.data<float>() + Uc.size());

        auto fn = [TOTAL_STEPS, INPUT_DIM, HIDDEN,
                   &Wg_data, &Ug_data, &Wc_data, &Uc_data](
            const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto& xs_full = inputs[0];
            auto h = mx::zeros({HIDDEN});

            for (int i = 0; i < TOTAL_STEPS; ++i) {
                // Per-step constant creation (like jax-mps HandleConstant inside @closed_call)
                auto Wg_local = mx::array(Wg_data.data(), {INPUT_DIM, HIDDEN}, mx::float32);
                auto Ug_local = mx::array(Ug_data.data(), {HIDDEN, HIDDEN}, mx::float32);
                auto Wc_local = mx::array(Wc_data.data(), {INPUT_DIM, HIDDEN}, mx::float32);
                auto Uc_local = mx::array(Uc_data.data(), {HIDDEN, HIDDEN}, mx::float32);

                auto x = mx::reshape(mx::slice(xs_full, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
                auto v0 = mx::matmul(x, Wg_local);
                auto v1 = mx::matmul(h, Ug_local);
                auto v2 = mx::add(v0, v1);
                auto gate = mx::sigmoid(v2);
                auto v3 = mx::matmul(x, Wc_local);
                auto v4 = mx::matmul(h, Uc_local);
                auto v5 = mx::multiply(gate, v4);
                auto v6 = mx::add(v3, v5);
                auto cand = mx::tanh(v6);
                auto one = mx::broadcast_to(mx::array(1.0f), gate.shape());
                auto v7 = mx::subtract(one, gate);
                h = mx::add(mx::multiply(v7, h), mx::multiply(gate, cand));
            }
            return {h};
        };
        auto id3 = reinterpret_cast<std::uintptr_t>(&fn);
        auto compiled = mx::detail::compile(fn, id3, false, {});

        // Measure tape
        auto wrap_fn = [&fn](const std::vector<mx::array>& ins)
            -> mx::detail::ArraysAndExtra {
            return {fn(ins), nullptr};
        };
        auto [ti, to, ex] = mx::detail::compile_trace(wrap_fn, {xs}, false);
        auto [tape, parents] = mx::detail::compile_dfs(ti, to, {xs});
        auto tape_before = tape.size();
        mx::detail::compile_simplify(tape, parents, to, 4);
        printf("Mode 3 tape: %zu → %zu\n", tape_before, tape.size());

        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto r = compiled({xs});
            mx::eval(r);
        }

        // Measure
        double total_replace = 0, total_eval = 0;
        int N = 30;
        for (int i = 0; i < N; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = compiled({xs});
            auto t1 = std::chrono::high_resolution_clock::now();
            mx::eval(r);
            auto t2 = std::chrono::high_resolution_clock::now();
            total_replace += std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_eval += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        printf("Mode 3 (weights PER-STEP): replace=%.3f ms, eval=%.3f ms, total=%.3f ms\n",
               total_replace/N, total_eval/N, (total_replace+total_eval)/N);
    }

    return 0;
}
