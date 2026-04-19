// Compare C++ compiled 200-step graph timing with different op counts
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <vector>
#include "mlx/mlx.h"
#include "mlx/compile_impl.h"

namespace mx = mlx::core;

mx::array gru_step(mx::array h, mx::array x,
                   mx::array Wg, mx::array Ug,
                   mx::array Wc, mx::array Uc) {
    auto gate = mx::sigmoid(mx::add(mx::matmul(x, Wg), mx::matmul(h, Ug)));
    auto cand = mx::tanh(mx::add(mx::matmul(x, Wc),
                                  mx::multiply(gate, mx::matmul(h, Uc))));
    return mx::add(mx::multiply(mx::subtract(mx::array(1.0f), gate), h),
                   mx::multiply(gate, cand));
}

struct BenchResult { double median_ms; };

BenchResult bench(std::function<void()> fn, int warmup = 10, int repeats = 100) {
    for (int i = 0; i < warmup; i++) fn();
    std::vector<double> times;
    for (int i = 0; i < repeats; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
    }
    std::sort(times.begin(), times.end());
    return {times[times.size()/2]};
}

int main() {
    constexpr int INPUT_DIM = 32, HIDDEN_DIM = 64, STEPS = 200;
    
    auto Wg = mx::random::normal({INPUT_DIM, HIDDEN_DIM}) * mx::array(0.01f);
    auto Ug = mx::random::normal({HIDDEN_DIM, HIDDEN_DIM}) * mx::array(0.01f);
    auto Wc = mx::random::normal({INPUT_DIM, HIDDEN_DIM}) * mx::array(0.01f);
    auto Uc = mx::random::normal({HIDDEN_DIM, HIDDEN_DIM}) * mx::array(0.01f);
    auto h0 = mx::zeros({HIDDEN_DIM});
    auto xs = mx::random::normal({STEPS, INPUT_DIM});
    mx::eval({Wg, Ug, Wc, Uc, h0, xs});

    // --- MODE 1: Simple pre-sliced (few ops per step) ---
    {
        auto body_fn = [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto h = inputs[0];
            auto& Wg_l = inputs[201]; auto& Ug_l = inputs[202];
            auto& Wc_l = inputs[203]; auto& Uc_l = inputs[204];
            for (int k = 0; k < 200; ++k) {
                h = gru_step(h, inputs[1 + k], Wg_l, Ug_l, Wc_l, Uc_l);
            }
            return {h};
        };
        static std::uintptr_t id1 = 201;
        auto compiled = mx::detail::compile(body_fn, id1, false, {});
        mx::detail::compile_erase(id1);

        auto run = [&]() {
            std::vector<mx::array> inputs;
            inputs.push_back(h0);
            for (int k = 0; k < STEPS; ++k) {
                inputs.push_back(mx::reshape(
                    mx::slice(xs, {k, 0}, {k + 1, INPUT_DIM}), {INPUT_DIM}));
            }
            inputs.push_back(Wg); inputs.push_back(Ug);
            inputs.push_back(Wc); inputs.push_back(Uc);
            auto result = compiled(inputs);
            mx::eval(result);
        };
        auto r = bench(run);
        std::printf("Mode A (simple, pre-sliced, 200 steps): %6.2f ms\n", r.median_ms);
    }

    // --- MODE 2: JAX-like with dynamic slice + counter (more ops per step) ---
    {
        auto body_fn = [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto counter = inputs[0];
            auto h = inputs[1];
            auto& xs_full = inputs[2];
            auto& Wg_l = inputs[3]; auto& Ug_l = inputs[4];
            auto& Wc_l = inputs[5]; auto& Uc_l = inputs[6];
            
            for (int k = 0; k < 200; ++k) {
                auto idx = mx::add(counter, mx::array(k, mx::int32));
                auto x = mx::reshape(mx::take(xs_full, idx, 0), {32});
                h = gru_step(h, x, Wg_l, Ug_l, Wc_l, Uc_l);
            }
            auto new_counter = mx::add(counter, mx::array(200, mx::int32));
            return {new_counter, h};
        };
        static std::uintptr_t id2 = 202;
        auto compiled = mx::detail::compile(body_fn, id2, false, {});
        mx::detail::compile_erase(id2);
        
        auto run = [&]() {
            auto counter = mx::array(0, mx::int32);
            std::vector<mx::array> inputs = {counter, h0, xs, Wg, Ug, Wc, Uc};
            auto result = compiled(inputs);
            mx::eval(result);
        };
        auto r = bench(run);
        std::printf("Mode B (JAX-like, dynamic slice, 200 steps): %6.2f ms\n", r.median_ms);
    }

    // --- MODE 3: Extra broadcasts/reshapes to mimic MLIR lowering bloat ---
    {
        auto body_fn = [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto counter = inputs[0];
            auto h = inputs[1];
            auto& xs_full = inputs[2];
            auto& Wg_l = inputs[3]; auto& Ug_l = inputs[4];
            auto& Wc_l = inputs[5]; auto& Uc_l = inputs[6];
            
            for (int k = 0; k < 200; ++k) {
                auto idx = mx::add(counter, mx::array(k, mx::int32));
                // Mimic MLIR: extra reshape + broadcast operations
                auto x_2d = mx::take(xs_full, idx, 0);
                auto x = mx::reshape(x_2d, {32});
                
                // Extra broadcast/reshape ops that MLIR might generate
                auto gate_pre = mx::add(mx::matmul(mx::reshape(x, {1, 32}), Wg_l),
                                         mx::matmul(mx::reshape(h, {1, 64}), Ug_l));
                auto gate = mx::sigmoid(mx::reshape(gate_pre, {64}));

                auto uc_out = mx::matmul(mx::reshape(h, {1, 64}), Uc_l);
                auto cand_pre = mx::add(mx::matmul(mx::reshape(x, {1, 32}), Wc_l),
                                         mx::multiply(mx::reshape(gate, {1, 64}), uc_out));
                auto cand = mx::tanh(mx::reshape(cand_pre, {64}));
                
                h = mx::add(mx::multiply(mx::subtract(mx::array(1.0f), gate), h),
                           mx::multiply(gate, cand));
            }
            auto new_counter = mx::add(counter, mx::array(200, mx::int32));
            return {new_counter, h};
        };
        static std::uintptr_t id3 = 203;
        auto compiled = mx::detail::compile(body_fn, id3, false, {});
        mx::detail::compile_erase(id3);
        
        auto run = [&]() {
            auto counter = mx::array(0, mx::int32);
            std::vector<mx::array> inputs = {counter, h0, xs, Wg, Ug, Wc, Uc};
            auto result = compiled(inputs);
            mx::eval(result);
        };
        auto r = bench(run);
        std::printf("Mode C (extra reshapes to mimic MLIR, 200 steps): %6.2f ms\n", r.median_ms);
    }

    return 0;
}
