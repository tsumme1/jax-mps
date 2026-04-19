// Benchmark a single GRU step (no loop) in C++ MLX for comparison
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <vector>
#include "mlx/mlx.h"
#include "mlx/compile_impl.h"

namespace mx = mlx::core;

int main() {
    auto Wg = mx::random::normal({32, 64}) * mx::array(0.01f);
    auto Ug = mx::random::normal({64, 64}) * mx::array(0.01f);
    auto Wc = mx::random::normal({32, 64}) * mx::array(0.01f);
    auto Uc = mx::random::normal({64, 64}) * mx::array(0.01f);
    auto h  = mx::random::normal({64});
    auto x  = mx::random::normal({32});
    mx::eval({Wg, Ug, Wc, Uc, h, x});

    // Direct (lazy eval) body
    auto gru_step = [](mx::array h, mx::array x,
                       mx::array Wg, mx::array Ug,
                       mx::array Wc, mx::array Uc) -> mx::array {
        auto gate = mx::sigmoid(mx::add(mx::matmul(x, Wg), mx::matmul(h, Ug)));
        auto cand = mx::tanh(mx::add(mx::matmul(x, Wc),
                                      mx::multiply(gate, mx::matmul(h, Uc))));
        return mx::add(mx::multiply(mx::subtract(mx::array(1.0f), gate), h),
                       mx::multiply(gate, cand));
    };

    // --- Mode 1: Direct (no compile) ---
    for (int i = 0; i < 20; i++) {
        auto r = gru_step(h, x, Wg, Ug, Wc, Uc);
        mx::eval(r);
    }
    std::vector<double> times;
    for (int i = 0; i < 200; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = gru_step(h, x, Wg, Ug, Wc, Uc);
        mx::eval(r);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
    }
    std::sort(times.begin(), times.end());
    std::printf("C++ direct single GRU step:   %.3f ms (p10=%.3f, p90=%.3f)\n",
                times[times.size()/2], times[times.size()/10], times[9*times.size()/10]);

    // --- Mode 2: Compiled ---
    static std::uintptr_t body_id = 99;
    auto compiled_step = mx::detail::compile(
        [&](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            return {gru_step(inputs[0], inputs[1], inputs[2], inputs[3],
                            inputs[4], inputs[5])};
        }, body_id, false, {});

    for (int i = 0; i < 20; i++) {
        auto r = compiled_step({h, x, Wg, Ug, Wc, Uc});
        mx::eval(r);
    }
    times.clear();
    for (int i = 0; i < 200; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = compiled_step({h, x, Wg, Ug, Wc, Uc});
        mx::eval(r);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
    }
    std::sort(times.begin(), times.end());
    std::printf("C++ compiled single GRU step: %.3f ms (p10=%.3f, p90=%.3f)\n",
                times[times.size()/2], times[times.size()/10], times[9*times.size()/10]);

    return 0;
}
