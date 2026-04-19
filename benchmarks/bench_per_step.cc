// Minimal while-loop benchmark: compile body once, run with per-step eval
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/compile.h"

namespace mx = mlx::core;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

int main() {
    const int STEPS = 200;
    const int HIDDEN = 64;
    const int INPUT_DIM = 32;

    auto Wg = mx::multiply(mx::random::normal({INPUT_DIM, HIDDEN}), mx::array(0.01f));
    auto Ug = mx::multiply(mx::random::normal({HIDDEN, HIDDEN}), mx::array(0.01f));
    auto Wc = mx::multiply(mx::random::normal({INPUT_DIM, HIDDEN}), mx::array(0.01f));
    auto Uc = mx::multiply(mx::random::normal({HIDDEN, HIDDEN}), mx::array(0.01f));
    auto xs = mx::random::normal({STEPS, INPUT_DIM});
    auto h0 = mx::zeros({HIDDEN});
    mx::eval({Wg, Ug, Wc, Uc, xs, h0});

    // Body: [h, x, Wg, Ug, Wc, Uc] -> [h_new]
    auto body_fn = [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto& h = inputs[0];
        auto& x = inputs[1];
        auto& Wg = inputs[2];
        auto& Ug = inputs[3];
        auto& Wc = inputs[4];
        auto& Uc = inputs[5];
        auto gate = mx::sigmoid(mx::add(mx::matmul(x, Wg), mx::matmul(h, Ug)));
        auto cand = mx::tanh(mx::add(mx::matmul(x, Wc),
                                      mx::multiply(gate, mx::matmul(h, Uc))));
        auto h_new = mx::add(mx::multiply(mx::subtract(mx::array(1.0f), gate), h),
                             mx::multiply(gate, cand));
        return {h_new};
    };

    auto compiled_body = mx::compile(body_fn);

    // Per-step eval loop (matching jax-mps while loop behavior)
    auto run_loop = [&]() {
        auto h = h0;
        for (int i = 0; i < STEPS; ++i) {
            auto x = mx::reshape(mx::slice(xs, {i, 0}, {i+1, INPUT_DIM}), {INPUT_DIM});
            auto result = compiled_body({h, x, Wg, Ug, Wc, Uc});
            h = result[0];
            mx::eval(h);
        }
        return h;
    };

    // Warmup
    for (int i = 0; i < 10; ++i) run_loop();

    // Measure
    int N = 20;
    std::vector<double> times;
    for (int i = 0; i < N; ++i) {
        auto t0 = Clock::now();
        run_loop();
        auto t1 = Clock::now();
        times.push_back(Duration(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    printf("C++ per-step eval (h=64, 200 steps): %.2f ms\n", times[N/2]);

    // Also test h=2048
    const int H2 = 2048;
    const int I2 = 1024;
    auto Wg2 = mx::multiply(mx::random::normal({I2, H2}), mx::array(0.01f));
    auto Ug2 = mx::multiply(mx::random::normal({H2, H2}), mx::array(0.01f));
    auto Wc2 = mx::multiply(mx::random::normal({I2, H2}), mx::array(0.01f));
    auto Uc2 = mx::multiply(mx::random::normal({H2, H2}), mx::array(0.01f));
    auto xs2 = mx::random::normal({STEPS, I2});
    auto h02 = mx::zeros({H2});
    mx::eval({Wg2, Ug2, Wc2, Uc2, xs2, h02});

    auto compiled_body2 = mx::compile(body_fn);

    auto run_loop2 = [&]() {
        auto h = h02;
        for (int i = 0; i < STEPS; ++i) {
            auto x = mx::reshape(mx::slice(xs2, {i, 0}, {i+1, I2}), {I2});
            auto result = compiled_body2({h, x, Wg2, Ug2, Wc2, Uc2});
            h = result[0];
            mx::eval(h);
        }
        return h;
    };

    for (int i = 0; i < 5; ++i) run_loop2();
    times.clear();
    for (int i = 0; i < N; ++i) {
        auto t0 = Clock::now();
        run_loop2();
        auto t1 = Clock::now();
        times.push_back(Duration(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    printf("C++ per-step eval (h=2048, 200 steps): %.2f ms\n", times[N/2]);

    return 0;
}
