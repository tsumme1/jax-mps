// Dump the primitive types in the compiled graph tape to compare C++ vs jax-mps structure
#include <chrono>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/compile_impl.h"
#include "mlx/primitives.h"

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

    auto wrap_fn = [&fn](const std::vector<mx::array>& ins)
        -> mx::detail::ArraysAndExtra {
        return {fn(ins), nullptr};
    };
    auto [ti, to, ex] = mx::detail::compile_trace(wrap_fn, {xs}, false);
    auto [tape, parents] = mx::detail::compile_dfs(ti, to, {xs});

    printf("=== C++ Baseline Graph ===\n");
    printf("Total tape: %zu\n", tape.size());

    // Count primitives by type
    std::map<std::string, int> prim_counts;
    int no_prim = 0;
    for (auto& arr : tape) {
        if (arr.has_primitive()) {
            prim_counts[arr.primitive().name()]++;
        } else {
            no_prim++;
        }
    }
    printf("No primitive (sources): %d\n", no_prim);
    for (auto& [name, count] : prim_counts) {
        printf("  %-25s %d\n", name.c_str(), count);
    }

    // After simplification
    mx::detail::compile_simplify(tape, parents, to, 4);
    printf("\nAfter simplify: %zu\n", tape.size());
    prim_counts.clear();
    no_prim = 0;
    for (auto& arr : tape) {
        if (arr.has_primitive()) {
            prim_counts[arr.primitive().name()]++;
        } else {
            no_prim++;
        }
    }
    printf("No primitive (sources): %d\n", no_prim);
    for (auto& [name, count] : prim_counts) {
        printf("  %-25s %d\n", name.c_str(), count);
    }

    return 0;
}
