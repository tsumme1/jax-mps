// Quick test: verify detail::compile actually fuses and produces speedup
#include <chrono>
#include <cstdio>
#include "mlx/mlx.h"
#include "mlx/compile_impl.h"

namespace mx = mlx::core;

int main() {
    // Check if compile is available
    auto dev = mx::default_device();
    bool avail = mx::detail::compile_available_for_device(dev);
    std::printf("Device: %s, compile available: %s\n",
                dev == mx::Device::gpu ? "gpu" : "cpu",
                avail ? "YES" : "NO");
    
    // Simple test: verify fusion works
    auto x = mx::random::normal({1000, 1000});
    mx::eval(x);
    
    auto fn = [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto x = inputs[0];
        // Chain of element-wise ops that SHOULD be fused
        auto y = mx::exp(mx::negative(x));
        y = mx::add(y, mx::array(1.0f));
        y = mx::multiply(x, y);
        y = mx::tanh(y);
        return {y};
    };
    
    // Without compile  
    auto run_direct = [&]() {
        auto result = fn({x});
        mx::eval(result);
    };
    
    // With compile
    static std::uintptr_t id1 = 42;
    auto compiled = mx::detail::compile(fn, id1, false, {});
    mx::detail::compile_erase(id1);
    
    auto run_compiled = [&]() {
        auto result = compiled({x});
        mx::eval(result);
    };
    
    // Warmup
    for (int i = 0; i < 10; i++) { run_direct(); run_compiled(); }
    
    // Measure
    auto bench = [](auto f, int n) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++) f();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1-t0).count() / n;
    };
    
    double direct_ms = bench(run_direct, 100);
    double compiled_ms = bench(run_compiled, 100);
    
    std::printf("Direct (no compile): %.3f ms\n", direct_ms);
    std::printf("Compiled:            %.3f ms\n", compiled_ms);
    std::printf("Speedup:             %.2fx\n", direct_ms / compiled_ms);
    
    if (compiled_ms > direct_ms * 0.95) {
        std::printf("WARNING: Compile is NOT providing speedup! Fusion may be broken.\n");
    } else {
        std::printf("OK: Compile fusion is working.\n");
    }
    
    return 0;
}
