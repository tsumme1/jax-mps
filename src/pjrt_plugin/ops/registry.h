// Stub op registry for Phase 0
// TODO: Phase 1 - implement MLX op registry

#pragma once

#include <string>
#include <unordered_set>

namespace jax_mps {

class OpRegistry {
public:
    // Returns empty set - no ops registered in Phase 0 stub
    static std::unordered_set<std::string> GetRegisteredOps() {
        return {};
    }
};

}  // namespace jax_mps
