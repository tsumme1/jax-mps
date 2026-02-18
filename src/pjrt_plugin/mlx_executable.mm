// Stub implementation for MLX executable
// TODO: Phase 1 - implement MLX backend

#include "pjrt_plugin/mlx_executable.h"

#include "pjrt_plugin/mlx_buffer.h"

namespace jax_mps {

MlxExecutable::MlxExecutable() {
    error_ = "MLX backend not implemented (Phase 0 stub)";
    valid_ = false;
    num_outputs_ = 1;
}

bool MlxExecutable::IsValid() const {
    return valid_;
}

std::string MlxExecutable::error() const {
    return error_;
}

size_t MlxExecutable::num_outputs() const {
    return num_outputs_;
}

MlxExecuteResult MlxExecutable::Execute(const std::vector<MlxBuffer*>& inputs, MlxDevice* device) {
    MlxExecuteResult result;
    // TODO: Phase 1 - implement execution
    return result;
}

}  // namespace jax_mps
