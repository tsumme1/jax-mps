// Stub implementation for MLX buffer
// TODO: Phase 1 - implement MLX backend

#include "pjrt_plugin/mlx_buffer.h"

namespace jax_mps {

MlxBuffer::MlxBuffer() = default;

int MlxBuffer::dtype() const {
    return dtype_;
}

const std::vector<int64_t>& MlxBuffer::dimensions() const {
    return dimensions_;
}

size_t MlxBuffer::byte_size() const {
    return byte_size_;
}

void MlxBuffer::Delete() {
    deleted_ = true;
}

bool MlxBuffer::IsDeleted() const {
    return deleted_;
}

void MlxBuffer::ToHostBuffer(void* dst, void* event) {
    // TODO: Phase 1 - implement host buffer copy
}

}  // namespace jax_mps
