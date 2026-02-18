// Stub implementation for MLX device
// TODO: Phase 1 - implement MLX backend

#include "pjrt_plugin/mlx_device.h"

namespace jax_mps {

MlxDevice::MlxDevice(int id) : id_(id) {}

int MlxDevice::id() const {
    return id_;
}

int MlxDevice::local_hardware_id() const {
    return id_;
}

}  // namespace jax_mps
