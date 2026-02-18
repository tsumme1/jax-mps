// Stub implementation for MLX client
// TODO: Phase 1 - implement MLX backend

#include "pjrt_plugin/mlx_client.h"

#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/mlx_device.h"
#include "pjrt_plugin/mlx_executable.h"
#include "pjrt_plugin/stablehlo_parser.h"

namespace jax_mps {

MlxClient::MlxClient() {
    // Create a single device for now
    devices_.push_back(std::make_unique<MlxDevice>(0));
}

MlxClient::~MlxClient() = default;

int MlxClient::device_count() const {
    return static_cast<int>(devices_.size());
}

MlxDevice* MlxClient::device(int index) {
    if (index >= 0 && index < static_cast<int>(devices_.size())) {
        return devices_[index].get();
    }
    return nullptr;
}

void* MlxClient::metal_device() const {
    // TODO: Phase 1 - return actual Metal device
    return nullptr;
}

std::unique_ptr<MlxExecutable> MlxClient::CompileStableHLO(mps::ParsedModule parsed_module,
                                                           void* options) {
    // TODO: Phase 1 - implement compilation
    return std::make_unique<MlxExecutable>();
}

std::unique_ptr<MlxBuffer> MlxClient::BufferFromHostBuffer(const void* data, int dtype,
                                                           const std::vector<int64_t>& dims,
                                                           const std::vector<int64_t>& byte_strides,
                                                           MlxDevice* device) {
    // TODO: Phase 1 - implement buffer creation
    return std::make_unique<MlxBuffer>();
}

}  // namespace jax_mps
