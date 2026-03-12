#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mps {
struct ParsedModule;
}

namespace jax_mps {

class MlxDevice;
class MlxBuffer;
class MlxExecutable;

class MlxClient {
public:
    MlxClient();
    ~MlxClient();

    int device_count() const;
    MlxDevice* device(int index);
    void* metal_device() const;

    std::unique_ptr<MlxExecutable> CompileStableHLO(mps::ParsedModule parsed_module, void* options);
    std::unique_ptr<MlxBuffer> BufferFromHostBuffer(const void* data, int dtype,
                                                    const std::vector<int64_t>& dims,
                                                    const std::vector<int64_t>& byte_strides,
                                                    MlxDevice* device);

private:
    std::vector<std::unique_ptr<MlxDevice>> devices_;
};

}  // namespace jax_mps
