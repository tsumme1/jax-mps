#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace jax_mps {

class MlxBuffer;
class MlxDevice;

struct MlxExecuteResult {
    std::vector<std::unique_ptr<MlxBuffer>> buffers;
};

class MlxExecutable {
public:
    MlxExecutable();
    ~MlxExecutable() = default;

    bool IsValid() const;
    std::string error() const;
    size_t num_outputs() const;

    MlxExecuteResult Execute(const std::vector<MlxBuffer*>& inputs, MlxDevice* device);

private:
    std::string error_;
    bool valid_ = false;
    size_t num_outputs_ = 0;
};

}  // namespace jax_mps
