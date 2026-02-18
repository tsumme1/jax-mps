#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jax_mps {

class MlxBuffer {
public:
    MlxBuffer();
    ~MlxBuffer() = default;

    int dtype() const;
    const std::vector<int64_t>& dimensions() const;
    size_t byte_size() const;
    void Delete();
    bool IsDeleted() const;
    void ToHostBuffer(void* dst, void* event);

private:
    std::vector<int64_t> dimensions_;
    int dtype_ = 0;
    size_t byte_size_ = 0;
    bool deleted_ = false;
};

}  // namespace jax_mps
