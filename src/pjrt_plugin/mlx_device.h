#pragma once

#include <cstdint>

namespace jax_mps {

class MlxDevice {
public:
    MlxDevice(int id);
    ~MlxDevice() = default;

    int id() const;
    int local_hardware_id() const;

private:
    int id_;
};

}  // namespace jax_mps
