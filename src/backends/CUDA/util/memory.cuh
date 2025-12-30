#pragma once

#include <cstddef>

namespace ichol::cuda
{
    __host__ __device__ __forceinline__ size_t align_up(size_t x, size_t a)
    {
        return (x + a - 1) & ~(a - 1);
    }
} // namespace ichol::cuda