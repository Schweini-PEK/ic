// backends/CUDA/host_cast.hpp

#pragma once

#include <cuda_fp16.h>

namespace ichol::cuda
{
    template <class G>
    static __host__ __forceinline__ G host_cast(double x) { return (G)x; }

    template <>
    __host__ __forceinline__ __half host_cast<__half>(double x) { return __float2half_rn((float)x); }

    template <class G>
    static __host__ __forceinline__ double host_to_double(G x) { return (double)x; }

    template <>
    __host__ __forceinline__ double host_to_double<__half>(__half x) { return (double)__half2float(x); }

} // namespace ichol::cuda