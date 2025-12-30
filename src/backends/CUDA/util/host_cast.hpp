// backends/CUDA/util/host_cast.hpp
#pragma once

#include <cuda_fp16.h>
#include <type_traits>
#include <ichol/half.hpp> // half_float::half

namespace ichol::cuda::util
{

    /**
     * host_cast:
     *   double              -> double
     *   float               -> float
     *   half_float::half   -> __half
     */
    template <class G, class S>
    __host__ __forceinline__ G host_cast(S x)
    {
        using GG = std::remove_cv_t<std::remove_reference_t<G>>;
        using SS = std::remove_cv_t<std::remove_reference_t<S>>;

        if constexpr (std::is_same_v<GG, double> && std::is_same_v<SS, double>)
        {
            return x;
        }
        else if constexpr (std::is_same_v<GG, float> && std::is_same_v<SS, float>)
        {
            return x;
        }
        else if constexpr (std::is_same_v<GG, __half> && std::is_same_v<SS, half_float::half>)
        {
            // explicit via half_float API, then explicit CUDA half conversion
            return __float2half_rn(static_cast<float>(x));
        }
        else if constexpr (std::is_same_v<GG, float> && std::is_same_v<SS, double>)
        {
            return static_cast<float>(x);
        }
        else if constexpr (std::is_same_v<GG, __half> && std::is_same_v<SS, double>)
        {
            return __float2half_rn(static_cast<float>(x));
        }
        else
        {
            static_assert(sizeof(GG) == 0, "host_cast: only supports double->double, float->float, half_float::half->__half");
        }
    }

    /**
     * host_to_double:
     *   double              -> double
     *   float               -> double
     *   __half              -> double
     */
    template <class G>
    __host__ __forceinline__ double host_to_double(G x)
    {
        using GG = std::remove_cv_t<std::remove_reference_t<G>>;

        if constexpr (std::is_same_v<GG, double>)
        {
            return x;
        }
        else if constexpr (std::is_same_v<GG, float>)
        {
            return static_cast<double>(x);
        }
        else if constexpr (std::is_same_v<GG, __half>)
        {
            return static_cast<double>(__half2float(x));
        }
        else
        {
            static_assert(sizeof(GG) == 0, "host_to_double: only supports double, float, __half");
        }
    }

} // namespace ichol::cuda::util
