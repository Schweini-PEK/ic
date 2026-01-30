// src/backends/cpu/util/cast.hpp
#pragma once

#include <type_traits>
#include "ichol/half.hpp"

namespace ichol::util
{
    template <typename T>
    inline std::vector<double> to_double_vec(const std::vector<T> &input)
    {
        std::vector<double> output;
        output.reserve(input.size());
        for (const auto &v : input)
            output.push_back(static_cast<double>(v));
        return output;
    }

    template <typename To, typename From>
    inline To cast_fp_type(const From &value)
    {
        if constexpr (std::is_same<To, half_float::half>::value && std::is_floating_point<From>::value)
        {
            return half_float::half(static_cast<float>(value));
        }
        else if constexpr (std::is_floating_point<To>::value && std::is_same<From, half_float::half>::value)
        {
            return static_cast<To>(static_cast<float>(value));
        }
        else
        {
            return static_cast<To>(value);
        }
    }
} // namespace ichol::util