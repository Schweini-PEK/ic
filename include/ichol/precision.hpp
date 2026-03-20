#ifndef ICHOL_PRECISION_HPP
#define ICHOL_PRECISION_HPP

namespace ichol::solver
{
    enum class ComputePrecision
    {
        FP64,     // Standard Double
        FP32,     // Standard Float
        TF32,     // Tensor Float 32 request; normalized to FP32 where unsupported
        FP16,     // Half Precision
        BF16,     // Brain Float 16
        FP8_E4M3, // Hopper FP8 (Max precision)
        FP8_E5M2  // Hopper FP8 (Max dynamic range)
    };
} // namespace ichol::solver

#endif // ICHOL_PRECISION_HPP
