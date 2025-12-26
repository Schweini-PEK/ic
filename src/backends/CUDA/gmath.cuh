// src/backends/CUDA/gmath.cuh

#pragma once

#include <cuda_fp16.h>

namespace ichol::cuda
{
    template <class G>
    struct GMath;

    template <>
    struct GMath<float>
    {
        __device__ __forceinline__ static float zero() { return 0.0f; }
        __device__ __forceinline__ static float add(float a, float b) { return a + b; }
        __device__ __forceinline__ static float sub(float a, float b) { return a - b; }
        __device__ __forceinline__ static float mul(float a, float b) { return a * b; }
        __device__ __forceinline__ static float div(float a, float b) { return a / b; }
        __device__ __forceinline__ static float fma(float a, float b, float c) { return fmaf(a, b, c); }
        __device__ __forceinline__ static float sqrt(float a) { return sqrtf(a); }
        __device__ __forceinline__ static float abs(float a) { return fabsf(a); }
        __device__ __forceinline__ static bool lt(float a, float b) { return a < b; }
        __device__ __forceinline__ static bool gt(float a, float b) { return a > b; }
        __device__ __forceinline__ static bool le(float a, float b) { return a <= b; }
        __device__ __forceinline__ static bool ge(float a, float b) { return a >= b; }
        __device__ __forceinline__ static bool eq0(float a) { return a == 0.0f; }
    };

    template <>
    struct GMath<double>
    {
        __device__ __forceinline__ static double zero() { return 0.0; }
        __device__ __forceinline__ static double add(double a, double b) { return a + b; }
        __device__ __forceinline__ static double sub(double a, double b) { return a - b; }
        __device__ __forceinline__ static double mul(double a, double b) { return a * b; }
        __device__ __forceinline__ static double div(double a, double b) { return a / b; }
        __device__ __forceinline__ static double fma(double a, double b, double c) { return ::fma(a, b, c); }
        __device__ __forceinline__ static double sqrt(double a) { return ::sqrt(a); }
        __device__ __forceinline__ static double abs(double a) { return ::fabs(a); }
        __device__ __forceinline__ static bool lt(double a, double b) { return a < b; }
        __device__ __forceinline__ static bool gt(double a, double b) { return a > b; }
        __device__ __forceinline__ static bool le(double a, double b) { return a <= b; }
        __device__ __forceinline__ static bool ge(double a, double b) { return a >= b; }
        __device__ __forceinline__ static bool eq0(double a) { return a == 0.0; }
    };

    template <>
    struct GMath<__half>
    {
        __device__ __forceinline__ static __half zero() { return __float2half_rn(0.0f); }
        __device__ __forceinline__ static __half add(__half a, __half b) { return __hadd(a, b); }
        __device__ __forceinline__ static __half sub(__half a, __half b) { return __hsub(a, b); }
        __device__ __forceinline__ static __half mul(__half a, __half b) { return __hmul(a, b); }
        __device__ __forceinline__ static __half div(__half a, __half b) { return __hdiv(a, b); }
        __device__ __forceinline__ static __half fma(__half a, __half b, __half c) { return __hadd(__hmul(a, b), c); }
        __device__ __forceinline__ static __half sqrt(__half a) { return hsqrt(a); }

        __device__ __forceinline__ static __half abs(__half a)
        {
            union
            {
                __half h;
                unsigned short u;
            } x;
            x.h = a;
            x.u &= 0x7FFFu;
            return x.h;
        }

        __device__ __forceinline__ static bool lt(__half a, __half b) { return __hlt(a, b); }
        __device__ __forceinline__ static bool gt(__half a, __half b) { return __hgt(a, b); }
        __device__ __forceinline__ static bool le(__half a, __half b) { return __hle(a, b); }
        __device__ __forceinline__ static bool ge(__half a, __half b) { return __hge(a, b); }

        __device__ __forceinline__ static bool eq0(__half a)
        {
            union
            {
                __half h;
                unsigned short u;
            } x;
            x.h = a;
            return (x.u & 0x7FFFu) == 0;
        }
    };
} // namespace ichol::cuda