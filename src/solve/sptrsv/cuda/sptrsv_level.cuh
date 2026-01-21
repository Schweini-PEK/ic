// sptrsv_levelsets.cuh
//
// CUDA 12 level-scheduled sparse triangular solve (SpTRSV) using precomputed level sets.
// Persistent-plan version: level rows live on device; status buffer is allocated once and reused.
//
// Return codes:
//   0  success
//  -1 invalid argument / unsupported type
//  -2 numerical issue detected (NaN/Inf, or missing/zero diagonal when unit_diag=false)
//  -3 CUDA runtime error
//  -4 cuSPARSE error

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <vector>

#include "factor/symbolic/symbolic.hpp"
#include "backends/CUDA/util/gmath.cuh"

struct DeviceLevelSets
{
    // Host copies (owned) to keep async H2D copies safe without requiring sync.
    std::vector<int> level_ptr;
    std::vector<int> levels;

    int *d_level_ptr = nullptr;
    int *d_levels = nullptr;

    int n = 0;
    int num_levels = 0;

    DeviceLevelSets() = default;
    DeviceLevelSets(const DeviceLevelSets &) = delete;
    DeviceLevelSets &operator=(const DeviceLevelSets &) = delete;

    ~DeviceLevelSets() { reset(); }

    void reset()
    {
        if (d_level_ptr)
            cudaFree(d_level_ptr);
        if (d_levels)
            cudaFree(d_levels);
        d_level_ptr = nullptr;
        d_levels = nullptr;

        level_ptr.clear();
        levels.clear();
        n = 0;
        num_levels = 0;
    }

    int init(const ichol::symbolic::LevelSets &host, cudaStream_t stream)
    {
        reset();

        level_ptr = host.level_ptr;
        levels = host.levels;

        n = static_cast<int>(levels.size());
        num_levels = static_cast<int>(level_ptr.size()) - 1;

        if (n == 0)
            return 0;

        auto e = cudaMalloc((void **)&d_level_ptr, sizeof(int) * level_ptr.size());
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        e = cudaMalloc((void **)&d_levels, sizeof(int) * n);
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        e = cudaMemcpyAsync(d_level_ptr, level_ptr.data(),
                            sizeof(int) * level_ptr.size(),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        e = cudaMemcpyAsync(d_levels, levels.data(),
                            sizeof(int) * n,
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        // No cudaStreamSynchronize: stream ordering guarantees correctness for later work
        // enqueued on the same stream. Host copies are owned by this object.
        return 0;
    }
};

/* Device finite check for float/double scalars. */
template <typename ValueT>
__device__ __forceinline__ bool sptrsv_device_isfinite(ValueT v)
{
    if constexpr (std::is_same_v<ValueT, float>)
        return isfinite(v);
    if constexpr (std::is_same_v<ValueT, double>)
        return isfinite(v);
    if constexpr (std::is_same_v<ValueT, __half>)
        return isfinite(__half2float(v));
    return isfinite(static_cast<double>(v));
}

/* Device exact zero check (simple safeguard). */
template <typename ValueT>
__device__ __forceinline__ bool sptrsv_device_iszero(ValueT v)
{
    return ichol::cuda::GMath<ValueT>::eq0(v);
}

/*
Kernel: solve all rows in one level in parallel (thread-per-row), assuming diagonal is the
LAST entry in each CSR row.

Required invariant for the CSR passed to this kernel:
  - For every row i, the diagonal entry A(i,i) is stored at position rowPtr[i+1]-1.
*/
template <typename IndexT, typename ValueT>
__global__ void sptrsv_trsv_level_kernel(
    int level_size,
    const int *__restrict__ d_level_rows,
    const IndexT *__restrict__ d_rowPtr,
    const IndexT *__restrict__ d_colInd,
    const ValueT *__restrict__ d_val,
    const ValueT *__restrict__ d_b,
    ValueT *__restrict__ d_x,
    bool unit_diag,
    int *__restrict__ d_status)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size)
        return;

    // Optional cheap early abort after an error was flagged.
    if (__ldg(d_status) != 0)
        return;

    const int i = d_level_rows[tid];
    const IndexT p = d_rowPtr[i];
    const IndexT q = d_rowPtr[i + 1];
    const IndexT end = q - 1; // last entry is diagonal by invariant

    ValueT s = d_b[i];

    for (IndexT k = p; k < end; ++k)
    {
        const IndexT j = d_colInd[k];
        const ValueT a = d_val[k];
        s = ichol::cuda::GMath<ValueT>::sub(s, ichol::cuda::GMath<ValueT>::mul(a, d_x[j]));
    }

    if (!unit_diag)
    {
        const ValueT diag = d_val[end];
        if (sptrsv_device_iszero(diag))
        {
            atomicExch(d_status, 1);
            d_x[i] = ichol::cuda::GMath<ValueT>::zero();
            return;
        }
        s = ichol::cuda::GMath<ValueT>::div(s, diag);
    }

    d_x[i] = s;

    if (!sptrsv_device_isfinite(s))
        atomicExch(d_status, 1);
}

/*
Persistent plan: owns DeviceLevelSets + reusable device status flag + pinned host status.
Call init() once, then solve() many times.
*/
struct SpTRSVLevelsetsPlan
{
    DeviceLevelSets ls;

    int *d_status = nullptr; // device flag (reused each solve)
    int *h_status = nullptr; // pinned host flag (reused each solve)

    int n = 0;

    SpTRSVLevelsetsPlan() = default;
    SpTRSVLevelsetsPlan(const SpTRSVLevelsetsPlan &) = delete;
    SpTRSVLevelsetsPlan &operator=(const SpTRSVLevelsetsPlan &) = delete;

    ~SpTRSVLevelsetsPlan() { reset(); }

    void reset()
    {
        if (d_status)
            cudaFree(d_status);
        if (h_status)
            cudaFreeHost(h_status);
        d_status = nullptr;
        h_status = nullptr;

        ls.reset();
        n = 0;
    }

    int init(const ichol::symbolic::LevelSets &host, cudaStream_t stream)
    {
        reset();

        int rc = ls.init(host, stream);
        if (rc != 0)
        {
            reset();
            return rc;
        }

        n = ls.n;
        if (n == 0)
            return 0;

        auto e = cudaMalloc((void **)&d_status, sizeof(int));
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        e = cudaHostAlloc((void **)&h_status, sizeof(int), cudaHostAllocDefault);
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        e = cudaMemsetAsync(d_status, 0, sizeof(int), stream);
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        // no sync required here
        return 0;
    }

    template <typename IndexT, typename ValueT>
    int solve(
        int n_,
        const IndexT *d_rowPtr,
        const IndexT *d_colInd,
        const ValueT *d_val,
        const ValueT *d_b,
        ValueT *d_x,
        bool unit_diag,
        cudaStream_t stream)
    {
        if (n_ < 0)
            return -1;
        if (n_ == 0)
            return 0;
        if (n_ != n)
            return -1;

        if (!d_rowPtr || !d_colInd || !d_val || !d_b || !d_x)
            return -1;
        if (!ls.d_levels)
            return -1;
        if (!d_status || !h_status)
            return -1;
        if (ls.level_ptr.size() < 2)
            return -1;
        if (ls.level_ptr.front() != 0)
            return -1;
        if (ls.level_ptr.back() != n)
            return -1;

        auto e = cudaMemsetAsync(d_status, 0, sizeof(int), stream);
        if (e != cudaSuccess)
            return -3;

        constexpr int THREADS = 128;

        for (int lvl = 0; lvl < ls.num_levels; ++lvl)
        {
            const int start = ls.level_ptr[lvl];
            const int end = ls.level_ptr[lvl + 1];
            const int level_size = end - start;
            if (level_size <= 0)
                continue;

            const int blocks = (level_size + THREADS - 1) / THREADS;

            sptrsv_trsv_level_kernel<IndexT, ValueT><<<blocks, THREADS, 0, stream>>>(
                level_size,
                ls.d_levels + start,
                d_rowPtr, d_colInd, d_val,
                d_b, d_x,
                unit_diag,
                d_status);
        }

        e = cudaPeekAtLastError();
        if (e != cudaSuccess)
            return -3;

        e = cudaMemcpyAsync(h_status, d_status, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess)
            return -3;

        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess)
            return -3;

        return (*h_status != 0) ? -2 : 0;
    }
};
