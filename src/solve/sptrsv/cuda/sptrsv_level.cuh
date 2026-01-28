// sptrsv_levelsets.cuh
//
// CUDA 12 level-scheduled sparse triangular solve (SpTRSV) using precomputed level sets.
// Persistent-plan version: level rows live on device; status buffer is allocated once and reused.
//
// Variant: Option A1 (single cooperative persistent kernel per solve)
// - Replaces "one kernel launch per level" with "one cooperative kernel launch per solve".
// - Uses exactly one grid.sync() per level.
// - Critically: launches a LIMITED number of blocks to reduce grid.sync() overhead.
//
// Notes:
// - Requires device support for cooperative launch (cudaDevAttrCooperativeLaunch == 1).
// - If cooperative launch is not supported, this code falls back to the original per-level launches.
//
// Return codes:
//   0  success
//  -1 invalid argument / unsupported type
//  -2 numerical issue detected (missing/zero diagonal when unit_diag=false)
//  -3 CUDA runtime error
//  -4 cuSPARSE error

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

#include <type_traits>
#include <vector>
#include <algorithm>

#include "factor/symbolic/symbolic.hpp"
#include "backends/CUDA/util/gmath.cuh"

namespace cg = cooperative_groups;

struct DeviceLevelSets
{
    std::vector<int> level_ptr;
    std::vector<int> levels;

    int *d_level_ptr = nullptr;
    int *d_levels = nullptr;

    int n = 0;
    int num_levels = 0;

    // max number of rows in any level; used to size cooperative grid
    int max_level_size = 0;

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
        max_level_size = 0;
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
        if (num_levels < 0)
            return -1;
        if ((int)level_ptr.size() < 2)
            return -1;
        if (level_ptr.front() != 0)
            return -1;
        if (level_ptr.back() != n)
            return -1;

        max_level_size = 0;
        for (int l = 0; l < num_levels; ++l)
        {
            const int sz = level_ptr[l + 1] - level_ptr[l];
            max_level_size = std::max(max_level_size, sz);
        }

        cudaError_t e = cudaMalloc((void **)&d_level_ptr, sizeof(int) * level_ptr.size());
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

        return 0;
    }
};

/* Device exact zero check (simple safeguard). */
template <typename ValueT>
__device__ __forceinline__ bool sptrsv_device_iszero(ValueT v)
{
    return ichol::cuda::GMath<ValueT>::eq0(v);
}

/*
Original per-level kernel (kept for fallback / non-cooperative launch).
Diagonal is the LAST entry of each CSR row by invariant.
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

    if (__ldg(d_status) != 0)
        return;

    const int i = d_level_rows[tid];
    const IndexT p = d_rowPtr[i];
    const IndexT q = d_rowPtr[i + 1];
    const IndexT end = q - 1;

    ValueT s = d_b[i];

    for (IndexT k = p; k < end; ++k)
    {
        const IndexT j = d_colInd[k];
        const ValueT a = d_val[k];
        s = ichol::cuda::GMath<ValueT>::sub(
            s,
            ichol::cuda::GMath<ValueT>::mul(a, __ldg(&d_x[j])));
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
}

/*
Original __half per-level kernel with half2 inner loop (kept for fallback).
Diagonal is last.
*/
template <typename IndexT>
__global__ void sptrsv_trsv_level_kernel_half2(
    int level_size,
    const int *__restrict__ d_level_rows,
    const IndexT *__restrict__ d_rowPtr,
    const IndexT *__restrict__ d_colInd,
    const __half *__restrict__ d_val,
    const __half *__restrict__ d_b,
    __half *__restrict__ d_x,
    bool unit_diag,
    int *__restrict__ d_status)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size)
        return;

    if (__ldg(d_status) != 0)
        return;

    const int i = d_level_rows[tid];
    IndexT p = d_rowPtr[i];
    IndexT q = d_rowPtr[i + 1];
    IndexT end = q - 1;

    __half2 acc2 = __float2half2_rn(0.0f);
    __half acc_tail = __float2half(0.0f);

    IndexT k = p;

    if ((k & 1) && k < end)
    {
        IndexT j = d_colInd[k];
        __half a = d_val[k];
        __half xj = __ldg(&d_x[j]);
        acc_tail = __hfma(a, xj, acc_tail);
        ++k;
    }

    for (; k + 1 < end; k += 2)
    {
        IndexT j0 = d_colInd[k];
        IndexT j1 = d_colInd[k + 1];

        __half2 a2 = __halves2half2(d_val[k], d_val[k + 1]);
        __half2 x2 = __halves2half2(__ldg(&d_x[j0]), __ldg(&d_x[j1]));

        acc2 = __hfma2(a2, x2, acc2);
    }

    for (; k < end; ++k)
    {
        IndexT j = d_colInd[k];
        acc_tail = __hfma(d_val[k], __ldg(&d_x[j]), acc_tail);
    }

    __half sum2 = __hadd(__low2half(acc2), __high2half(acc2));
    __half ax = __hadd(sum2, acc_tail);

    __half s = __hsub(d_b[i], ax);

    if (!unit_diag)
    {
        __half diag = d_val[end];
        if (__heq(diag, __float2half(0.0f)))
        {
            atomicExch(d_status, 1);
            d_x[i] = __float2half(0.0f);
            return;
        }
        s = __hdiv(s, diag);
    }

    d_x[i] = s;
}

/* Row solve helpers used by the persistent kernel */
template <typename IndexT, typename ValueT>
__device__ __forceinline__ void sptrsv_solve_row_diag_last(
    int i,
    const IndexT *__restrict__ d_rowPtr,
    const IndexT *__restrict__ d_colInd,
    const ValueT *__restrict__ d_val,
    const ValueT *__restrict__ d_b,
    ValueT *__restrict__ d_x,
    bool unit_diag,
    int *__restrict__ d_status)
{
    const IndexT p = d_rowPtr[i];
    const IndexT q = d_rowPtr[i + 1];
    const IndexT end = q - 1;

    ValueT s = d_b[i];

    for (IndexT k = p; k < end; ++k)
    {
        const IndexT j = d_colInd[k];
        const ValueT a = d_val[k];
        s = ichol::cuda::GMath<ValueT>::sub(
            s,
            ichol::cuda::GMath<ValueT>::mul(a, __ldg(&d_x[j])));
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
}

template <typename IndexT>
__device__ __forceinline__ void sptrsv_solve_row_diag_last_half2(
    int i,
    const IndexT *__restrict__ d_rowPtr,
    const IndexT *__restrict__ d_colInd,
    const __half *__restrict__ d_val,
    const __half *__restrict__ d_b,
    __half *__restrict__ d_x,
    bool unit_diag,
    int *__restrict__ d_status)
{
    IndexT p = d_rowPtr[i];
    IndexT q = d_rowPtr[i + 1];
    IndexT end = q - 1;

    __half2 acc2 = __float2half2_rn(0.0f);
    __half acc1 = __float2half(0.0f);

    IndexT k = p;

    if ((k & 1) && k < end)
    {
        IndexT j = d_colInd[k];
        acc1 = __hfma(d_val[k], __ldg(&d_x[j]), acc1);
        ++k;
    }

    for (; k + 1 < end; k += 2)
    {
        IndexT j0 = d_colInd[k];
        IndexT j1 = d_colInd[k + 1];

        __half2 a2 = __halves2half2(d_val[k], d_val[k + 1]);
        __half2 x2 = __halves2half2(__ldg(&d_x[j0]), __ldg(&d_x[j1]));

        acc2 = __hfma2(a2, x2, acc2);
    }

    for (; k < end; ++k)
    {
        IndexT j = d_colInd[k];
        acc1 = __hfma(d_val[k], __ldg(&d_x[j]), acc1);
    }

    __half ax = __hadd(__hadd(__low2half(acc2), __high2half(acc2)), acc1);
    __half s = __hsub(d_b[i], ax);

    if (!unit_diag)
    {
        __half diag = d_val[end];
        if (__heq(diag, __float2half(0.0f)))
        {
            atomicExch(d_status, 1);
            d_x[i] = __float2half(0.0f);
            return;
        }
        s = __hdiv(s, diag);
    }

    d_x[i] = s;
}

/*
Persistent cooperative kernel (generic)
- exactly one grid.sync per level
- status checked at top of each level (after previous sync) for global visibility
*/
template <typename IndexT, typename ValueT>
__global__ void sptrsv_trsv_level_persistent_kernel(
    int num_levels,
    const int *__restrict__ d_level_ptr,
    const int *__restrict__ d_levels,
    const IndexT *__restrict__ d_rowPtr,
    const IndexT *__restrict__ d_colInd,
    const ValueT *__restrict__ d_val,
    const ValueT *__restrict__ d_b,
    ValueT *__restrict__ d_x,
    bool unit_diag,
    int *__restrict__ d_status)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = int(gridDim.x * blockDim.x);

    for (int lvl = 0; lvl < num_levels; ++lvl)
    {
        // After previous grid.sync, global memory is visible
        if (*d_status != 0)
            break;

        const int start = d_level_ptr[lvl];
        const int end = d_level_ptr[lvl + 1];

        for (int t = start + tid; t < end; t += stride)
        {
            const int i = d_levels[t];
            sptrsv_solve_row_diag_last<IndexT, ValueT>(
                i, d_rowPtr, d_colInd, d_val, d_b, d_x, unit_diag, d_status);
        }

        grid.sync();
    }
}

/*
Persistent cooperative kernel (half2)
*/
template <typename IndexT>
__global__ void sptrsv_trsv_level_persistent_kernel_half2(
    int num_levels,
    const int *__restrict__ d_level_ptr,
    const int *__restrict__ d_levels,
    const IndexT *__restrict__ d_rowPtr,
    const IndexT *__restrict__ d_colInd,
    const __half *__restrict__ d_val,
    const __half *__restrict__ d_b,
    __half *__restrict__ d_x,
    bool unit_diag,
    int *__restrict__ d_status)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = int(gridDim.x * blockDim.x);

    for (int lvl = 0; lvl < num_levels; ++lvl)
    {
        if (*d_status != 0)
            break;

        const int start = d_level_ptr[lvl];
        const int end = d_level_ptr[lvl + 1];

        for (int t = start + tid; t < end; t += stride)
        {
            const int i = d_levels[t];
            sptrsv_solve_row_diag_last_half2<IndexT>(
                i, d_rowPtr, d_colInd, d_val, d_b, d_x, unit_diag, d_status);
        }

        grid.sync();
    }
}

/*
Persistent plan: owns DeviceLevelSets + reusable device status flag + pinned host status.
Call init() once, then solve() many times.
*/
struct SpTRSVLevelsetsPlan
{
    DeviceLevelSets ls;

    int *d_status = nullptr;
    int *h_status = nullptr;

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

        cudaError_t e = cudaMalloc((void **)&d_status, sizeof(int));
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
        if (!ls.d_levels || !ls.d_level_ptr)
            return -1;
        if (!d_status || !h_status)
            return -1;
        if (ls.level_ptr.size() < 2)
            return -1;
        if (ls.level_ptr.front() != 0)
            return -1;
        if (ls.level_ptr.back() != n)
            return -1;

        cudaError_t e = cudaMemsetAsync(d_status, 0, sizeof(int), stream);
        if (e != cudaSuccess)
            return -3;

        // Check cooperative launch support
        int coop = 0;
        e = cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
        if (e != cudaSuccess)
            return -3;

        // Kernel config (tuned to reduce barrier participants)
        constexpr int THREADS = 256;

        // Fallback path: original per-level launches (if coop unsupported)
        if (!coop)
        {
            for (int lvl = 0; lvl < ls.num_levels; ++lvl)
            {
                const int start = ls.level_ptr[lvl];
                const int end = ls.level_ptr[lvl + 1];
                const int level_size = end - start;
                if (level_size <= 0)
                    continue;

                const int blocks = (level_size + THREADS - 1) / THREADS;

                if constexpr (std::is_same_v<ValueT, __half>)
                {
                    sptrsv_trsv_level_kernel_half2<IndexT><<<blocks, THREADS, 0, stream>>>(
                        level_size,
                        ls.d_levels + start,
                        d_rowPtr, d_colInd,
                        reinterpret_cast<const __half *>(d_val),
                        reinterpret_cast<const __half *>(d_b),
                        reinterpret_cast<__half *>(d_x),
                        unit_diag,
                        d_status);
                }
                else
                {
                    sptrsv_trsv_level_kernel<IndexT, ValueT><<<blocks, THREADS, 0, stream>>>(
                        level_size,
                        ls.d_levels + start,
                        d_rowPtr, d_colInd,
                        d_val, d_b, d_x,
                        unit_diag,
                        d_status);
                }
            }

            e = cudaPeekAtLastError();
            if (e != cudaSuccess)
                return -3;

            e = cudaMemcpyAsync(h_status, d_status, sizeof(int), cudaMemcpyDeviceToHost, stream);
            if (e != cudaSuccess)
                return -3;
            e = cudaStreamSynchronize(stream);
            if (e != cudaSuccess)
                return -3;

            return (*h_status != 0) ? -2 : 0;
        }

        // Cooperative path: size blocks to reduce grid.sync overhead
        int numSM = 0;
        e = cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
        if (e != cudaSuccess)
            return -3;

        int maxBlocksPerSM = 0;
        if constexpr (std::is_same_v<ValueT, __half>)
        {
            e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                sptrsv_trsv_level_persistent_kernel_half2<IndexT>,
                THREADS, 0);
        }
        else
        {
            e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                sptrsv_trsv_level_persistent_kernel<IndexT, ValueT>,
                THREADS, 0);
        }
        if (e != cudaSuccess)
            return -3;
        if (maxBlocksPerSM <= 0)
            return -3;

        const int coop_limit = maxBlocksPerSM * numSM;

        // Work-driven blocks: enough to cover the biggest level once
        int blocks_for_work = 1;
        if (ls.max_level_size > 0)
            blocks_for_work = (ls.max_level_size + THREADS - 1) / THREADS;

        int blocks = blocks_for_work;
        blocks = std::min(blocks, coop_limit);
        blocks = std::min(blocks, 2 * numSM);
        blocks = std::max(blocks, 1);

        void *args[] = {
            &ls.num_levels,
            &ls.d_level_ptr,
            &ls.d_levels,
            (void *)&d_rowPtr,
            (void *)&d_colInd,
            (void *)&d_val,
            (void *)&d_b,
            (void *)&d_x,
            &unit_diag,
            &d_status};

        if constexpr (std::is_same_v<ValueT, __half>)
        {
            e = cudaLaunchCooperativeKernel(
                (void *)sptrsv_trsv_level_persistent_kernel_half2<IndexT>,
                blocks, THREADS, args, 0, stream);
        }
        else
        {
            e = cudaLaunchCooperativeKernel(
                (void *)sptrsv_trsv_level_persistent_kernel<IndexT, ValueT>,
                blocks, THREADS, args, 0, stream);
        }
        if (e != cudaSuccess)
            return -3;

        e = cudaPeekAtLastError();
        if (e != cudaSuccess)
            return -3;

        e = cudaMemcpyAsync(h_status, d_status, sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess)
            return -3;
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess)
            return -3;

        return (*h_status != 0) ? -2 : 0;
    }
};
