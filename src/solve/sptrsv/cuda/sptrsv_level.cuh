// sptrsv_levelsets.cuh
//
// CUDA 12 level-scheduled sparse triangular solve (SpTRSV) using precomputed level sets.
// This header only handles the non-transpose solve; transpose handling is external.
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

/* Enum for fill mode (kept for API compatibility; not needed for correctness in this implementation). */
enum class FillMode
{
    LOWER,
    UPPER
};

struct DeviceLevelSets
{
    std::vector<int> level_ptr;
    int *d_level_ptr = nullptr;
    int *d_levels = nullptr;
    int n = 0;
    int num_levels = 0;

    DeviceLevelSets() = default;
    DeviceLevelSets(const DeviceLevelSets &) = delete;
    DeviceLevelSets &operator=(const DeviceLevelSets &) = delete;

    ~DeviceLevelSets()
    {
        reset();
    }

    void reset()
    {
        if (d_level_ptr)
            cudaFree(d_level_ptr);
        if (d_levels)
            cudaFree(d_levels);
        d_level_ptr = nullptr;
        d_levels = nullptr;
        level_ptr.clear();
        n = 0;
        num_levels = 0;
    }

    int init(const ichol::symbolic::LevelSets &host, cudaStream_t stream)
    {
        reset();
        level_ptr = host.level_ptr;
        n = static_cast<int>(host.levels.size());
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
                            sizeof(int) * level_ptr.size(), cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        e = cudaMemcpyAsync(d_levels, host.levels.data(),
                            sizeof(int) * n, cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess)
        {
            reset();
            return -3;
        }

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

Math (A is CSR(op(L)) that satisfies the invariant above):
  s := b_i - sum_{k=rowPtr[i]}^{rowPtr[i+1]-2} A(i, colInd[k]) * x[colInd[k]]
  if unit_diag: x_i := s
  else:         x_i := s / A(i,i)   where A(i,i) == val[rowPtr[i+1]-1]

Numerical safeguard:
  - Flag d_status if x_i is NaN/Inf or if diag==0 when unit_diag=false.

Documented choice:
  - No validation that the last entry is diagonal; the kernel assumes it unconditionally
    per user-provided invariant.
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

    int i = d_level_rows[tid];

    IndexT p = d_rowPtr[i];
    IndexT q = d_rowPtr[i + 1];

    // s := b[i]
    ValueT s = d_b[i];

    // Off-diagonals are all entries except the last one.
    // s -= sum_{k=p}^{q-2} A(i, col[k]) * x[col[k]]
    for (IndexT k = p; k + 1 < q; ++k)
    {
        IndexT j = d_colInd[k];
        ValueT a = d_val[k];
        s = ichol::cuda::GMath<ValueT>::sub(s, ichol::cuda::GMath<ValueT>::mul(a, d_x[j]));
    }

    if (!unit_diag)
    {
        // diag := A(i,i) assumed stored at last position.
        ValueT diag = d_val[q - 1];
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
Level-scheduled sparse triangular solve using precomputed host-side level sets.

Mechanism:
  - Use CSR(L) directly.
  - For each level (host loop over level_ptr):
      launch sptrsv_trsv_level_kernel on the rows in that level
    Stream ordering enforces level ordering (no additional sync needed between levels).

Numerical safeguard:
  - Uses a device status flag; if any row produces NaN/Inf or invalid diagonal, returns -2.
*/
template <typename IndexT, typename ValueT>
int SpTRSV_solve_levelsets(
    int n,
    const IndexT *d_rowPtr,
    const IndexT *d_colInd,
    const ValueT *d_val,
    const ValueT *d_b,
    ValueT *d_x,
    FillMode /*fill_mode*/,
    bool unit_diag,
    const ichol::symbolic::LevelSets &levelsets,
    cudaStream_t stream)
{
    if (n < 0)
        return -1;
    if (n == 0)
        return 0;
    if (!d_rowPtr || !d_colInd || !d_val || !d_b || !d_x)
        return -1;

    if ((int)levelsets.levels.size() != n)
        return -1;
    if (levelsets.level_ptr.size() < 2)
        return -1;
    if (levelsets.level_ptr.front() != 0)
        return -1;
    if (levelsets.level_ptr.back() != n)
        return -1;

    // Deterministic initialization.
    auto e = cudaMemsetAsync(d_x, 0, sizeof(ValueT) * n, stream);
    if (e != cudaSuccess)
        return -3;

    // Upload level rows (level_ptr stays on host; it only controls launches).
    int *d_levels = nullptr;
    e = cudaMalloc((void **)&d_levels, sizeof(int) * n);
    if (e != cudaSuccess)
        return -3;

    e = cudaMemcpyAsync(d_levels, levelsets.levels.data(),
                        sizeof(int) * n, cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess)
    {
        cudaFree(d_levels);
        return -3;
    }

    // Device status flag.
    int *d_status = nullptr;
    e = cudaMalloc((void **)&d_status, sizeof(int));
    if (e != cudaSuccess)
    {
        cudaFree(d_levels);
        return -3;
    }

    e = cudaMemsetAsync(d_status, 0, sizeof(int), stream);
    if (e != cudaSuccess)
    {
        cudaFree(d_status);
        cudaFree(d_levels);
        return -3;
    }

    constexpr int THREADS = 128;
    int num_levels = (int)levelsets.level_ptr.size() - 1;

    for (int lvl = 0; lvl < num_levels; ++lvl)
    {
        int start = levelsets.level_ptr[lvl];
        int end = levelsets.level_ptr[lvl + 1];
        int level_size = end - start;
        if (level_size <= 0)
            continue;

        int blocks = (level_size + THREADS - 1) / THREADS;

        sptrsv_trsv_level_kernel<IndexT, ValueT><<<blocks, THREADS, 0, stream>>>(
            level_size,
            d_levels + start,
            d_rowPtr, d_colInd, d_val,
            d_b, d_x,
            unit_diag,
            d_status);

        e = cudaGetLastError();
        if (e != cudaSuccess)
        {
            cudaFree(d_status);
            cudaFree(d_levels);
            return -3;
        }
    }

    // Retrieve status and synchronize so the caller gets a definite result.
    int h_status = 0;
    e = cudaMemcpyAsync(&h_status, d_status, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess)
    {
        cudaFree(d_status);
        cudaFree(d_levels);
        return -3;
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess)
    {
        cudaFree(d_status);
        cudaFree(d_levels);
        return -3;
    }

    cudaFree(d_status);
    cudaFree(d_levels);

    return (h_status != 0) ? -2 : 0;
}

template <typename IndexT, typename ValueT>
int SpTRSV_solve_levelsets_device(
    int n,
    const IndexT *d_rowPtr,
    const IndexT *d_colInd,
    const ValueT *d_val,
    const ValueT *d_b,
    ValueT *d_x,
    FillMode /*fill_mode*/,
    bool unit_diag,
    const DeviceLevelSets &levelsets,
    cudaStream_t stream)
{
    if (n < 0)
        return -1;
    if (n == 0)
        return 0;
    if (!d_rowPtr || !d_colInd || !d_val || !d_b || !d_x)
        return -1;
    if (levelsets.n != n)
        return -1;
    if (!levelsets.d_levels || !levelsets.d_level_ptr)
        return -1;
    if (levelsets.level_ptr.size() < 2)
        return -1;
    if (levelsets.level_ptr.front() != 0)
        return -1;
    if (levelsets.level_ptr.back() != n)
        return -1;

    (void)levelsets.d_level_ptr;

    auto e = cudaMemsetAsync(d_x, 0, sizeof(ValueT) * n, stream);
    if (e != cudaSuccess)
        return -3;

    int *d_status = nullptr;
    e = cudaMalloc((void **)&d_status, sizeof(int));
    if (e != cudaSuccess)
        return -3;

    e = cudaMemsetAsync(d_status, 0, sizeof(int), stream);
    if (e != cudaSuccess)
    {
        cudaFree(d_status);
        return -3;
    }

    constexpr int THREADS = 128;
    int num_levels = levelsets.num_levels;

    for (int lvl = 0; lvl < num_levels; ++lvl)
    {
        int start = levelsets.level_ptr[lvl];
        int end = levelsets.level_ptr[lvl + 1];
        int level_size = end - start;
        if (level_size <= 0)
            continue;

        int blocks = (level_size + THREADS - 1) / THREADS;

        sptrsv_trsv_level_kernel<IndexT, ValueT><<<blocks, THREADS, 0, stream>>>(
            level_size,
            levelsets.d_levels + start,
            d_rowPtr, d_colInd, d_val,
            d_b, d_x,
            unit_diag,
            d_status);

        e = cudaGetLastError();
        if (e != cudaSuccess)
        {
            cudaFree(d_status);
            return -3;
        }
    }

    int h_status = 0;
    e = cudaMemcpyAsync(&h_status, d_status, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess)
    {
        cudaFree(d_status);
        return -3;
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess)
    {
        cudaFree(d_status);
        return -3;
    }

    cudaFree(d_status);

    return (h_status != 0) ? -2 : 0;
}
