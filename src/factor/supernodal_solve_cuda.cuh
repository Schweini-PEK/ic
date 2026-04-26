#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "backends/CUDA/util/gmath.cuh"
#include "factor/supernodal_solve.hpp"
#include "factor/symbolic/super_sym.hpp"

namespace ichol::supernodal::cuda_reference
{
    namespace cg = cooperative_groups;
    constexpr int kSupernodeSolveThreads = 32;

    inline void check_cuda(cudaError_t err, const char *what)
    {
        if (err != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }

    template <typename T>
    class DeviceArray
    {
    public:
        DeviceArray() = default;
        explicit DeviceArray(size_t count) { alloc(count); }
        DeviceArray(const DeviceArray &) = delete;
        DeviceArray &operator=(const DeviceArray &) = delete;

        ~DeviceArray()
        {
            if (ptr_ != nullptr)
                cudaFree(ptr_);
        }

        void alloc(size_t count)
        {
            if (ptr_ != nullptr)
                cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = count;
            if (count > 0)
                check_cuda(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc");
        }

        void copy_from_host(const std::vector<T> &host, cudaStream_t stream)
        {
            alloc(host.size());
            if (!host.empty())
                check_cuda(cudaMemcpyAsync(ptr_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync H2D");
        }

        void copy_to_host(std::vector<T> &host, cudaStream_t stream) const
        {
            host.resize(size_);
            if (size_ > 0)
                check_cuda(cudaMemcpyAsync(host.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream),
                           "cudaMemcpyAsync D2H");
        }

        void memset_zero(cudaStream_t stream)
        {
            if (size_ > 0)
                check_cuda(cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), stream), "cudaMemsetAsync");
        }

        T *get() const { return ptr_; }
        size_t size() const { return size_; }

    private:
        T *ptr_ = nullptr;
        size_t size_ = 0;
    };

    template <typename ValueT>
    __device__ __forceinline__ ValueT neg_value(ValueT v)
    {
        return ichol::cuda::GMath<ValueT>::sub(ichol::cuda::GMath<ValueT>::zero(), v);
    }

    template <typename ValueT>
    __device__ void solve_lower_supernode_block(
        int k,
        const int *__restrict__ super,
        const int *__restrict__ pi,
        const int *__restrict__ px,
        const int *__restrict__ s,
        const ValueT *__restrict__ packed,
        ValueT *__restrict__ work,
        ValueT *__restrict__ x,
        int *__restrict__ status)
    {
        __shared__ int active;
        __shared__ int ok;

        if (threadIdx.x == 0)
        {
            active = (*status == 0);
            ok = 1;
        }
        __syncthreads();
        if (!active)
            return;

        const int c0 = super[k];
        const int c1 = super[k + 1];
        const int nscol = c1 - c0;
        const int pi0 = pi[k];
        const int pi1 = pi[k + 1];
        const int nsrow = pi1 - pi0;
        const int px0 = px[k];

        if (threadIdx.x == 0)
        {
            for (int j = 0; j < nscol; ++j)
            {
                ValueT rhs = work[c0 + j];
                for (int i = 0; i < j; ++i)
                {
                    const ValueT lij = packed[(size_t)px0 + (size_t)i * (size_t)nsrow + (size_t)j];
                    rhs = ichol::cuda::GMath<ValueT>::sub(
                        rhs,
                        ichol::cuda::GMath<ValueT>::mul(lij, x[c0 + i]));
                }

                const ValueT diag = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)j];
                if (ichol::cuda::GMath<ValueT>::eq0(diag))
                {
                    atomicExch(status, 1);
                    ok = 0;
                    break;
                }

                x[c0 + j] = ichol::cuda::GMath<ValueT>::div(rhs, diag);
            }
        }
        __syncthreads();
        if (!ok)
            return;

        for (int local_row = nscol + threadIdx.x; local_row < nsrow; local_row += blockDim.x)
        {
            ValueT update = ichol::cuda::GMath<ValueT>::zero();
            for (int j = 0; j < nscol; ++j)
            {
                const ValueT lij = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row];
                update = ichol::cuda::GMath<ValueT>::add(
                    update,
                    ichol::cuda::GMath<ValueT>::mul(lij, x[c0 + j]));
            }

            const int grow = s[pi0 + local_row];
            atomicAdd(&work[grow], neg_value(update));
        }
        __syncthreads();
    }

    template <typename ValueT>
    __device__ void solve_lower_transpose_supernode_block(
        int k,
        const int *__restrict__ super,
        const int *__restrict__ pi,
        const int *__restrict__ px,
        const int *__restrict__ s,
        const ValueT *__restrict__ packed,
        const ValueT *__restrict__ b,
        ValueT *__restrict__ x,
        int *__restrict__ status)
    {
        __shared__ int active;
        __shared__ int ok;
        __shared__ ValueT partial[kSupernodeSolveThreads];

        if (threadIdx.x == 0)
        {
            active = (*status == 0);
            ok = 1;
        }
        __syncthreads();
        if (!active)
            return;

        const int c0 = super[k];
        const int c1 = super[k + 1];
        const int nscol = c1 - c0;
        const int pi0 = pi[k];
        const int pi1 = pi[k + 1];
        const int nsrow = pi1 - pi0;
        const int px0 = px[k];

        for (int j = nscol - 1; j >= 0; --j)
        {
            ValueT local_sum = ichol::cuda::GMath<ValueT>::zero();
            for (int local_row = nscol + threadIdx.x; local_row < nsrow; local_row += blockDim.x)
            {
                const int grow = s[pi0 + local_row];
                const ValueT lij = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row];
                local_sum = ichol::cuda::GMath<ValueT>::add(
                    local_sum,
                    ichol::cuda::GMath<ValueT>::mul(lij, x[grow]));
            }

            partial[threadIdx.x] = local_sum;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (threadIdx.x < stride)
                    partial[threadIdx.x] = ichol::cuda::GMath<ValueT>::add(
                        partial[threadIdx.x],
                        partial[threadIdx.x + stride]);
                __syncthreads();
            }

            if (threadIdx.x == 0)
            {
                ValueT rhs = ichol::cuda::GMath<ValueT>::sub(b[c0 + j], partial[0]);

                for (int i = j + 1; i < nscol; ++i)
                {
                    const ValueT lij = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)i];
                    rhs = ichol::cuda::GMath<ValueT>::sub(
                        rhs,
                        ichol::cuda::GMath<ValueT>::mul(lij, x[c0 + i]));
                }

                const ValueT diag = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)j];
                if (ichol::cuda::GMath<ValueT>::eq0(diag))
                {
                    atomicExch(status, 1);
                    ok = 0;
                }
                else
                {
                    x[c0 + j] = ichol::cuda::GMath<ValueT>::div(rhs, diag);
                }
            }
            __syncthreads();
            if (!ok)
                return;
        }
    }

    inline void flatten_buckets(
        const std::vector<std::vector<int>> &buckets,
        std::vector<int> &bucket_ptr,
        std::vector<int> &bucket_nodes)
    {
        bucket_ptr.clear();
        bucket_nodes.clear();
        bucket_ptr.reserve(buckets.size() + 1);
        bucket_ptr.push_back(0);

        for (const auto &bucket : buckets)
        {
            if (bucket_nodes.size() + bucket.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("flatten_buckets: too many supernode entries");
            bucket_nodes.insert(bucket_nodes.end(), bucket.begin(), bucket.end());
            bucket_ptr.push_back(static_cast<int>(bucket_nodes.size()));
        }
    }

    template <typename ValueT>
    __global__ void lower_bucket_kernel(
        int bucket_size,
        const int *__restrict__ bucket_nodes,
        const int *__restrict__ super,
        const int *__restrict__ pi,
        const int *__restrict__ px,
        const int *__restrict__ s,
        const ValueT *__restrict__ packed,
        ValueT *__restrict__ work,
        ValueT *__restrict__ x,
        int *__restrict__ status)
    {
        const int t = blockIdx.x;
        if (t >= bucket_size)
            return;

        const int k = bucket_nodes[t];
        solve_lower_supernode_block(k, super, pi, px, s, packed, work, x, status);
    }

    template <typename ValueT>
    __global__ void lower_transpose_bucket_kernel(
        int bucket_size,
        const int *__restrict__ bucket_nodes,
        const int *__restrict__ super,
        const int *__restrict__ pi,
        const int *__restrict__ px,
        const int *__restrict__ s,
        const ValueT *__restrict__ packed,
        const ValueT *__restrict__ b,
        ValueT *__restrict__ x,
        int *__restrict__ status)
    {
        const int t = blockIdx.x;
        if (t >= bucket_size)
            return;

        const int k = bucket_nodes[t];
        solve_lower_transpose_supernode_block(k, super, pi, px, s, packed, b, x, status);
    }

    template <typename ValueT>
    __global__ void lower_persistent_kernel(
        int num_levels,
        const int *__restrict__ bucket_ptr,
        const int *__restrict__ bucket_nodes,
        const int *__restrict__ super,
        const int *__restrict__ pi,
        const int *__restrict__ px,
        const int *__restrict__ s,
        const ValueT *__restrict__ packed,
        ValueT *__restrict__ work,
        ValueT *__restrict__ x,
        int *__restrict__ status)
    {
        cg::grid_group grid = cg::this_grid();

        for (int level = 0; level < num_levels; ++level)
        {
            const int start = bucket_ptr[level];
            const int end = bucket_ptr[level + 1];

            for (int node_pos = start + blockIdx.x; node_pos < end; node_pos += gridDim.x)
                solve_lower_supernode_block(bucket_nodes[node_pos], super, pi, px, s, packed, work, x, status);

            grid.sync();
            if (*status != 0)
                break;
        }
    }

    template <typename ValueT>
    __global__ void lower_transpose_persistent_kernel(
        int num_levels,
        const int *__restrict__ bucket_ptr,
        const int *__restrict__ bucket_nodes,
        const int *__restrict__ super,
        const int *__restrict__ pi,
        const int *__restrict__ px,
        const int *__restrict__ s,
        const ValueT *__restrict__ packed,
        const ValueT *__restrict__ b,
        ValueT *__restrict__ x,
        int *__restrict__ status)
    {
        cg::grid_group grid = cg::this_grid();

        for (int level = num_levels - 1; level >= 0; --level)
        {
            const int start = bucket_ptr[level];
            const int end = bucket_ptr[level + 1];

            for (int node_pos = start + blockIdx.x; node_pos < end; node_pos += gridDim.x)
                solve_lower_transpose_supernode_block(bucket_nodes[node_pos], super, pi, px, s, packed, b, x, status);

            grid.sync();
            if (*status != 0)
                break;
        }
    }

    inline int cooperative_block_count(
        void *kernel,
        int max_bucket_size)
    {
        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");

        int num_sm = 0;
        check_cuda(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device),
                   "cudaDeviceGetAttribute cudaDevAttrMultiProcessorCount");

        int max_blocks_per_sm = 0;
        constexpr int threads = kSupernodeSolveThreads;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &max_blocks_per_sm, kernel, threads, 0),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

        if (num_sm <= 0 || max_blocks_per_sm <= 0)
            throw std::runtime_error("cooperative_block_count: invalid occupancy result");

        const int coop_limit = num_sm * max_blocks_per_sm;
        int blocks = std::max(1, max_bucket_size);
        blocks = std::min(blocks, coop_limit);
        blocks = std::min(blocks, 2 * num_sm);
        return std::max(1, blocks);
    }

    inline bool cooperative_launch_supported()
    {
        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");

        int coop = 0;
        check_cuda(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device),
                   "cudaDeviceGetAttribute cudaDevAttrCooperativeLaunch");
        return coop != 0;
    }

    template <typename ValueT>
    void solve_lower_device(
        int n,
        const std::vector<int> &bucket_ptr,
        const int *d_bucket_nodes,
        const int *d_super,
        const int *d_pi,
        const int *d_px,
        const int *d_s,
        const ValueT *d_packed,
        const ValueT *d_b,
        ValueT *d_work,
        ValueT *d_x,
        int *d_status,
        cudaStream_t stream,
        bool reset_status = true)
    {
        static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>,
                      "CUDA reference supernodal solve currently supports float/double.");

        if (n < 0 || bucket_ptr.empty() || bucket_ptr.front() != 0)
            throw std::runtime_error("solve_lower_device: invalid dimensions");
        if (!d_bucket_nodes || !d_super || !d_pi || !d_px || !d_s || !d_packed || !d_b || !d_work || !d_x || !d_status)
            throw std::runtime_error("solve_lower_device: null device pointer");

        if (reset_status)
            check_cuda(cudaMemsetAsync(d_status, 0, sizeof(int), stream), "cudaMemsetAsync status");
        if (d_work != d_b && n > 0)
            check_cuda(cudaMemcpyAsync(d_work, d_b, (size_t)n * sizeof(ValueT), cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpyAsync lower work");
        if (n > 0)
            check_cuda(cudaMemsetAsync(d_x, 0, (size_t)n * sizeof(ValueT), stream), "cudaMemsetAsync lower x");

        constexpr int threads = kSupernodeSolveThreads;
        for (size_t level = 0; level + 1 < bucket_ptr.size(); ++level)
        {
            const int start = bucket_ptr[level];
            const int end = bucket_ptr[level + 1];
            const int count = end - start;
            if (count <= 0)
                continue;

            lower_bucket_kernel<ValueT><<<count, threads, 0, stream>>>(
                count,
                d_bucket_nodes + start,
                d_super, d_pi, d_px, d_s,
                d_packed, d_work, d_x, d_status);
            check_cuda(cudaPeekAtLastError(), "lower_bucket_kernel launch");
        }
    }

    template <typename ValueT>
    void solve_lower_transpose_device(
        int n,
        const std::vector<int> &bucket_ptr,
        const int *d_bucket_nodes,
        const int *d_super,
        const int *d_pi,
        const int *d_px,
        const int *d_s,
        const ValueT *d_packed,
        const ValueT *d_b,
        ValueT *d_x,
        int *d_status,
        cudaStream_t stream,
        bool reset_status = true)
    {
        static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>,
                      "CUDA reference supernodal solve currently supports float/double.");

        if (n < 0 || bucket_ptr.empty() || bucket_ptr.front() != 0)
            throw std::runtime_error("solve_lower_transpose_device: invalid dimensions");
        if (!d_bucket_nodes || !d_super || !d_pi || !d_px || !d_s || !d_packed || !d_b || !d_x || !d_status)
            throw std::runtime_error("solve_lower_transpose_device: null device pointer");

        if (reset_status)
            check_cuda(cudaMemsetAsync(d_status, 0, sizeof(int), stream), "cudaMemsetAsync status");
        if (n > 0)
            check_cuda(cudaMemsetAsync(d_x, 0, (size_t)n * sizeof(ValueT), stream), "cudaMemsetAsync lower transpose x");

        constexpr int threads = kSupernodeSolveThreads;
        for (int level = static_cast<int>(bucket_ptr.size()) - 2; level >= 0; --level)
        {
            const int start = bucket_ptr[(size_t)level];
            const int end = bucket_ptr[(size_t)level + 1];
            const int count = end - start;
            if (count <= 0)
                continue;

            lower_transpose_bucket_kernel<ValueT><<<count, threads, 0, stream>>>(
                count,
                d_bucket_nodes + start,
                d_super, d_pi, d_px, d_s,
                d_packed, d_b, d_x, d_status);
            check_cuda(cudaPeekAtLastError(), "lower_transpose_bucket_kernel launch");
        }
    }

    template <typename ValueT>
    bool solve_lower_device_persistent(
        int n,
        int num_levels,
        int max_bucket_size,
        const int *d_bucket_ptr,
        const int *d_bucket_nodes,
        const int *d_super,
        const int *d_pi,
        const int *d_px,
        const int *d_s,
        const ValueT *d_packed,
        const ValueT *d_b,
        ValueT *d_work,
        ValueT *d_x,
        int *d_status,
        cudaStream_t stream,
        bool reset_status = true)
    {
        static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>,
                      "CUDA reference supernodal solve currently supports float/double.");

        if (!cooperative_launch_supported())
            return false;
        if (n < 0 || num_levels < 0 || max_bucket_size < 0)
            throw std::runtime_error("solve_lower_device_persistent: invalid dimensions");
        if (!d_bucket_ptr || !d_bucket_nodes || !d_super || !d_pi || !d_px || !d_s ||
            !d_packed || !d_b || !d_work || !d_x || !d_status)
            throw std::runtime_error("solve_lower_device_persistent: null device pointer");

        if (reset_status)
            check_cuda(cudaMemsetAsync(d_status, 0, sizeof(int), stream), "cudaMemsetAsync status");
        if (d_work != d_b && n > 0)
            check_cuda(cudaMemcpyAsync(d_work, d_b, (size_t)n * sizeof(ValueT), cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpyAsync lower persistent work");
        if (n > 0)
            check_cuda(cudaMemsetAsync(d_x, 0, (size_t)n * sizeof(ValueT), stream), "cudaMemsetAsync lower persistent x");

        constexpr int threads = kSupernodeSolveThreads;
        const int blocks = cooperative_block_count((void *)lower_persistent_kernel<ValueT>, max_bucket_size);
        void *args[] = {
            &num_levels,
            (void *)&d_bucket_ptr,
            (void *)&d_bucket_nodes,
            (void *)&d_super,
            (void *)&d_pi,
            (void *)&d_px,
            (void *)&d_s,
            (void *)&d_packed,
            (void *)&d_work,
            (void *)&d_x,
            (void *)&d_status};

        check_cuda(cudaLaunchCooperativeKernel(
                       (void *)lower_persistent_kernel<ValueT>,
                       blocks, threads, args, 0, stream),
                   "cudaLaunchCooperativeKernel lower_persistent_kernel");
        check_cuda(cudaPeekAtLastError(), "lower_persistent_kernel launch");
        return true;
    }

    template <typename ValueT>
    bool solve_lower_transpose_device_persistent(
        int n,
        int num_levels,
        int max_bucket_size,
        const int *d_bucket_ptr,
        const int *d_bucket_nodes,
        const int *d_super,
        const int *d_pi,
        const int *d_px,
        const int *d_s,
        const ValueT *d_packed,
        const ValueT *d_b,
        ValueT *d_x,
        int *d_status,
        cudaStream_t stream,
        bool reset_status = true)
    {
        static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>,
                      "CUDA reference supernodal solve currently supports float/double.");

        if (!cooperative_launch_supported())
            return false;
        if (n < 0 || num_levels < 0 || max_bucket_size < 0)
            throw std::runtime_error("solve_lower_transpose_device_persistent: invalid dimensions");
        if (!d_bucket_ptr || !d_bucket_nodes || !d_super || !d_pi || !d_px || !d_s ||
            !d_packed || !d_b || !d_x || !d_status)
            throw std::runtime_error("solve_lower_transpose_device_persistent: null device pointer");

        if (reset_status)
            check_cuda(cudaMemsetAsync(d_status, 0, sizeof(int), stream), "cudaMemsetAsync status");
        if (n > 0)
            check_cuda(cudaMemsetAsync(d_x, 0, (size_t)n * sizeof(ValueT), stream),
                       "cudaMemsetAsync lower transpose persistent x");

        constexpr int threads = kSupernodeSolveThreads;
        const int blocks = cooperative_block_count((void *)lower_transpose_persistent_kernel<ValueT>, max_bucket_size);
        void *args[] = {
            &num_levels,
            (void *)&d_bucket_ptr,
            (void *)&d_bucket_nodes,
            (void *)&d_super,
            (void *)&d_pi,
            (void *)&d_px,
            (void *)&d_s,
            (void *)&d_packed,
            (void *)&d_b,
            (void *)&d_x,
            (void *)&d_status};

        check_cuda(cudaLaunchCooperativeKernel(
                       (void *)lower_transpose_persistent_kernel<ValueT>,
                       blocks, threads, args, 0, stream),
                   "cudaLaunchCooperativeKernel lower_transpose_persistent_kernel");
        check_cuda(cudaPeekAtLastError(), "lower_transpose_persistent_kernel launch");
        return true;
    }

    template <typename ValueT>
    std::vector<ValueT> solve_lower(
        const symbolic::SuperSym &sym,
        const std::vector<ValueT> &packed,
        const std::vector<std::vector<int>> &buckets,
        const std::vector<ValueT> &b,
        cudaStream_t stream = 0)
    {
        static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>,
                      "CUDA reference supernodal solve currently supports float/double.");

        validate_supernodal_solve_inputs(sym, packed, b);
        validate_forward_solve_buckets(sym, buckets);

        std::vector<int> bucket_ptr, bucket_nodes;
        flatten_buckets(buckets, bucket_ptr, bucket_nodes);
        int max_bucket_size = 0;
        for (size_t level = 0; level + 1 < bucket_ptr.size(); ++level)
            max_bucket_size = std::max(max_bucket_size, bucket_ptr[level + 1] - bucket_ptr[level]);

        DeviceArray<int> d_super, d_pi, d_px, d_s, d_bucket_ptr, d_bucket_nodes, d_status;
        DeviceArray<ValueT> d_packed, d_work, d_x;
        d_super.copy_from_host(sym.super, stream);
        d_pi.copy_from_host(sym.pi, stream);
        d_px.copy_from_host(sym.px, stream);
        d_s.copy_from_host(sym.s, stream);
        d_bucket_ptr.copy_from_host(bucket_ptr, stream);
        d_bucket_nodes.copy_from_host(bucket_nodes, stream);
        d_packed.copy_from_host(packed, stream);
        d_work.copy_from_host(b, stream);
        d_x.alloc(b.size());
        d_x.memset_zero(stream);

        std::vector<int> status = {0};
        d_status.copy_from_host(status, stream);

        const bool used_persistent = solve_lower_device_persistent(
            static_cast<int>(b.size()), static_cast<int>(bucket_ptr.size()) - 1, max_bucket_size,
            d_bucket_ptr.get(), d_bucket_nodes.get(),
            d_super.get(), d_pi.get(), d_px.get(), d_s.get(),
            d_packed.get(), d_work.get(), d_work.get(), d_x.get(), d_status.get(),
            stream, true);
        if (!used_persistent)
        {
            solve_lower_device(
                static_cast<int>(b.size()), bucket_ptr, d_bucket_nodes.get(),
                d_super.get(), d_pi.get(), d_px.get(), d_s.get(),
                d_packed.get(), d_work.get(), d_work.get(), d_x.get(), d_status.get(),
                stream, true);
        }

        d_status.copy_to_host(status, stream);
        std::vector<ValueT> x;
        d_x.copy_to_host(x, stream);
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        if (status[0] != 0)
            throw std::runtime_error("solve_lower: zero diagonal encountered on device");
        return x;
    }

    template <typename ValueT>
    std::vector<ValueT> solve_lower_transpose(
        const symbolic::SuperSym &sym,
        const std::vector<ValueT> &packed,
        const std::vector<std::vector<int>> &buckets,
        const std::vector<ValueT> &b,
        cudaStream_t stream = 0)
    {
        static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>,
                      "CUDA reference supernodal solve currently supports float/double.");

        validate_supernodal_solve_inputs(sym, packed, b);
        validate_forward_solve_buckets(sym, buckets);

        std::vector<int> bucket_ptr, bucket_nodes;
        flatten_buckets(buckets, bucket_ptr, bucket_nodes);
        int max_bucket_size = 0;
        for (size_t level = 0; level + 1 < bucket_ptr.size(); ++level)
            max_bucket_size = std::max(max_bucket_size, bucket_ptr[level + 1] - bucket_ptr[level]);

        DeviceArray<int> d_super, d_pi, d_px, d_s, d_bucket_ptr, d_bucket_nodes, d_status;
        DeviceArray<ValueT> d_packed, d_b, d_x;
        d_super.copy_from_host(sym.super, stream);
        d_pi.copy_from_host(sym.pi, stream);
        d_px.copy_from_host(sym.px, stream);
        d_s.copy_from_host(sym.s, stream);
        d_bucket_ptr.copy_from_host(bucket_ptr, stream);
        d_bucket_nodes.copy_from_host(bucket_nodes, stream);
        d_packed.copy_from_host(packed, stream);
        d_b.copy_from_host(b, stream);
        d_x.alloc(b.size());
        d_x.memset_zero(stream);

        std::vector<int> status = {0};
        d_status.copy_from_host(status, stream);

        const bool used_persistent = solve_lower_transpose_device_persistent(
            static_cast<int>(b.size()), static_cast<int>(bucket_ptr.size()) - 1, max_bucket_size,
            d_bucket_ptr.get(), d_bucket_nodes.get(),
            d_super.get(), d_pi.get(), d_px.get(), d_s.get(),
            d_packed.get(), d_b.get(), d_x.get(), d_status.get(),
            stream, true);
        if (!used_persistent)
        {
            solve_lower_transpose_device(
                static_cast<int>(b.size()), bucket_ptr, d_bucket_nodes.get(),
                d_super.get(), d_pi.get(), d_px.get(), d_s.get(),
                d_packed.get(), d_b.get(), d_x.get(), d_status.get(),
                stream, true);
        }

        d_status.copy_to_host(status, stream);
        std::vector<ValueT> x;
        d_x.copy_to_host(x, stream);
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        if (status[0] != 0)
            throw std::runtime_error("solve_lower_transpose: zero diagonal encountered on device");
        return x;
    }
} // namespace ichol::supernodal::cuda_reference
