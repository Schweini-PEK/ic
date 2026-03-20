#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ichol/matrix_formats.hpp"
#include "ichol/subdomain_preconditioner_gpu.hpp"

namespace ichol::precond::detail::subdomain_common
{
    inline void cuda_check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
    }

    inline void cusparse_check(cusparseStatus_t s)
    {
        if (s != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("cuSPARSE error");
    }

    __host__ __device__ inline int flatten_local_3d(int x, int y, int z, int w, int h)
    {
        return x + y * w + z * (w * h);
    }

    __host__ __device__ inline void unflatten_global_3d(int gi, int gw, int gh, int &x, int &y, int &z)
    {
        const int plane = gw * gh;
        z = gi / plane;
        const int rem = gi - z * plane;
        y = rem / gw;
        x = rem - y * gw;
    }

    inline int local_from_global_host(
        int gj,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg,
        int lw,
        int lh)
    {
        int x = 0;
        int y = 0;
        int z = 0;
        unflatten_global_3d(gj, global.w, global.h, x, y, z);
        if (x < reg.x0 || x >= reg.x1 || y < reg.y0 || y >= reg.y1 || z < reg.z0 || z >= reg.z1)
            return -1;
        return flatten_local_3d(x - reg.x0, y - reg.y0, z - reg.z0, lw, lh);
    }

    inline ichol::matrix::CsrMatrix<double> extract_lower_subdomain_csr(
        const ichol::matrix::CsrMatrix<double> &A,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg)
    {
        const int lw = reg.x1 - reg.x0;
        const int lh = reg.y1 - reg.y0;
        const int ld = reg.z1 - reg.z0;
        const int nsub = lw * lh * ld;

        ichol::matrix::CsrMatrix<double> sub;
        sub.num_rows = nsub;
        sub.num_cols = nsub;
        sub.row_ptr.resize(static_cast<std::size_t>(nsub) + 1, 0);

        std::vector<int> cols;
        std::vector<double> vals;

        for (int li = 0; li < nsub; ++li)
        {
            const int plane = lw * lh;
            const int lz = li / plane;
            const int rem = li - lz * plane;
            const int ly = rem / lw;
            const int lx = rem - ly * lw;
            const int gi = (reg.x0 + lx) + (reg.y0 + ly) * global.w + (reg.z0 + lz) * (global.w * global.h);

            std::vector<std::pair<int, double>> row_entries;
            row_entries.reserve(static_cast<std::size_t>(A.row_ptr[gi + 1] - A.row_ptr[gi]));
            for (int kk = A.row_ptr[gi]; kk < A.row_ptr[gi + 1]; ++kk)
            {
                const int lj = local_from_global_host(A.col_ind[kk], global, reg, lw, lh);
                if (lj < 0 || lj > li)
                    continue;
                row_entries.push_back({lj, A.values[kk]});
            }

            std::sort(row_entries.begin(), row_entries.end(), [](const auto &a, const auto &b)
                      { return a.first < b.first; });

            int diag_pos = -1;
            for (int i = 0; i < static_cast<int>(row_entries.size()); ++i)
            {
                if (row_entries[static_cast<std::size_t>(i)].first == li)
                {
                    diag_pos = i;
                    break;
                }
            }
            if (diag_pos < 0)
                throw std::runtime_error("extract_lower_subdomain_csr: missing diagonal entry");

            for (int i = 0; i < static_cast<int>(row_entries.size()); ++i)
            {
                if (i == diag_pos)
                    continue;
                cols.push_back(row_entries[static_cast<std::size_t>(i)].first);
                vals.push_back(row_entries[static_cast<std::size_t>(i)].second);
            }
            cols.push_back(li);
            vals.push_back(row_entries[static_cast<std::size_t>(diag_pos)].second);
            sub.row_ptr[static_cast<std::size_t>(li) + 1] = static_cast<int>(cols.size());
        }

        sub.col_ind = std::move(cols);
        sub.values = std::move(vals);
        sub.nnz = static_cast<int>(sub.values.size());
        return sub;
    }

    inline void build_csr_transpose(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<double> &val,
        std::vector<int> &row_ptr_t,
        std::vector<int> &col_ind_t,
        std::vector<double> &val_t)
    {
        const int nnz = static_cast<int>(val.size());
        row_ptr_t.assign(static_cast<std::size_t>(n) + 1, 0);
        col_ind_t.assign(static_cast<std::size_t>(nnz), 0);
        val_t.assign(static_cast<std::size_t>(nnz), 0.0);

        for (int i = 0; i < nnz; ++i)
            ++row_ptr_t[static_cast<std::size_t>(col_ind[i]) + 1];

        for (int i = 0; i < n; ++i)
            row_ptr_t[static_cast<std::size_t>(i) + 1] += row_ptr_t[static_cast<std::size_t>(i)];

        std::vector<int> next = row_ptr_t;
        for (int i = 0; i < n; ++i)
        {
            for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
            {
                const int j = col_ind[p];
                const int dst = next[static_cast<std::size_t>(j)]++;
                col_ind_t[static_cast<std::size_t>(dst)] = i;
                val_t[static_cast<std::size_t>(dst)] = val[static_cast<std::size_t>(p)];
            }
        }
    }

    inline ichol::solver::ComputePrecision normalize_sparse_solve_precision(ichol::solver::ComputePrecision prec)
    {
        using Prec = ichol::solver::ComputePrecision;
        switch (prec)
        {
        case Prec::FP64:
            return Prec::FP64;
        case Prec::FP32:
        case Prec::TF32:
            return Prec::FP32;
        default:
            throw std::runtime_error("subdomain sparse-solve backends support FP64 and FP32 only");
        }
    }

    template <typename T>
    constexpr cudaDataType_t cuda_data_type();

    template <>
    constexpr cudaDataType_t cuda_data_type<double>()
    {
        return CUDA_R_64F;
    }

    template <>
    constexpr cudaDataType_t cuda_data_type<float>()
    {
        return CUDA_R_32F;
    }

    template <typename T>
    __global__ void k_gather_subvec(const T *src, const int *gidx, T *dst, int nsub)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nsub)
            dst[i] = src[gidx[i]];
    }

    template <typename T>
    __global__ void k_scatter_subvec(const T *src, const int *gidx, T *dst, int nsub)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nsub)
            dst[gidx[i]] = src[i];
    }

    template <typename To, typename From>
    __global__ void k_cast_vec(int n, const From *src, To *dst)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = static_cast<To>(src[i]);
    }

    static __global__ void k_build_gidx(
        int *gidx,
        int lw,
        int lh,
        int ld,
        int gw,
        int gh,
        int x0,
        int y0,
        int z0)
    {
        const int li = blockIdx.x * blockDim.x + threadIdx.x;
        const int nsub = lw * lh * ld;
        if (li >= nsub)
            return;

        const int plane = lw * lh;
        const int lz = li / plane;
        const int rem = li - lz * plane;
        const int ly = rem / lw;
        const int lx = rem - ly * lw;

        const int gx = x0 + lx;
        const int gy = y0 + ly;
        const int gz = z0 + lz;
        gidx[li] = gx + gy * gw + gz * (gw * gh);
    }

    inline void build_subdomain_gidx(
        int *d_gidx,
        int lw,
        int lh,
        int ld,
        int gw,
        int gh,
        int x0,
        int y0,
        int z0,
        cudaStream_t stream = 0)
    {
        const int nsub = lw * lh * ld;
        const int threads = 256;
        const int blocks = (nsub + threads - 1) / threads;
        k_build_gidx<<<blocks, threads, 0, stream>>>(d_gidx, lw, lh, ld, gw, gh, x0, y0, z0);
        cuda_check(cudaGetLastError());
    }

    template <typename T>
    inline void upload_values(std::vector<T> &dst, const std::vector<double> &src)
    {
        dst.resize(src.size());
        for (std::size_t i = 0; i < src.size(); ++i)
            dst[i] = static_cast<T>(src[i]);
    }
} // namespace ichol::precond::detail::subdomain_common
