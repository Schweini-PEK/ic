// src/factor/numerical/cuda/detail/ickdt_factorize_impl.hpp
#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <type_traits>

#include "ichol/cuda_utils.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/detail/numeric_plan.hpp"
#include "factor/numerical/cuda/detail/ickdt_factorize_impl.hpp"
#include "backends/CUDA/util/gmath.cuh"
#include "backends/CUDA/util/host_cast.hpp"
#include "backends/CUDA/util/hash.cuh"
#include "backends/CUDA/util/memory.cuh"
#include "backends/CUDA/util/sort.cuh"
#include "backends/CUDA/factor/ickdt/ickdt_kernels.cuh"
#include "backends/CUDA/factor/ickdt/ickdt_launch.cuh"

namespace ichol::numeric::cuda
{
    template <class T, class G>
    void icdtk_gpu(
        const ichol::matrix::CsrMatrix<T> &Ahost,
        const ichol::symbolic::FactorPattern &L_pattern,
        const ichol::symbolic::LevelSets &level_sets,
        std::vector<int> &L_row_ptr_out,
        std::vector<int> &L_col_ind_out,
        std::vector<G> &L_val_fixed_out,
        int &fail_row_out,
        const ichol::IncompleteCholeskyOptions &options)
    {
        const int n = Ahost.num_rows;

        std::vector<int> col_ptr, col_row, col_csr_pos;
        ichol::matrix::csr_to_csc_pattern_only(n, L_pattern.row_ptr_L, L_pattern.col_ind_L, col_ptr, col_row, col_csr_pos);

        std::vector<int> level_ptr, level_rows;
        level_ptr = level_sets.level_ptr;
        level_rows = level_sets.levels;

        int *d_rowPtrA = nullptr, *d_colIndA = nullptr;
        G *d_valA = nullptr;

        int *d_rowPtrL = nullptr, *d_colIndL = nullptr;
        G *d_valL = nullptr;

        int *d_colPtrL = nullptr, *d_colRowL = nullptr, *d_colCsrPosL = nullptr;
        int *d_levelRows = nullptr;

        int *d_status = nullptr, *d_fail_row = nullptr;

        const int nnzA = static_cast<int>(Ahost.col_ind.size());
        const int nnzL = static_cast<int>(L_pattern.col_ind_L.size());

        CUDA_CHECK(cudaMalloc(&d_rowPtrA, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIndA, nnzA * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_valA, nnzA * sizeof(G)));

        CUDA_CHECK(cudaMemcpy(d_rowPtrA, Ahost.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIndA, Ahost.col_ind.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));

        {
            std::vector<G> tmp(nnzA);
            for (int p = 0; p < nnzA; ++p)
                tmp[p] = ichol::cuda::util::to_cuda_type<G>(Ahost.values[p]);
            CUDA_CHECK(cudaMemcpy(d_valA, tmp.data(), nnzA * sizeof(G), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMalloc(&d_rowPtrL, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIndL, nnzL * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_valL, nnzL * sizeof(G)));

        CUDA_CHECK(cudaMemcpy(d_rowPtrL, L_pattern.row_ptr_L.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIndL, L_pattern.col_ind_L.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_valL, 0, nnzL * sizeof(G)));

        CUDA_CHECK(cudaMalloc(&d_colPtrL, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colRowL, nnzL * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colCsrPosL, nnzL * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_colPtrL, col_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colRowL, col_row.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colCsrPosL, col_csr_pos.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_levelRows, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_levelRows, level_rows.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_status, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_fail_row, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_status, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_fail_row, -1, sizeof(int)));

        int threads = 128;

        fail_row_out = -1;
        int h_status = 0;

        // shared memory capability check
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        int maxShmOptin = 0;
        int maxShmDefault = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&maxShmOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxShmDefault, cudaDevAttrMaxSharedMemoryPerBlock, dev));

        int maxlvl = (int)level_ptr.size() - 2;
        for (int lev = 1; lev <= maxlvl; ++lev)
        {
            int lev_begin = level_ptr[lev - 1];
            int lev_end = level_ptr[lev];
            int nrows = lev_end - lev_begin;
            if (nrows <= 0)
                continue;

            int max_off_level = 0;
            for (int bi = lev_begin; bi < lev_end; ++bi)
            {
                int row = level_rows[bi];
                int r0 = L_pattern.row_ptr_L[row];
                int r1 = L_pattern.row_ptr_L[row + 1];
                int m_off = (r1 - r0) - 1;
                max_off_level = std::max(max_off_level, m_off);
            }
            if (max_off_level < 1)
                max_off_level = 1;

            int H_level = 1;
            while (H_level < 4 * max_off_level)
                H_level <<= 1;

            int N_level = 1;
            while (N_level < max_off_level)
                N_level <<= 1;

            size_t shmem = ichol::cuda::shmem_bytes_for_level<G>(max_off_level, H_level, N_level);

            // if dynamic shmem > default, opt in; if > opt-in limit, fail early (prevents launch invalid argument)
            if ((int)shmem > maxShmOptin)
                throw std::runtime_error("ictp_par: required dynamic shared memory exceeds device limit");

            if ((int)shmem > maxShmDefault)
            {
                CUDA_CHECK(cudaFuncSetAttribute(
                    ichol::cuda::ictp_level_kernel<G>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)shmem));
            }

            ichol::cuda::ictp_level_kernel<G><<<nrows, threads, shmem>>>(
                lev_begin, lev_end,
                d_levelRows,
                d_rowPtrA, d_colIndA, d_valA,
                d_rowPtrL, d_colIndL, d_valL,
                d_colPtrL, d_colRowL, d_colCsrPosL,
                options.lfil,
                ichol::cuda::util::to_cuda_type<G>(options.drop_tol),
                ichol::cuda::util::to_cuda_type<G>(options.pivot_tol),
                max_off_level, H_level, N_level,
                d_status, d_fail_row);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&h_status, d_status, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_status != 0)
            {
                CUDA_CHECK(cudaMemcpy(&fail_row_out, d_fail_row, sizeof(int), cudaMemcpyDeviceToHost));
                break;
            }
        }

        L_row_ptr_out = L_pattern.row_ptr_L;
        L_col_ind_out = L_pattern.col_ind_L;
        L_val_fixed_out.resize(nnzL);
        CUDA_CHECK(cudaMemcpy(L_val_fixed_out.data(), d_valL, nnzL * sizeof(G), cudaMemcpyDeviceToHost));

        cudaFree(d_rowPtrA);
        cudaFree(d_colIndA);
        cudaFree(d_valA);
        cudaFree(d_rowPtrL);
        cudaFree(d_colIndL);
        cudaFree(d_valL);
        cudaFree(d_colPtrL);
        cudaFree(d_colRowL);
        cudaFree(d_colCsrPosL);
        cudaFree(d_levelRows);
        cudaFree(d_status);
        cudaFree(d_fail_row);
    }
} // namespace ichol::numeric::cuda