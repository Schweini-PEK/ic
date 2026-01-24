// supernodal_numeric_ll_cuda.cu
#include "factor/numerical/supernodal_numeric_ll.hpp"
#include "factor/symbolic/super_sym.hpp"
#include "factor/symbolic/snode_schedule.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace ichol::numeric
{
#define CUDA_TRY(call)                                                                                     \
    do                                                                                                     \
    {                                                                                                      \
        cudaError_t _e = (call);                                                                           \
        if (_e != cudaSuccess)                                                                             \
        {                                                                                                  \
            std::cerr << "CUDA Error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                                        \
            out.ok = false;                                                                                \
            goto cleanup;                                                                                  \
        }                                                                                                  \
    } while (0)

#define CUBLAS_TRY(call)                                                          \
    do                                                                            \
    {                                                                             \
        cublasStatus_t _s = (call);                                               \
        if (_s != CUBLAS_STATUS_SUCCESS)                                          \
        {                                                                         \
            std::cerr << "cuBLAS Error: status=" << int(_s) << " at " << __FILE__ \
                      << ":" << __LINE__ << std::endl;                            \
            out.ok = false;                                                       \
            goto cleanup;                                                         \
        }                                                                         \
    } while (0)

#define CUSOLVER_TRY(call)                                                          \
    do                                                                              \
    {                                                                               \
        cusolverStatus_t _s = (call);                                               \
        if (_s != CUSOLVER_STATUS_SUCCESS)                                          \
        {                                                                           \
            std::cerr << "cuSOLVER Error: status=" << int(_s) << " at " << __FILE__ \
                      << ":" << __LINE__ << std::endl;                              \
            out.ok = false;                                                         \
            goto cleanup;                                                           \
        }                                                                           \
    } while (0)

    // ---------------- Device kernels ----------------

    __global__ void k_build_g2p_stamp(
        int pi0, int nsrow,
        const int *__restrict__ s,
        int *__restrict__ g2p,
        int *__restrict__ stamp,
        int epoch)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t < nsrow)
        {
            int r = s[pi0 + t];
            g2p[r] = t;
            stamp[r] = epoch;
        }
    }

    __device__ __forceinline__ int lookup_local(
        int global_idx,
        const int *__restrict__ g2p,
        const int *__restrict__ stamp,
        int epoch)
    {
        return (stamp[global_idx] == epoch) ? g2p[global_idx] : -1;
    }

    __global__ void k_assemble_A(
        int scol, int ecol,
        const int *__restrict__ col_ptr,
        const int *__restrict__ row_ind,
        const double *__restrict__ val,
        const int *__restrict__ g2p,
        const int *__restrict__ stamp,
        int epoch,
        double *__restrict__ F,
        int ld)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int nscol = ecol - scol;
        if (j >= nscol)
            return;

        int global_col = scol + j;
        int local_col = lookup_local(global_col, g2p, stamp, epoch);
        if (local_col < 0)
            return;

        for (int p = col_ptr[global_col]; p < col_ptr[global_col + 1]; ++p)
        {
            int global_row = row_ind[p];
            if (global_row < global_col)
                continue; // lower only
            int local_row = lookup_local(global_row, g2p, stamp, epoch);
            if (local_row >= 0)
            {
                F[local_row + local_col * ld] += val[p];
            }
        }
    }

    __global__ void k_assemble_child(
        int nupd_child,
        const int *__restrict__ child_idx,
        const double *__restrict__ child_S, // ld = nupd_child, lower valid
        const int *__restrict__ g2p,
        const int *__restrict__ stamp,
        int epoch,
        double *__restrict__ F,
        int ld)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        if (j < nupd_child && i >= j && i < nupd_child)
        {
            double v = child_S[i + j * nupd_child];

            int global_row = child_idx[i];
            int global_col = child_idx[j];

            int local_row = lookup_local(global_row, g2p, stamp, epoch);
            int local_col = lookup_local(global_col, g2p, stamp, epoch);

            if (local_row >= 0 && local_col >= 0)
            {
                // No atomics needed: one launch per child, sequential in the same stream.
                F[local_row + local_col * ld] += v;
            }
        }
    }

    __global__ void k_store_factors_2d(
        int nsrow, int nscol,
        const double *__restrict__ F, int ld,
        double *__restrict__ x, int px0)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (j < nscol && i < nsrow && i >= j)
        {
            x[px0 + i + j * nsrow] = F[i + j * ld];
        }
    }

    __global__ void k_pack_F22_to_S_lower(
        int nupd, int nsrow, int nscol,
        const double *__restrict__ F,
        double *__restrict__ S) // ld=nupd, lower filled, upper zeroed
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < nupd && j < nupd)
        {
            if (i >= j)
                S[i + j * nupd] = F[(nscol + i) + (nscol + j) * nsrow];
            else
                S[i + j * nupd] = 0.0;
        }
    }

    // ---------------- Per-stream resources ----------------

    struct PerStream
    {
        cudaStream_t stream = nullptr;
        cublasHandle_t blas = nullptr;
        cusolverDnHandle_t solver = nullptr;

        double *dF = nullptr;   // max_nsrow^2
        int *d_g2p = nullptr;   // size n
        int *d_stamp = nullptr; // size n
        int epoch = 1;

        double *d_potrf_work = nullptr; // size potrf_lwork doubles
        int potrf_lwork = 0;

        void destroy()
        {
            if (d_potrf_work)
                cudaFree(d_potrf_work);
            if (d_stamp)
                cudaFree(d_stamp);
            if (d_g2p)
                cudaFree(d_g2p);
            if (dF)
                cudaFree(dF);

            if (solver)
                cusolverDnDestroy(solver);
            if (blas)
                cublasDestroy(blas);
            if (stream)
                cudaStreamDestroy(stream);

            d_potrf_work = nullptr;
            d_stamp = nullptr;
            d_g2p = nullptr;
            dF = nullptr;
            solver = nullptr;
            blas = nullptr;
            stream = nullptr;
            potrf_lwork = 0;
            epoch = 1;
        }
    };

    static bool ensure_potrf_workspace(PerStream &ps, int nscol, int lda)
    {
        int lwork = 0;
        cusolverStatus_t s = cusolverDnDpotrf_bufferSize(ps.solver, CUBLAS_FILL_MODE_LOWER, nscol, ps.dF, lda, &lwork);
        if (s != CUSOLVER_STATUS_SUCCESS)
        {
            std::cerr << "cuSOLVER Error: potrf_bufferSize status=" << int(s) << "\n";
            return false;
        }
        if (lwork > ps.potrf_lwork)
        {
            if (ps.d_potrf_work)
            {
                cudaError_t e = cudaFree(ps.d_potrf_work);
                if (e != cudaSuccess)
                {
                    std::cerr << "CUDA Error: cudaFree potrf_work " << cudaGetErrorString(e) << "\n";
                    return false;
                }
            }
            cudaError_t e = cudaMalloc(&ps.d_potrf_work, (size_t)lwork * sizeof(double));
            if (e != cudaSuccess)
            {
                std::cerr << "CUDA Error: cudaMalloc potrf_work " << cudaGetErrorString(e) << "\n";
                return false;
            }
            ps.potrf_lwork = lwork;
        }
        return true;
    }

    // ---------------- Host entry ----------------

    numeric::SuperNumeric factorize_supernodal_ll_gpu(
        const ichol::matrix::CscMatrix<double> &A,
        const symbolic::SupernodalLLPlan &plan)
    {
        numeric::SuperNumeric out;
        out.ok = true;
        out.fail_snode = -1;
        out.fail_col_in_snode = -1;
        out.sym = plan.sym;
        out.x.assign((size_t)plan.sym.px.back(), 0.0);

        const int n = A.num_rows; // assume SPD => num_rows == num_cols

        int *d_col_ptr = nullptr, *d_row_ind = nullptr;
        double *d_val = nullptr;
        int *d_s = nullptr;

        double *d_x = nullptr;
        double *d_update_pool = nullptr;
        int *d_potrf_info_all = nullptr;

        int n_streams = 4;
        std::vector<PerStream> ps;

        const int nsuper = (int)plan.sym.super.size() - 1;
        std::vector<size_t> update_offsets((size_t)nsuper + 1, 0);

        const std::vector<std::vector<int>> *ptr_buckets = &plan.buckets;
        std::vector<std::vector<int>> fallback_buckets;

        std::vector<int> h_potrf_info((size_t)nsuper, 0);

        size_t max_f_elems = 0;

        // ---- static device inputs ----
        CUDA_TRY(cudaMalloc(&d_col_ptr, (size_t)(A.num_cols + 1) * sizeof(int)));
        CUDA_TRY(cudaMalloc(&d_row_ind, (size_t)A.nnz * sizeof(int)));
        CUDA_TRY(cudaMalloc(&d_val, (size_t)A.nnz * sizeof(double)));

        CUDA_TRY(cudaMemcpy(d_col_ptr, A.col_ptr.data(), (size_t)(A.num_cols + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_row_ind, A.row_ind.data(), (size_t)A.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_val, A.values.data(), (size_t)A.nnz * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_TRY(cudaMalloc(&d_s, (size_t)plan.sym.s.size() * sizeof(int)));
        CUDA_TRY(cudaMemcpy(d_s, plan.sym.s.data(), (size_t)plan.sym.s.size() * sizeof(int), cudaMemcpyHostToDevice));

        // ---- x ----
        {
            size_t x_size = (size_t)plan.sym.px.back();
            CUDA_TRY(cudaMalloc(&d_x, x_size * sizeof(double)));
            CUDA_TRY(cudaMemset(d_x, 0, x_size * sizeof(double)));
        }

        // ---- potrf info ----
        CUDA_TRY(cudaMalloc(&d_potrf_info_all, (size_t)nsuper * sizeof(int)));
        CUDA_TRY(cudaMemset(d_potrf_info_all, 0, (size_t)nsuper * sizeof(int)));

        // ---- update offsets + max frontal ----
        for (int k = 0; k < nsuper; ++k)
        {
            int pi0 = plan.sym.pi[(size_t)k];
            int pi1 = plan.sym.pi[(size_t)k + 1];
            int scol = plan.sym.super[(size_t)k];
            int ecol = plan.sym.super[(size_t)k + 1];

            int nsrow = pi1 - pi0;
            int nscol = ecol - scol;
            int nupd = nsrow - nscol;

            update_offsets[(size_t)k + 1] = update_offsets[(size_t)k];
            if (nupd > 0)
                update_offsets[(size_t)k + 1] += (size_t)nupd * (size_t)nupd;

            size_t f_elems = (size_t)nsrow * (size_t)nsrow;
            if (f_elems > max_f_elems)
                max_f_elems = f_elems;
        }

        if (update_offsets.back() > 0)
        {
            CUDA_TRY(cudaMalloc(&d_update_pool, update_offsets.back() * sizeof(double)));
        }

        // ---- buckets fallback ----
        if (plan.buckets.empty())
        {
            fallback_buckets.resize(1);
            fallback_buckets[0].reserve((size_t)nsuper);
            for (int k = 0; k < nsuper; ++k)
                fallback_buckets[0].push_back(k);
            ptr_buckets = &fallback_buckets;
        }

        // ---- per-stream init ----
        ps.resize((size_t)n_streams);
        for (int i = 0; i < n_streams; ++i)
        {
            CUDA_TRY(cudaStreamCreateWithFlags(&ps[(size_t)i].stream, cudaStreamNonBlocking));

            CUBLAS_TRY(cublasCreate(&ps[(size_t)i].blas));
            CUSOLVER_TRY(cusolverDnCreate(&ps[(size_t)i].solver));

            CUBLAS_TRY(cublasSetStream(ps[(size_t)i].blas, ps[(size_t)i].stream));
            CUSOLVER_TRY(cusolverDnSetStream(ps[(size_t)i].solver, ps[(size_t)i].stream));

            CUDA_TRY(cudaMalloc(&ps[(size_t)i].dF, max_f_elems * sizeof(double)));

            CUDA_TRY(cudaMalloc(&ps[(size_t)i].d_g2p, (size_t)n * sizeof(int)));
            CUDA_TRY(cudaMalloc(&ps[(size_t)i].d_stamp, (size_t)n * sizeof(int)));
            CUDA_TRY(cudaMemsetAsync(ps[(size_t)i].d_stamp, 0, (size_t)n * sizeof(int), ps[(size_t)i].stream));
        }
        for (int i = 0; i < n_streams; ++i)
            CUDA_TRY(cudaStreamSynchronize(ps[(size_t)i].stream));

        // ---- main execution ----
        for (const auto &nodes : *ptr_buckets)
        {
            if (!out.ok)
                break;

            for (size_t ii = 0; ii < nodes.size(); ++ii)
            {
                int k = nodes[ii];
                int sid = (int)(ii % (size_t)n_streams);
                PerStream &S = ps[(size_t)sid];

                int pi0 = plan.sym.pi[(size_t)k];
                int pi1 = plan.sym.pi[(size_t)k + 1];
                int scol = plan.sym.super[(size_t)k];
                int ecol = plan.sym.super[(size_t)k + 1];
                int px0 = plan.sym.px[(size_t)k];

                int nsrow = pi1 - pi0;
                int nscol = ecol - scol;
                int nupd = nsrow - nscol;

                int epoch = ++S.epoch;
                if (epoch <= 0)
                {
                    S.epoch = 1;
                    epoch = 1;
                    CUDA_TRY(cudaMemsetAsync(S.d_stamp, 0, (size_t)n * sizeof(int), S.stream));
                }

                // build map for this supernode
                {
                    int threads = 256;
                    int blocks = (nsrow + threads - 1) / threads;
                    k_build_g2p_stamp<<<blocks, threads, 0, S.stream>>>(pi0, nsrow, d_s, S.d_g2p, S.d_stamp, epoch);
                    CUDA_TRY(cudaGetLastError());
                }

                // zero frontal
                CUDA_TRY(cudaMemsetAsync(S.dF, 0, (size_t)nsrow * (size_t)nsrow * sizeof(double), S.stream));

                // assemble A
                {
                    int threads = 128;
                    int blocks = (nscol + threads - 1) / threads;
                    k_assemble_A<<<blocks, threads, 0, S.stream>>>(
                        scol, ecol, d_col_ptr, d_row_ind, d_val,
                        S.d_g2p, S.d_stamp, epoch,
                        S.dF, nsrow);
                    CUDA_TRY(cudaGetLastError());
                }

                // assemble children updates
                for (int child : plan.children[(size_t)k])
                {
                    int c_pi0 = plan.sym.pi[(size_t)child];
                    int c_pi1 = plan.sym.pi[(size_t)child + 1];
                    int c_scol = plan.sym.super[(size_t)child];
                    int c_ecol = plan.sym.super[(size_t)child + 1];
                    int c_nsrow = c_pi1 - c_pi0;
                    int c_nscol = c_ecol - c_scol;
                    int c_nupd = c_nsrow - c_nscol;
                    if (c_nupd <= 0)
                        continue;

                    double *child_S = d_update_pool + update_offsets[(size_t)child];
                    const int *c_update_idx = d_s + c_pi0 + c_nscol;

                    dim3 block(16, 16);
                    dim3 grid((c_nupd + 15) / 16, (c_nupd + 15) / 16);
                    k_assemble_child<<<grid, block, 0, S.stream>>>(
                        c_nupd, c_update_idx, child_S,
                        S.d_g2p, S.d_stamp, epoch,
                        S.dF, nsrow);
                    CUDA_TRY(cudaGetLastError());
                }

                // POTRF
                if (!ensure_potrf_workspace(S, nscol, nsrow))
                {
                    out.ok = false;
                    goto cleanup;
                }
                {
                    int *d_info_k = d_potrf_info_all + k;
                    CUSOLVER_TRY(cusolverDnDpotrf(
                        S.solver,
                        CUBLAS_FILL_MODE_LOWER,
                        nscol,
                        S.dF, nsrow,
                        S.d_potrf_work, S.potrf_lwork,
                        d_info_k));
                }

                // TRSM
                if (nupd > 0)
                {
                    const double alpha = 1.0;
                    double *dL21 = S.dF + (size_t)nscol;
                    CUBLAS_TRY(cublasDtrsm(
                        S.blas,
                        CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT,
                        nupd, nscol,
                        &alpha,
                        S.dF, nsrow,
                        dL21, nsrow));
                }

                // store factors
                {
                    dim3 block(16, 16);
                    dim3 grid((nscol + 15) / 16, (nsrow + 15) / 16);
                    k_store_factors_2d<<<grid, block, 0, S.stream>>>(nsrow, nscol, S.dF, nsrow, d_x, px0);
                    CUDA_TRY(cudaGetLastError());
                }

                // update S_k
                if (nupd > 0)
                {
                    double *Sk = d_update_pool + update_offsets[(size_t)k];

                    // pack F22 -> Sk (lower)
                    {
                        dim3 block(16, 16);
                        dim3 grid((nupd + 15) / 16, (nupd + 15) / 16);
                        k_pack_F22_to_S_lower<<<grid, block, 0, S.stream>>>(nupd, nsrow, nscol, S.dF, Sk);
                        CUDA_TRY(cudaGetLastError());
                    }

                    const double alpha = -1.0;
                    const double beta = 1.0;
                    const double *dL21 = S.dF + (size_t)nscol;

                    CUBLAS_TRY(cublasDsyrk(
                        S.blas,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N,
                        nupd, nscol,
                        &alpha,
                        dL21, nsrow,
                        &beta,
                        Sk, nupd));
                }
            }

            // bucket barrier
            for (int i = 0; i < n_streams; ++i)
                CUDA_TRY(cudaStreamSynchronize(ps[(size_t)i].stream));

            // potrf failure check (only nodes in this bucket)
            CUDA_TRY(cudaMemcpy(h_potrf_info.data(), d_potrf_info_all, (size_t)nsuper * sizeof(int), cudaMemcpyDeviceToHost));
            for (size_t ii = 0; ii < nodes.size(); ++ii)
            {
                int k = nodes[ii];
                int info = h_potrf_info[(size_t)k];
                if (info != 0)
                {
                    out.ok = false;
                    out.fail_snode = k;
                    out.fail_col_in_snode = std::max(0, info - 1);
                    goto cleanup;
                }
            }
        }

        // copy x back
        if (out.ok)
        {
            CUDA_TRY(cudaMemcpy(out.x.data(), d_x, (size_t)plan.sym.px.back() * sizeof(double), cudaMemcpyDeviceToHost));
        }

    cleanup:
        if (d_potrf_info_all)
            cudaFree(d_potrf_info_all);
        if (d_update_pool)
            cudaFree(d_update_pool);
        if (d_x)
            cudaFree(d_x);

        if (d_s)
            cudaFree(d_s);
        if (d_val)
            cudaFree(d_val);
        if (d_row_ind)
            cudaFree(d_row_ind);
        if (d_col_ptr)
            cudaFree(d_col_ptr);

        for (auto &r : ps)
            r.destroy();

        return out;
    }

} // namespace ichol::numeric
