// src/solver/mpcg_cuda.cu

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cstring>

#include "ichol/pcg.hpp"
#include "ichol/preconditioner.hpp"

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t _e = (call);                                   \
        if (_e != cudaSuccess)                                     \
        {                                                          \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(_e));      \
        }                                                          \
    } while (0)

#define CUSPARSE_CHECK(call)                            \
    do                                                  \
    {                                                   \
        cusparseStatus_t _s = (call);                   \
        if (_s != CUSPARSE_STATUS_SUCCESS)              \
        {                                               \
            throw std::runtime_error("cuSPARSE error"); \
        }                                               \
    } while (0)

#define CUBLAS_CHECK(call)                            \
    do                                                \
    {                                                 \
        cublasStatus_t _s = (call);                   \
        if (_s != CUBLAS_STATUS_SUCCESS)              \
        {                                             \
            throw std::runtime_error("cuBLAS error"); \
        }                                             \
    } while (0)

#define CUSOLVER_CHECK(call)                            \
    do                                                  \
    {                                                   \
        cusolverStatus_t _s = (call);                   \
        if (_s != CUSOLVER_STATUS_SUCCESS)              \
        {                                               \
            throw std::runtime_error("cuSOLVER error"); \
        }                                               \
    } while (0)

// ------------------------- Device kernels -------------------------

__global__ void k_row_scaling(double *T, const double *S_inv, int k, int nrhs)
{
    int row = threadIdx.x;
    if (row < k)
    {
        double s = S_inv[row];
        for (int col = 0; col < nrhs; ++col)
        {
            T[row + col * k] *= s;
        }
    }
}

// ------------------------- SVD Pseudo-Inverse (cuSOLVER) -------------------------
// Compute X = pinv(G) * B using Singular Value Decomposition
static void pinv_svd_cuda(cusolverDnHandle_t cusolver,
                          cublasHandle_t cublas,
                          const double *d_G, int k,
                          const double *d_B, int nrhs,
                          double *d_X,
                          cudaStream_t stream,
                          double rcond = 1e-15)
{
    // cusolverDnDgesvd overwrites input matrix, create a working copy
    double *d_G_copy;
    CUDA_CHECK(cudaMallocAsync(&d_G_copy, k * k * sizeof(double), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_G_copy, d_G, k * k * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    double *d_S, *d_U, *d_VT;
    CUDA_CHECK(cudaMallocAsync(&d_S, k * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_U, k * k * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_VT, k * k * sizeof(double), stream));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolver, k, k, &lwork));
    double *d_work;
    CUDA_CHECK(cudaMallocAsync(&d_work, lwork * sizeof(double), stream));

    int *d_info;
    CUDA_CHECK(cudaMallocAsync(&d_info, sizeof(int), stream));

    // 1. Compute SVD: G = U * S * VT
    CUSOLVER_CHECK(cusolverDnDgesvd(cusolver, 'A', 'A', k, k, d_G_copy, k, d_S, d_U, k, d_VT, k, d_work, lwork, nullptr, d_info));

    // 2. Threshold singular values
    std::vector<double> h_S(k);
    CUDA_CHECK(cudaMemcpyAsync(h_S.data(), d_S, k * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double max_s = h_S[0]; // S is sorted descending
    double thresh = rcond * max_s;
    std::vector<double> h_S_inv(k, 0.0);
    for (int i = 0; i < k; ++i)
    {
        if (h_S[i] > thresh)
            h_S_inv[i] = 1.0 / h_S[i];
    }

    double *d_S_inv;
    CUDA_CHECK(cudaMallocAsync(&d_S_inv, k * sizeof(double), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_S_inv, h_S_inv.data(), k * sizeof(double), cudaMemcpyHostToDevice, stream));

    // 3. Solve: X = V * S_inv * U^T * B
    double *d_T1;
    CUDA_CHECK(cudaMallocAsync(&d_T1, k * nrhs * sizeof(double), stream));

    double one = 1.0, zero = 0.0;

    // T1 = U^T * B
    CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, nrhs, k, &one, d_U, k, d_B, k, &zero, d_T1, k));

    // T1 = S_inv * T1
    k_row_scaling<<<1, k, 0, stream>>>(d_T1, d_S_inv, k, nrhs);

    // X = V * T1 (cuSOLVER provides VT, so V is VT^T)
    CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, nrhs, k, &one, d_VT, k, d_T1, k, &zero, d_X, k));

    // Cleanup
    CUDA_CHECK(cudaFreeAsync(d_G_copy, stream));
    CUDA_CHECK(cudaFreeAsync(d_S, stream));
    CUDA_CHECK(cudaFreeAsync(d_U, stream));
    CUDA_CHECK(cudaFreeAsync(d_VT, stream));
    CUDA_CHECK(cudaFreeAsync(d_work, stream));
    CUDA_CHECK(cudaFreeAsync(d_info, stream));
    CUDA_CHECK(cudaFreeAsync(d_S_inv, stream));
    CUDA_CHECK(cudaFreeAsync(d_T1, stream));
}

namespace ichol::solver
{
    template <typename T_L>
    void mpcg(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        int maxits,
        double tol,
        int restart,
        int &iterations,
        double &finalRes)
    {
        const int n = static_cast<int>(h_csrRowPtrA.size()) - 1;
        if (n <= 0)
            throw std::runtime_error("mpcg: invalid n");
        if ((int)h_b.size() != n)
            throw std::runtime_error("mpcg: b size mismatch");
        if ((int)h_x.size() != n)
            throw std::runtime_error("mpcg: x size mismatch");
        if ((int)h_csrColIndA.size() != (int)h_valA.size())
            throw std::runtime_error("mpcg: A nnz mismatch");

        const int64_t nnzA = (int64_t)h_valA.size();
        const int k = static_cast<int>(preconds.size());

        if (k <= 0)
            throw std::runtime_error("mpcg: need at least one preconditioner");
        if (maxits <= 0)
            throw std::runtime_error("mpcg: maxits must be > 0");
        if (tol <= 0.0)
            throw std::runtime_error("mpcg: tol must be > 0");

        const int m = (restart <= 0) ? maxits : restart;

        cublasHandle_t cublas = nullptr;
        cusparseHandle_t cusparse = nullptr;
        cusolverDnHandle_t cusolver = nullptr;

        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));

        int *d_rowPtrA = nullptr;
        int *d_colIndA = nullptr;
        double *d_valA = nullptr;
        double *d_b = nullptr, *d_x = nullptr, *d_r = nullptr, *d_tmp = nullptr;

        CUDA_CHECK(cudaMalloc((void **)&d_rowPtrA, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_colIndA, nnzA * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_valA, nnzA * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_r, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_tmp, n * sizeof(double)));

        CUDA_CHECK(cudaMemcpyAsync(d_rowPtrA, h_csrRowPtrA.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_colIndA, h_csrColIndA.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_valA, h_valA.data(), nnzA * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_x, h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));

        cusparseSpMatDescr_t matA = nullptr;
        CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, n, nnzA, d_rowPtrA, d_colIndA, d_valA,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        cusparseDnVecDescr_t vecX = nullptr, vecTmp = nullptr;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecTmp, n, d_tmp, CUDA_R_64F));

        CUDA_CHECK(cudaMemcpyAsync(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice, stream));

        size_t spmv_bufSize = 0;
        void *d_spmvBuf = nullptr;
        double one = 1.0, zero = 0.0, minus_one = -1.0;

        CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX,
                                               &zero, vecTmp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_bufSize));
        CUDA_CHECK(cudaMalloc(&d_spmvBuf, spmv_bufSize));

        CUSPARSE_CHECK(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX,
                                    &zero, vecTmp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf));

        CUBLAS_CHECK(cublasDaxpy(cublas, n, &minus_one, d_tmp, 1, d_r, 1));

        double bnorm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublas, n, d_b, 1, &bnorm));
        if (bnorm == 0.0)
            bnorm = 1.0;

        double *d_Znew = nullptr, *d_Pnew = nullptr, *d_Wnew = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_Znew, (size_t)n * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Pnew, (size_t)n * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Wnew, (size_t)n * k * sizeof(double)));

        double *d_P_hist = nullptr, *d_W_hist = nullptr, *d_G_hist = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_P_hist, (size_t)m * n * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_W_hist, (size_t)m * n * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_G_hist, (size_t)m * k * k * sizeof(double)));

        double *d_C = nullptr, *d_Gnew = nullptr, *d_rhs = nullptr, *d_Y = nullptr, *d_alpha = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_C, (size_t)k * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Gnew, (size_t)k * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_rhs, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Y, (size_t)k * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_alpha, (size_t)k * sizeof(double)));

        cusparseDnMatDescr_t dnB = nullptr, dnC = nullptr;
        CUSPARSE_CHECK(cusparseCreateDnMat(&dnB, n, k, n, d_Pnew, CUDA_R_64F, CUSPARSE_ORDER_COL));
        CUSPARSE_CHECK(cusparseCreateDnMat(&dnC, n, k, n, d_Wnew, CUDA_R_64F, CUSPARSE_ORDER_COL));

        size_t spmm_bufSize = 0;
        void *d_spmmBuf = nullptr;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &one, matA, dnB, &zero, dnC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &spmm_bufSize));
        CUDA_CHECK(cudaMalloc(&d_spmmBuf, spmm_bufSize));

        CUSPARSE_CHECK(cusparseSpMM_preprocess(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &one, matA, dnB, &zero, dnC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmmBuf));

        iterations = 0;
        finalRes = 0.0;

        for (int iter = 0; iter < maxits; ++iter)
        {
            double res = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res));

            if (res <= tol * bnorm)
            {
                iterations = iter;
                finalRes = res;
                break;
            }

            for (int t = 0; t < k; ++t)
            {
                double *d_z_col = d_Znew + (size_t)t * n;
                preconds[t].apply(preconds[t].ctx, d_r, d_z_col, n, stream);
            }

            CUDA_CHECK(cudaMemcpyAsync(d_Pnew, d_Znew, (size_t)n * k * sizeof(double), cudaMemcpyDeviceToDevice, stream));

            //
            const int hist_count = std::min(iter, m);
            for (int j = iter - hist_count; j < iter; ++j)
            {
                const int slot = j % m;

                double *d_Pj = d_P_hist + (size_t)slot * n * k;
                double *d_Wj = d_W_hist + (size_t)slot * n * k;
                double *d_Gj = d_G_hist + (size_t)slot * k * k;

                // C = Wj^T * Pnew (Modified Gram-Schmidt update using Pnew instead of Znew)
                CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n,
                                         &one, d_Wj, n, d_Pnew, n, &zero, d_C, k));

                // Y = pinv(Gj) * C
                pinv_svd_cuda(cusolver, cublas, d_Gj, k, d_C, k, d_Y, stream, 1e-15);

                // Pnew = Pnew - Pj * Y
                CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, k, k,
                                         &minus_one, d_Pj, n, d_Y, k, &one, d_Pnew, n));
            }

            CUSPARSE_CHECK(cusparseDnMatSetValues(dnB, d_Pnew));
            CUSPARSE_CHECK(cusparseDnMatSetValues(dnC, d_Wnew));
            CUSPARSE_CHECK(cusparseSpMM(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, dnB, &zero, dnC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmmBuf));

            // Gnew = Pnew^T * Wnew
            CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n,
                                     &one, d_Pnew, n, d_Wnew, n, &zero, d_Gnew, k));

            // rhs = Pnew^T * r
            CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_T, n, k,
                                     &one, d_Pnew, n, d_r, 1, &zero, d_rhs, 1));

            // alpha = pinv(Gnew) * rhs
            pinv_svd_cuda(cusolver, cublas, d_Gnew, k, d_rhs, 1, d_alpha, stream, 1e-15);

            // x = x + Pnew * alpha
            CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n, k,
                                     &one, d_Pnew, n, d_alpha, 1, &one, d_x, 1));

            // r = r - Wnew * alpha
            CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n, k,
                                     &minus_one, d_Wnew, n, d_alpha, 1, &one, d_r, 1));

            const int slot = iter % m;
            CUDA_CHECK(cudaMemcpyAsync(d_P_hist + (size_t)slot * n * k, d_Pnew, (size_t)n * k * sizeof(double), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_W_hist + (size_t)slot * n * k, d_Wnew, (size_t)n * k * sizeof(double), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_G_hist + (size_t)slot * k * k, d_Gnew, (size_t)k * k * sizeof(double), cudaMemcpyDeviceToDevice, stream));

            iterations = iter + 1;

            if (iter == maxits - 1)
            {
                double res_end = 0.0;
                CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res_end));
                finalRes = res_end;
            }
        }

        if (finalRes == 0.0)
        {
            double res_end = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res_end));
            finalRes = res_end;
        }

        CUDA_CHECK(cudaMemcpyAsync(h_x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (dnB)
            CUSPARSE_CHECK(cusparseDestroyDnMat(dnB));
        if (dnC)
            CUSPARSE_CHECK(cusparseDestroyDnMat(dnC));
        if (vecX)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
        if (vecTmp)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecTmp));
        if (matA)
            CUSPARSE_CHECK(cusparseDestroySpMat(matA));

        if (d_spmvBuf)
            CUDA_CHECK(cudaFree(d_spmvBuf));
        if (d_spmmBuf)
            CUDA_CHECK(cudaFree(d_spmmBuf));

        CUDA_CHECK(cudaFree(d_rowPtrA));
        CUDA_CHECK(cudaFree(d_colIndA));
        CUDA_CHECK(cudaFree(d_valA));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaFree(d_tmp));

        CUDA_CHECK(cudaFree(d_Znew));
        CUDA_CHECK(cudaFree(d_Pnew));
        CUDA_CHECK(cudaFree(d_Wnew));

        CUDA_CHECK(cudaFree(d_P_hist));
        CUDA_CHECK(cudaFree(d_W_hist));
        CUDA_CHECK(cudaFree(d_G_hist));

        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_Gnew));
        CUDA_CHECK(cudaFree(d_rhs));
        CUDA_CHECK(cudaFree(d_Y));
        CUDA_CHECK(cudaFree(d_alpha));

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUSPARSE_CHECK(cusparseDestroy(cusparse));
        CUBLAS_CHECK(cublasDestroy(cublas));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    }

    template void mpcg<double>(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        int maxits,
        double tol,
        int restart,
        int &iterations,
        double &finalRes);
} // namespace ichol::solver