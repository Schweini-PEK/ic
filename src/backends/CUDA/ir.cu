#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ichol/pcg.hpp"

#define CUDA_CHECK(call)                                                              \
    do                                                                                \
    {                                                                                 \
        cudaError_t _e = (call);                                                      \
        if (_e != cudaSuccess)                                                        \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
    } while (0)

#define CUBLAS_CHECK(call)                                                                             \
    do                                                                                                 \
    {                                                                                                  \
        cublasStatus_t _s = (call);                                                                    \
        if (_s != CUBLAS_STATUS_SUCCESS)                                                               \
            throw std::runtime_error(std::string("cuBLAS Error at line ") + std::to_string(__LINE__) + \
                                     " status=" + std::to_string(static_cast<int>(_s)));               \
    } while (0)

#define CUSPARSE_CHECK(call)                            \
    do                                                  \
    {                                                   \
        cusparseStatus_t _s = (call);                   \
        if (_s != CUSPARSE_STATUS_SUCCESS)              \
            throw std::runtime_error("cuSPARSE Error"); \
    } while (0)

namespace ichol::solver
{
    template <typename T_L>
    PCGResult mpcg_low_storage_device(
        int n,
        int64_t nnzA,
        int *d_csrRowPtrA,
        int *d_csrColIndA,
        double *d_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const double *d_b,
        double *d_x,
        const PCGParams &params);

    template <typename T_L>
    PCGResult ir(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const IterativeRefinementParams &params)
    {
        using Clock = std::chrono::steady_clock;

        const int n = static_cast<int>(h_b.size());
        const int64_t nnzA = static_cast<int64_t>(h_valA.size());
        if (n <= 0)
            throw std::runtime_error("ir: system size must be positive");
        if (params.maxits <= 0)
            throw std::runtime_error("ir: maxits must be positive");
        if (params.tol <= 0.0)
            throw std::runtime_error("ir: tol must be positive");

        if (h_x.size() != static_cast<std::size_t>(n))
            h_x.assign(static_cast<std::size_t>(n), 0.0);

        const auto total_wall_start = Clock::now();

        cublasHandle_t cublas = nullptr;
        cusparseHandle_t cusparse = nullptr;
        cudaStream_t stream = nullptr;
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));

        int *d_rowPtrA = nullptr;
        int *d_colIndA = nullptr;
        double *d_valA = nullptr;
        double *d_b = nullptr;
        double *d_x = nullptr;
        double *d_r = nullptr;
        double *d_ax = nullptr;
        double *d_dx = nullptr;

        cusparseSpMatDescr_t matA = nullptr;
        cusparseDnVecDescr_t vecX = nullptr;
        cusparseDnVecDescr_t vecAx = nullptr;
        void *d_spmvBuf = nullptr;

        PCGResult result{};
        try
        {
            CUDA_CHECK(cudaMalloc(&d_rowPtrA, static_cast<size_t>(n + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_colIndA, static_cast<size_t>(nnzA) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_valA, static_cast<size_t>(nnzA) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(n) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_r, static_cast<size_t>(n) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_ax, static_cast<size_t>(n) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_dx, static_cast<size_t>(n) * sizeof(double)));

            CUDA_CHECK(cudaMemcpyAsync(d_rowPtrA, h_csrRowPtrA.data(), static_cast<size_t>(n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_colIndA, h_csrColIndA.data(), static_cast<size_t>(nnzA) * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_valA, h_valA.data(), static_cast<size_t>(nnzA) * sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), static_cast<size_t>(n) * sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_x, h_x.data(), static_cast<size_t>(n) * sizeof(double), cudaMemcpyHostToDevice, stream));

            CUSPARSE_CHECK(cusparseCreateCsr(
                &matA,
                n, n, nnzA,
                d_rowPtrA, d_colIndA, d_valA,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_64F));
            CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
            CUSPARSE_CHECK(cusparseCreateDnVec(&vecAx, n, d_ax, CUDA_R_64F));

            const double one = 1.0;
            const double zero = 0.0;
            const double minus_one = -1.0;

            size_t spmv_buf_size = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA,
                vecX,
                &zero,
                vecAx,
                CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &spmv_buf_size));
            CUDA_CHECK(cudaMalloc(&d_spmvBuf, spmv_buf_size));

            auto recompute_residual = [&](double &res_norm)
            {
                CUSPARSE_CHECK(cusparseSpMV(
                    cusparse,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one,
                    matA,
                    vecX,
                    &zero,
                    vecAx,
                    CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf));
                CUDA_CHECK(cudaMemcpyAsync(d_r, d_b, static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToDevice, stream));
                CUBLAS_CHECK(cublasDaxpy(cublas, n, &minus_one, d_ax, 1, d_r, 1));
                CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res_norm));
            };

            double bnorm = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas, n, d_b, 1, &bnorm));
            if (bnorm == 0.0)
                bnorm = 1.0;

            double current_res_norm = 0.0;
            recompute_residual(current_res_norm);

            result.relResiduals.reserve(static_cast<std::size_t>(params.maxits) + 1);
            result.relResiduals.push_back(current_res_norm / bnorm);

            const auto iter_wall_start = Clock::now();
            bool converged = false;
            for (int iter = 0; iter < params.maxits; ++iter)
            {
                if (current_res_norm <= params.tol * bnorm)
                {
                    result.iterations = iter;
                    result.finalRes = current_res_norm;
                    converged = true;
                    break;
                }

                CUDA_CHECK(cudaMemsetAsync(d_dx, 0, static_cast<size_t>(n) * sizeof(double), stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                const PCGResult inner = mpcg_low_storage_device<T_L>(
                    n,
                    nnzA,
                    d_rowPtrA,
                    d_colIndA,
                    d_valA,
                    preconds,
                    d_r,
                    d_dx,
                    params.inner_params);

                result.timing.preconditioner_apply_ms += inner.timing.preconditioner_apply_ms;
                result.timing.orthogonalization_ms += inner.timing.orthogonalization_ms;
                result.timing.spmm_ms += inner.timing.spmm_ms;
                result.timing.dense_ms += inner.timing.dense_ms;
                result.timing.residual_reset_ms += inner.timing.residual_reset_ms;
                result.timing.other_iter_ms += inner.timing.other_iter_ms;

                CUBLAS_CHECK(cublasDaxpy(cublas, n, &one, d_dx, 1, d_x, 1));
                recompute_residual(current_res_norm);

                result.iterations = iter + 1;
                result.relResiduals.push_back(current_res_norm / bnorm);
            }

            const auto iter_wall_end = Clock::now();
            if (!converged)
                result.finalRes = current_res_norm;

            const auto finalize_wall_start = Clock::now();
            CUDA_CHECK(cudaMemcpyAsync(h_x.data(), d_x, static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto finalize_wall_end = Clock::now();

            result.timing.total_ms =
                std::chrono::duration<double, std::milli>(finalize_wall_end - total_wall_start).count();
            result.timing.setup_ms =
                std::chrono::duration<double, std::milli>(iter_wall_start - total_wall_start).count();
            result.timing.iter_ms =
                std::chrono::duration<double, std::milli>(iter_wall_end - iter_wall_start).count();
            result.timing.finalize_ms =
                std::chrono::duration<double, std::milli>(finalize_wall_end - finalize_wall_start).count();
        }
        catch (...)
        {
            if (d_spmvBuf)
                cudaFree(d_spmvBuf);
            if (vecAx)
                cusparseDestroyDnVec(vecAx);
            if (vecX)
                cusparseDestroyDnVec(vecX);
            if (matA)
                cusparseDestroySpMat(matA);
            if (d_dx)
                cudaFree(d_dx);
            if (d_ax)
                cudaFree(d_ax);
            if (d_r)
                cudaFree(d_r);
            if (d_x)
                cudaFree(d_x);
            if (d_b)
                cudaFree(d_b);
            if (d_valA)
                cudaFree(d_valA);
            if (d_colIndA)
                cudaFree(d_colIndA);
            if (d_rowPtrA)
                cudaFree(d_rowPtrA);
            if (stream)
                cudaStreamDestroy(stream);
            if (cusparse)
                cusparseDestroy(cusparse);
            if (cublas)
                cublasDestroy(cublas);
            throw;
        }

        if (d_spmvBuf)
            CUDA_CHECK(cudaFree(d_spmvBuf));
        if (vecAx)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecAx));
        if (vecX)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
        if (matA)
            CUSPARSE_CHECK(cusparseDestroySpMat(matA));
        if (d_dx)
            CUDA_CHECK(cudaFree(d_dx));
        if (d_ax)
            CUDA_CHECK(cudaFree(d_ax));
        if (d_r)
            CUDA_CHECK(cudaFree(d_r));
        if (d_x)
            CUDA_CHECK(cudaFree(d_x));
        if (d_b)
            CUDA_CHECK(cudaFree(d_b));
        if (d_valA)
            CUDA_CHECK(cudaFree(d_valA));
        if (d_colIndA)
            CUDA_CHECK(cudaFree(d_colIndA));
        if (d_rowPtrA)
            CUDA_CHECK(cudaFree(d_rowPtrA));

        CUSPARSE_CHECK(cusparseDestroy(cusparse));
        CUBLAS_CHECK(cublasDestroy(cublas));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return result;
    }

    template PCGResult ir<double>(
        const std::vector<int> &,
        const std::vector<int> &,
        const std::vector<double> &,
        const std::vector<ichol::precond::PrecondApply> &,
        const std::vector<double> &,
        std::vector<double> &,
        const IterativeRefinementParams &);
} // namespace ichol::solver
