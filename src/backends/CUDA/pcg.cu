#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <random>
#include <type_traits>
#include "ichol/half.hpp"

// ---------------------- mixed-precision helpers ----------------------
template <typename To, typename From>
__global__ void cast_vec(int n, const From *__restrict__ in, To *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = static_cast<To>(in[i]);
}

/**
 * Element-Wise product
 */
__global__ void ew_mul(int n, const double *a, const double *b, double *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * b[i];
}

__global__ void diag_sub(int n,
                         const int *__restrict__ rowPtr,
                         const double *__restrict__ val,
                         const double *__restrict__ p,
                         double *__restrict__ q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double di = val[rowPtr[i + 1] - 1]; // last entry in row = diagonal
        q[i] -= di * p[i];
    }
}

/**
 * q_i \leftarrow q_i - A_{ii} p_i
 */
__global__ void diag_sub_from_diag(int n,
                                   const double *__restrict__ diagA,
                                   const double *__restrict__ p,
                                   double *__restrict__ q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        q[i] -= diagA[i] * p[i];
}

#define CUDA_CHECK(call)                                                \
    do                                                                  \
    {                                                                   \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess)                                         \
        {                                                               \
            std::cerr << "CUDA error " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort();                                               \
        }                                                               \
    } while (0)

#define CUBLAS_CHECK(call)                                              \
    do                                                                  \
    {                                                                   \
        cublasStatus_t st = (call);                                     \
        if (st != CUBLAS_STATUS_SUCCESS)                                \
        {                                                               \
            std::cerr << "cuBLAS error " << st                          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort();                                               \
        }                                                               \
    } while (0)

#define CUSPARSE_CHECK(call)                                            \
    do                                                                  \
    {                                                                   \
        cusparseStatus_t st = (call);                                   \
        if (st != CUSPARSE_STATUS_SUCCESS)                              \
        {                                                               \
            std::cerr << "cuSPARSE error " << st                        \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort();                                               \
        }                                                               \
    } while (0)

namespace ichol
{
    template <typename T_L>
    void icPreconditionedCG_GPU(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<T_L> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        int &iterations,
        double &finalRes)
    {
        /*
        I. Allocate A, L on GPU
        */
        cusparseHandle_t cusparseHandle = nullptr;
        CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

        cublasHandle_t cublasHandle = nullptr;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        const int n = static_cast<int>(h_csrRowPtrA.size()) - 1;
        const int nnzA = static_cast<int>(h_valA.size());
        const int nnzL = static_cast<int>(h_valL.size());

        std::vector<double> h_diagA(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            h_diagA[i] = h_valA[h_csrRowPtrA[i + 1] - 1];
        }
        double *d_diagA = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_diagA, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_diagA, h_diagA.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        // Copy A to device
        int *d_csrRowPtrA = nullptr, *d_csrColIndA = nullptr;
        double *d_valA = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtrA, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_csrColIndA, nnzA * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_valA, nnzA * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndA, h_csrColIndA.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valA, h_valA.data(), nnzA * sizeof(double), cudaMemcpyHostToDevice));

        cusparseSpMatDescr_t spMatA = nullptr;
        CUSPARSE_CHECK(cusparseCreateCsr(&spMatA, n, n, nnzA,
                                         d_csrRowPtrA, d_csrColIndA, d_valA,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        // Copy L to device
        int *d_csrRowPtrL = nullptr, *d_csrColIndL = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtrL, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_csrColIndL, nnzL * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_csrRowPtrL, h_csrRowPtrL.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndL, h_csrColIndL.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));

        // ---------------------- Preconditioner precision selection ----------------------
        // Outer PCG stays FP64; only the preconditioner application (SpSV) uses the selected work precision.
        constexpr bool L_is_fp64 = std::is_same<T_L, double>::value;
        constexpr bool L_is_fp32 = std::is_same<T_L, float>::value;
        constexpr bool L_is_fp16 = !L_is_fp64 && !L_is_fp32; // (half_float::half)

        using PrecWork = typename std::conditional<L_is_fp64, double, float>::type;
        const cudaDataType precWork_dataType = L_is_fp64 ? CUDA_R_64F : CUDA_R_32F;

        // For FP16 storage, cuSPARSE SpSV does not support FP16. Convert L once at setup to FP32.
        void *d_valL_storage = nullptr; // optional: original L values, if we keep them
        void *d_valLptr = nullptr;      // L values used by SpSV (PrecWork)

        if constexpr (L_is_fp64 || L_is_fp32)
        {
            CUDA_CHECK(cudaMalloc(&d_valLptr, nnzL * sizeof(T_L)));
            CUDA_CHECK(cudaMemcpy(d_valLptr, h_valL.data(), nnzL * sizeof(T_L), cudaMemcpyHostToDevice));
            d_valL_storage = d_valLptr;
        }
        else
        {
            std::vector<float> h_valL32(nnzL);
            for (int i = 0; i < nnzL; ++i)
                h_valL32[i] = static_cast<float>(h_valL[i]);
            CUDA_CHECK(cudaMalloc(&d_valLptr, nnzL * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_valLptr, h_valL32.data(), nnzL * sizeof(float), cudaMemcpyHostToDevice));
            d_valL_storage = nullptr; // not needed on device for SpSV
        }

        cusparseSpMatDescr_t spMatL = nullptr;
        CUSPARSE_CHECK(cusparseCreateCsr(&spMatL, n, n, nnzL,
                                         d_csrRowPtrL, d_csrColIndL, d_valLptr,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, precWork_dataType));

        // Tell cuSPARSE that L is triangular, lower fill, with a non-unit diagonal
        cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(spMatL, CUSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode)));
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(spMatL, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType)));

        // 3) Allocate vectors x, b, etc. in double
        double *d_x = nullptr, *d_b = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double))); // x=0 initial
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        // new: set up D
        double *d_D = nullptr;
        CUDA_CHECK(cudaMalloc(&d_D, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_D, h_D.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        double *d_Dr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Dr, n * sizeof(double)));

        double *d_b_orig = nullptr;
        CUDA_CHECK(cudaMalloc(&d_b_orig, n * sizeof(double)));

        int block = 256;
        int grid = (n + block - 1) / block;
        ew_mul<<<grid, block>>>(n, d_D, d_b, d_b_orig); // d_b_orig = D .* d_b

        /**
         * Note that \tilde{b} = D^{-1} b is passed into this func.
         * This computes the norm of original b for convergence check:
         */
        double bnorm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_b_orig, 1, &bnorm));

        cudaFree(d_b_orig);

        // 4) Setup Triangular Solve descriptors for L & L^T
        cusparseSpSVDescr_t spSVDescrFwd = nullptr, spSVDescrBwd = nullptr;
        CUSPARSE_CHECK(cusparseSpSV_createDescr(&spSVDescrFwd));
        CUSPARSE_CHECK(cusparseSpSV_createDescr(&spSVDescrBwd));

        // Dummy dense vectors to measure buffer sizes / run analysis in the same precision used by SpSV.
        PrecWork *d_dummyVec_spsv = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dummyVec_spsv, n * sizeof(PrecWork)));
        cusparseDnVecDescr_t dummyVecR = nullptr, dummyVecSol = nullptr;
        CUSPARSE_CHECK(cusparseCreateDnVec(&dummyVecR, n, (void *)d_dummyVec_spsv, precWork_dataType));
        CUSPARSE_CHECK(cusparseCreateDnVec(&dummyVecSol, n, (void *)d_dummyVec_spsv, precWork_dataType));

        size_t bufSizeFwd = 0, bufSizeBwd = 0;
        void *d_solveBufFwd = nullptr, *d_solveBufBwd = nullptr;
        const PrecWork alpha_one = static_cast<PrecWork>(1);

        CUSPARSE_CHECK(cusparseSpSV_bufferSize(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_one, spMatL,
            dummyVecR, dummyVecSol,
            precWork_dataType,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrFwd, &bufSizeFwd));
        CUDA_CHECK(cudaMalloc(&d_solveBufFwd, bufSizeFwd));

        CUSPARSE_CHECK(cusparseSpSV_analysis(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_one, spMatL,
            dummyVecR, dummyVecSol,
            precWork_dataType, CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrFwd, d_solveBufFwd));

        CUSPARSE_CHECK(cusparseSpSV_bufferSize(
            cusparseHandle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha_one, spMatL,
            dummyVecR, dummyVecSol,
            precWork_dataType,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrBwd, &bufSizeBwd));
        CUDA_CHECK(cudaMalloc(&d_solveBufBwd, bufSizeBwd));

        CUSPARSE_CHECK(cusparseSpSV_analysis(
            cusparseHandle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha_one, spMatL,
            dummyVecR, dummyVecSol,
            precWork_dataType, CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrBwd, d_solveBufBwd));

        CUSPARSE_CHECK(cusparseDestroyDnVec(dummyVecR));
        CUSPARSE_CHECK(cusparseDestroyDnVec(dummyVecSol));
        CUDA_CHECK(cudaFree(d_dummyVec_spsv));

        // For A * p
        cusparseDnVecDescr_t vecP_dev = nullptr, vecQ_dev = nullptr;
        void *d_dummyVec = nullptr; // (kept for backward compatibility with existing cleanup)
        double *d_p = nullptr, *d_q = nullptr, *d_r = nullptr, *d_z = nullptr;
        CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(double))); // outer z (FP64) used in dot products

        CUSPARSE_CHECK(cusparseCreateDnVec(&vecP_dev, n, d_p, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecQ_dev, n, d_q, CUDA_R_64F));

        // A spMV buffer for A
        size_t spmvBufSize = 0;
        void *d_spmvBuf = nullptr;
        {
            size_t spmvBufSizeNT = 0, spmvBufSizeT = 0;

            double alpha1 = 1.0;
            double beta0 = 0.0;

            // NON_TRANSPOSE buffer
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha1, spMatA, vecP_dev,
                &beta0, vecQ_dev,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &spmvBufSizeNT));

            // TRANSPOSE buffer
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_TRANSPOSE,
                &alpha1, spMatA, vecP_dev,
                &beta0, vecQ_dev,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &spmvBufSizeT));

            // assign to outer variable (no shadowing)
            spmvBufSize = (spmvBufSizeNT > spmvBufSizeT) ? spmvBufSizeNT : spmvBufSizeT;
            if (spmvBufSize > 0)
            {
                CUDA_CHECK(cudaMalloc(&d_spmvBuf, spmvBufSize));
            }
            else
            {
                d_spmvBuf = nullptr;
            }
        }

        // Preconditioner work vectors (PrecWork precision). No per-iteration allocations.
        // r_work is the (possibly casted) copy of r for the triangular solves.
        // w_work holds the intermediate solve, and z_work is the preconditioned result in PrecWork.
        PrecWork *d_r_work = nullptr, *d_w_work = nullptr, *d_z_work = nullptr;
        if constexpr (L_is_fp64)
        {
            d_r_work = reinterpret_cast<PrecWork *>(d_r); // alias (FP64)
            d_z_work = reinterpret_cast<PrecWork *>(d_z); // alias (FP64)
            CUDA_CHECK(cudaMalloc(&d_w_work, n * sizeof(PrecWork)));
        }
        else
        {
            CUDA_CHECK(cudaMalloc(&d_r_work, n * sizeof(PrecWork)));
            CUDA_CHECK(cudaMalloc(&d_w_work, n * sizeof(PrecWork)));
            CUDA_CHECK(cudaMalloc(&d_z_work, n * sizeof(PrecWork)));
        }

        //======================================================
        // PHASE 2: CG Initialization
        //======================================================
        // r = b - A*x => but x=0, so r=b
        cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice);

        // Norm of r
        double nrmr0 = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_r, 1, &nrmr0));

        // Stopping criteria
        double tol = 1e-6;

        //======================================================
        // PHASE 3: CG iteration
        //======================================================
        double rho = 0.0, rhoOld = 0.0;
        int maxIters = 1000; // or some big
        iterations = 0;

        cusparseDnVecDescr_t vecR_work = nullptr, vecW_work = nullptr;
        cusparseDnVecDescr_t vecR2_work = nullptr, vecZ_work = nullptr;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecR_work, n, (void *)d_r_work, precWork_dataType));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecW_work, n, (void *)d_w_work, precWork_dataType));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecR2_work, n, (void *)d_w_work, precWork_dataType));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecZ_work, n, (void *)d_z_work, precWork_dataType));

        for (int k = 1; k <= maxIters; k++)
        {
            // (1) z = M^-1 r
            //   L * w = r => w = L^-1 r
            //   L^T z = w => z = L^-T w
            {
                if constexpr (!L_is_fp64)
                {
                    // Cast r (FP64) -> r_work (PrecWork=FP32) once per iteration.
                    cast_vec<PrecWork, double><<<grid, block>>>(n, d_r, d_r_work);
                    CUDA_CHECK(cudaGetLastError());
                }

                // forward: L * w_work = r_work
                CUSPARSE_CHECK(cusparseSpSV_solve(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha_one,
                    spMatL,
                    vecR_work, vecW_work,
                    precWork_dataType,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spSVDescrFwd));

                // backward: L^T * z_work = w_work
                CUSPARSE_CHECK(cusparseSpSV_solve(
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha_one,
                    spMatL,
                    vecR2_work, vecZ_work,
                    precWork_dataType,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spSVDescrBwd));

                if constexpr (!L_is_fp64)
                {
                    // Cast z_work (PrecWork=FP32) -> z (FP64) for outer FP64 dot products and updates.
                    cast_vec<double, PrecWork><<<grid, block>>>(n, d_z_work, d_z);
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            // (2) rho = r^T z
            rhoOld = rho;
            cublasDdot(cublasHandle, n, d_r, 1, d_z, 1, &rho);

            // (3) if k==1 => p=z else => p=z + beta p
            if (k == 1)
            {
                cudaMemcpy(d_p, d_z, n * sizeof(double), cudaMemcpyDeviceToDevice);
            }
            else
            {
                double beta = (rho / rhoOld);
                cublasDscal(cublasHandle, n, &beta, d_p, 1);
                double alphaOne = 1.0;
                cublasDaxpy(cublasHandle, n, &alphaOne, d_z, 1, d_p, 1);
            }

            // (4) q = A p, where A is symmetric but stored as (L+D) only.
            // q = (L+D)*p + (L^T)*p - diag(A).*p
            {
                CUSPARSE_CHECK(cusparseDnVecSetValues(vecP_dev, d_p));
                CUSPARSE_CHECK(cusparseDnVecSetValues(vecQ_dev, d_q));

                double alpha1 = 1.0;
                double beta0 = 0.0;
                double beta1 = 1.0;

                // q = (L+D) * p
                CUSPARSE_CHECK(cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta0, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf));

                // q += (L+D)^T * p   (adds diag again too)
                CUSPARSE_CHECK(cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta1, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf));

                // remove one diag
                // diag_sub<<<grid, block>>>(n, d_csrRowPtrA, d_valA, d_p, d_q);
                diag_sub_from_diag<<<grid, block>>>(n, d_diagA, d_p, d_q);

                CUDA_CHECK(cudaGetLastError());
            }

            // (5) alpha = rho / (p^T q)
            double denom = 0.0;
            cublasDdot(cublasHandle, n, d_p, 1, d_q, 1, &denom);

            if (denom <= 0.0 || std::isnan(denom) || std::isinf(denom))
            {
                std::cerr << "ERROR: denom invalid in iter " << k << ": " << denom << "\n";
                iterations = k;
                finalRes = std::numeric_limits<double>::infinity();
                break;
            }

            double alpha = rho / denom;

            // (6) x = x + alpha p
            cublasDaxpy(cublasHandle, n, &alpha, d_p, 1, d_x, 1);

            // (7) r = r - alpha q
            double negAlpha = -alpha;
            cublasDaxpy(cublasHandle, n, &negAlpha, d_q, 1, d_r, 1);

            // (8) check convergence
            ew_mul<<<grid, block>>>(n, d_D, d_r, d_Dr); // d_Dr = D .* r

            double nrmDr = 0.0;
            cublasDnrm2(cublasHandle, n, d_Dr, 1, &nrmDr);

            if (nrmDr / bnorm < tol)
            {
                iterations = k;
                finalRes = nrmDr / bnorm;
                break;
            }

            // Example 2: Check intermediate values during CG iterations
            if (std::isnan(nrmDr) || std::isinf(nrmDr))
            {
                std::cerr << "ERROR: nrmDr is NaN or Inf in iteration " << k << ": " << nrmDr << std::endl;
                break;
            }
        }

        // If never converges
        if (iterations == 0)
        {
            iterations = maxIters;
        }

        CUSPARSE_CHECK(cusparseDestroyDnVec(vecR_work));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecW_work));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecR2_work));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecZ_work));

        // copy x back
        h_x.resize(n);
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

        // free
        if (vecP_dev)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecP_dev));
        if (vecQ_dev)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecQ_dev));
        if (d_spmvBuf)
            cudaFree(d_spmvBuf);

        if (d_solveBufFwd)
            cudaFree(d_solveBufFwd);
        if (d_solveBufBwd)
            cudaFree(d_solveBufBwd);

        if (spSVDescrFwd)
            CUSPARSE_CHECK(cusparseSpSV_destroyDescr(spSVDescrFwd));
        if (spSVDescrBwd)
            CUSPARSE_CHECK(cusparseSpSV_destroyDescr(spSVDescrBwd));

        // Preconditioner work buffers
        if constexpr (!L_is_fp64)
        {
            CUDA_CHECK(cudaFree(d_r_work));
            CUDA_CHECK(cudaFree(d_z_work));
        }
        CUDA_CHECK(cudaFree(d_w_work));

        if (d_dummyVec)
            CUDA_CHECK(cudaFree(d_dummyVec));
        CUDA_CHECK(cudaFree(d_p));
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaFree(d_z));

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_csrRowPtrA));
        CUDA_CHECK(cudaFree(d_csrColIndA));
        CUDA_CHECK(cudaFree(d_valA));
        CUDA_CHECK(cudaFree(d_csrRowPtrL));
        CUDA_CHECK(cudaFree(d_csrColIndL));
        if (d_valL_storage && d_valL_storage != d_valLptr)
            CUDA_CHECK(cudaFree(d_valL_storage));
        CUDA_CHECK(cudaFree(d_valLptr));

        CUDA_CHECK(cudaFree(d_diagA));
        CUDA_CHECK(cudaFree(d_D));
        CUDA_CHECK(cudaFree(d_Dr));

        CUSPARSE_CHECK(cusparseDestroySpMat(spMatA));
        CUSPARSE_CHECK(cusparseDestroySpMat(spMatL));

        CUBLAS_CHECK(cublasDestroy(cublasHandle));
        CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));
    }

    template void icPreconditionedCG_GPU<double>(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<double> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        int &iterations,
        double &finalRes);

    template void icPreconditionedCG_GPU<float>(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<float> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        int &iterations,
        double &finalRes);

    template void icPreconditionedCG_GPU<half_float::half>(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<half_float::half> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        int &iterations,
        double &finalRes);

} // namespace ichol