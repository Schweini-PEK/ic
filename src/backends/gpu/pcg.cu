#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <type_traits>
#include "ichol/half.hpp"

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

static void debug_check_and_extract_diagA(
    int n,
    const std::vector<int> &rowPtr,
    const std::vector<int> &colInd,
    const std::vector<double> &val,
    std::vector<double> &diagA_out,
    int max_print = 20)
{
    diagA_out.assign(n, 0.0);

    int bad_last = 0;
    int missing = 0;

    for (int i = 0; i < n; ++i)
    {
        int row_start = rowPtr[i];
        int row_end = rowPtr[i + 1];
        if (row_end <= row_start)
        {
            ++missing;
            continue;
        }

        int last_p = row_end - 1;
        int last_j = colInd[last_p];

        if (last_j == i)
        {
            diagA_out[i] = val[last_p];
            continue;
        }

        // If the last one is not the diag, look for the diag
        ++bad_last;
        bool found = false;
        for (int p = row_start; p < row_end; ++p)
        {
            if (colInd[p] == i)
            {
                diagA_out[i] = val[p];
                found = true;
                break;
            }
        }
        if (!found)
            ++missing;

        if (bad_last <= max_print)
        {
            std::cerr << "[diag-check] row " << i
                      << ": last col = " << last_j
                      << ", scanned diag " << (found ? "FOUND" : "MISSING")
                      << " (row nnz=" << (row_end - row_start) << ")\n";
        }
    }

    if (bad_last > 0)
        std::cerr << "[diag-check] rows with diag not last: " << bad_last << " / " << n << "\n";
    if (missing > 0)
        std::cerr << "[diag-check] rows missing diagonal entry: " << missing << " / " << n << "\n";
}

// ---------------------- GPU CHECKS (abort-on-fail) ----------------------
namespace test_checks
{

    inline double host_l2_norm(const std::vector<double> &v)
    {
        long double s = 0.0L;
        for (double a : v)
            s += (long double)a * (long double)a;
        return std::sqrt((double)s);
    }

    inline void symm_lower_csr_matvec_raw(int n,
                                          const std::vector<int> &rowPtr,
                                          const std::vector<int> &colInd,
                                          const std::vector<double> &val,
                                          const std::vector<double> &x,
                                          std::vector<double> &y)
    {
        y.assign(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            for (int p = rowPtr[i]; p < rowPtr[i + 1]; ++p)
            {
                const int j = colInd[p];
                const double aij = val[p];
                y[i] += aij * x[j];
                if (j != i)
                    y[j] += aij * x[i];
            }
        }
    }

    inline void abort_fail(const char *msg)
    {
        std::cerr << "CHECK FAILED: " << msg << "\n";
        std::abort();
    }

    // Verifies the GPU matvec used by CG matches the CPU symmetric matvec for a deterministic p.
    // Call this ONCE after spMatA + vec descriptors + spmv buffer are ready.
    inline void check_gpu_matvec_matches_cpu(cusparseHandle_t cusparseHandle,
                                             cusparseSpMatDescr_t spMatA,
                                             cusparseDnVecDescr_t vecP,
                                             cusparseDnVecDescr_t vecQ,
                                             void *d_spmvBuf,
                                             int n,
                                             const std::vector<int> &h_rowPtrA,
                                             const std::vector<int> &h_colIndA,
                                             const std::vector<double> &h_valA,
                                             const double *d_diagA,
                                             double *d_p,
                                             double *d_q)
    {
        // deterministic p
        std::mt19937 gen(0);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> p(n);
        for (int i = 0; i < n; ++i)
            p[i] = dist(gen);

        CUDA_CHECK(cudaMemcpy(d_p, p.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        // q = A p
        double alpha = 1.0, beta0 = 0.0, beta1 = 1.0;
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecP, d_p));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecQ, d_q));
        CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, spMatA, vecP,
                                    &beta0, vecQ,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf));
        // q += A^T p
        CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                    &alpha, spMatA, vecP,
                                    &beta1, vecQ,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf));
        // q -= diag(A) * p  (remove the duplicated diagonal)
        int block = 256;
        int grid = (n + block - 1) / block;
        diag_sub_from_diag<<<grid, block>>>(n, d_diagA, d_p, d_q);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> q_gpu(n);
        CUDA_CHECK(cudaMemcpy(q_gpu.data(), d_q, n * sizeof(double), cudaMemcpyDeviceToHost));

        // CPU reference: symmetric matvec on lower+diag CSR
        std::vector<double> q_cpu;
        symm_lower_csr_matvec_raw(n, h_rowPtrA, h_colIndA, h_valA, p, q_cpu);

        // relative error
        std::vector<double> diff(n);
        for (int i = 0; i < n; ++i)
            diff[i] = q_gpu[i] - q_cpu[i];
        const double nd = host_l2_norm(diff);
        const double nr = host_l2_norm(q_cpu);
        const double rel = (nr == 0.0) ? nd : nd / nr;

        if (!(std::isfinite(rel) && rel <= 1e-12))
        {
            std::cerr << "GPU matvec mismatch: rel_err=" << rel << "\n";
            abort_fail("GPU matvec != CPU matvec (format/dup/diag handling bug)");
        }
    }

    inline void check_cg_scalar_pos_finite(const char *name, double v, int k)
    {
        if (!(std::isfinite(v) && v > 0.0))
        {
            std::cerr << "CG invariant failed: " << name << " = " << v << " at iter " << k << "\n";
            abort_fail("CG saw non-positive/invalid scalar (operator/preconditioner application bug)");
        }
    }

} // namespace test_checks

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

        int n = static_cast<int>(h_csrRowPtrA.size()) - 1;
        int nnzA = static_cast<int>(h_valA.size());
        int nnzL = static_cast<int>(h_valL.size());

        std::vector<double> h_diagA(n, 0.0);
        // for (int i = 0; i < n; ++i)
        // {
        //     h_diagA[i] = h_valA[h_csrRowPtrA[i + 1] - 1];
        // }
        debug_check_and_extract_diagA(
            n, h_csrRowPtrA, h_csrColIndA, h_valA, h_diagA);
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
        cusparseCreateCsr(&spMatA, n, n, nnzA,
                          d_csrRowPtrA, d_csrColIndA, d_valA,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        // Copy L to device
        int *d_csrRowPtrL = nullptr, *d_csrColIndL = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtrL, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_csrColIndL, nnzL * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_csrRowPtrL, h_csrRowPtrL.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndL, h_csrColIndL.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));

        // Determine L data type based on T_L
        cudaDataType L_dataType;
        if (std::is_same<T_L, double>::value)
        {
            L_dataType = CUDA_R_64F;
        }
        else if (std::is_same<T_L, float>::value)
        {
            L_dataType = CUDA_R_32F;
        }
        else
        {
            L_dataType = CUDA_R_16F; // Assuming half_float::half or __half
        }

        // Allocate memory for L values on device and copy data
        void *d_valLptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_valLptr, nnzL * sizeof(T_L)));
        CUDA_CHECK(cudaMemcpy(d_valLptr, h_valL.data(), nnzL * sizeof(T_L), cudaMemcpyHostToDevice));

        cusparseSpMatDescr_t spMatL = nullptr;
        cusparseCreateCsr(&spMatL, n, n, nnzL,
                          d_csrRowPtrL, d_csrColIndL, d_valLptr,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, L_dataType);

        // Tell cuSPARSE that L is triangular, lower fill, with a non-unit diagonal
        cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(spMatL, CUSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
        cusparseSpMatSetAttribute(spMatL, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));

        // 3) Allocate vectors x, b, etc. in double
        double *d_x = nullptr, *d_b = nullptr;
        cudaMalloc((void **)&d_x, n * sizeof(double));
        cudaMalloc((void **)&d_b, n * sizeof(double));
        cudaMemset(d_x, 0, n * sizeof(double)); // x=0 initial
        cudaMemcpy(d_b, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice);

        // new: set up D
        double *d_D = nullptr;
        cudaMalloc(&d_D, n * sizeof(double));
        cudaMemcpy(d_D, h_D.data(), n * sizeof(double), cudaMemcpyHostToDevice);

        double *d_Dr = nullptr;
        cudaMalloc(&d_Dr, n * sizeof(double));

        double *d_b_orig = nullptr;
        cudaMalloc(&d_b_orig, n * sizeof(double));

        int block = 256;
        int grid = (n + block - 1) / block;
        ew_mul<<<grid, block>>>(n, d_D, d_b, d_b_orig); // d_b_orig = D .* d_b

        /**
         * Note that \tilde{b} = D^{-1} b is passed into this func.
         * This computes the norm of original b for convergence check:
         */
        double bnorm = 0.0;
        cublasDnrm2(cublasHandle, n, d_b_orig, 1, &bnorm);

        cudaFree(d_b_orig);

        // 4) Setup Triangular Solve descriptors for L & L^T
        cusparseSpSVDescr_t spSVDescrFwd = nullptr, spSVDescrBwd = nullptr;
        cusparseSpSV_createDescr(&spSVDescrFwd);
        cusparseSpSV_createDescr(&spSVDescrBwd);

        // We need a dummy dense vector to measure buffer sizes
        double *d_dummyVec = nullptr;
        cudaMalloc(&d_dummyVec, n * sizeof(double));
        cusparseDnVecDescr_t dummyVecR = nullptr, dummyVecSol = nullptr;
        cusparseCreateDnVec(&dummyVecR, n, (void *)d_dummyVec, CUDA_R_64F);
        cusparseCreateDnVec(&dummyVecSol, n, (void *)d_dummyVec, CUDA_R_64F);

        // Query buffer sizes
        size_t bufSizeFwd = 0, bufSizeBwd = 0;
        void *d_solveBufFwd = nullptr, *d_solveBufBwd = nullptr;
        double alpha_one = 1.0;

        cusparseSpSV_bufferSize(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_one, spMatL,
            dummyVecR, dummyVecSol,
            L_dataType,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrFwd, &bufSizeFwd);
        cudaMalloc(&d_solveBufFwd, bufSizeFwd);

        cusparseSpSV_analysis(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_one, spMatL,
            dummyVecR, dummyVecSol,
            L_dataType, CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrFwd, d_solveBufFwd);

        cusparseSpSV_bufferSize(
            cusparseHandle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha_one, spMatL,
            dummyVecR, dummyVecSol,
            L_dataType,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrBwd, &bufSizeBwd);
        cudaMalloc(&d_solveBufBwd, bufSizeBwd);

        cusparseSpSV_analysis(
            cusparseHandle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha_one, spMatL,
            dummyVecR, dummyVecSol,
            L_dataType, CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrBwd, d_solveBufBwd);

        cusparseDestroyDnVec(dummyVecR);
        cusparseDestroyDnVec(dummyVecSol);
        cudaFree(d_dummyVec);

        // For A * p
        cusparseDnVecDescr_t vecP_dev = nullptr, vecQ_dev = nullptr;
        cudaMalloc(&d_dummyVec, n * sizeof(double)); // reuse for spmv output if needed
        double *d_p = nullptr, *d_q = nullptr, *d_r = nullptr, *d_z = nullptr, *d_w = nullptr;
        cudaMalloc(&d_p, n * sizeof(double));
        cudaMalloc(&d_q, n * sizeof(double));
        cudaMalloc(&d_r, n * sizeof(double));
        cudaMalloc(&d_w, n * sizeof(double)); // for L^-1*r
        cudaMalloc(&d_z, n * sizeof(double)); // for M^-1*r

        cusparseCreateDnVec(&vecP_dev, n, d_p, CUDA_R_64F);
        cusparseCreateDnVec(&vecQ_dev, n, d_q, CUDA_R_64F);

        // A spMV buffer for A
        size_t spmvBufSize = 0;
        void *d_spmvBuf = nullptr;
        {
            size_t spmvBufSizeNT = 0, spmvBufSizeT = 0;

            double alpha1 = 1.0;
            double beta0 = 0.0;

            // NON_TRANSPOSE buffer
            cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha1, spMatA, vecP_dev,
                &beta0, vecQ_dev,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &spmvBufSizeNT);

            // TRANSPOSE buffer
            cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_TRANSPOSE,
                &alpha1, spMatA, vecP_dev,
                &beta0, vecQ_dev,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &spmvBufSizeT);

            // assign to outer variable (no shadowing)
            spmvBufSize = (spmvBufSizeNT > spmvBufSizeT) ? spmvBufSizeNT : spmvBufSizeT;
            if (spmvBufSize > 0)
            {
                cudaMalloc(&d_spmvBuf, spmvBufSize);
            }
            else
            {
                d_spmvBuf = nullptr;
            }

            test_checks::check_gpu_matvec_matches_cpu(
                cusparseHandle, spMatA,
                vecP_dev, vecQ_dev,
                d_spmvBuf,
                n,
                h_csrRowPtrA, h_csrColIndA, h_valA,
                d_diagA,
                d_p, d_q);
        }

        //======================================================
        // PHASE 2: CG Initialization
        //======================================================
        // r = b - A*x => but x=0, so r=b
        cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice);

        // Norm of r
        double nrmr0 = 0.0;
        cublasDnrm2(cublasHandle, n, d_r, 1, &nrmr0);

        // Stopping criteria
        double tol = 1e-6;

        //======================================================
        // PHASE 3: CG iteration
        //======================================================
        double rho = 0.0, rhoOld = 0.0;
        int maxIters = 1000; // or some big
        iterations = 0;

        cusparseDnVecDescr_t vecR = nullptr, vecW = nullptr;
        cusparseCreateDnVec(&vecR, n, (void *)d_r, CUDA_R_64F);
        cusparseCreateDnVec(&vecW, n, (void *)d_w, CUDA_R_64F);
        cusparseDnVecDescr_t vecR2 = nullptr, vecZ = nullptr;
        cusparseCreateDnVec(&vecR2, n, (void *)d_w, CUDA_R_64F); // input = w from forward
        cusparseCreateDnVec(&vecZ, n, (void *)d_z, CUDA_R_64F);  // output = z

        for (int k = 1; k <= maxIters; k++)
        {
            // (1) z = M^-1 r
            //   L * w = r => w = L^-1 r
            //   L^T z = w => z = L^-T w
            {
                // forward
                cusparseSpSV_solve(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha_one,
                    spMatL,
                    vecR, vecW,
                    L_dataType,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spSVDescrFwd);

                // backward
                cusparseSpSV_solve(
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha_one,
                    spMatL,
                    vecR2, vecZ,
                    L_dataType,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spSVDescrBwd);
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
                cusparseDnVecSetValues(vecP_dev, d_p);
                cusparseDnVecSetValues(vecQ_dev, d_q);

                double alpha1 = 1.0;
                double beta0 = 0.0;
                double beta1 = 1.0;

                // q = (L+D) * p
                cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta0, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf);

                // q += (L+D)^T * p   (adds diag again too)
                cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta1, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf);

                // remove one diag
                // diag_sub<<<grid, block>>>(n, d_csrRowPtrA, d_valA, d_p, d_q);
                diag_sub_from_diag<<<grid, block>>>(n, d_diagA, d_p, d_q);

                CUDA_CHECK(cudaGetLastError());
            }

            // (5) alpha = rho / (p^T q)
            double denom = 0.0;
            cublasDdot(cublasHandle, n, d_p, 1, d_q, 1, &denom);

            test_checks::check_cg_scalar_pos_finite("rho = r^T z", rho, k);
            test_checks::check_cg_scalar_pos_finite("denom = p^T A p", denom, k);

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

        cusparseDestroyDnVec(vecR);
        cusparseDestroyDnVec(vecW);
        cusparseDestroyDnVec(vecR2);
        cusparseDestroyDnVec(vecZ);

        // copy x back
        h_x.resize(n);
        cudaMemcpy(h_x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

        // free
        if (vecP_dev)
            cusparseDestroyDnVec(vecP_dev);
        if (vecQ_dev)
            cusparseDestroyDnVec(vecQ_dev);
        if (d_spmvBuf)
            cudaFree(d_spmvBuf);

        if (d_solveBufFwd)
            cudaFree(d_solveBufFwd);
        if (d_solveBufBwd)
            cudaFree(d_solveBufBwd);

        if (spSVDescrFwd)
            cusparseSpSV_destroyDescr(spSVDescrFwd);
        if (spSVDescrBwd)
            cusparseSpSV_destroyDescr(spSVDescrBwd);

        cudaFree(d_dummyVec);
        cudaFree(d_p);
        cudaFree(d_q);
        cudaFree(d_r);
        cudaFree(d_z);
        cudaFree(d_w);

        cudaFree(d_x);
        cudaFree(d_b);
        cudaFree(d_csrRowPtrA);
        cudaFree(d_csrColIndA);
        cudaFree(d_valA);
        cudaFree(d_csrRowPtrL);
        cudaFree(d_csrColIndL);
        cudaFree(d_valLptr);

        cudaFree(d_diagA);
        cudaFree(d_D);
        cudaFree(d_Dr);

        cusparseDestroySpMat(spMatA);
        cusparseDestroySpMat(spMatL);

        cublasDestroy(cublasHandle);
        cusparseDestroy(cusparseHandle);
    }

    // Explicit template instantiations for the types we support
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