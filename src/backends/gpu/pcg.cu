#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <type_traits>
#include "ichol/half.hpp"

namespace ichol
{

    /**
     * @brief Solves a linear system using a GPU-based CG solver with IC preconditioning.
     *
     * Applies the preconditioner M^{-1} = L^{-T} L^{-1} and uses CG
     * to solve A x = b.
     *
     * @param h_csrRowPtrA CSR row pointer for matrix A (host).
     * @param h_csrColIndA CSR column indices for matrix A (host).
     * @param h_valA       Nonzero values of matrix A (host).
     * @param h_csrRowPtrL CSR row pointer for factor L (host).
     * @param h_csrColIndL CSR column indices for factor L (host).
     * @param h_valL       Nonzero values for factor L (host) in precision T_L.
     * @param h_b          Right-hand side vector b (host).
     * @param h_x          Solution vector x (host, output).
     * @param iterations   Number of iterations performed (output).
     * @param finalRes     Final residual norm (output).
     */
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
        int &iterations,
        double &finalRes)
    {

        //======================================================
        // PHASE 1: Allocate A, L on GPU, create cuSPARSE/cuBLAS
        //======================================================
        cusparseHandle_t cusparseHandle = nullptr;
        cusparseCreate(&cusparseHandle);

        cublasHandle_t cublasHandle = nullptr;
        cublasCreate(&cublasHandle);

        int n = static_cast<int>(h_csrRowPtrA.size()) - 1;
        int nnzA = static_cast<int>(h_valA.size());
        int nnzL = static_cast<int>(h_valL.size());

        // 1) Copy A to device
        int *d_csrRowPtrA = nullptr, *d_csrColIndA = nullptr;
        double *d_valA = nullptr;
        cudaMalloc((void **)&d_csrRowPtrA, (n + 1) * sizeof(int));
        cudaMalloc((void **)&d_csrColIndA, nnzA * sizeof(int));
        cudaMalloc((void **)&d_valA, nnzA * sizeof(double));
        cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrColIndA, h_csrColIndA.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_valA, h_valA.data(), nnzA * sizeof(double), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t spMatA = nullptr;
        cusparseCreateCsr(&spMatA, n, n, nnzA,
                          d_csrRowPtrA, d_csrColIndA, d_valA,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        // 2) Copy L to device
        int *d_csrRowPtrL = nullptr, *d_csrColIndL = nullptr;
        cudaMalloc((void **)&d_csrRowPtrL, (n + 1) * sizeof(int));
        cudaMalloc((void **)&d_csrColIndL, nnzL * sizeof(int));
        cudaMemcpy(d_csrRowPtrL, h_csrRowPtrL.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrColIndL, h_csrColIndL.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice);

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
        cudaMalloc(&d_valLptr, nnzL * sizeof(T_L));
        cudaMemcpy(d_valLptr, h_valL.data(), nnzL * sizeof(T_L), cudaMemcpyHostToDevice);

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

        // Also need a spMV buffer for A
        size_t spmvBufSize = 0;
        void *d_spmvBuf = nullptr;
        {
            double beta0 = 0.0;
            cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_one, spMatA, vecP_dev,
                &beta0, vecQ_dev,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &spmvBufSize);
            cudaMalloc(&d_spmvBuf, spmvBufSize);
        }

        //======================================================
        // PHASE 2: CG Initialization
        //======================================================
        // r = b - A*x => but x=0, so r=b
        cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice);

        // Norm of r
        double nrmr0 = 0.0;
        cublasDnrm2(cublasHandle, n, d_r, 1, &nrmr0);

        double tol = 1e-12;

        //======================================================
        // PHASE 3: CG iteration
        //======================================================
        double rho = 0.0, rhoOld = 0.0;
        int maxIters = 1000; // or some big
        iterations = 0;

        for (int k = 1; k <= maxIters; k++)
        {
            // (1) z = M^-1 r => we do forward solve + backward solve
            //   L * w = r => w = L^-1 r
            //   L^T z = w => z = L^-T w
            {
                // forward
                cusparseDnVecDescr_t vecR = nullptr, vecW = nullptr;
                cusparseCreateDnVec(&vecR, n, (void *)d_r, CUDA_R_64F);
                // cusparseCreateDnVec(&vecW, n, (void *)d_z, CUDA_R_64F);
                cusparseCreateDnVec(&vecW, n, (void *)d_w, CUDA_R_64F);

                cusparseSpSV_solve(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha_one,
                    spMatL,
                    vecR, vecW,
                    L_dataType,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spSVDescrFwd);

                cusparseDestroyDnVec(vecR);
                cusparseDestroyDnVec(vecW);

                // backward
                // cusparseCreateDnVec(&vecR, n, (void *)d_z, CUDA_R_64F);
                // cusparseCreateDnVec(&vecW, n, (void *)d_z, CUDA_R_64F);
                cusparseDnVecDescr_t vecR2 = nullptr, vecZ = nullptr;
                cusparseCreateDnVec(&vecR2, n, (void *)d_w, CUDA_R_64F); // input = w from forward
                cusparseCreateDnVec(&vecZ, n, (void *)d_z, CUDA_R_64F);  // output = z
                cusparseSpSV_solve(
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha_one,
                    spMatL,
                    vecR2, vecZ,
                    L_dataType,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spSVDescrBwd);

                cusparseDestroyDnVec(vecR2);
                cusparseDestroyDnVec(vecZ);
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

            // (4) q = A p
            {
                double alpha1 = 1.0, beta0 = 0.0;
                cusparseDnVecSetValues(vecP_dev, d_p);
                cusparseDnVecSetValues(vecQ_dev, d_q);

                cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta0, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf);
            }

            // (5) alpha = rho / (p^T q)
            double denom = 0.0;
            cublasDdot(cublasHandle, n, d_p, 1, d_q, 1, &denom);

            double alpha = rho / denom;

            // (6) x = x + alpha p
            cublasDaxpy(cublasHandle, n, &alpha, d_p, 1, d_x, 1);

            // (7) r = r - alpha q
            double negAlpha = -alpha;
            cublasDaxpy(cublasHandle, n, &negAlpha, d_q, 1, d_r, 1);

            // (8) check convergence
            double nrmr = 0.0;
            cublasDnrm2(cublasHandle, n, d_r, 1, &nrmr);
            iterations = k;
            std::cout << "  CG Iteration " << k << ": Residual = " << nrmr << std::endl;
            finalRes = nrmr;
            if (nrmr / nrmr0 < tol)
            {
                break;
            }

            // Example 2: Check intermediate values during CG iterations
            if (std::isnan(nrmr) || std::isinf(nrmr))
            {
                std::cerr << "ERROR: nrmr is NaN or Inf in iteration " << k << ": " << nrmr << std::endl;
                break;
            }
        }

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
        int &iterations,
        double &finalRes);

} // namespace ichol