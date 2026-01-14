#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <random>
#include <type_traits>
#include <numeric>
#include <algorithm>
#include <cstring>

#include "ichol/half.hpp"
#include "solve/sptrsv/cuda/sptrsv_level.cuh"

// ---------------------- mixed-precision helpers ----------------------
template <typename To, typename From>
__global__ void cast_vec(int n, const From *__restrict__ in, To *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        if constexpr (std::is_same_v<To, __half>)
        {
            out[i] = __float2half(static_cast<float>(in[i]));
        }
        else if constexpr (std::is_same_v<From, __half>)
        {
            out[i] = static_cast<To>(__half2float(in[i]));
        }
        else
        {
            out[i] = static_cast<To>(in[i]);
        }
    }
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

// ---------------------- host helpers: CSR transpose + level sets ----------------------
template <typename ValueT>
static void build_csr_transpose_diag_last(
    int n,
    const std::vector<int> &row_ptr,
    const std::vector<int> &col_ind,
    const std::vector<ValueT> &val,
    std::vector<int> &row_ptr_T,
    std::vector<int> &col_ind_T,
    std::vector<ValueT> &val_T)
{
    const int nnz = static_cast<int>(val.size());
    row_ptr_T.assign(n + 1, 0);
    col_ind_T.resize(nnz);
    val_T.resize(nnz);

    // Count nnz per row in transpose (i.e., per column of original).
    for (int i = 0; i < n; ++i)
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
        {
            int j = col_ind[p];
            row_ptr_T[j + 1]++;
        }
    }

    std::partial_sum(row_ptr_T.begin(), row_ptr_T.end(), row_ptr_T.begin());
    std::vector<int> next = row_ptr_T;

    // Fill transpose. Within each transpose row j, the inserted columns are i in increasing order.
    for (int i = 0; i < n; ++i)
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
        {
            int j = col_ind[p];
            int dst = next[j]++;
            col_ind_T[dst] = i;
            val_T[dst] = val[p];
        }
    }

    // Move diagonal to last position per row (keep off-diagonals sorted).
    for (int r = 0; r < n; ++r)
    {
        int s = row_ptr_T[r];
        int e = row_ptr_T[r + 1];
        // assume diagonal exists
        int d = -1;
        for (int k = s; k < e; ++k)
        {
            if (col_ind_T[k] == r)
            {
                d = k;
                break;
            }
        }
        if (d < 0)
        {
            // no diagonal: leave as-is (SpTRSV will flag -2 later)
            continue;
        }
        if (d == e - 1)
            continue;

        int diag_col = col_ind_T[d];
        ValueT diag_val = val_T[d];

        int move_count = (e - 1) - d;
        std::memmove(&col_ind_T[d], &col_ind_T[d + 1], sizeof(int) * move_count);
        std::memmove(&val_T[d], &val_T[d + 1], sizeof(ValueT) * move_count);

        col_ind_T[e - 1] = diag_col; // == r
        val_T[e - 1] = diag_val;
    }
}

static ichol::symbolic::LevelSets build_level_sets_lower_csr_diag_last(
    int n, const std::vector<int> &row_ptr, const std::vector<int> &col_ind)
{
    int max_level = -1;
    std::vector<int> level_of(n, -1);

    for (int i = 0; i < n; ++i)
    {
        int best = -1;
        for (int p = row_ptr[i]; p < row_ptr[i + 1] - 1; ++p) // skip last (diag)
        {
            best = std::max(best, level_of[col_ind[p]]);
        }
        level_of[i] = best + 1;
        max_level = std::max(max_level, level_of[i]);
    }

    const int num_levels = max_level + 1;
    std::vector<int> counts(num_levels, 0);
    for (int i = 0; i < n; ++i)
        counts[level_of[i]]++;

    ichol::symbolic::LevelSets out;
    out.level_ptr.resize(num_levels + 1);
    out.level_ptr[0] = 0;
    std::partial_sum(counts.begin(), counts.end(), out.level_ptr.begin() + 1);

    out.levels.resize(n);
    std::vector<int> next(out.level_ptr.begin(), out.level_ptr.end() - 1);
    for (int i = 0; i < n; ++i)
    {
        int L = level_of[i];
        out.levels[next[L]++] = i;
    }
    return out;
}

static ichol::symbolic::LevelSets build_level_sets_upper_csr_diag_last(
    int n, const std::vector<int> &row_ptr, const std::vector<int> &col_ind)
{
    int max_level = -1;
    std::vector<int> level_of(n, -1);

    for (int ii = 0; ii < n; ++ii)
    {
        int i = n - 1 - ii;
        int best = -1;
        for (int p = row_ptr[i]; p < row_ptr[i + 1] - 1; ++p) // skip last (diag)
        {
            best = std::max(best, level_of[col_ind[p]]);
        }
        level_of[i] = best + 1;
        max_level = std::max(max_level, level_of[i]);
    }

    const int num_levels = max_level + 1;
    std::vector<int> counts(num_levels, 0);
    for (int i = 0; i < n; ++i)
        counts[level_of[i]]++;

    ichol::symbolic::LevelSets out;
    out.level_ptr.resize(num_levels + 1);
    out.level_ptr[0] = 0;
    std::partial_sum(counts.begin(), counts.end(), out.level_ptr.begin() + 1);

    out.levels.resize(n);
    std::vector<int> next(out.level_ptr.begin(), out.level_ptr.end() - 1);
    for (int i = 0; i < n; ++i)
    {
        int L = level_of[i];
        out.levels[next[L]++] = i;
    }
    return out;
}

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

        // ---------------------- Preconditioner value type selection for YOUR SpTRSV ----------------------
        constexpr bool L_is_fp64 = std::is_same<T_L, double>::value;
        constexpr bool L_is_fp32 = std::is_same<T_L, float>::value;
        constexpr bool L_is_fp16 = !L_is_fp64 && !L_is_fp32; // (half_float::half)

        using SolveT = std::conditional_t<L_is_fp64, double, std::conditional_t<L_is_fp32, float, __half>>;

        // Build host SolveT values for L
        std::vector<SolveT> h_valL_solve(nnzL);
        if constexpr (L_is_fp64)
        {
            for (int i = 0; i < nnzL; ++i)
                h_valL_solve[i] = static_cast<double>(h_valL[i]);
        }
        else if constexpr (L_is_fp32)
        {
            for (int i = 0; i < nnzL; ++i)
                h_valL_solve[i] = static_cast<float>(h_valL[i]);
        }
        else
        {
            for (int i = 0; i < nnzL; ++i)
                h_valL_solve[i] = __float2half_rn(static_cast<float>(h_valL[i]));
        }

        // Build CSR(L^T) on host with diag-last.
        std::vector<int> h_csrRowPtrLt, h_csrColIndLt;
        std::vector<SolveT> h_valLt_solve;
        build_csr_transpose_diag_last<SolveT>(
            n, h_csrRowPtrL, h_csrColIndL, h_valL_solve,
            h_csrRowPtrLt, h_csrColIndLt, h_valLt_solve);

        // Build level sets for L (lower) and L^T (upper).
        const ichol::symbolic::LevelSets levelsets_L =
            build_level_sets_lower_csr_diag_last(n, h_csrRowPtrL, h_csrColIndL);

        const ichol::symbolic::LevelSets levelsets_Lt =
            build_level_sets_upper_csr_diag_last(n, h_csrRowPtrLt, h_csrColIndLt);

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

        // Copy L (CSR) to device (indices + values in SolveT)
        int *d_csrRowPtrL = nullptr, *d_csrColIndL = nullptr;
        SolveT *d_valL = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtrL, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_csrColIndL, nnzL * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_valL, nnzL * sizeof(SolveT)));
        CUDA_CHECK(cudaMemcpy(d_csrRowPtrL, h_csrRowPtrL.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndL, h_csrColIndL.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valL, h_valL_solve.data(), nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));

        // Copy L^T (CSR) to device (indices + values in SolveT)
        int *d_csrRowPtrLt = nullptr, *d_csrColIndLt = nullptr;
        SolveT *d_valLt = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtrLt, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_csrColIndLt, nnzL * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_valLt, nnzL * sizeof(SolveT)));
        CUDA_CHECK(cudaMemcpy(d_csrRowPtrLt, h_csrRowPtrLt.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndLt, h_csrColIndLt.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valLt, h_valLt_solve.data(), nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));

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

        // For A * p
        cusparseDnVecDescr_t vecP_dev = nullptr, vecQ_dev = nullptr;
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

            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha1, spMatA, vecP_dev,
                &beta0, vecQ_dev,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &spmvBufSizeNT));

            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_TRANSPOSE,
                &alpha1, spMatA, vecP_dev,
                &beta0, vecQ_dev,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                &spmvBufSizeT));

            spmvBufSize = (spmvBufSizeNT > spmvBufSizeT) ? spmvBufSizeNT : spmvBufSizeT;
            if (spmvBufSize > 0)
                CUDA_CHECK(cudaMalloc(&d_spmvBuf, spmvBufSize));
        }

        // Preconditioner work vectors (SolveT precision). No per-iteration allocations.
        SolveT *d_r_work = nullptr, *d_w_work = nullptr, *d_z_work = nullptr;
        if constexpr (std::is_same_v<SolveT, double>)
        {
            d_r_work = reinterpret_cast<SolveT *>(d_r); // alias
            d_z_work = reinterpret_cast<SolveT *>(d_z); // alias
            CUDA_CHECK(cudaMalloc(&d_w_work, n * sizeof(SolveT)));
        }
        else
        {
            CUDA_CHECK(cudaMalloc(&d_r_work, n * sizeof(SolveT)));
            CUDA_CHECK(cudaMalloc(&d_w_work, n * sizeof(SolveT)));
            CUDA_CHECK(cudaMalloc(&d_z_work, n * sizeof(SolveT)));
        }

        //======================================================
        // PHASE 2: CG Initialization
        //======================================================
        cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice);

        double nrmr0 = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_r, 1, &nrmr0));

        double tol = 1e-6;

        //======================================================
        // PHASE 3: CG iteration
        //======================================================
        double rho = 0.0, rhoOld = 0.0;
        int maxIters = 1000;
        iterations = 0;

        cudaStream_t stream = 0;

        for (int k = 1; k <= maxIters; k++)
        {
            // (1) z = M^-1 r
            //   L * w = r
            //   L^T * z = w
            {
                if constexpr (!std::is_same_v<SolveT, double>)
                {
                    cast_vec<SolveT, double><<<grid, block>>>(n, d_r, d_r_work);
                    CUDA_CHECK(cudaGetLastError());
                }

                int rc1 = SpTRSV_solve_levelsets<int, SolveT>(
                    n,
                    d_csrRowPtrL,
                    d_csrColIndL,
                    d_valL,
                    d_r_work,
                    d_w_work,
                    FillMode::LOWER,
                    /*unit_diag=*/false,
                    levelsets_L,
                    stream);

                if (rc1 != 0)
                {
                    std::cerr << "ERROR: SpTRSV(L) failed with code " << rc1 << " at iter " << k << "\n";
                    iterations = k;
                    finalRes = std::numeric_limits<double>::infinity();
                    break;
                }

                int rc2 = SpTRSV_solve_levelsets<int, SolveT>(
                    n,
                    d_csrRowPtrLt,
                    d_csrColIndLt,
                    d_valLt,
                    d_w_work,
                    d_z_work,
                    FillMode::UPPER,
                    /*unit_diag=*/false,
                    levelsets_Lt,
                    stream);

                if (rc2 != 0)
                {
                    std::cerr << "ERROR: SpTRSV(L^T) failed with code " << rc2 << " at iter " << k << "\n";
                    iterations = k;
                    finalRes = std::numeric_limits<double>::infinity();
                    break;
                }

                if constexpr (!std::is_same_v<SolveT, double>)
                {
                    cast_vec<double, SolveT><<<grid, block>>>(n, d_z_work, d_z);
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            // (2) rho = r^T z
            rhoOld = rho;
            cublasDdot(cublasHandle, n, d_r, 1, d_z, 1, &rho);

            // (3) p update
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

                CUSPARSE_CHECK(cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta0, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf));

                CUSPARSE_CHECK(cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta1, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf));

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
            ew_mul<<<grid, block>>>(n, d_D, d_r, d_Dr);

            double nrmDr = 0.0;
            cublasDnrm2(cublasHandle, n, d_Dr, 1, &nrmDr);

            if (nrmDr / bnorm < tol)
            {
                iterations = k;
                finalRes = nrmDr / bnorm;
                break;
            }

            if (std::isnan(nrmDr) || std::isinf(nrmDr))
            {
                std::cerr << "ERROR: nrmDr is NaN or Inf in iteration " << k << ": " << nrmDr << std::endl;
                iterations = k;
                finalRes = std::numeric_limits<double>::infinity();
                break;
            }
        }

        if (iterations == 0)
            iterations = maxIters;

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

        if constexpr (!std::is_same_v<SolveT, double>)
        {
            CUDA_CHECK(cudaFree(d_r_work));
            CUDA_CHECK(cudaFree(d_z_work));
        }
        CUDA_CHECK(cudaFree(d_w_work));

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
        CUDA_CHECK(cudaFree(d_valL));

        CUDA_CHECK(cudaFree(d_csrRowPtrLt));
        CUDA_CHECK(cudaFree(d_csrColIndLt));
        CUDA_CHECK(cudaFree(d_valLt));

        CUDA_CHECK(cudaFree(d_diagA));
        CUDA_CHECK(cudaFree(d_D));
        CUDA_CHECK(cudaFree(d_Dr));

        CUSPARSE_CHECK(cusparseDestroySpMat(spMatA));

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
