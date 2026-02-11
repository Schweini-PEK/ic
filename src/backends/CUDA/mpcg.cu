// src/solver/mpcg_cuda.cu
//
// CUDA 12 baseline implementation of Bridson & Greif (2005) MPCG,
// mirroring the MATLAB reference (mpcg.m) with truncation window "restart".
//
// Key idea: each iteration builds a dense block P_i (n×k) of k search directions.
// It maintains last m blocks (m = restart) and A-orthogonalizes the new block
// against those blocks.
//
// Implementation choices (baseline):
// - A*P uses cuSPARSE SpMM (CUDA 12 generic API).
// - All block inner products and updates use cuBLAS GEMM/GEMV.
// - The "pinv" on small k×k matrices is done on CPU via a self-contained
//   Jacobi eigen-decomposition pseudo-inverse (copy small matrices host<->device).
// - Preconditioner apply is a placeholder "identity" kernel by default; replace
//   apply_precond_column_* with your existing IC/triangular solve pipeline.
//
// Build notes:
//   nvcc -O3 -std=c++17 mpcg_cuda.cu -lcusparse -lcublas -o mpcg_test
//
// Dense block layout on GPU:
//   Column-major (leading dimension = n).
//   Column t corresponds to the t-th preconditioner output.
//
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cstring>

#include "ichol/pcg.hpp"
#include "ichol/preconditioner.hpp"

// ------------------------- Error handling -------------------------
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

// ------------------------- Device kernels (baseline helpers) -------------------------
// Identity "preconditioner": z(:,t) = r
__global__ void k_copy_vec_to_col(double *__restrict__ Z, int ldZ,
                                  const double *__restrict__ r, int n,
                                  int col_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        Z[col_id * ldZ + i] = r[i];
}

// ------------------------- CPU: Jacobi eigen-decomposition for symmetric k×k -------------------------
// Baseline pinv for small symmetric matrices G (k×k):
//   G ≈ Q diag(lambda) Q^T
//   pinv(G) = Q diag(1/lambda_i if lambda_i > thresh else 0) Q^T
//
// This is used to realize MATLAB pinv(G) in mpcg.m.
//
// Notes:
// - Designed for small k (e.g., 1..32).
// - G is symmetrized before factorization (0.5*(G+G^T)).
//
static void symmetrize_inplace(std::vector<double> &A, int k)
{
    for (int j = 0; j < k; ++j)
    {
        for (int i = j + 1; i < k; ++i)
        {
            double aij = A[i + j * k];
            double aji = A[j + i * k];
            double s = 0.5 * (aij + aji);
            A[i + j * k] = s;
            A[j + i * k] = s;
        }
    }
}

static void jacobi_eig_sym(const std::vector<double> &A_in, int k,
                           std::vector<double> &Q_out,
                           std::vector<double> &eval_out,
                           int max_sweeps = 50,
                           double tol = 1e-12)
{
    // A (working copy), Q initialized to identity
    std::vector<double> A = A_in;
    Q_out.assign(k * k, 0.0);
    eval_out.assign(k, 0.0);
    for (int i = 0; i < k; ++i)
        Q_out[i + i * k] = 1.0;

    auto idx = [k](int r, int c)
    { return r + c * k; }; // column-major k×k

    for (int sweep = 0; sweep < max_sweeps; ++sweep)
    {
        // Find largest off-diagonal magnitude
        int p = 0, q = 1;
        double max_off = 0.0;
        for (int j = 0; j < k; ++j)
        {
            for (int i = 0; i < j; ++i)
            {
                double v = std::abs(A[idx(i, j)]);
                if (v > max_off)
                {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < tol)
            break;

        double app = A[idx(p, p)];
        double aqq = A[idx(q, q)];
        double apq = A[idx(p, q)];

        // Jacobi rotation parameters
        double tau = (aqq - app) / (2.0 * apq);
        double t = (tau >= 0.0)
                       ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                       : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;

        // Apply rotation to A: A <- J^T A J
        // Update rows/cols p,q
        for (int i = 0; i < k; ++i)
        {
            if (i == p || i == q)
                continue;
            double aip = A[idx(i, p)];
            double aiq = A[idx(i, q)];
            // New entries in columns p,q
            A[idx(i, p)] = c * aip - s * aiq;
            A[idx(i, q)] = s * aip + c * aiq;
        }
        for (int i = 0; i < k; ++i)
        {
            if (i == p || i == q)
                continue;
            // Mirror symmetry
            A[idx(p, i)] = A[idx(i, p)];
            A[idx(q, i)] = A[idx(i, q)];
        }

        // Update diagonal & apq
        double app_new = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        double aqq_new = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[idx(p, p)] = app_new;
        A[idx(q, q)] = aqq_new;
        A[idx(p, q)] = 0.0;
        A[idx(q, p)] = 0.0;

        // Update eigenvectors: Q <- Q J
        for (int i = 0; i < k; ++i)
        {
            double qip = Q_out[idx(i, p)];
            double qiq = Q_out[idx(i, q)];
            Q_out[idx(i, p)] = c * qip - s * qiq;
            Q_out[idx(i, q)] = s * qip + c * qiq;
        }
    }

    // Eigenvalues are diagonal
    for (int i = 0; i < k; ++i)
        eval_out[i] = A[idx(i, i)];
}

// Compute X = pinv(G) * B, where:
// - G is symmetric (k×k)
// - B is (k×nrhs) in column-major with ldB=k
// - X is (k×nrhs) in column-major with ldX=k
//
// This realizes the MATLAB blocks:
//   Y = pinv(Gj) * C
//   alpha = pinv(Gnew) * rhs
static void pinv_sym_apply(const std::vector<double> &G_in, int k,
                           const std::vector<double> &B_in, int nrhs,
                           std::vector<double> &X_out,
                           double rcond = 1e-12)
{
    // Symmetrize G to guard numerical drift
    std::vector<double> G = G_in;
    symmetrize_inplace(G, k);

    // Eigendecompose G
    std::vector<double> Q, eval;
    jacobi_eig_sym(G, k, Q, eval);

    // Determine threshold for pseudo-inverse
    double max_abs = 0.0;
    for (int i = 0; i < k; ++i)
        max_abs = std::max(max_abs, std::abs(eval[i]));
    double thresh = rcond * (max_abs > 0.0 ? max_abs : 1.0);

    // Precompute inv eigenvalues
    std::vector<double> inv_eval(k, 0.0);
    for (int i = 0; i < k; ++i)
    {
        if (std::abs(eval[i]) > thresh)
            inv_eval[i] = 1.0 / eval[i];
        else
            inv_eval[i] = 0.0; // pseudo-inverse: drop small eigenvalues
    }

    // X = Q * diag(inv_eval) * (Q^T * B)
    X_out.assign(k * nrhs, 0.0);

    // T = Q^T * B : (k×nrhs)
    std::vector<double> T(k * nrhs, 0.0);
    for (int col = 0; col < nrhs; ++col)
    {
        for (int i = 0; i < k; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < k; ++j)
            {
                sum += Q[j + i * k] * B_in[j + col * k]; // Q^T: (i,j) = Q(j,i)
            }
            T[i + col * k] = sum;
        }
    }

    // Scale rows by inv_eval: T <- diag(inv_eval) * T
    for (int col = 0; col < nrhs; ++col)
    {
        for (int i = 0; i < k; ++i)
        {
            T[i + col * k] *= inv_eval[i];
        }
    }

    // X = Q * T
    for (int col = 0; col < nrhs; ++col)
    {
        for (int i = 0; i < k; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < k; ++j)
            {
                sum += Q[i + j * k] * T[j + col * k]; // Q(i,j)
            }
            X_out[i + col * k] = sum;
        }
    }
}

namespace ichol::solver
{
    // ------------------------- MPCG solver (CUDA 12) -------------------------
    //
    // This function is designed to be wrapped similarly to your pcg(...) interface.
    // A is CSR on host (double). b,x are host dense vectors (double).
    // preconds is a list of k preconditioners (IC factors, etc.).
    //
    // Math mapping (per iteration i):
    //   r_i = b - A x_i
    //   Z_{i+1} = [M1^{-1} r_i | ... | Mk^{-1} r_i]                         (n×k)
    //   P_{i+1} = Z_{i+1} - sum_{j in window} P_j * pinv(P_j^T A P_j) * (P_j^T A Z_{i+1})
    //   alpha_{i+1} = pinv(P_{i+1}^T A P_{i+1}) * (P_{i+1}^T r_i)          (k×1)
    //   x_{i+1} = x_i + P_{i+1} * alpha_{i+1}
    //   r_{i+1} = r_i - A P_{i+1} * alpha_{i+1}
    //
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
        int restart, // 0 => treat as "full" (allocate maxits blocks)
        int &iterations,
        double &finalRes)
    {

        // ------------------------- Basic validation -------------------------
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
            throw std::runtime_error("mpcg: need at least one preconditioner (k>=1)");
        if (maxits <= 0)
            throw std::runtime_error("mpcg: maxits must be > 0");
        if (tol <= 0.0)
            throw std::runtime_error("mpcg: tol must be > 0");

        const int m = (restart <= 0) ? maxits : restart; // truncation window length

        // ------------------------- Create handles -------------------------
        cublasHandle_t cublas = nullptr;
        cusparseHandle_t cusparse = nullptr;
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSPARSE_CHECK(cusparseCreate(&cusparse));

        // Use a dedicated stream for all ops (baseline).
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));

        // ------------------------- Upload CSR(A), b, x -------------------------
        int *d_rowPtrA = nullptr;
        int *d_colIndA = nullptr;
        double *d_valA = nullptr;
        double *d_b = nullptr;
        double *d_x = nullptr;
        double *d_r = nullptr;
        double *d_tmp = nullptr;

        CUDA_CHECK(cudaMalloc((void **)&d_rowPtrA, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_colIndA, nnzA * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_valA, nnzA * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_r, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_tmp, n * sizeof(double)));

        CUDA_CHECK(cudaMemcpyAsync(d_rowPtrA, h_csrRowPtrA.data(), (n + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_colIndA, h_csrColIndA.data(), nnzA * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_valA, h_valA.data(), nnzA * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_x, h_x.data(), n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));

        // ------------------------- cuSPARSE descriptors for A and vectors -------------------------
        cusparseSpMatDescr_t matA = nullptr;
        CUSPARSE_CHECK(cusparseCreateCsr(
            &matA,
            n, n, nnzA,
            d_rowPtrA, d_colIndA, d_valA,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F));

        // DnVec descriptors for SpMV (used to form initial residual r0 = b - A x0)
        cusparseDnVecDescr_t vecX = nullptr, vecTmp = nullptr;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecTmp, n, d_tmp, CUDA_R_64F));

        // ------------------------- Compute initial residual r = b - A*x -------------------------
        // Step: r0 = b - A*x0
        //   - copy r <- b
        //   - tmp <- A*x via cuSPARSE SpMV
        //   - r <- r - tmp via cuBLAS AXPY
        CUDA_CHECK(cudaMemcpyAsync(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice, stream));

        size_t spmv_bufSize = 0;
        void *d_spmvBuf = nullptr;
        double one = 1.0, zero = 0.0, minus_one = -1.0;

        // This block realizes: tmp = A * x  using cuSPARSE SpMV
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA,
            vecX,
            &zero,
            vecTmp,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &spmv_bufSize));
        CUDA_CHECK(cudaMalloc(&d_spmvBuf, spmv_bufSize));

        CUSPARSE_CHECK(cusparseSpMV(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA,
            vecX,
            &zero,
            vecTmp,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            d_spmvBuf));

        // This block realizes: r = r - tmp  using cuBLAS daxpy
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &minus_one, d_tmp, 1, d_r, 1));

        // ------------------------- Norm(b) for stopping criterion -------------------------
        double bnorm = 0.0;
        // This block realizes: bnorm = ||b||_2  using cuBLAS nrm2
        CUBLAS_CHECK(cublasDnrm2(cublas, n, d_b, 1, &bnorm));
        if (bnorm == 0.0)
            bnorm = 1.0; // avoid division by zero in relative stopping

        // ------------------------- Allocate dense blocks (n×k) -------------------------
        // Znew: n×k, Pnew: n×k, Wnew: n×k (Wnew = A*Pnew)
        double *d_Znew = nullptr;
        double *d_Pnew = nullptr;
        double *d_Wnew = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_Znew, (size_t)n * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Pnew, (size_t)n * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Wnew, (size_t)n * k * sizeof(double)));

        // ------------------------- Allocate ring buffers for last m blocks -------------------------
        // P_hist[slot] : n×k, W_hist[slot] : n×k, G_hist[slot] : k×k
        double *d_P_hist = nullptr;
        double *d_W_hist = nullptr;
        double *d_G_hist = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_P_hist, (size_t)m * n * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_W_hist, (size_t)m * n * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_G_hist, (size_t)m * k * k * sizeof(double)));

        // ------------------------- Small device buffers (k×k, k×1) -------------------------
        double *d_C = nullptr;     // C = Wj^T * Znew (k×k)
        double *d_Gnew = nullptr;  // Gnew = Pnew^T * Wnew (k×k)
        double *d_rhs = nullptr;   // rhs = Pnew^T * r (k×1)
        double *d_Y = nullptr;     // Y = pinv(Gj) * C (k×k)
        double *d_alpha = nullptr; // alpha = pinv(Gnew) * rhs (k×1)
        CUDA_CHECK(cudaMalloc((void **)&d_C, (size_t)k * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Gnew, (size_t)k * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_rhs, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Y, (size_t)k * k * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_alpha, (size_t)k * sizeof(double)));

        // Host-side small matrices/vectors for CPU pinv
        std::vector<double> h_G(k * k), h_C(k * k), h_Y(k * k), h_rhs(k), h_alpha(k);

        // ------------------------- cuSPARSE SpMM descriptors for A * (dense n×k) -------------------------
        // Dense matrices are column-major with leading dimension ld = n.
        cusparseDnMatDescr_t dnB = nullptr, dnC = nullptr;
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &dnB, n, k, n,
            d_Pnew, CUDA_R_64F, CUSPARSE_ORDER_COL));
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &dnC, n, k, n,
            d_Wnew, CUDA_R_64F, CUSPARSE_ORDER_COL));

        // Preprocess SpMM because matA stays constant across iterations
        size_t spmm_bufSize = 0;
        void *d_spmmBuf = nullptr;

        // This block realizes: Wnew = A * Pnew  via cuSPARSE SpMM
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA,
            dnB,
            &zero,
            dnC,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            &spmm_bufSize));
        CUDA_CHECK(cudaMalloc(&d_spmmBuf, spmm_bufSize));

        CUSPARSE_CHECK(cusparseSpMM_preprocess(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA,
            dnB,
            &zero,
            dnC,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            d_spmmBuf));

        // ------------------------- Main MPCG loop -------------------------
        iterations = 0;
        finalRes = 0.0;

        for (int iter = 0; iter < maxits; ++iter)
        {
            // Step: compute residual norm ||r||
            // This block realizes: res = ||r||_2 via cuBLAS nrm2
            double res = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res));

            // Stopping criterion: ||r|| <= tol * ||b||
            if (res <= tol * bnorm)
            {
                iterations = iter;
                finalRes = res;
                break;
            }

            // ------------------------------------------------------------------
            // Step (MATLAB): Z(:,istart:iend) = multiprecondition(..., r)
            // Math: Znew = [M1^{-1} r | ... | Mk^{-1} r]  (n×k)
            // ------------------------------------------------------------------
            // Znew(:,t) = M_t^{-1} r  (Option A: callback)
            for (int t = 0; t < k; ++t)
            {
                double *d_z_col = d_Znew + (size_t)t * n; // column t, ld=n

                // This block realizes: z_t = M_t^{-1} r using preconds[t].apply(...)
                preconds[t].apply(preconds[t].ctx, d_r, d_z_col, n, stream);
            }

            // ------------------------------------------------------------------
            // Step: Pnew = Znew  (initialize new search block)
            // In MATLAB: P(:,istart:iend)=Z(:,istart:iend)
            // ------------------------------------------------------------------
            CUDA_CHECK(cudaMemcpyAsync(d_Pnew, d_Znew, (size_t)n * k * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));

            // ------------------------------------------------------------------
            // Step (MATLAB loop over previous blocks):
            //   Pnew = Pnew - Pj * pinv(Pj^T A Pj) * (Pj^T A Znew)
            // We implement:
            //   C = Pj^T A Znew  == (A Pj)^T Znew == Wj^T Znew
            //   Y = pinv(Gj) * C,  where Gj = Pj^T A Pj == Pj^T Wj
            //   Pnew -= Pj * Y
            // ------------------------------------------------------------------
            const int hist_count = std::min(iter, m); // number of stored blocks available
            for (int j = iter - hist_count; j < iter; ++j)
            {
                const int slot = j % m;

                double *d_Pj = d_P_hist + (size_t)slot * n * k; // n×k
                double *d_Wj = d_W_hist + (size_t)slot * n * k; // n×k
                double *d_Gj = d_G_hist + (size_t)slot * k * k; // k×k

                // 1) C = Wj^T * Znew  (k×k)
                // This block realizes: C = (A Pj)^T * Znew = Pj^T A Znew
                // using cuBLAS GEMM: C = Wj^T Znew
                CUBLAS_CHECK(cublasDgemm(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    k, k, n,
                    &one,
                    d_Wj, n,
                    d_Znew, n,
                    &zero,
                    d_C, k));

                // Copy Gj and C to CPU for pinv
                CUDA_CHECK(cudaMemcpyAsync(h_G.data(), d_Gj, (size_t)k * k * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(h_C.data(), d_C, (size_t)k * k * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // 2) Y = pinv(Gj) * C   (k×k), CPU baseline
                // This realizes MATLAB: pinv(Pj^T A Pj) * (Pj^T A Znew)
                pinv_sym_apply(h_G, k, h_C, /*nrhs=*/k, h_Y, /*rcond=*/1e-12);

                // Copy Y back to GPU
                CUDA_CHECK(cudaMemcpyAsync(d_Y, h_Y.data(), (size_t)k * k * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));

                // 3) Pnew = Pnew - Pj * Y  (n×k)
                // This realizes MATLAB: Pnew -= Pj * (...) using cuBLAS GEMM
                CUBLAS_CHECK(cublasDgemm(
                    cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, k, k,
                    &minus_one,
                    d_Pj, n,
                    d_Y, k,
                    &one,
                    d_Pnew, n));
            }

            // ------------------------------------------------------------------
            // Step: Wnew = A * Pnew  (n×k)
            // This realizes MATLAB's repeated "A*Pnew" in alpha/residual update
            // using cuSPARSE SpMM.
            // ------------------------------------------------------------------
            CUSPARSE_CHECK(cusparseDnMatSetValues(dnB, d_Pnew));
            CUSPARSE_CHECK(cusparseDnMatSetValues(dnC, d_Wnew));
            CUSPARSE_CHECK(cusparseSpMM(
                cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA,
                dnB,
                &zero,
                dnC,
                CUDA_R_64F,
                CUSPARSE_SPMM_ALG_DEFAULT,
                d_spmmBuf));

            // ------------------------------------------------------------------
            // Step: Gnew = Pnew^T * Wnew  (k×k)
            // Math: Gnew = P_{i+1}^T A P_{i+1}
            // This realizes MATLAB: (Pnew'*A*Pnew) using cuBLAS GEMM
            // ------------------------------------------------------------------
            CUBLAS_CHECK(cublasDgemm(
                cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                k, k, n,
                &one,
                d_Pnew, n,
                d_Wnew, n,
                &zero,
                d_Gnew, k));

            // ------------------------------------------------------------------
            // Step: rhs = Pnew^T * r  (k×1)
            // This realizes MATLAB: (Pnew' * r) using cuBLAS GEMV
            // ------------------------------------------------------------------
            CUBLAS_CHECK(cublasDgemv(
                cublas,
                CUBLAS_OP_T,
                n, k,
                &one,
                d_Pnew, n,
                d_r, 1,
                &zero,
                d_rhs, 1));

            // Copy Gnew and rhs to CPU for pinv solve
            CUDA_CHECK(cudaMemcpyAsync(h_G.data(), d_Gnew, (size_t)k * k * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_rhs.data(), d_rhs, (size_t)k * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // ------------------------------------------------------------------
            // Step: alpha = pinv(Gnew) * rhs
            // This realizes MATLAB: alpha = pinv(Pnew'*A*Pnew) * (Pnew'*r)
            // CPU baseline pinv for k×k
            // ------------------------------------------------------------------
            {
                // Treat rhs as k×1 (nrhs=1)
                std::vector<double> rhs_mat = h_rhs; // already length k
                pinv_sym_apply(h_G, k, rhs_mat, /*nrhs=*/1, h_alpha, /*rcond=*/1e-12);
            }

            // Copy alpha back to GPU
            CUDA_CHECK(cudaMemcpyAsync(d_alpha, h_alpha.data(), (size_t)k * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

            // ------------------------------------------------------------------
            // Step: x = x + Pnew * alpha
            // This realizes MATLAB: x(:,i+1)=x(:,i)+Pnew*alpha
            // using cuBLAS GEMV with beta=1 (in-place accumulate)
            // ------------------------------------------------------------------
            CUBLAS_CHECK(cublasDgemv(
                cublas,
                CUBLAS_OP_N,
                n, k,
                &one,
                d_Pnew, n,
                d_alpha, 1,
                &one,
                d_x, 1));

            // ------------------------------------------------------------------
            // Step: r = r - Wnew * alpha
            // This realizes MATLAB: r(:,i+1)=r(:,i)-A*Pnew*alpha
            // using cuBLAS GEMV with alpha=-1, beta=1
            // ------------------------------------------------------------------
            CUBLAS_CHECK(cublasDgemv(
                cublas,
                CUBLAS_OP_N,
                n, k,
                &minus_one,
                d_Wnew, n,
                d_alpha, 1,
                &one,
                d_r, 1));

            // ------------------------------------------------------------------
            // Cache current block into ring:
            //   P_hist[slot] = Pnew
            //   W_hist[slot] = Wnew
            //   G_hist[slot] = Gnew
            // This realizes storing P_j and (P_j^T A P_j) needed for future projections.
            // ------------------------------------------------------------------
            const int slot = iter % m;
            CUDA_CHECK(cudaMemcpyAsync(d_P_hist + (size_t)slot * n * k, d_Pnew,
                                       (size_t)n * k * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_W_hist + (size_t)slot * n * k, d_Wnew,
                                       (size_t)n * k * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_G_hist + (size_t)slot * k * k, d_Gnew,
                                       (size_t)k * k * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));

            iterations = iter + 1;

            // If we are at the last iteration and didn't early-exit, record finalRes.
            if (iter == maxits - 1)
            {
                double res_end = 0.0;
                CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res_end));
                finalRes = res_end;
            }
        }

        // If early-exit happened with finalRes unset in loop, set it now.
        if (finalRes == 0.0)
        {
            double res_end = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res_end));
            finalRes = res_end;
        }

        // ------------------------- Download x -------------------------
        CUDA_CHECK(cudaMemcpyAsync(h_x.data(), d_x, n * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ------------------------- Cleanup -------------------------
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
