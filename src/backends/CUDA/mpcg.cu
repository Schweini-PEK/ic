// src/solver/mpcg.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cmath>

#include "ichol/pcg.hpp"
#include "ichol/preconditioner.hpp"

#define CUDA_CHECK(call)                                                              \
    do                                                                                \
    {                                                                                 \
        cudaError_t _e = (call);                                                      \
        if (_e != cudaSuccess)                                                        \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
    } while (0)

#define CUBLAS_CHECK(call)                            \
    do                                                \
    {                                                 \
        cublasStatus_t _s = (call);                   \
        if (_s != CUBLAS_STATUS_SUCCESS)              \
            throw std::runtime_error("cuBLAS Error"); \
    } while (0)

#define CUSPARSE_CHECK(call)                            \
    do                                                  \
    {                                                   \
        cusparseStatus_t _s = (call);                   \
        if (_s != CUSPARSE_STATUS_SUCCESS)              \
            throw std::runtime_error("cuSPARSE Error"); \
    } while (0)

#define CUSOLVER_CHECK(call)                            \
    do                                                  \
    {                                                   \
        cusolverStatus_t _s = (call);                   \
        if (_s != CUSOLVER_STATUS_SUCCESS)              \
            throw std::runtime_error("cuSOLVER Error"); \
    } while (0)

struct CublasScalars
{
    // cuBLAS APIs take host pointers to scalar alpha/beta values.
    // Keep both fp32/fp64 constants in one place so we can pick at runtime.
    float s32_one = 1.0f;
    float s32_zero = 0.0f;
    float s32_m_one = -1.0f;

    double s64_one = 1.0;
    double s64_zero = 0.0;
    double s64_m_one = -1.0;

    // Return pointer to scalar with the right host type for selected compute type.
    void *one(cublasComputeType_t ct) { return (ct == CUBLAS_COMPUTE_64F) ? (void *)&s64_one : (void *)&s32_one; }
    void *zero(cublasComputeType_t ct) { return (ct == CUBLAS_COMPUTE_64F) ? (void *)&s64_zero : (void *)&s32_zero; }
    void *m_one(cublasComputeType_t ct) { return (ct == CUBLAS_COMPUTE_64F) ? (void *)&s64_m_one : (void *)&s32_m_one; }
};

struct PrecisionMap
{
    cudaDataType_t data_type;
    cublasComputeType_t compute_type;
    size_t el_size;
};

/**
 * Map user precision choice to CUDA data type and cuBLAS compute type.
 */
static PrecisionMap get_precision_map(ichol::solver::ComputePrecision prec, ichol::solver::ComputePrecision /*acc*/)
{
    PrecisionMap m{};
    if (prec == ichol::solver::ComputePrecision::FP64)
    {
        m.data_type = CUDA_R_64F;
        m.compute_type = CUBLAS_COMPUTE_64F;
        m.el_size = 8;
    }
    else
    {
        if (prec == ichol::solver::ComputePrecision::FP16)
            m.data_type = CUDA_R_16F;
        else if (prec == ichol::solver::ComputePrecision::BF16)
            m.data_type = CUDA_R_16BF;
        else
            m.data_type = CUDA_R_32F;

        if (prec == ichol::solver::ComputePrecision::TF32)
            m.compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
        else
            m.compute_type = CUBLAS_COMPUTE_32F;

        m.el_size = (prec == ichol::solver::ComputePrecision::FP16 ||
                     prec == ichol::solver::ComputePrecision::BF16)
                        ? 2
                        : 4;
    }
    return m;
}

struct StorageMap
{
    cudaDataType_t data_type;
    size_t el_size;
};

/**
 * Map user precision choice to CUDA data type for history storage buffers.
 */
static StorageMap get_storage_map(ichol::solver::ComputePrecision prec)
{
    StorageMap m{};
    if (prec == ichol::solver::ComputePrecision::FP64)
    {
        m.data_type = CUDA_R_64F;
        m.el_size = 8;
    }
    else if (prec == ichol::solver::ComputePrecision::FP16)
    {
        m.data_type = CUDA_R_16F;
        m.el_size = 2;
    }
    else if (prec == ichol::solver::ComputePrecision::BF16)
    {
        m.data_type = CUDA_R_16BF;
        m.el_size = 2;
    }
    else
    {
        m.data_type = CUDA_R_32F;
        m.el_size = 4;
    }
    return m;
}

static double get_safe_rcond(ichol::solver::ComputePrecision prec, double base_rcond)
{
    // rcond controls singular-value cutoff in pseudo-inverse:
    // singular values <= rcond * sigma_max are treated as zero.
    //
    // Lower precision needs a larger cutoff to avoid unstable inversions.
    if (prec == ichol::solver::ComputePrecision::FP64)
        return base_rcond;
    if (prec == ichol::solver::ComputePrecision::FP32 || prec == ichol::solver::ComputePrecision::TF32)
        return std::max(base_rcond, 1e-7);
    return std::max(base_rcond, 1e-3);
}

__global__ void k_cast_d2any(const double *src, void *dst, int N, ichol::solver::ComputePrecision prec)
{
    // Generic elementwise cast kernel:
    // each thread handles one index i.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    if (prec == ichol::solver::ComputePrecision::FP64)
        ((double *)dst)[i] = src[i];
    else if (prec == ichol::solver::ComputePrecision::FP32 || prec == ichol::solver::ComputePrecision::TF32)
        ((float *)dst)[i] = (float)src[i];
    else if (prec == ichol::solver::ComputePrecision::FP16)
        ((__half *)dst)[i] = (__half)src[i];
    else if (prec == ichol::solver::ComputePrecision::BF16)
        ((__nv_bfloat16 *)dst)[i] = (__nv_bfloat16)src[i];
}

__global__ void k_cast_any2d_vec(const void *src, double *dst, int N, ichol::solver::ComputePrecision prec)
{
    // Reverse cast: lower-precision storage back to fp64 workspace.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    if (prec == ichol::solver::ComputePrecision::FP64)
        dst[i] = ((const double *)src)[i];
    else if (prec == ichol::solver::ComputePrecision::FP32 || prec == ichol::solver::ComputePrecision::TF32)
        dst[i] = (double)((const float *)src)[i];
    else if (prec == ichol::solver::ComputePrecision::FP16)
        dst[i] = (double)((const __half *)src)[i];
    else if (prec == ichol::solver::ComputePrecision::BF16)
        dst[i] = (double)((const __nv_bfloat16 *)src)[i];
}

__global__ void k_cast_f2d_mat(const float *src, double *dst, int N)
{
    // Specialized cast for GEMM outputs produced in fp32.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    dst[i] = (double)src[i];
}

__global__ void k_row_scaling(double *T, const double *S_inv, int k, int nrhs)
{
    // T is k x nrhs (column-major).
    // One block with k threads; thread "row" scales one row across all rhs cols.
    // This applies diag(S_inv) * T.
    int row = threadIdx.x;
    if (row < k)
    {
        double s = S_inv[row];
        for (int col = 0; col < nrhs; ++col)
            T[row + col * k] *= s;
    }
}

struct PinvSVDWorkspace
{
    double *d_G_copy = nullptr;
    double *d_S = nullptr;
    double *d_U = nullptr;
    double *d_VT = nullptr;
    double *d_work = nullptr;
    double *d_S_inv = nullptr;
    double *d_T1 = nullptr;
    int *d_info = nullptr;
    int lwork = 0;
    int k = 0;
    int max_nrhs = 0;
};

// For very small systems, cuSOLVER launch/setup overhead can dominate.
// This kernel solves G * X = B using in-kernel LU with partial pivoting.
// It runs with one thread because k is tiny (<= 32).
__global__ void k_small_lu_solve(
    const double *G,
    const double *B,
    double *X,
    int k,
    int nrhs,
    double rcond)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    constexpr int KMAX = 32;
    double A[KMAX * KMAX];
    double RHS[KMAX * KMAX];

    double max_abs = 0.0;
    for (int col = 0; col < k; ++col)
    {
        for (int row = 0; row < k; ++row)
        {
            const double v = G[row + col * k];
            A[row + col * k] = v;
            max_abs = fmax(max_abs, fabs(v));
        }
    }
    for (int col = 0; col < nrhs; ++col)
        for (int row = 0; row < k; ++row)
            RHS[row + col * k] = B[row + col * k];

    const double pivot_tol = fmax(1e-15, rcond * fmax(max_abs, 1.0));

    // LU factorization with partial pivoting.
    for (int col = 0; col < k; ++col)
    {
        int piv = col;
        double piv_abs = fabs(A[col + col * k]);
        for (int row = col + 1; row < k; ++row)
        {
            const double cur = fabs(A[row + col * k]);
            if (cur > piv_abs)
            {
                piv_abs = cur;
                piv = row;
            }
        }

        if (piv != col)
        {
            for (int j = col; j < k; ++j)
            {
                const double tmp = A[col + j * k];
                A[col + j * k] = A[piv + j * k];
                A[piv + j * k] = tmp;
            }
            for (int j = 0; j < nrhs; ++j)
            {
                const double tmp = RHS[col + j * k];
                RHS[col + j * k] = RHS[piv + j * k];
                RHS[piv + j * k] = tmp;
            }
        }

        // Diagonal floor avoids numerical blow-up on near-singular columns.
        if (fabs(A[col + col * k]) < pivot_tol)
            A[col + col * k] = (A[col + col * k] < 0.0) ? -pivot_tol : pivot_tol;

        const double inv_diag = 1.0 / A[col + col * k];
        for (int row = col + 1; row < k; ++row)
        {
            const double f = A[row + col * k] * inv_diag;
            A[row + col * k] = 0.0;
            for (int j = col + 1; j < k; ++j)
                A[row + j * k] -= f * A[col + j * k];
            for (int j = 0; j < nrhs; ++j)
                RHS[row + j * k] -= f * RHS[col + j * k];
        }
    }

    // Back-substitution.
    for (int j = 0; j < nrhs; ++j)
    {
        for (int i = k - 1; i >= 0; --i)
        {
            double sum = RHS[i + j * k];
            for (int c = i + 1; c < k; ++c)
                sum -= A[i + c * k] * X[c + j * k];
            double d = A[i + i * k];
            if (fabs(d) < pivot_tol)
                d = (d < 0.0) ? -pivot_tol : pivot_tol;
            X[i + j * k] = sum / d;
        }
    }
}

/**
 * k_small_pinv_jacobi_svd:
 * Performs Moore-Penrose Pseudo-inverse for G * X = B for small K.
 * Uses a one-sided Jacobi SVD algorithm.
 */
__global__ void k_small_pinv_jacobi_svd(
    const double *G, // k x k
    const double *B, // k x nrhs
    double *X,       // k x nrhs
    int k,
    int nrhs,
    double rcond)
{
    // Use a single thread for k <= 32 to avoid shared memory sync overhead
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    constexpr int KMAX = 32;
    double U[KMAX * KMAX];
    double V[KMAX * KMAX];
    double S[KMAX];

    // Initialize U as symmetrized G and V as Identity.
    // G should be symmetric (P^T A P), but GEMM roundoff can introduce drift.
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            const double gij = G[i + j * k];
            const double gji = G[j + i * k];
            U[i + j * k] = 0.5 * (gij + gji);
            V[i + j * k] = (i == j) ? 1.0 : 0.0;
        }
    }

    // One-sided Jacobi SVD iterations
    for (int iter = 0; iter < 20; ++iter)
    {
        for (int i = 0; i < k - 1; ++i)
        {
            for (int j = i + 1; j < k; ++j)
            {
                double a = 0, b = 0, c = 0;
                for (int m = 0; m < k; ++m)
                {
                    a += U[m + i * k] * U[m + i * k];
                    b += U[m + j * k] * U[m + j * k];
                    c += U[m + i * k] * U[m + j * k];
                }

                if (fabs(c) <= 1e-15 * sqrt(a * b))
                    continue;

                double zeta = (b - a) / (2.0 * c);
                double t = (zeta > 0 ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                double cos_th = 1.0 / sqrt(1.0 + t * t);
                double sin_th = cos_th * t;

                for (int m = 0; m < k; ++m)
                {
                    double u_i = U[m + i * k];
                    double u_j = U[m + j * k];
                    U[m + i * k] = cos_th * u_i - sin_th * u_j;
                    U[m + j * k] = sin_th * u_i + cos_th * u_j;

                    double v_i = V[m + i * k];
                    double v_j = V[m + j * k];
                    V[m + i * k] = cos_th * v_i - sin_th * v_j;
                    V[m + j * k] = sin_th * v_i + cos_th * v_j;
                }
            }
        }
    }

    // Compute singular values (norms of columns of U) and normalize U
    for (int j = 0; j < k; ++j)
    {
        double norm = 0;
        for (int i = 0; i < k; ++i)
            norm += U[i + j * k] * U[i + j * k];
        S[j] = sqrt(norm);
        if (S[j] > 1e-15)
        {
            for (int i = 0; i < k; ++i)
                U[i + j * k] /= S[j];
        }
    }

    // Thresholding (pinv logic)
    double s_max = 0;
    for (int i = 0; i < k; ++i)
        if (S[i] > s_max)
            s_max = S[i];
    if (s_max == 0.0)
    {
        for (int rhs_col = 0; rhs_col < nrhs; ++rhs_col)
        {
            for (int i = 0; i < k; ++i)
                X[i + rhs_col * k] = 0.0;
        }
        return;
    }

    // Robust threshold: relative + absolute floor.
    double threshold = fmax(rcond * s_max, 1e-15 * s_max);
    threshold = fmax(threshold, 1e-15);

    // Solve: X = V * S_inv * U^T * B
    for (int rhs_col = 0; rhs_col < nrhs; ++rhs_col)
    {
        double ut_b[KMAX];
        for (int i = 0; i < k; ++i)
        {
            double sum = 0;
            for (int m = 0; m < k; ++m)
                sum += U[m + i * k] * B[m + rhs_col * k];
            ut_b[i] = (S[i] > threshold) ? (sum / S[i]) : 0.0;
        }

        for (int i = 0; i < k; ++i)
        {
            double res = 0;
            for (int m = 0; m < k; ++m)
                res += V[i + m * k] * ut_b[m];
            X[i + rhs_col * k] = res;
        }
    }
}

__global__ void k_build_s_inv_from_s(const double *S, double *S_inv, int k, double rcond)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k)
        return;

    double a = rcond * S[0];
    const double thresh = fmax(a, 1e-15);
    const double si = S[i];
    S_inv[i] = (si > thresh) ? (1.0 / si) : 0.0;
}

static void pinv_svd_cuda(
    cusolverDnHandle_t cusolver,
    cublasHandle_t cublas,
    const double *d_G,
    int k,
    const double *d_B,
    int nrhs,
    double *d_X,
    cudaStream_t stream,
    double rcond,
    PinvSVDWorkspace &ws)
{
    // Solve X = pinv(G) * B using SVD on GPU.
    //
    // For G = U * diag(S) * V^T, Moore-Penrose pseudo-inverse is:
    // pinv(G) = V * diag(S_inv) * U^T, where small singular values are dropped.
    //
    // We then compute:
    // T1 = U^T * B
    // T1 = diag(S_inv) * T1
    // X  = V * T1
    //
    // This is used for small kxk systems that can be ill-conditioned.
    // Fast path for tiny systems: direct LU solve in one kernel launch.
    if (k <= 32)
    {
        // k_small_lu_solve<<<1, 1, 0, stream>>>(d_G, d_B, d_X, k, nrhs, rcond);
        k_small_pinv_jacobi_svd<<<1, 1, 0, stream>>>(d_G, d_B, d_X, k, nrhs, rcond);
        return;
    }

    if (k != ws.k || nrhs > ws.max_nrhs)
        throw std::runtime_error("pinv_svd_cuda: workspace shape mismatch");

    CUDA_CHECK(cudaMemcpyAsync(ws.d_G_copy, d_G, k * k * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    // Full SVD: all singular vectors (jobu='A', jobvt='A').
    CUSOLVER_CHECK(cusolverDnDgesvd(
        cusolver, 'A', 'A',
        k, k,
        ws.d_G_copy, k,
        ws.d_S,
        ws.d_U, k,
        ws.d_VT, k,
        ws.d_work, ws.lwork,
        nullptr,
        ws.d_info));

    // Build reciprocal singular values with thresholding on device:
    // S_inv[i] = 1/S[i] if S[i] > rcond*S[0], else 0.
    const int threads = 128;
    const int blocks = (k + threads - 1) / threads;
    k_build_s_inv_from_s<<<blocks, threads, 0, stream>>>(ws.d_S, ws.d_S_inv, k, rcond);

    double one = 1.0, zero = 0.0;
    // d_T1 = U^T * B
    CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, nrhs, k, &one, ws.d_U, k, d_B, k, &zero, ws.d_T1, k));
    // d_T1 = diag(S_inv) * d_T1
    k_row_scaling<<<1, k, 0, stream>>>(ws.d_T1, ws.d_S_inv, k, nrhs);
    // d_X = V * d_T1  (note: cuSOLVER gives V^T, so V is VT^T)
    CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, nrhs, k, &one, ws.d_VT, k, ws.d_T1, k, &zero, d_X, k));
}

namespace ichol::solver
{

    template <typename T_L>
    PCGResult mpcg(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const PCGParams &params)
    {
        const int n = static_cast<int>(h_b.size());                           // matrix size
        const int k = static_cast<int>(preconds.size());                      // # of precond
        const int m = (params.restart <= 0) ? params.maxits : params.restart; // history size (0 or negative means no restarts, i.e. full history up to maxits)
        const int64_t nnzA = (int64_t)h_valA.size();

        // Create library handles and bind them to one non-blocking stream so
        // all operations are ordered on the same stream.
        cublasHandle_t cublas;
        CUBLAS_CHECK(cublasCreate(&cublas));
        cusparseHandle_t cusparse;
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        cusolverDnHandle_t cusolver;
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));

        CublasScalars scalars;
        PinvSVDWorkspace pinv_ws{};

        // Precision controls:
        PrecisionMap g_map = get_precision_map(params.prec_gemm, params.prec_acc);

        StorageMap P_hist_map = get_storage_map(params.store_P_hist);
        StorageMap W_hist_map = get_storage_map(params.store_W_hist);

        int *d_rowPtrA = nullptr, *d_colIndA = nullptr;
        double *d_valA = nullptr, *d_b = nullptr, *d_x = nullptr, *d_r = nullptr, *d_tmp = nullptr;

        CUDA_CHECK(cudaMalloc(&d_rowPtrA, (size_t)(n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIndA, (size_t)nnzA * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_valA, (size_t)nnzA * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)n * sizeof(double)));

        // Upload host problem data to GPU.
        CUDA_CHECK(cudaMemcpyAsync(d_rowPtrA, h_csrRowPtrA.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_colIndA, h_csrColIndA.data(), (size_t)nnzA * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_valA, h_valA.data(), (size_t)nnzA * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_x, h_x.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice, stream));

        void *d_Pnew_low = nullptr, *d_Wnew_low = nullptr, *d_Wj_low = nullptr;
        float *d_C_gemm = nullptr;

        // Extra low-precision buffers only needed when GEMM is not fp64.
        if (params.prec_gemm != ComputePrecision::FP64)
        {
            CUDA_CHECK(cudaMalloc(&d_Pnew_low, (size_t)n * k * g_map.el_size));
            CUDA_CHECK(cudaMalloc(&d_Wnew_low, (size_t)n * k * g_map.el_size));
            CUDA_CHECK(cudaMalloc(&d_Wj_low, (size_t)n * k * g_map.el_size));
            CUDA_CHECK(cudaMalloc(&d_C_gemm, (size_t)k * k * sizeof(float)));
        }

        const size_t nk = (size_t)n * (size_t)k;

        // Main block vectors (all n x k, column-major):
        // Znew: preconditioned directions (raw)
        // Pnew: orthogonalized search directions
        // Wnew: A * Pnew
        double *d_Znew = nullptr, *d_Pnew = nullptr, *d_Wnew = nullptr;
        double *d_G_hist = nullptr, *d_C = nullptr, *d_Gnew = nullptr, *d_rhs = nullptr, *d_Y = nullptr, *d_alpha = nullptr;

        CUDA_CHECK(cudaMalloc(&d_Znew, nk * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Pnew, nk * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Wnew, nk * sizeof(double)));

        CUDA_CHECK(cudaMalloc(&d_G_hist, (size_t)m * (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Gnew, (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rhs, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Y, (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_alpha, (size_t)k * sizeof(double)));

        // Workspace for repeated pseudo-inverse solves on larger systems.
        // For k <= 32 we use the tiny LU kernel path and skip this allocation.
        if (k > 32)
        {
            pinv_ws.k = k;
            pinv_ws.max_nrhs = k;
            CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolver, k, k, &pinv_ws.lwork));
            CUDA_CHECK(cudaMalloc(&pinv_ws.d_G_copy, (size_t)k * (size_t)k * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&pinv_ws.d_S, (size_t)k * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&pinv_ws.d_U, (size_t)k * (size_t)k * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&pinv_ws.d_VT, (size_t)k * (size_t)k * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&pinv_ws.d_work, (size_t)pinv_ws.lwork * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&pinv_ws.d_S_inv, (size_t)k * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&pinv_ws.d_T1, (size_t)k * (size_t)k * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&pinv_ws.d_info, sizeof(int)));
        }

        double *d_P_hist64 = nullptr, *d_W_hist64 = nullptr;
        void *d_P_histLP = nullptr, *d_W_histLP = nullptr;

        // Circular history buffers (size m) store previous P/W/G blocks.
        // Can be fp64 or compressed precision depending on params.
        if (params.store_P_hist == ComputePrecision::FP64)
            CUDA_CHECK(cudaMalloc(&d_P_hist64, (size_t)m * nk * sizeof(double)));
        else
            CUDA_CHECK(cudaMalloc(&d_P_histLP, (size_t)m * nk * P_hist_map.el_size));

        if (params.store_W_hist == ComputePrecision::FP64)
            CUDA_CHECK(cudaMalloc(&d_W_hist64, (size_t)m * nk * sizeof(double)));
        else
            CUDA_CHECK(cudaMalloc(&d_W_histLP, (size_t)m * nk * W_hist_map.el_size));

        double *d_Pj_tmp = nullptr, *d_Wj_tmp = nullptr;
        if (d_P_histLP)
            CUDA_CHECK(cudaMalloc(&d_Pj_tmp, nk * sizeof(double)));
        if (d_W_histLP)
            CUDA_CHECK(cudaMalloc(&d_Wj_tmp, nk * sizeof(double)));

        void *d_scratch_LP = nullptr;
        size_t max_lp_size = std::max(P_hist_map.el_size, W_hist_map.el_size);
        // Scratch buffer for round-trip quantization when storing in lower precision.
        if (params.store_P_hist != ComputePrecision::FP64 || params.store_W_hist != ComputePrecision::FP64)
        {
            CUDA_CHECK(cudaMalloc(&d_scratch_LP, nk * max_lp_size));
        }

        // Create cuSPARSE descriptors:
        // matA: sparse CSR matrix A
        // dnB/dnC: dense n x k mats for SpMM
        // vecX/vecTmp: dense vectors for SpMV
        cusparseSpMatDescr_t matA;
        CUSPARSE_CHECK(cusparseCreateCsr(
            &matA, n, n, nnzA,
            d_rowPtrA, d_colIndA, d_valA,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F));

        cusparseDnMatDescr_t dnB, dnC;
        CUSPARSE_CHECK(cusparseCreateDnMat(&dnB, n, k, n, d_Pnew, CUDA_R_64F, CUSPARSE_ORDER_COL));
        CUSPARSE_CHECK(cusparseCreateDnMat(&dnC, n, k, n, d_Wnew, CUDA_R_64F, CUSPARSE_ORDER_COL));

        cusparseDnVecDescr_t vecX, vecTmp;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecTmp, n, d_tmp, CUDA_R_64F));

        size_t spmv_bufSize = 0;
        void *d_spmvBuf = nullptr;

        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &scalars.s64_one, matA, vecX,
            &scalars.s64_zero, vecTmp,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_bufSize));
        CUDA_CHECK(cudaMalloc(&d_spmvBuf, spmv_bufSize));

        // tmp = A * x
        CUSPARSE_CHECK(cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &scalars.s64_one, matA, vecX,
            &scalars.s64_zero, vecTmp,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf));

        // Initial residual r = b - A*x
        CUDA_CHECK(cudaMemcpyAsync(d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &scalars.s64_m_one, d_tmp, 1, d_r, 1));

        // Use ||b|| as denominator for relative residual stopping criterion.
        double bnorm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublas, n, d_b, 1, &bnorm));
        if (bnorm == 0.0)
            bnorm = 1.0;

        size_t spmm_bufSize = 0;
        void *d_spmmBuf = nullptr;

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &scalars.s64_one, matA, dnB,
            &scalars.s64_zero, dnC,
            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &spmm_bufSize));
        CUDA_CHECK(cudaMalloc(&d_spmmBuf, spmm_bufSize));

        PCGResult result{};
        bool converged = false;
        const int threads = 256;
        const int blocks_nk = (int)((nk + (size_t)threads - 1) / (size_t)threads);
        const int blocks_kk = (int)(((size_t)k * (size_t)k + (size_t)threads - 1) / (size_t)threads);
        double hist_rcond = get_safe_rcond(params.store_P_hist, params.rcond_base);
        hist_rcond = std::max(hist_rcond, 1e-15);

        // Main MPCG iteration.
        for (int iter = 0; iter < params.maxits; ++iter)
        {
            // Check convergence: ||r|| <= tol * ||b||
            double res_norm = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res_norm));
            if (res_norm <= params.tol * bnorm)
            {
                result.iterations = iter;
                result.finalRes = res_norm;
                converged = true;
                break;
            }

            // Keep descriptor pointers in sync with current data buffers.
            CUSPARSE_CHECK(cusparseDnMatSetValues(dnB, d_Pnew));
            CUSPARSE_CHECK(cusparseDnMatSetValues(dnC, d_Wnew));

            // CPU-side preconditioners may read d_r on host via D2H.
            // Ensure all prior stream work that produces d_r is complete first.
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Apply each preconditioner to the same residual r.
            // Output columns form Znew(:, t).
            for (int t = 0; t < k; ++t)
            {
                // Each preconditioner owns only its column buffer.
                CUDA_CHECK(cudaMemsetAsync(d_Znew + (size_t)t * (size_t)n, 0, (size_t)n * sizeof(double), stream));
                preconds[t].apply(preconds[t].ctx, d_r, d_Znew + (size_t)t * (size_t)n, n, stream);
            }

            // Start from Znew, then orthogonalize against history below.
            CUDA_CHECK(cudaMemcpyAsync(d_Pnew, d_Znew, nk * sizeof(double), cudaMemcpyDeviceToDevice, stream));

            // Number of previous iterations currently available (capped by m).
            const int hist_count = std::min(iter, m);
            for (int jj = iter - hist_count; jj < iter; ++jj)
            {
                // Ring-buffer slot for historical block jj.
                const int slot = jj % m;

                double *d_Pj = nullptr;
                double *d_Wj = nullptr;

                if (d_P_hist64)
                {
                    // Fast path: history already in fp64.
                    d_Pj = d_P_hist64 + (size_t)slot * nk;
                }
                else
                {
                    // Compressed path: cast slot -> temporary fp64 workspace.
                    const size_t offB = (size_t)slot * nk * P_hist_map.el_size;
                    const void *src = (const void *)((const char *)d_P_histLP + offB);
                    k_cast_any2d_vec<<<blocks_nk, threads, 0, stream>>>(src, d_Pj_tmp, (int)nk, params.store_P_hist);
                    d_Pj = d_Pj_tmp;
                }

                if (d_W_hist64)
                {
                    d_Wj = d_W_hist64 + (size_t)slot * nk;
                }
                else
                {
                    const size_t offB = (size_t)slot * nk * W_hist_map.el_size;
                    const void *src = (const void *)((const char *)d_W_histLP + offB);
                    k_cast_any2d_vec<<<blocks_nk, threads, 0, stream>>>(src, d_Wj_tmp, (int)nk, params.store_W_hist);
                    d_Wj = d_Wj_tmp;
                }

                double *d_Gj = d_G_hist + (size_t)slot * (size_t)k * (size_t)k;

                if (params.prec_gemm == ComputePrecision::FP64)
                {
                    // C = Wj^T * Pnew (k x k), computed directly in fp64.
                    CUBLAS_CHECK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        k, k, n,
                        scalars.one(g_map.compute_type),
                        d_Wj, CUDA_R_64F, n,
                        d_Pnew, CUDA_R_64F, n,
                        scalars.zero(g_map.compute_type),
                        d_C, CUDA_R_64F, k,
                        g_map.compute_type, CUBLAS_GEMM_DEFAULT));
                }
                else
                {
                    // Cast inputs to selected low precision, run GEMM, cast result back.
                    k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Wj, d_Wj_low, (int)nk, params.prec_gemm);
                    k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Pnew, d_Pnew_low, (int)nk, params.prec_gemm);

                    CUBLAS_CHECK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        k, k, n,
                        scalars.one(g_map.compute_type),
                        d_Wj_low, g_map.data_type, n,
                        d_Pnew_low, g_map.data_type, n,
                        scalars.zero(g_map.compute_type),
                        d_C_gemm, CUDA_R_32F, k,
                        g_map.compute_type, CUBLAS_GEMM_DEFAULT));

                    k_cast_f2d_mat<<<blocks_kk, threads, 0, stream>>>(d_C_gemm, d_C, k * k);
                }

                // Solve Y = pinv(Gj) * C.
                // Y are projection coefficients that remove old-direction components.
                pinv_svd_cuda(cusolver, cublas, d_Gj, k, d_C, k, d_Y, stream, hist_rcond, pinv_ws);

                CUBLAS_CHECK(cublasDgemm(
                    cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, k, k,
                    &scalars.s64_m_one,
                    d_Pj, n,
                    d_Y, k,
                    &scalars.s64_one,
                    d_Pnew, n));
            }

            // Normalize each search-direction column to keep Gram diagonals well-scaled.
            // Here we enforce ||p_t||_2 = 1.
            for (int t = 0; t < k; ++t)
            {
                double *d_pt = d_Pnew + (size_t)t * (size_t)n;
                double pnorm = 0.0;
                CUBLAS_CHECK(cublasDnrm2(cublas, n, d_pt, 1, &pnorm));
                if (pnorm > 1e-30)
                {
                    const double inv_pnorm = 1.0 / pnorm;
                    CUBLAS_CHECK(cublasDscal(cublas, n, &inv_pnorm, d_pt, 1));
                }
                else
                {
                    CUDA_CHECK(cudaMemsetAsync(d_pt, 0, (size_t)n * sizeof(double), stream));
                }
            }

            if (params.store_P_hist != ComputePrecision::FP64)
            {
                // Optional "store precision" simulation:
                // quantize then dequantize Pnew so later computations see that loss.
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Pnew, d_scratch_LP, (int)nk, params.store_P_hist);
                k_cast_any2d_vec<<<blocks_nk, threads, 0, stream>>>(d_scratch_LP, d_Pnew, (int)nk, params.store_P_hist);
            }

            // Wnew = A * Pnew (sparse-dense matrix multiply).
            CUSPARSE_CHECK(cusparseSpMM(
                cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &scalars.s64_one, matA, dnB,
                &scalars.s64_zero, dnC,
                CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmmBuf));

            if (params.store_W_hist != ComputePrecision::FP64)
            {
                // Same quantize/dequantize for Wnew if requested.
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Wnew, d_scratch_LP, (int)nk, params.store_W_hist);
                k_cast_any2d_vec<<<blocks_nk, threads, 0, stream>>>(d_scratch_LP, d_Wnew, (int)nk, params.store_W_hist);
            }

            if (params.prec_gemm == ComputePrecision::FP64)
            {
                // Gnew = Pnew^T * Wnew (k x k Gram-like matrix).
                CUBLAS_CHECK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    k, k, n,
                    scalars.one(g_map.compute_type),
                    d_Pnew, CUDA_R_64F, n,
                    d_Wnew, CUDA_R_64F, n,
                    scalars.zero(g_map.compute_type),
                    d_Gnew, CUDA_R_64F, k,
                    g_map.compute_type, CUBLAS_GEMM_DEFAULT));
            }
            else
            {
                // Low-precision GEMM path for Gnew.
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Pnew, d_Pnew_low, (int)nk, params.prec_gemm);
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Wnew, d_Wnew_low, (int)nk, params.prec_gemm);

                CUBLAS_CHECK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    k, k, n,
                    scalars.one(g_map.compute_type),
                    d_Pnew_low, g_map.data_type, n,
                    d_Wnew_low, g_map.data_type, n,
                    scalars.zero(g_map.compute_type),
                    d_C_gemm, CUDA_R_32F, k,
                    g_map.compute_type, CUBLAS_GEMM_DEFAULT));

                k_cast_f2d_mat<<<blocks_kk, threads, 0, stream>>>(d_C_gemm, d_Gnew, k * k);
            }

            // rhs = Pnew^T * r (size k)
            CUBLAS_CHECK(cublasDgemv(
                cublas, CUBLAS_OP_T,
                n, k,
                &scalars.s64_one,
                d_Pnew, n,
                d_r, 1,
                &scalars.s64_zero,
                d_rhs, 1));

            // alpha = pinv(Gnew) * rhs (block step coefficients).
            pinv_svd_cuda(cusolver, cublas, d_Gnew, k, d_rhs, 1, d_alpha, stream, hist_rcond, pinv_ws);

            // x = x + Pnew * alpha
            CUBLAS_CHECK(cublasDgemv(
                cublas, CUBLAS_OP_N,
                n, k,
                &scalars.s64_one,
                d_Pnew, n,
                d_alpha, 1,
                &scalars.s64_one,
                d_x, 1));

            // r = r - Wnew * alpha
            CUBLAS_CHECK(cublasDgemv(
                cublas, CUBLAS_OP_N,
                n, k,
                &scalars.s64_m_one,
                d_Wnew, n,
                d_alpha, 1,
                &scalars.s64_one,
                d_r, 1));

            // Periodically reset recursive residual drift: r = b - A*x.
            if ((iter + 1) % 50 == 0)
            {
                CUSPARSE_CHECK(cusparseSpMV(
                    cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &scalars.s64_one, matA, vecX,
                    &scalars.s64_zero, vecTmp,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf));
                CUDA_CHECK(cudaMemcpyAsync(d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
                CUBLAS_CHECK(cublasDaxpy(cublas, n, &scalars.s64_m_one, d_tmp, 1, d_r, 1));
            }

            // Save current blocks/matrix in ring-buffer slot for future orthogonalization.
            const int slot = iter % m;

            if (d_P_hist64)
            {
                double *Pdst = d_P_hist64 + (size_t)slot * nk;
                CUDA_CHECK(cudaMemcpyAsync(Pdst, d_Pnew, nk * sizeof(double), cudaMemcpyDeviceToDevice, stream));
            }
            else
            {
                const size_t offB = (size_t)slot * nk * P_hist_map.el_size;
                void *dstP = (void *)((char *)d_P_histLP + offB);
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Pnew, dstP, (int)nk, params.store_P_hist);
            }

            if (d_W_hist64)
            {
                double *Wdst = d_W_hist64 + (size_t)slot * nk;
                CUDA_CHECK(cudaMemcpyAsync(Wdst, d_Wnew, nk * sizeof(double), cudaMemcpyDeviceToDevice, stream));
            }
            else
            {
                const size_t offB = (size_t)slot * nk * W_hist_map.el_size;
                void *dstW = (void *)((char *)d_W_histLP + offB);
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Wnew, dstW, (int)nk, params.store_W_hist);
            }

            double *Gdst = d_G_hist + (size_t)slot * (size_t)k * (size_t)k;
            CUDA_CHECK(cudaMemcpyAsync(Gdst, d_Gnew, k * k * sizeof(double), cudaMemcpyDeviceToDevice, stream));

            result.iterations = iter + 1;
        }

        // If we stopped due to max iterations (or maxits==0), return the latest residual.
        if (!converged)
        {
            double res_norm = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &res_norm));
            result.finalRes = res_norm;
        }

        // Copy final solution back to host and wait for all queued GPU work.
        CUDA_CHECK(cudaMemcpyAsync(h_x.data(), d_x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Cleanup descriptors, buffers, and handles.
        CUSPARSE_CHECK(cusparseDestroyDnMat(dnB));
        CUSPARSE_CHECK(cusparseDestroyDnMat(dnC));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecTmp));
        CUSPARSE_CHECK(cusparseDestroySpMat(matA));

        CUDA_CHECK(cudaFree(d_spmvBuf));
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

        if (d_P_hist64)
            CUDA_CHECK(cudaFree(d_P_hist64));
        if (d_W_hist64)
            CUDA_CHECK(cudaFree(d_W_hist64));
        if (d_P_histLP)
            CUDA_CHECK(cudaFree(d_P_histLP));
        if (d_W_histLP)
            CUDA_CHECK(cudaFree(d_W_histLP));

        if (d_Pj_tmp)
            CUDA_CHECK(cudaFree(d_Pj_tmp));
        if (d_Wj_tmp)
            CUDA_CHECK(cudaFree(d_Wj_tmp));
        if (d_scratch_LP)
            CUDA_CHECK(cudaFree(d_scratch_LP));

        CUDA_CHECK(cudaFree(d_G_hist));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_Gnew));
        CUDA_CHECK(cudaFree(d_rhs));
        CUDA_CHECK(cudaFree(d_Y));
        CUDA_CHECK(cudaFree(d_alpha));
        CUDA_CHECK(cudaFree(pinv_ws.d_G_copy));
        CUDA_CHECK(cudaFree(pinv_ws.d_S));
        CUDA_CHECK(cudaFree(pinv_ws.d_U));
        CUDA_CHECK(cudaFree(pinv_ws.d_VT));
        CUDA_CHECK(cudaFree(pinv_ws.d_work));
        CUDA_CHECK(cudaFree(pinv_ws.d_S_inv));
        CUDA_CHECK(cudaFree(pinv_ws.d_T1));
        CUDA_CHECK(cudaFree(pinv_ws.d_info));

        if (d_Pnew_low)
        {
            CUDA_CHECK(cudaFree(d_Pnew_low));
            CUDA_CHECK(cudaFree(d_Wnew_low));
            CUDA_CHECK(cudaFree(d_Wj_low));
            CUDA_CHECK(cudaFree(d_C_gemm));
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUSPARSE_CHECK(cusparseDestroy(cusparse));
        CUBLAS_CHECK(cublasDestroy(cublas));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver));

        return result;
    }

    template PCGResult mpcg<double>(
        const std::vector<int> &,
        const std::vector<int> &,
        const std::vector<double> &,
        const std::vector<ichol::precond::PrecondApply> &,
        const std::vector<double> &,
        std::vector<double> &,
        const PCGParams &);
} // namespace ichol::solver
