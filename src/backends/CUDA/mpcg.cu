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
#include <chrono>
#include <cstdio>

#include "ichol/pcg.hpp"
#include "ichol/preconditioner.hpp"
#include "backends/CUDA/mpcg_debug.hpp"

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

    const void *one(cudaDataType_t ct) const { return (ct == CUDA_R_64F) ? (const void *)&s64_one : (const void *)&s32_one; }
    const void *zero(cudaDataType_t ct) const { return (ct == CUDA_R_64F) ? (const void *)&s64_zero : (const void *)&s32_zero; }
    const void *m_one(cudaDataType_t ct) const { return (ct == CUDA_R_64F) ? (const void *)&s64_m_one : (const void *)&s32_m_one; }
};

struct DeviceCublasScalars
{
    float *d_s32 = nullptr;
    double *d_s64 = nullptr;

    const void *one(cublasComputeType_t ct) const { return (ct == CUBLAS_COMPUTE_64F) ? (const void *)(d_s64 + 0) : (const void *)(d_s32 + 0); }
    const void *zero(cublasComputeType_t ct) const { return (ct == CUBLAS_COMPUTE_64F) ? (const void *)(d_s64 + 1) : (const void *)(d_s32 + 1); }
    const void *m_one(cublasComputeType_t ct) const { return (ct == CUBLAS_COMPUTE_64F) ? (const void *)(d_s64 + 2) : (const void *)(d_s32 + 2); }

    const double *one64() const { return d_s64 + 0; }
    const double *zero64() const { return d_s64 + 1; }
    const double *m_one64() const { return d_s64 + 2; }
};

struct PrecisionMap
{
    ichol::solver::ComputePrecision io_prec;
    ichol::solver::ComputePrecision acc_prec;
    cudaDataType_t data_type;
    cudaDataType_t output_type;
    cublasComputeType_t compute_type;
    size_t el_size;
};

struct SpmmMap
{
    ichol::solver::ComputePrecision io_prec;
    ichol::solver::ComputePrecision acc_prec;
    cudaDataType_t data_type;
    cudaDataType_t compute_type;
    size_t el_size;
};

struct ProfilePhase
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    double total_ms = 0.0;
};

static ichol::solver::ComputePrecision normalize_precision(ichol::solver::ComputePrecision prec)
{
    using Prec = ichol::solver::ComputePrecision;
    switch (prec)
    {
    case Prec::FP64:
    case Prec::FP32:
    case Prec::FP16:
    case Prec::BF16:
        return prec;
    case Prec::TF32:
        // TF32 uses fp32 storage with tensor-core execution.
        return Prec::FP32;
    default:
        throw std::runtime_error("mpcg: unsupported precision mode for CUDA mixed-precision path");
    }
}

static cudaDataType_t get_cuda_data_type(ichol::solver::ComputePrecision prec)
{
    using Prec = ichol::solver::ComputePrecision;
    switch (normalize_precision(prec))
    {
    case Prec::FP64:
        return CUDA_R_64F;
    case Prec::FP32:
        return CUDA_R_32F;
    case Prec::FP16:
        return CUDA_R_16F;
    case Prec::BF16:
        return CUDA_R_16BF;
    default:
        throw std::runtime_error("mpcg: unsupported CUDA data type");
    }
}

static size_t get_precision_el_size(ichol::solver::ComputePrecision prec)
{
    using Prec = ichol::solver::ComputePrecision;
    switch (normalize_precision(prec))
    {
    case Prec::FP64:
        return sizeof(double);
    case Prec::FP32:
        return sizeof(float);
    case Prec::FP16:
        return sizeof(__half);
    case Prec::BF16:
        return sizeof(__nv_bfloat16);
    default:
        throw std::runtime_error("mpcg: unsupported CUDA element size");
    }
}

static ichol::solver::ComputePrecision resolve_accum_precision(
    ichol::solver::ComputePrecision input_prec,
    ichol::solver::ComputePrecision requested_acc)
{
    using Prec = ichol::solver::ComputePrecision;
    const Prec io_prec = normalize_precision(input_prec);
    const Prec acc_prec = normalize_precision(requested_acc);

    if (io_prec == Prec::FP64)
        return Prec::FP64;

    if (io_prec == Prec::FP32)
    {
        // fp32/tf32 inputs stay on fp32 accumulation. prec_acc is interpreted
        // as a request and normalized to the supported mixed-precision policy.
        (void)acc_prec;
        return Prec::FP32;
    }

    // fp16/bf16 inputs accumulate in fp32.
    return Prec::FP32;
}

/**
 * Map user precision choice to CUDA data type and cuBLAS compute type.
 */
static PrecisionMap get_precision_map(ichol::solver::ComputePrecision prec, ichol::solver::ComputePrecision acc)
{
    using Prec = ichol::solver::ComputePrecision;
    PrecisionMap m{};
    m.io_prec = normalize_precision(prec);
    m.acc_prec = resolve_accum_precision(prec, acc);
    m.data_type = get_cuda_data_type(prec);
    m.el_size = get_precision_el_size(prec);
    m.output_type = (m.acc_prec == Prec::FP64) ? CUDA_R_64F : CUDA_R_32F;

    if (m.acc_prec == Prec::FP64)
    {
        m.compute_type = CUBLAS_COMPUTE_64F;
    }
    else
    {
        if (prec == ichol::solver::ComputePrecision::TF32)
            m.compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
        else
            m.compute_type = CUBLAS_COMPUTE_32F;
    }
    return m;
}

struct StorageMap
{
    ichol::solver::ComputePrecision storage_prec;
    cudaDataType_t data_type;
    size_t el_size;
};

static SpmmMap get_spmm_map(ichol::solver::ComputePrecision prec, ichol::solver::ComputePrecision acc)
{
    SpmmMap m{};
    m.io_prec = normalize_precision(prec);
    m.acc_prec = resolve_accum_precision(prec, acc);
    m.data_type = get_cuda_data_type(prec);
    m.compute_type = (m.acc_prec == ichol::solver::ComputePrecision::FP64) ? CUDA_R_64F : CUDA_R_32F;
    m.el_size = get_precision_el_size(prec);
    return m;
}

/**
 * Map user precision choice to CUDA data type for history storage buffers.
 */
static StorageMap get_storage_map(ichol::solver::ComputePrecision prec)
{
    StorageMap m{};
    m.storage_prec = normalize_precision(prec);
    m.data_type = get_cuda_data_type(prec);
    m.el_size = get_precision_el_size(prec);
    return m;
}

static StorageMap get_precond_map(ichol::solver::ComputePrecision prec)
{
    using Prec = ichol::solver::ComputePrecision;
    switch (prec)
    {
    case Prec::FP64:
        return get_storage_map(Prec::FP64);
    case Prec::FP32:
    case Prec::TF32:
        return get_storage_map(Prec::FP32);
    default:
        throw std::runtime_error("mpcg: preconditioner apply path supports FP64 and FP32 only");
    }
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
        return std::max(base_rcond, 1e-6);
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

static void sync_block_to_storage(
    const double *src,
    void *dst,
    const StorageMap &storage_map,
    int count,
    int threads,
    cudaStream_t stream)
{
    if (storage_map.storage_prec == ichol::solver::ComputePrecision::FP64 || !dst)
        return;

    const int blocks = (count + threads - 1) / threads;
    // Large live blocks can stay compressed in memory, but sensitive kernels
    // still consume fp64 workspaces explicitly loaded at the next boundary.
    k_cast_d2any<<<blocks, threads, 0, stream>>>(src, dst, count, storage_map.storage_prec);
}

static void sync_block_from_storage(
    const void *src,
    double *dst,
    const StorageMap &storage_map,
    int count,
    int threads,
    cudaStream_t stream)
{
    if (storage_map.storage_prec == ichol::solver::ComputePrecision::FP64 || !src)
        return;

    const int blocks = (count + threads - 1) / threads;
    k_cast_any2d_vec<<<blocks, threads, 0, stream>>>(src, dst, count, storage_map.storage_prec);
}

static void *prepare_spmm_input(
    const double *src_fp64,
    void *storage_ptr,
    const StorageMap &storage_map,
    const SpmmMap &spmm_map,
    void *tmp_ptr,
    int count,
    int threads,
    cudaStream_t stream)
{
    if (spmm_map.io_prec == ichol::solver::ComputePrecision::FP64)
        return (void *)src_fp64;

    if (storage_ptr && storage_map.storage_prec == spmm_map.io_prec)
    {
        sync_block_to_storage(src_fp64, storage_ptr, storage_map, count, threads, stream);
        return storage_ptr;
    }

    if (!tmp_ptr)
        throw std::runtime_error("mpcg: missing temporary SpMM input buffer");

    const StorageMap tmp_map = get_storage_map(spmm_map.io_prec);
    sync_block_to_storage(src_fp64, tmp_ptr, tmp_map, count, threads, stream);
    return tmp_ptr;
}

static void finalize_spmm_output(
    const void *spmm_out,
    const SpmmMap &spmm_map,
    void *storage_ptr,
    const StorageMap &storage_map,
    double *dst_fp64,
    int count,
    int threads,
    cudaStream_t stream)
{
    if (spmm_map.io_prec == ichol::solver::ComputePrecision::FP64)
        return;

    const StorageMap spmm_storage = get_storage_map(spmm_map.io_prec);
    sync_block_from_storage(spmm_out, dst_fp64, spmm_storage, count, threads, stream);

    if (storage_ptr && storage_ptr != spmm_out)
        sync_block_to_storage(dst_fp64, storage_ptr, storage_map, count, threads, stream);
}

static void build_znew_columns_device(
    const std::vector<ichol::precond::PrecondApply> &preconds,
    const double *d_r,
    void *d_r_precond,
    double *d_Znew,
    void *d_Znew_precond,
    int n,
    const StorageMap &precond_map,
    int threads,
    cudaStream_t stream,
    cudaEvent_t main_stream_ready,
    const std::vector<cudaStream_t> &precond_streams,
    const std::vector<cudaEvent_t> &precond_events,
    bool serial,
    bool sync_after_each_apply)
{
    const int k = static_cast<int>(preconds.size());
    sync_block_to_storage(d_r, d_r_precond, precond_map, n, threads, stream);

    if (serial)
    {
        for (int t = 0; t < k; ++t)
        {
            const size_t offset = static_cast<size_t>(t) * static_cast<size_t>(n);
            if (precond_map.storage_prec == ichol::solver::ComputePrecision::FP64)
            {
                CUDA_CHECK(cudaMemsetAsync(d_Znew + offset, 0, static_cast<size_t>(n) * sizeof(double), stream));
                preconds[t].apply(preconds[t].ctx, d_r, d_Znew + offset, n, precond_map.storage_prec, stream);
            }
            else
            {
                void *d_Zlow = static_cast<void *>(static_cast<char *>(d_Znew_precond) + offset * precond_map.el_size);
                CUDA_CHECK(cudaMemsetAsync(d_Zlow, 0, static_cast<size_t>(n) * precond_map.el_size, stream));
                preconds[t].apply(preconds[t].ctx, d_r_precond, d_Zlow, n, precond_map.storage_prec, stream);
            }
            if (sync_after_each_apply)
                CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    else
    {
        CUDA_CHECK(cudaEventRecord(main_stream_ready, stream));
        for (int t = 0; t < k; ++t)
        {
            CUDA_CHECK(cudaStreamWaitEvent(precond_streams[t], main_stream_ready, 0));
            const size_t offset = static_cast<size_t>(t) * static_cast<size_t>(n);
            if (precond_map.storage_prec == ichol::solver::ComputePrecision::FP64)
            {
                CUDA_CHECK(cudaMemsetAsync(d_Znew + offset, 0, static_cast<size_t>(n) * sizeof(double), precond_streams[t]));
                preconds[t].apply(preconds[t].ctx, d_r, d_Znew + offset, n, precond_map.storage_prec, precond_streams[t]);
            }
            else
            {
                void *d_Zlow = static_cast<void *>(static_cast<char *>(d_Znew_precond) + offset * precond_map.el_size);
                CUDA_CHECK(cudaMemsetAsync(d_Zlow, 0, static_cast<size_t>(n) * precond_map.el_size, precond_streams[t]));
                preconds[t].apply(preconds[t].ctx, d_r_precond, d_Zlow, n, precond_map.storage_prec, precond_streams[t]);
            }
            CUDA_CHECK(cudaEventRecord(precond_events[t], precond_streams[t]));
        }
        for (int t = 0; t < k; ++t)
            CUDA_CHECK(cudaStreamWaitEvent(stream, precond_events[t], 0));
    }

    sync_block_from_storage(d_Znew_precond, d_Znew, precond_map, n * k, threads, stream);
}

__global__ void k_build_col_scale_from_gdiag(const double *G, double *col_scale, int k, double diag_floor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k)
        return;

    const double gii = G[i + i * k];
    if (!isfinite(gii) || gii <= diag_floor)
    {
        col_scale[i] = 0.0;
        return;
    }
    col_scale[i] = 1.0 / sqrt(gii);
}

__global__ void k_fuse_scale_P_W(double *P, double *W, const double *col_scale, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nk = n * k;
    if (idx >= nk)
        return;

    const int col = idx / n;
    const double scale = col_scale[col];
    P[idx] *= scale;
    W[idx] *= scale;
}

__global__ void k_congruence_scale_gram(double *G, const double *col_scale, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int kk = k * k;
    if (idx >= kk)
        return;

    const int i = idx % k;
    const int j = idx / k;
    G[idx] *= col_scale[i] * col_scale[j];
}

__global__ void column_norms_kernel(const double *X, int ld, int n, int k, double *out)
{
    const int col = blockIdx.x;
    if (col >= k)
        return;

    extern __shared__ double shared[];
    double sum = 0.0;
    for (int row = threadIdx.x; row < n; row += blockDim.x)
    {
        const double v = X[row + (size_t)col * (size_t)ld];
        sum += v * v;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[col] = sqrt(shared[0]);
}

__global__ void column_dots_kernel(const double *X, const double *Y, int ldX, int ldY, int n, int k, double *out)
{
    const int col = blockIdx.x;
    if (col >= k)
        return;

    extern __shared__ double shared[];
    double sum = 0.0;
    for (int row = threadIdx.x; row < n; row += blockDim.x)
        sum += X[row + (size_t)col * (size_t)ldX] * Y[row + (size_t)col * (size_t)ldY];

    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[col] = shared[0];
}

__global__ void k_scale_columns_from_norms(double *X, int n, int k, const double *norms, double floor)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nk = n * k;
    if (idx >= nk)
        return;

    const int col = idx / n;
    const double norm = norms[col];
    if (norm > floor)
        X[idx] /= norm;
    else
        X[idx] = 0.0;
}

__global__ void k_set_identity(double *A, int k)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int kk = k * k;
    if (idx >= kk)
        return;

    const int row = idx % k;
    const int col = idx / k;
    A[idx] = (row == col) ? 1.0 : 0.0;
}

__global__ void k_should_reproject(const double *z_dots, const double *p_dots, int k, double ratio_sq, int *flag)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    double z_sum = 0.0;
    double p_sum = 0.0;
    for (int i = 0; i < k; ++i)
    {
        z_sum += fmax(0.0, z_dots[i]);
        p_sum += fmax(0.0, p_dots[i]);
    }
    *flag = (z_sum > 0.0 && p_sum < ratio_sq * z_sum) ? 1 : 0;
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

// Workspace for cuSOLVER Cholesky-based k×k solver (k > 32).
struct CholWorkspace
{
    double *d_G_copy = nullptr; // k×k scratch: potrf overwrites in-place
    double *d_work = nullptr;   // potrf device workspace
    int *d_info = nullptr;      // potrf return code
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

// --------------------------------------------------------------------------
// Cholesky solver for small k (k <= 32), with LU fallback.
// --------------------------------------------------------------------------

// In-register LU factorisation + solve, factored out as a __device__ helper
// so both k_small_lu_solve and k_small_chol_solve can call it.
__device__ static void d_lu_solve_k32(
    const double *G, // k×k input
    const double *B, // k×nrhs input
    double *X,       // k×nrhs output
    int k,
    int nrhs,
    double rcond)
{
    constexpr int KMAX = 32;
    double A[KMAX * KMAX];
    double RHS[KMAX * KMAX];

    double max_abs = 0.0;
    for (int col = 0; col < k; ++col)
        for (int row = 0; row < k; ++row)
        {
            const double v = G[row + col * k];
            A[row + col * k] = v;
            max_abs = fmax(max_abs, fabs(v));
        }
    for (int col = 0; col < nrhs; ++col)
        for (int row = 0; row < k; ++row)
            RHS[row + col * k] = B[row + col * k];

    const double pivot_tol = fmax(1e-15, rcond * fmax(max_abs, 1.0));

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
                double tmp = A[col + j * k];
                A[col + j * k] = A[piv + j * k];
                A[piv + j * k] = tmp;
            }
            for (int j = 0; j < nrhs; ++j)
            {
                double tmp = RHS[col + j * k];
                RHS[col + j * k] = RHS[piv + j * k];
                RHS[piv + j * k] = tmp;
            }
        }
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

    for (int j = 0; j < nrhs; ++j)
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

/**
 * k_small_chol_solve:
 * Solves G * X = B for small k (k <= 32) using in-register Cholesky.
 * G must be SPD; if a non-positive pivot is detected the kernel falls back
 * to LU with partial pivoting.  Runs single-threaded (one block, one thread).
 */
__global__ void k_small_chol_solve(
    const double *G, // k×k SPD matrix
    const double *B, // k×nrhs
    double *X,       // k×nrhs output
    int k,
    int nrhs,
    double rcond)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    constexpr int KMAX = 32;
    double L[KMAX * KMAX]; // lower-triangular Cholesky factor (column-major)
    double Y[KMAX * KMAX]; // intermediate forward-solve result

    // Compute max diagonal for the non-SPD pivot threshold.
    double max_diag = 0.0;
    for (int i = 0; i < k; ++i)
        max_diag = fmax(max_diag, G[i + i * k]);
    const double pivot_tol = rcond * fmax(max_diag, 1e-15);

    // Symmetrize G into the lower triangle of L.
    for (int i = 0; i < k; ++i)
        for (int j = 0; j <= i; ++j)
            L[i + j * k] = 0.5 * (G[i + j * k] + G[j + i * k]);

    // In-place Cholesky (column-major, lower triangle).
    for (int j = 0; j < k; ++j)
    {
        double diag = L[j + j * k];
        for (int m = 0; m < j; ++m)
            diag -= L[j + m * k] * L[j + m * k];

        if (diag <= pivot_tol)
        {
            // Non-positive pivot: G is not SPD — fall back to LU.
            d_lu_solve_k32(G, B, X, k, nrhs, rcond);
            return;
        }

        L[j + j * k] = sqrt(diag);
        const double inv_ljj = 1.0 / L[j + j * k];
        for (int i = j + 1; i < k; ++i)
        {
            double s = L[i + j * k];
            for (int m = 0; m < j; ++m)
                s -= L[i + m * k] * L[j + m * k];
            L[i + j * k] = s * inv_ljj;
        }
    }

    // Forward substitution: L * Y = B
    for (int col = 0; col < nrhs; ++col)
        for (int i = 0; i < k; ++i)
        {
            double s = B[i + col * k];
            for (int m = 0; m < i; ++m)
                s -= L[i + m * k] * Y[m + col * k];
            Y[i + col * k] = s / L[i + i * k];
        }

    // Backward substitution: L^T * X = Y  (L^T[i,m] = L[m,i])
    for (int col = 0; col < nrhs; ++col)
        for (int i = k - 1; i >= 0; --i)
        {
            double s = Y[i + col * k];
            for (int m = i + 1; m < k; ++m)
                s -= L[m + i * k] * X[m + col * k];
            X[i + col * k] = s / L[i + i * k];
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
    PinvSVDWorkspace &ws,
    const DeviceCublasScalars &dev_scalars)
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

    // d_T1 = U^T * B
    CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, nrhs, k, dev_scalars.one64(), ws.d_U, k, d_B, k, dev_scalars.zero64(), ws.d_T1, k));
    // d_T1 = diag(S_inv) * d_T1
    k_row_scaling<<<1, k, 0, stream>>>(ws.d_T1, ws.d_S_inv, k, nrhs);
    // d_X = V * d_T1  (note: cuSOLVER gives V^T, so V is VT^T)
    CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, nrhs, k, dev_scalars.one64(), ws.d_VT, k, ws.d_T1, k, dev_scalars.zero64(), d_X, k));
}

// --------------------------------------------------------------------------
// chol_solve_cuda: Cholesky-based k×k solve, with SVD fallback.
// For k <= 32 dispatches to the in-register k_small_chol_solve kernel.
// For k > 32 uses cusolverDnDpotrf + cusolverDnDpotrs; if potrf reports a
// non-positive pivot (d_info > 0) it falls back to pinv_svd_cuda.
// --------------------------------------------------------------------------
static void chol_solve_cuda(
    cusolverDnHandle_t cusolver,
    cublasHandle_t cublas,
    const double *d_G,
    int k,
    const double *d_B,
    int nrhs,
    double *d_X,
    cudaStream_t stream,
    double rcond,
    CholWorkspace &chol_ws,
    PinvSVDWorkspace &svd_ws,
    const DeviceCublasScalars &dev_scalars)
{
    if (k <= 32)
    {
        k_small_chol_solve<<<1, 1, 0, stream>>>(d_G, d_B, d_X, k, nrhs, rcond);
        return;
    }

    if (k != chol_ws.k || nrhs > chol_ws.max_nrhs)
        throw std::runtime_error("chol_solve_cuda: workspace shape mismatch");

    // Copy G for in-place factorisation (potrf overwrites the matrix).
    CUDA_CHECK(cudaMemcpyAsync(chol_ws.d_G_copy, d_G,
                               (size_t)k * k * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));

    // Cholesky factorisation (lower triangular, in-place on d_G_copy).
    CUSOLVER_CHECK(cusolverDnDpotrf(
        cusolver, CUBLAS_FILL_MODE_LOWER, k,
        chol_ws.d_G_copy, k,
        chol_ws.d_work, chol_ws.lwork,
        chol_ws.d_info));

    // Retrieve factorisation status — requires a stream sync.
    int info = 0;
    CUDA_CHECK(cudaMemcpyAsync(&info, chol_ws.d_info, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (info > 0)
    {
        // G is not positive definite: fall back to pseudo-inverse SVD.
        pinv_svd_cuda(cusolver, cublas, d_G, k, d_B, nrhs, d_X,
                      stream, rcond, svd_ws, dev_scalars);
        return;
    }

    // Copy B into X; potrs solves in-place (overwrites X with the solution).
    CUDA_CHECK(cudaMemcpyAsync(d_X, d_B,
                               (size_t)k * nrhs * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));

    CUSOLVER_CHECK(cusolverDnDpotrs(
        cusolver, CUBLAS_FILL_MODE_LOWER, k, nrhs,
        chol_ws.d_G_copy, k,
        d_X, k,
        chol_ws.d_info));
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
        using Clock = std::chrono::steady_clock;
        const int n = static_cast<int>(h_b.size());                           // matrix size
        const int k = static_cast<int>(preconds.size());                      // # of precond
        const int m = (params.restart <= 0) ? params.maxits : params.restart; // history size (0 or negative means no restarts, i.e. full history up to maxits)
        const int64_t nnzA = (int64_t)h_valA.size();
        const bool collect_timing = true;
        const bool profile_enabled = params.verbose;
        const auto total_wall_start = Clock::now();

        ProfilePhase phase_precond{};
        ProfilePhase phase_ortho{};
        ProfilePhase phase_spmm{};
        ProfilePhase phase_dense{};
        ProfilePhase phase_reset{};

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

        auto init_phase = [&](ProfilePhase &phase)
        {
            if (!collect_timing)
                return;
            CUDA_CHECK(cudaEventCreate(&phase.start));
            CUDA_CHECK(cudaEventCreate(&phase.stop));
        };
        auto destroy_phase = [&](ProfilePhase &phase)
        {
            if (!phase.start)
                return;
            CUDA_CHECK(cudaEventDestroy(phase.start));
            CUDA_CHECK(cudaEventDestroy(phase.stop));
        };
        auto phase_begin = [&](ProfilePhase &phase)
        {
            if (collect_timing)
                CUDA_CHECK(cudaEventRecord(phase.start, stream));
        };
        auto phase_end = [&](ProfilePhase &phase)
        {
            if (collect_timing)
                CUDA_CHECK(cudaEventRecord(phase.stop, stream));
        };
        auto phase_accumulate = [&](ProfilePhase &phase)
        {
            if (!collect_timing || !phase.start)
                return;
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, phase.start, phase.stop));
            phase.total_ms += static_cast<double>(elapsed_ms);
        };

        // One stream/event per preconditioner so their apply() calls can run concurrently.
        std::vector<cudaStream_t> precond_streams(k);
        std::vector<cudaEvent_t> precond_events(k);
        cudaEvent_t main_stream_ready = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&main_stream_ready, cudaEventDisableTiming));
        for (int t = 0; t < k; ++t)
        {
            CUDA_CHECK(cudaStreamCreateWithFlags(&precond_streams[t], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&precond_events[t], cudaEventDisableTiming));
        }

        init_phase(phase_precond);
        init_phase(phase_ortho);
        init_phase(phase_spmm);
        init_phase(phase_dense);
        init_phase(phase_reset);

        CublasScalars scalars;
        DeviceCublasScalars dev_scalars;
        PinvSVDWorkspace pinv_ws{};
        CholWorkspace chol_ws{};

        // Precision controls:
        PrecisionMap g_map = get_precision_map(params.prec_gemm, params.prec_acc);
        SpmmMap spmm_map = get_spmm_map(params.prec_spmm, params.prec_acc);
        StorageMap precond_map = get_precond_map(params.prec_precond);
        StorageMap Znew_store_map = get_storage_map(params.store_Znew);
        StorageMap Pnew_store_map = get_storage_map(params.store_Pnew);
        StorageMap Wnew_store_map = get_storage_map(params.store_Wnew);

        StorageMap P_hist_map = get_storage_map(params.store_P_hist);
        StorageMap W_hist_map = get_storage_map(params.store_W_hist);
        const bool use_lowp_history =
            (P_hist_map.storage_prec != ComputePrecision::FP64) ||
            (W_hist_map.storage_prec != ComputePrecision::FP64);
        const bool enable_anorm_reprojection =
            use_lowp_history && params.projection_anorm_drop_tol > 0.0;
        const bool use_mixed_spmm = (spmm_map.io_prec != ComputePrecision::FP64);

        const double anorm_drop_tol_sq =
            params.projection_anorm_drop_tol * params.projection_anorm_drop_tol;

        int *d_rowPtrA = nullptr, *d_colIndA = nullptr;
        double *d_valA = nullptr, *d_b = nullptr, *d_x = nullptr, *d_r = nullptr, *d_tmp = nullptr;
        void *d_valA_spmm = nullptr;

        CUDA_CHECK(cudaMalloc(&d_rowPtrA, (size_t)(n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIndA, (size_t)nnzA * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_valA, (size_t)nnzA * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)n * sizeof(double)));

        const float h_s32[] = {1.0f, 0.0f, -1.0f};
        const double h_s64[] = {1.0, 0.0, -1.0};
        CUDA_CHECK(cudaMalloc(&dev_scalars.d_s32, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_scalars.d_s64, 3 * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(dev_scalars.d_s32, h_s32, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dev_scalars.d_s64, h_s64, 3 * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUBLAS_CHECK(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE));

        // Upload host problem data to GPU.
        CUDA_CHECK(cudaMemcpyAsync(d_rowPtrA, h_csrRowPtrA.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_colIndA, h_csrColIndA.data(), (size_t)nnzA * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_valA, h_valA.data(), (size_t)nnzA * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_x, h_x.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice, stream));

        if (use_mixed_spmm)
        {
            CUDA_CHECK(cudaMalloc(&d_valA_spmm, (size_t)nnzA * spmm_map.el_size));
            const int spmm_cast_threads = 256;
            const int spmm_cast_blocks = (int)((nnzA + spmm_cast_threads - 1) / spmm_cast_threads);
            k_cast_d2any<<<spmm_cast_blocks, spmm_cast_threads, 0, stream>>>(
                d_valA, d_valA_spmm, (int)nnzA, spmm_map.io_prec);
        }

        void *d_Pnew_low = nullptr, *d_Wnew_low = nullptr, *d_Wj_low = nullptr;
        float *d_C_gemm = nullptr;

        // Extra low-precision buffers only needed when GEMM accumulates in fp32.
        if (g_map.output_type != CUDA_R_64F)
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
        double *d_Znew = nullptr, *d_Pnew = nullptr, *d_Wnew = nullptr, *d_Wz = nullptr;
        void *d_r_precond = nullptr, *d_Znew_precond = nullptr;
        void *d_Znew_store = nullptr, *d_Pnew_store = nullptr, *d_Wnew_store = nullptr;
        void *d_spmm_in_tmp = nullptr, *d_spmm_out_tmp = nullptr;
        double *d_G_hist = nullptr, *d_Ginv_hist = nullptr, *d_hist_C = nullptr, *d_hist_Y = nullptr;
        double *d_Gnew = nullptr, *d_rhs = nullptr, *d_alpha = nullptr, *d_col_scale = nullptr;
        double *d_scalar_tmp = nullptr, *d_col_norms = nullptr, *d_z_anorm_cols = nullptr, *d_p_anorm_cols = nullptr, *d_eye = nullptr;
        int *d_reproject_flag = nullptr;

        CUDA_CHECK(cudaMalloc(&d_Znew, nk * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Pnew, nk * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Wnew, nk * sizeof(double)));
        if (enable_anorm_reprojection)
            CUDA_CHECK(cudaMalloc(&d_Wz, nk * sizeof(double)));
        if (precond_map.storage_prec != ComputePrecision::FP64)
        {
            CUDA_CHECK(cudaMalloc(&d_r_precond, (size_t)n * precond_map.el_size));
            CUDA_CHECK(cudaMalloc(&d_Znew_precond, nk * precond_map.el_size));
        }

        if (Znew_store_map.storage_prec != ComputePrecision::FP64)
            CUDA_CHECK(cudaMalloc(&d_Znew_store, nk * Znew_store_map.el_size));
        if (Pnew_store_map.storage_prec != ComputePrecision::FP64)
            CUDA_CHECK(cudaMalloc(&d_Pnew_store, nk * Pnew_store_map.el_size));
        if (Wnew_store_map.storage_prec != ComputePrecision::FP64)
            CUDA_CHECK(cudaMalloc(&d_Wnew_store, nk * Wnew_store_map.el_size));

        const bool need_spmm_input_tmp =
            use_mixed_spmm &&
            ((Znew_store_map.storage_prec != spmm_map.io_prec) ||
             (Pnew_store_map.storage_prec != spmm_map.io_prec));
        const bool need_spmm_output_tmp =
            use_mixed_spmm &&
            ((Wnew_store_map.storage_prec != spmm_map.io_prec) ||
             enable_anorm_reprojection);
        if (need_spmm_input_tmp)
            CUDA_CHECK(cudaMalloc(&d_spmm_in_tmp, nk * spmm_map.el_size));
        if (need_spmm_output_tmp)
            CUDA_CHECK(cudaMalloc(&d_spmm_out_tmp, nk * spmm_map.el_size));

        CUDA_CHECK(cudaMalloc(&d_G_hist, (size_t)m * (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ginv_hist, (size_t)m * (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_hist_C, (size_t)m * (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_hist_Y, (size_t)m * (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Gnew, (size_t)k * (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rhs, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_alpha, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_col_scale, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_scalar_tmp, 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_col_norms, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z_anorm_cols, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_p_anorm_cols, (size_t)k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_reproject_flag, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_eye, (size_t)k * (size_t)k * sizeof(double)));
        k_set_identity<<<((k * k) + 255) / 256, 256, 0, stream>>>(d_eye, k);

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

        // Workspace for Cholesky-based k×k solves (only when use_svd == false).
        // For k <= 32 the device kernel needs no extra allocation.
        if (!params.use_svd && k > 32)
        {
            chol_ws.k = k;
            chol_ws.max_nrhs = k;
            CUDA_CHECK(cudaMalloc(&chol_ws.d_G_copy, (size_t)k * (size_t)k * sizeof(double)));
            CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
                cusolver, CUBLAS_FILL_MODE_LOWER, k,
                chol_ws.d_G_copy, k, &chol_ws.lwork));
            CUDA_CHECK(cudaMalloc(&chol_ws.d_work, (size_t)chol_ws.lwork * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&chol_ws.d_info, sizeof(int)));
        }

        double *d_P_hist64 = nullptr, *d_W_hist64 = nullptr;
        void *d_P_histLP = nullptr, *d_W_histLP = nullptr;
        const size_t P_hist_bytes =
            (P_hist_map.storage_prec == ComputePrecision::FP64)
                ? ((size_t)m * nk * sizeof(double))
                : ((size_t)m * nk * P_hist_map.el_size);
        const size_t W_hist_bytes =
            (W_hist_map.storage_prec == ComputePrecision::FP64)
                ? ((size_t)m * nk * sizeof(double))
                : ((size_t)m * nk * W_hist_map.el_size);

        // Circular history buffers (size m) store previous P/W/G blocks.
        // Can be fp64 or compressed precision depending on params.
        if (P_hist_map.storage_prec == ComputePrecision::FP64)
            CUDA_CHECK(cudaMalloc(&d_P_hist64, (size_t)m * nk * sizeof(double)));
        else
            CUDA_CHECK(cudaMalloc(&d_P_histLP, (size_t)m * nk * P_hist_map.el_size));

        if (W_hist_map.storage_prec == ComputePrecision::FP64)
            CUDA_CHECK(cudaMalloc(&d_W_hist64, (size_t)m * nk * sizeof(double)));
        else
            CUDA_CHECK(cudaMalloc(&d_W_histLP, (size_t)m * nk * W_hist_map.el_size));

        double *d_Pj_tmp = nullptr, *d_Wj_tmp = nullptr;
        if (d_P_histLP)
            CUDA_CHECK(cudaMalloc(&d_Pj_tmp, nk * sizeof(double)));
        if (d_W_histLP)
            CUDA_CHECK(cudaMalloc(&d_Wj_tmp, nk * sizeof(double)));

        if (params.verbose)
        {
            std::fprintf(stderr,
                         "[MPCG history] m=%d nk=%zu P_hist=%.3f MiB W_hist=%.3f MiB total=%.3f MiB\n",
                         m, nk,
                         (double)P_hist_bytes / (1024.0 * 1024.0),
                         (double)W_hist_bytes / (1024.0 * 1024.0),
                         (double)(P_hist_bytes + W_hist_bytes) / (1024.0 * 1024.0));
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
        cusparseSpMatDescr_t matA_spmm = nullptr;
        if (use_mixed_spmm)
        {
            CUSPARSE_CHECK(cusparseCreateCsr(
                &matA_spmm, n, n, nnzA,
                d_rowPtrA, d_colIndA, d_valA_spmm,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                spmm_map.data_type));
        }

        cusparseDnMatDescr_t dnB, dnC;
        void *spmm_B_ptr = use_mixed_spmm
                               ? (d_spmm_in_tmp ? d_spmm_in_tmp : d_Pnew_store)
                               : (void *)d_Pnew;
        void *spmm_C_ptr = use_mixed_spmm
                               ? (d_spmm_out_tmp ? d_spmm_out_tmp : d_Wnew_store)
                               : (void *)d_Wnew;
        CUSPARSE_CHECK(cusparseCreateDnMat(&dnB, n, k, n, spmm_B_ptr, use_mixed_spmm ? spmm_map.data_type : CUDA_R_64F, CUSPARSE_ORDER_COL));
        CUSPARSE_CHECK(cusparseCreateDnMat(&dnC, n, k, n, spmm_C_ptr, use_mixed_spmm ? spmm_map.data_type : CUDA_R_64F, CUSPARSE_ORDER_COL));

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
        CUBLAS_CHECK(cublasDaxpy(cublas, n, dev_scalars.m_one64(), d_tmp, 1, d_r, 1));

        // Use ||b|| as denominator for relative residual stopping criterion.
        double bnorm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublas, n, d_b, 1, d_scalar_tmp + 0));
        CUDA_CHECK(cudaMemcpyAsync(&bnorm, d_scalar_tmp + 0, sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (bnorm == 0.0)
            bnorm = 1.0;

        double current_res_norm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, d_scalar_tmp + 1));
        CUDA_CHECK(cudaMemcpyAsync(&current_res_norm, d_scalar_tmp + 1, sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t spmm_bufSize = 0;
        void *d_spmmBuf = nullptr;

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            scalars.one(use_mixed_spmm ? spmm_map.compute_type : CUDA_R_64F), use_mixed_spmm ? matA_spmm : matA, dnB,
            scalars.zero(use_mixed_spmm ? spmm_map.compute_type : CUDA_R_64F), dnC,
            use_mixed_spmm ? spmm_map.compute_type : CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &spmm_bufSize));
        CUDA_CHECK(cudaMalloc(&d_spmmBuf, spmm_bufSize));

        PCGResult result{};
        result.relResiduals.reserve((size_t)params.maxits + 1);
        result.relResiduals.push_back(current_res_norm / bnorm);
        bool converged = false;
        const int threads = 256;
        const int blocks_nk = (int)((nk + (size_t)threads - 1) / (size_t)threads);
        const int blocks_kk = (int)(((size_t)k * (size_t)k + (size_t)threads - 1) / (size_t)threads);
        double hist_rcond = get_safe_rcond(P_hist_map.storage_prec, params.rcond_base);
        hist_rcond = std::max(hist_rcond, 1e-15);
        std::vector<int> hist_slots;
        hist_slots.reserve((size_t)m);

        int reset_iter = 50;
        if (P_hist_map.storage_prec != ComputePrecision::FP64)
        {
            reset_iter = 10;
        }

        const StorageMap fp64_storage = get_storage_map(ComputePrecision::FP64);
        auto run_spmm = [&](const double *src_fp64,
                            void *src_store,
                            const StorageMap &src_store_map,
                            double *dst_fp64,
                            void *dst_store,
                            const StorageMap &dst_store_map)
        {
            if (!use_mixed_spmm)
            {
                CUSPARSE_CHECK(cusparseDnMatSetValues(dnB, (void *)src_fp64));
                CUSPARSE_CHECK(cusparseDnMatSetValues(dnC, dst_fp64));
                CUSPARSE_CHECK(cusparseSpMM(
                    cusparse,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &scalars.s64_one, matA, dnB,
                    &scalars.s64_zero, dnC,
                    CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_spmmBuf));
                sync_block_to_storage(dst_fp64, dst_store, dst_store_map, (int)nk, threads, stream);
                return;
            }

            void *b_ptr = prepare_spmm_input(
                src_fp64, src_store, src_store_map, spmm_map,
                d_spmm_in_tmp, (int)nk, threads, stream);
            void *c_ptr = nullptr;
            if (dst_store && dst_store_map.storage_prec == spmm_map.io_prec)
                c_ptr = dst_store;
            else
                c_ptr = d_spmm_out_tmp;

            if (!c_ptr)
                throw std::runtime_error("mpcg: missing temporary SpMM output buffer");

            CUSPARSE_CHECK(cusparseDnMatSetValues(dnB, b_ptr));
            CUSPARSE_CHECK(cusparseDnMatSetValues(dnC, c_ptr));
            CUSPARSE_CHECK(cusparseSpMM(
                cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                scalars.one(spmm_map.compute_type), matA_spmm, dnB,
                scalars.zero(spmm_map.compute_type), dnC,
                spmm_map.compute_type, CUSPARSE_SPMM_ALG_DEFAULT, d_spmmBuf));

            finalize_spmm_output(
                c_ptr, spmm_map, dst_store, dst_store_map,
                dst_fp64, (int)nk, threads, stream);
        };

        const auto iter_wall_start = Clock::now();

        // Unified k×k solver: Cholesky path (default) or SVD (params.use_svd == true).
        // Wraps both pinv_svd_cuda and chol_solve_cuda so call-sites stay uniform.
        auto solve_kk = [&](const double *d_G_in, int nrhs_in,
                            const double *d_B_in, double *d_X_out)
        {
            if (params.use_svd)
                pinv_svd_cuda(cusolver, cublas, d_G_in, k, d_B_in, nrhs_in,
                              d_X_out, stream, hist_rcond, pinv_ws, dev_scalars);
            else
                chol_solve_cuda(cusolver, cublas, d_G_in, k, d_B_in, nrhs_in,
                                d_X_out, stream, hist_rcond,
                                chol_ws, pinv_ws, dev_scalars);
        };

        // Main MPCG iteration.
        for (int iter = 0; iter < params.maxits; ++iter)
        {
            if (current_res_norm <= params.tol * bnorm)
            {
                result.iterations = iter;
                result.finalRes = current_res_norm;
                converged = true;
                break;
            }

            // Apply each preconditioner to the same residual r.
            // Output columns form Znew(:, t).
            phase_begin(phase_precond);
            build_znew_columns_device(
                preconds,
                d_r,
                d_r_precond,
                d_Znew,
                d_Znew_precond,
                n,
                precond_map,
                threads,
                stream,
                main_stream_ready,
                precond_streams,
                precond_events,
                false,
                false);

            // Znew can be stored in lower precision for the optional SpMM path,
            // but projection always starts from the fp64 copy used by the
            // preconditioner outputs.
            sync_block_to_storage(d_Znew, d_Znew_store, Znew_store_map, (int)nk, threads, stream);

            // Start from Znew, then orthogonalize against history below.
            CUDA_CHECK(cudaMemcpyAsync(d_Pnew, d_Znew, nk * sizeof(double), cudaMemcpyDeviceToDevice, stream));
            phase_end(phase_precond);

            const int hist_count = std::min(iter, m);
            hist_slots.clear();
            for (int jj = iter - hist_count; jj < iter; ++jj)
                hist_slots.push_back(jj % m);

            auto project_against_history = [&]()
            {
                if (hist_count == 0)
                    return;

                // ---------------------------------------------------------------
                // Fast batched path: fp64 history buffers + fp64 GEMM output.
                //
                // All hist_count C matrices are computed from the same d_Pnew
                // in a single cublasGemmStridedBatched call (classical A-orthog).
                // History is always at contiguous slots 0..hist_count-1 because:
                //   - When iter < m: hist_slots == [0..iter-1] (ring not yet full)
                //   - When iter >= m: hist_count == m, ALL m slots are valid and
                //     the projection sum is order-independent, so iterating over
                //     slots 0..m-1 is mathematically equivalent.
                // ---------------------------------------------------------------
                if (d_W_hist64 && d_P_hist64 && g_map.output_type == CUDA_R_64F)
                {
                    // Batch 1: C[slot] = W[slot]^T * Pnew
                    //   W[slot] is n×k stored at d_W_hist64 + slot*nk (stride = nk).
                    //   Pnew is the same n×k matrix for all batches (stride = 0).
                    //   C[slot] is k×k stored at d_hist_C + slot*k*k  (stride = k*k).
                    CUBLAS_CHECK(cublasDgemmStridedBatched(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        k, k, n,
                        dev_scalars.one64(),
                        d_W_hist64, n, (long long)nk,
                        d_Pnew, n, 0LL,
                        dev_scalars.zero64(),
                        d_hist_C, k, (long long)(k * k),
                        hist_count));

                    // Batch 2: Y[slot] = Ginv[slot] * C[slot]
                    CUBLAS_CHECK(cublasDgemmStridedBatched(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        k, k, k,
                        dev_scalars.one64(),
                        d_Ginv_hist, k, (long long)(k * k),
                        d_hist_C, k, (long long)(k * k),
                        dev_scalars.zero64(),
                        d_hist_Y, k, (long long)(k * k),
                        hist_count));

                    // Subtraction loop: Pnew -= P[slot] * Y[slot]  (n×k, kept as loop)
                    for (int slot = 0; slot < hist_count; ++slot)
                    {
                        double *d_Pj = d_P_hist64 + (size_t)slot * nk;
                        double *d_Yj = d_hist_Y + (size_t)slot * (size_t)k * (size_t)k;
                        CUBLAS_CHECK(cublasDgemm(
                            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, k, k,
                            dev_scalars.m_one64(),
                            d_Pj, n,
                            d_Yj, k,
                            dev_scalars.one64(),
                            d_Pnew, n));
                    }
                    return;
                }

                // ---------------------------------------------------------------
                // Original sequential loop — handles LP history, LP GEMM, or any
                // mix where the fp64 strided-batch path is not applicable.
                // ---------------------------------------------------------------
                for (int hist_idx = 0; hist_idx < hist_count; ++hist_idx)
                {
                    const int slot = hist_slots[(size_t)hist_idx];
                    double *d_Pj = nullptr;
                    double *d_Wj = nullptr;
                    double *d_Cj = d_hist_C + (size_t)hist_idx * (size_t)k * (size_t)k;
                    double *d_Yj = d_hist_Y + (size_t)hist_idx * (size_t)k * (size_t)k;
                    double *d_Ginvj = d_Ginv_hist + (size_t)slot * (size_t)k * (size_t)k;

                    if (d_P_hist64)
                    {
                        d_Pj = d_P_hist64 + (size_t)slot * nk;
                    }
                    else
                    {
                        const size_t offB = (size_t)slot * nk * P_hist_map.el_size;
                        const void *src = (const void *)((const char *)d_P_histLP + offB);
                        k_cast_any2d_vec<<<blocks_nk, threads, 0, stream>>>(src, d_Pj_tmp, (int)nk, P_hist_map.storage_prec);
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
                        k_cast_any2d_vec<<<blocks_nk, threads, 0, stream>>>(src, d_Wj_tmp, (int)nk, W_hist_map.storage_prec);
                        d_Wj = d_Wj_tmp;
                    }

                    if (g_map.output_type == CUDA_R_64F)
                    {
                        CUBLAS_CHECK(cublasGemmEx(
                            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            k, k, n,
                            dev_scalars.one(g_map.compute_type),
                            d_Wj, CUDA_R_64F, n,
                            d_Pnew, CUDA_R_64F, n,
                            dev_scalars.zero(g_map.compute_type),
                            d_Cj, CUDA_R_64F, k,
                            g_map.compute_type, CUBLAS_GEMM_DEFAULT));
                    }
                    else
                    {
                        k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Wj, d_Wj_low, (int)nk, g_map.io_prec);
                        k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Pnew, d_Pnew_low, (int)nk, g_map.io_prec);

                        CUBLAS_CHECK(cublasGemmEx(
                            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            k, k, n,
                            dev_scalars.one(g_map.compute_type),
                            d_Wj_low, g_map.data_type, n,
                            d_Pnew_low, g_map.data_type, n,
                            dev_scalars.zero(g_map.compute_type),
                            d_C_gemm, CUDA_R_32F, k,
                            g_map.compute_type, CUBLAS_GEMM_DEFAULT));

                        k_cast_f2d_mat<<<blocks_kk, threads, 0, stream>>>(d_C_gemm, d_Cj, k * k);
                    }

                    CUBLAS_CHECK(cublasDgemm(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        k, k, k,
                        dev_scalars.one64(),
                        d_Ginvj, k,
                        d_Cj, k,
                        dev_scalars.zero64(),
                        d_Yj, k));

                    CUBLAS_CHECK(cublasDgemm(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        n, k, k,
                        dev_scalars.m_one64(),
                        d_Pj, n,
                        d_Yj, k,
                        dev_scalars.one64(),
                        d_Pnew, n));
                }
            };
            phase_begin(phase_ortho);
            project_against_history();

            if (enable_anorm_reprojection && hist_count > 0)
            {
                // Check if projection removed too much A-norm and do one extra pass if needed.
                run_spmm(d_Znew, d_Znew_store, Znew_store_map, d_Wz, nullptr, fp64_storage);
                run_spmm(d_Pnew, d_Pnew_store, Pnew_store_map, d_Wnew, d_Wnew_store, Wnew_store_map);

                column_dots_kernel<<<k, threads, threads * sizeof(double), stream>>>(d_Znew, d_Wz, n, n, n, k, d_z_anorm_cols);
                column_dots_kernel<<<k, threads, threads * sizeof(double), stream>>>(d_Pnew, d_Wnew, n, n, n, k, d_p_anorm_cols);
                k_should_reproject<<<1, 1, 0, stream>>>(d_z_anorm_cols, d_p_anorm_cols, k, anorm_drop_tol_sq, d_reproject_flag);

                int reproject_flag = 0;
                CUDA_CHECK(cudaMemcpyAsync(&reproject_flag, d_reproject_flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (reproject_flag != 0)
                    project_against_history();
            }

            // Normalize each search-direction column to keep Gram diagonals well-scaled.
            // Here we enforce ||p_t||_2 = 1.
            column_norms_kernel<<<k, threads, threads * sizeof(double), stream>>>(d_Pnew, n, n, k, d_col_norms);
            k_scale_columns_from_norms<<<blocks_nk, threads, 0, stream>>>(d_Pnew, n, k, d_col_norms, 1e-30);
            sync_block_to_storage(d_Pnew, d_Pnew_store, Pnew_store_map, (int)nk, threads, stream);
            phase_end(phase_ortho);

            // Wnew = A * Pnew (sparse-dense matrix multiply).
            phase_begin(phase_spmm);
            run_spmm(d_Pnew, d_Pnew_store, Pnew_store_map, d_Wnew, d_Wnew_store, Wnew_store_map);
            phase_end(phase_spmm);

            phase_begin(phase_dense);
            if (g_map.output_type == CUDA_R_64F)
            {
                // Gnew = Pnew^T * Wnew (k x k Gram-like matrix).
                CUBLAS_CHECK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    k, k, n,
                    dev_scalars.one(g_map.compute_type),
                    d_Pnew, CUDA_R_64F, n,
                    d_Wnew, CUDA_R_64F, n,
                    dev_scalars.zero(g_map.compute_type),
                    d_Gnew, CUDA_R_64F, k,
                    g_map.compute_type, CUBLAS_GEMM_DEFAULT));
            }
            else
            {
                // Low-precision GEMM path for Gnew.
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Pnew, d_Pnew_low, (int)nk, g_map.io_prec);
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Wnew, d_Wnew_low, (int)nk, g_map.io_prec);

                CUBLAS_CHECK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    k, k, n,
                    dev_scalars.one(g_map.compute_type),
                    d_Pnew_low, g_map.data_type, n,
                    d_Wnew_low, g_map.data_type, n,
                    dev_scalars.zero(g_map.compute_type),
                    d_C_gemm, CUDA_R_32F, k,
                    g_map.compute_type, CUBLAS_GEMM_DEFAULT));

                k_cast_f2d_mat<<<blocks_kk, threads, 0, stream>>>(d_C_gemm, d_Gnew, k * k);
            }

            // Cheap A-norm normalization:
            // scale columns by 1/sqrt(diag(Gnew)) so diag(Pnew^T A Pnew) is close to 1.
            const int blocks_k = (k + threads - 1) / threads;
            k_build_col_scale_from_gdiag<<<blocks_k, threads, 0, stream>>>(d_Gnew, d_col_scale, k, 1e-30);
            k_fuse_scale_P_W<<<blocks_nk, threads, 0, stream>>>(d_Pnew, d_Wnew, d_col_scale, n, k);
            k_congruence_scale_gram<<<blocks_kk, threads, 0, stream>>>(d_Gnew, d_col_scale, k);
            // Pnew/Wnew stay stored in low precision when requested, then are
            // re-expanded to fp64 only for G, rhs, alpha, x, and r updates.
            sync_block_to_storage(d_Pnew, d_Pnew_store, Pnew_store_map, (int)nk, threads, stream);
            sync_block_to_storage(d_Wnew, d_Wnew_store, Wnew_store_map, (int)nk, threads, stream);

            // rhs = Pnew^T * r (size k)
            CUBLAS_CHECK(cublasDgemv(
                cublas, CUBLAS_OP_T,
                n, k,
                dev_scalars.one64(),
                d_Pnew, n,
                d_r, 1,
                dev_scalars.zero64(),
                d_rhs, 1));

            // alpha = G^{-1} * rhs  (Cholesky by default, SVD when params.use_svd).
            solve_kk(d_Gnew, 1, d_rhs, d_alpha);

            // x = x + Pnew * alpha
            CUBLAS_CHECK(cublasDgemv(
                cublas, CUBLAS_OP_N,
                n, k,
                dev_scalars.one64(),
                d_Pnew, n,
                d_alpha, 1,
                dev_scalars.one64(),
                d_x, 1));

            // r = r - Wnew * alpha
            CUBLAS_CHECK(cublasDgemv(
                cublas, CUBLAS_OP_N,
                n, k,
                dev_scalars.m_one64(),
                d_Wnew, n,
                d_alpha, 1,
                dev_scalars.one64(),
                d_r, 1));

            // Periodically reset recursive residual drift: r = b - A*x.
            if ((iter + 1) % reset_iter == 0)
            {
                phase_begin(phase_reset);
                CUSPARSE_CHECK(cusparseSpMV(
                    cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &scalars.s64_one, matA, vecX,
                    &scalars.s64_zero, vecTmp,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf));
                CUDA_CHECK(cudaMemcpyAsync(d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
                CUBLAS_CHECK(cublasDaxpy(cublas, n, dev_scalars.m_one64(), d_tmp, 1, d_r, 1));
                phase_end(phase_reset);
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
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Pnew, dstP, (int)nk, P_hist_map.storage_prec);
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
                k_cast_d2any<<<blocks_nk, threads, 0, stream>>>(d_Wnew, dstW, (int)nk, W_hist_map.storage_prec);
            }

            double *Gdst = d_G_hist + (size_t)slot * (size_t)k * (size_t)k;
            CUDA_CHECK(cudaMemcpyAsync(Gdst, d_Gnew, k * k * sizeof(double), cudaMemcpyDeviceToDevice, stream));
            solve_kk(d_Gnew, k, d_eye,
                     d_Ginv_hist + (size_t)slot * (size_t)k * (size_t)k);

            CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, d_scalar_tmp + 1));
            CUDA_CHECK(cudaMemcpyAsync(&current_res_norm, d_scalar_tmp + 1, sizeof(double), cudaMemcpyDeviceToHost, stream));
            phase_end(phase_dense);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            phase_accumulate(phase_precond);
            phase_accumulate(phase_ortho);
            phase_accumulate(phase_spmm);
            phase_accumulate(phase_dense);
            if ((iter + 1) % reset_iter == 0)
                phase_accumulate(phase_reset);

            result.iterations = iter + 1;
            result.relResiduals.push_back(current_res_norm / bnorm);
        }

        const auto iter_wall_end = Clock::now();

        // If we stopped due to max iterations (or maxits==0), return the latest residual.
        if (!converged)
            result.finalRes = current_res_norm;

        // Copy final solution back to host and wait for all queued GPU work.
        const auto finalize_wall_start = Clock::now();
        CUDA_CHECK(cudaMemcpyAsync(h_x.data(), d_x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto finalize_wall_end = Clock::now();

        const double total_wall_ms = std::chrono::duration<double, std::milli>(finalize_wall_end - total_wall_start).count();
        const double setup_wall_ms = std::chrono::duration<double, std::milli>(iter_wall_start - total_wall_start).count();
        const double iter_wall_ms = std::chrono::duration<double, std::milli>(iter_wall_end - iter_wall_start).count();
        const double finalize_wall_ms = std::chrono::duration<double, std::milli>(finalize_wall_end - finalize_wall_start).count();
        const double dense_exclusive_ms = std::max(0.0, phase_dense.total_ms - phase_reset.total_ms);
        const double accounted_iter_ms =
            phase_precond.total_ms + phase_ortho.total_ms + phase_spmm.total_ms + dense_exclusive_ms + phase_reset.total_ms;
        const double other_iter_ms = std::max(0.0, iter_wall_ms - accounted_iter_ms);

        result.timing.total_ms = total_wall_ms;
        result.timing.setup_ms = setup_wall_ms;
        result.timing.iter_ms = iter_wall_ms;
        result.timing.finalize_ms = finalize_wall_ms;
        result.timing.preconditioner_apply_ms = phase_precond.total_ms;
        result.timing.orthogonalization_ms = phase_ortho.total_ms;
        result.timing.spmm_ms = phase_spmm.total_ms;
        result.timing.dense_ms = dense_exclusive_ms;
        result.timing.residual_reset_ms = phase_reset.total_ms;
        result.timing.other_iter_ms = other_iter_ms;

        if (profile_enabled)
        {
            std::fprintf(stderr,
                         "[MPCG profile] n=%d k=%d iters=%d total=%.3fms setup=%.3fms iter=%.3fms finalize=%.3fms\n",
                         n, k, result.iterations, total_wall_ms, setup_wall_ms, iter_wall_ms, finalize_wall_ms);
            std::fprintf(stderr,
                         "[MPCG profile] precond=%.3fms ortho=%.3fms spmm=%.3fms dense=%.3fms reset=%.3fms other_iter=%.3fms\n",
                         phase_precond.total_ms, phase_ortho.total_ms, phase_spmm.total_ms,
                         dense_exclusive_ms, phase_reset.total_ms, other_iter_ms);
        }

        // Cleanup descriptors, buffers, and handles.
        CUSPARSE_CHECK(cusparseDestroyDnMat(dnB));
        CUSPARSE_CHECK(cusparseDestroyDnMat(dnC));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecTmp));
        if (matA_spmm)
            CUSPARSE_CHECK(cusparseDestroySpMat(matA_spmm));
        CUSPARSE_CHECK(cusparseDestroySpMat(matA));

        CUDA_CHECK(cudaFree(d_spmvBuf));
        CUDA_CHECK(cudaFree(d_spmmBuf));

        CUDA_CHECK(cudaFree(d_rowPtrA));
        CUDA_CHECK(cudaFree(d_colIndA));
        CUDA_CHECK(cudaFree(d_valA));
        if (d_valA_spmm)
            CUDA_CHECK(cudaFree(d_valA_spmm));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaFree(d_tmp));

        CUDA_CHECK(cudaFree(d_Znew));
        CUDA_CHECK(cudaFree(d_Pnew));
        CUDA_CHECK(cudaFree(d_Wnew));
        if (d_r_precond)
            CUDA_CHECK(cudaFree(d_r_precond));
        if (d_Znew_precond)
            CUDA_CHECK(cudaFree(d_Znew_precond));
        if (d_Wz)
            CUDA_CHECK(cudaFree(d_Wz));
        if (d_Znew_store)
            CUDA_CHECK(cudaFree(d_Znew_store));
        if (d_Pnew_store)
            CUDA_CHECK(cudaFree(d_Pnew_store));
        if (d_Wnew_store)
            CUDA_CHECK(cudaFree(d_Wnew_store));
        if (d_spmm_in_tmp)
            CUDA_CHECK(cudaFree(d_spmm_in_tmp));
        if (d_spmm_out_tmp)
            CUDA_CHECK(cudaFree(d_spmm_out_tmp));

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

        CUDA_CHECK(cudaFree(d_G_hist));
        CUDA_CHECK(cudaFree(d_Ginv_hist));
        CUDA_CHECK(cudaFree(d_hist_C));
        CUDA_CHECK(cudaFree(d_hist_Y));
        CUDA_CHECK(cudaFree(d_Gnew));
        CUDA_CHECK(cudaFree(d_rhs));
        CUDA_CHECK(cudaFree(d_alpha));
        CUDA_CHECK(cudaFree(d_col_scale));
        CUDA_CHECK(cudaFree(d_scalar_tmp));
        CUDA_CHECK(cudaFree(d_col_norms));
        CUDA_CHECK(cudaFree(d_z_anorm_cols));
        CUDA_CHECK(cudaFree(d_p_anorm_cols));
        CUDA_CHECK(cudaFree(d_reproject_flag));
        CUDA_CHECK(cudaFree(d_eye));
        CUDA_CHECK(cudaFree(dev_scalars.d_s32));
        CUDA_CHECK(cudaFree(dev_scalars.d_s64));
        CUDA_CHECK(cudaFree(pinv_ws.d_G_copy));
        CUDA_CHECK(cudaFree(pinv_ws.d_S));
        CUDA_CHECK(cudaFree(pinv_ws.d_U));
        CUDA_CHECK(cudaFree(pinv_ws.d_VT));
        CUDA_CHECK(cudaFree(pinv_ws.d_work));
        CUDA_CHECK(cudaFree(pinv_ws.d_S_inv));
        CUDA_CHECK(cudaFree(pinv_ws.d_T1));
        CUDA_CHECK(cudaFree(pinv_ws.d_info));

        if (chol_ws.d_G_copy)
            CUDA_CHECK(cudaFree(chol_ws.d_G_copy));
        if (chol_ws.d_work)
            CUDA_CHECK(cudaFree(chol_ws.d_work));
        if (chol_ws.d_info)
            CUDA_CHECK(cudaFree(chol_ws.d_info));

        if (d_Pnew_low)
        {
            CUDA_CHECK(cudaFree(d_Pnew_low));
            CUDA_CHECK(cudaFree(d_Wnew_low));
            CUDA_CHECK(cudaFree(d_Wj_low));
            CUDA_CHECK(cudaFree(d_C_gemm));
        }

        for (int t = 0; t < k; ++t)
        {
            CUDA_CHECK(cudaEventDestroy(precond_events[t]));
            CUDA_CHECK(cudaStreamDestroy(precond_streams[t]));
        }
        CUDA_CHECK(cudaEventDestroy(main_stream_ready));
        destroy_phase(phase_precond);
        destroy_phase(phase_ortho);
        destroy_phase(phase_spmm);
        destroy_phase(phase_dense);
        destroy_phase(phase_reset);

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUSPARSE_CHECK(cusparseDestroy(cusparse));
        CUBLAS_CHECK(cublasDestroy(cublas));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver));

        return result;
    }

    namespace debug
    {
        std::vector<double> build_mpcg_znew_columns(
            const std::vector<ichol::precond::PrecondApply> &preconds,
            const std::vector<double> &h_r,
            const ZnewBuildOptions &options)
        {
            const int n = static_cast<int>(h_r.size());
            const int k = static_cast<int>(preconds.size());
            const size_t nk = static_cast<size_t>(n) * static_cast<size_t>(k);
            const int threads = 256;

            StorageMap precond_map = get_precond_map(options.prec_precond);

            cudaStream_t stream = nullptr;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

            std::vector<cudaStream_t> precond_streams(k, nullptr);
            std::vector<cudaEvent_t> precond_events(k, nullptr);
            cudaEvent_t main_stream_ready = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&main_stream_ready, cudaEventDisableTiming));
            for (int t = 0; t < k; ++t)
            {
                CUDA_CHECK(cudaStreamCreateWithFlags(&precond_streams[t], cudaStreamNonBlocking));
                CUDA_CHECK(cudaEventCreateWithFlags(&precond_events[t], cudaEventDisableTiming));
            }

            double *d_r = nullptr;
            double *d_Znew = nullptr;
            void *d_r_precond = nullptr;
            void *d_Znew_precond = nullptr;
            std::vector<double> h_Znew(nk, 0.0);

            CUDA_CHECK(cudaMalloc(&d_r, static_cast<size_t>(n) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Znew, nk * sizeof(double)));
            CUDA_CHECK(cudaMemcpyAsync(d_r, h_r.data(), static_cast<size_t>(n) * sizeof(double), cudaMemcpyHostToDevice, stream));

            if (precond_map.storage_prec != ComputePrecision::FP64)
            {
                CUDA_CHECK(cudaMalloc(&d_r_precond, static_cast<size_t>(n) * precond_map.el_size));
                CUDA_CHECK(cudaMalloc(&d_Znew_precond, nk * precond_map.el_size));
            }

            build_znew_columns_device(
                preconds,
                d_r,
                d_r_precond,
                d_Znew,
                d_Znew_precond,
                n,
                precond_map,
                threads,
                stream,
                main_stream_ready,
                precond_streams,
                precond_events,
                options.serial,
                options.sync_after_each_apply);

            CUDA_CHECK(cudaMemcpyAsync(h_Znew.data(), d_Znew, nk * sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaFree(d_r));
            CUDA_CHECK(cudaFree(d_Znew));
            if (d_r_precond)
                CUDA_CHECK(cudaFree(d_r_precond));
            if (d_Znew_precond)
                CUDA_CHECK(cudaFree(d_Znew_precond));

            for (int t = 0; t < k; ++t)
            {
                CUDA_CHECK(cudaEventDestroy(precond_events[t]));
                CUDA_CHECK(cudaStreamDestroy(precond_streams[t]));
            }
            CUDA_CHECK(cudaEventDestroy(main_stream_ready));
            CUDA_CHECK(cudaStreamDestroy(stream));
            return h_Znew;
        }
    } // namespace debug

    template PCGResult mpcg<double>(
        const std::vector<int> &,
        const std::vector<int> &,
        const std::vector<double> &,
        const std::vector<ichol::precond::PrecondApply> &,
        const std::vector<double> &,
        std::vector<double> &,
        const PCGParams &);
} // namespace ichol::solver
