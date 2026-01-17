#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
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

struct CusparseHandle
{
    cusparseHandle_t handle = nullptr;
    CusparseHandle() { CUSPARSE_CHECK(cusparseCreate(&handle)); }
    ~CusparseHandle()
    {
        if (handle)
            CUSPARSE_CHECK(cusparseDestroy(handle));
    }
    cusparseHandle_t get() const { return handle; }
    operator cusparseHandle_t() const { return handle; }
};

struct CublasHandle
{
    cublasHandle_t handle = nullptr;
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle)); }
    ~CublasHandle()
    {
        if (handle)
            CUBLAS_CHECK(cublasDestroy(handle));
    }
    cublasHandle_t get() const { return handle; }
    operator cublasHandle_t() const { return handle; }
};

struct CusparseSpMat
{
    cusparseSpMatDescr_t mat = nullptr;
    ~CusparseSpMat()
    {
        if (mat)
            CUSPARSE_CHECK(cusparseDestroySpMat(mat));
    }
    void create(int rows, int cols, int nnz, int *row_ptr, int *col_ind, double *values)
    {
        CUSPARSE_CHECK(cusparseCreateCsr(&mat, rows, cols, nnz,
                                         row_ptr, col_ind, values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    }
    cusparseSpMatDescr_t get() const { return mat; }
    operator cusparseSpMatDescr_t() const { return mat; }
};

struct CusparseDnVec
{
    cusparseDnVecDescr_t vec = nullptr;
    ~CusparseDnVec()
    {
        if (vec)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vec));
    }
    void create(int n, double *data)
    {
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec, n, data, CUDA_R_64F));
    }
    cusparseDnVecDescr_t get() const { return vec; }
    operator cusparseDnVecDescr_t() const { return vec; }
};

struct CudaEvent
{
    cudaEvent_t evt = nullptr;
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&evt)); }
    ~CudaEvent()
    {
        if (evt)
            CUDA_CHECK(cudaEventDestroy(evt));
    }
    cudaEvent_t get() const { return evt; }
    operator cudaEvent_t() const { return evt; }
};

template <typename T>
struct DeviceBuffer
{
    T *ptr = nullptr;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { alloc(count); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other)
        {
            release();
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    ~DeviceBuffer() { release(); }
    void alloc(size_t count)
    {
        release();
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }
    void release()
    {
        if (ptr)
            CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
    T *get() const { return ptr; }
    operator T *() const { return ptr; }
};

/**
 * Given a CSR matrix in lower-triangular + diagonal form, build its transpose
 *
 * Each row is assumed to be sorted by column index, with the diagonal entry last.
 */
template <typename ValueT>
static void build_csr_trans(
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

    // Count nnz per row in transpose
    // Split off-diagonals and diagonal to avoid loading diag col_ind.
    for (int i = 0; i < n; ++i)
    {
        const int s = row_ptr[i];
        const int e = row_ptr[i + 1];
        const int end = e - 1; // diag position in row i

        for (int p = s; p < end; ++p)
        {
            const int j = col_ind[p]; // j < i
            ++row_ptr_T[j + 1];
        }

        // diagonal (i,i)
        ++row_ptr_T[i + 1];
    }

    // Prefix sum to CSR row_ptr_T.
    for (int r = 0; r < n; ++r)
        row_ptr_T[r + 1] += row_ptr_T[r];

    // Write heads for off-diagonals only; diagonal always goes to last slot.
    std::vector<int> next(n);
    for (int r = 0; r < n; ++r)
        next[r] = row_ptr_T[r];

    // Fill transpose with diagonal-last directly (no per-row scan/memmove).
    for (int i = 0; i < n; ++i)
    {
        const int s = row_ptr[i];
        const int e = row_ptr[i + 1];
        const int end = e - 1; // diag

        // off-diagonals (i > j): go from the front, i increases => sorted in T rows
        for (int p = s; p < end; ++p)
        {
            const int j = col_ind[p];
            const int dst = next[j]++; // uses [row_ptr_T[j], row_ptr_T[j+1)-1)
            col_ind_T[dst] = i;        // sorted by construction
            val_T[dst] = val[p];
        }

        // diagonal goes to the last position of row i in transpose
        const int diag_dst = row_ptr_T[i + 1] - 1;
        col_ind_T[diag_dst] = i;
        val_T[diag_dst] = val[end];
    }
}

static ichol::symbolic::LevelSets build_level_sets_csr_diag_last(
    int n,
    const std::vector<int> &row_ptr,
    const std::vector<int> &col_ind,
    bool reverse)
{
    int max_level = -1;
    std::vector<int> level_of(n, -1);

    for (int ii = 0; ii < n; ++ii)
    {
        int i = reverse ? (n - 1 - ii) : ii;
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

namespace ichol::solver
{
    template <typename T_L>
    void pcg(
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
        CusparseHandle cusparseHandle;
        CublasHandle cublasHandle;

        const int n = static_cast<int>(h_csrRowPtrA.size()) - 1;
        const int nnzA = static_cast<int>(h_valA.size());
        const int nnzL = static_cast<int>(h_valL.size());

        constexpr bool L_is_fp64 = std::is_same<T_L, double>::value;
        constexpr bool L_is_fp32 = std::is_same<T_L, float>::value;

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
        build_csr_trans<SolveT>(
            n, h_csrRowPtrL, h_csrColIndL, h_valL_solve,
            h_csrRowPtrLt, h_csrColIndLt, h_valLt_solve);

        // Build level sets for L (lower) and L^T (upper).
        const ichol::symbolic::LevelSets levelsets_L =
            build_level_sets_csr_diag_last(n, h_csrRowPtrL, h_csrColIndL, false);

        const ichol::symbolic::LevelSets levelsets_Lt =
            build_level_sets_csr_diag_last(n, h_csrRowPtrLt, h_csrColIndLt, true);

        cudaStream_t stream = 0;
        DeviceLevelSets d_levelsets_L;
        DeviceLevelSets d_levelsets_Lt;
        int rc_levels = d_levelsets_L.init(levelsets_L, stream);
        if (rc_levels != 0)
        {
            std::cerr << "ERROR: SpTRSV levelset upload (L) failed\n";
            iterations = 0;
            finalRes = std::numeric_limits<double>::infinity();
            return;
        }
        rc_levels = d_levelsets_Lt.init(levelsets_Lt, stream);
        if (rc_levels != 0)
        {
            std::cerr << "ERROR: SpTRSV levelset upload (L^T) failed\n";
            iterations = 0;
            finalRes = std::numeric_limits<double>::infinity();
            return;
        }

        std::vector<double> h_diagA(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            h_diagA[i] = h_valA[h_csrRowPtrA[i + 1] - 1];
        }
        DeviceBuffer<double> d_diagA(n);
        CUDA_CHECK(cudaMemcpy(d_diagA.get(), h_diagA.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        // Copy A to device
        DeviceBuffer<int> d_csrRowPtrA(n + 1);
        DeviceBuffer<int> d_csrColIndA(nnzA);
        DeviceBuffer<double> d_valA(nnzA);
        CUDA_CHECK(cudaMemcpy(d_csrRowPtrA.get(), h_csrRowPtrA.data(),
                              (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndA.get(), h_csrColIndA.data(),
                              nnzA * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valA.get(), h_valA.data(),
                              nnzA * sizeof(double), cudaMemcpyHostToDevice));

        CusparseSpMat spMatA;
        spMatA.create(n, n, nnzA, d_csrRowPtrA.get(), d_csrColIndA.get(), d_valA.get());

        // Copy L (CSR) to device (indices + values in SolveT)
        DeviceBuffer<int> d_csrRowPtrL(n + 1);
        DeviceBuffer<int> d_csrColIndL(nnzL);
        DeviceBuffer<SolveT> d_valL(nnzL);
        CUDA_CHECK(cudaMemcpy(d_csrRowPtrL.get(), h_csrRowPtrL.data(),
                              (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndL.get(), h_csrColIndL.data(),
                              nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valL.get(), h_valL_solve.data(),
                              nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));

        // Copy L^T (CSR) to device (indices + values in SolveT)
        DeviceBuffer<int> d_csrRowPtrLt(n + 1);
        DeviceBuffer<int> d_csrColIndLt(nnzL);
        DeviceBuffer<SolveT> d_valLt(nnzL);
        CUDA_CHECK(cudaMemcpy(d_csrRowPtrLt.get(), h_csrRowPtrLt.data(),
                              (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndLt.get(), h_csrColIndLt.data(),
                              nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valLt.get(), h_valLt_solve.data(),
                              nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));

        // 3) Allocate vectors x, b, etc. in double
        DeviceBuffer<double> d_x(n);
        DeviceBuffer<double> d_b(n);
        CUDA_CHECK(cudaMemset(d_x.get(), 0, n * sizeof(double))); // x=0 initial
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        // new: set up D
        DeviceBuffer<double> d_D(n);
        CUDA_CHECK(cudaMemcpy(d_D.get(), h_D.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        DeviceBuffer<double> d_Dr(n);

        int block = 256;
        int grid = (n + block - 1) / block;
        ew_mul<<<grid, block>>>(n, d_D.get(), d_b.get(), d_Dr.get()); // d_Dr = D .* d_b

        /**
         * Note that \tilde{b} = D^{-1} b is passed into this func.
         * This computes the norm of original b for convergence check:
         */
        double bnorm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_Dr.get(), 1, &bnorm));

        // For A * p
        CusparseDnVec vecP_dev;
        CusparseDnVec vecQ_dev;
        DeviceBuffer<double> d_p(n);
        DeviceBuffer<double> d_q(n);
        DeviceBuffer<double> d_r(n);
        DeviceBuffer<double> d_z(n); // outer z (FP64) used in dot products

        vecP_dev.create(n, d_p.get());
        vecQ_dev.create(n, d_q.get());

        // A spMV buffer for A
        size_t spmvBufSize = 0;
        DeviceBuffer<char> d_spmvBuf;
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
                d_spmvBuf.alloc(spmvBufSize);
        }

        // Preconditioner work vectors (SolveT precision). No per-iteration allocations.
        SolveT *d_r_work = nullptr, *d_w_work = nullptr, *d_z_work = nullptr;
        DeviceBuffer<SolveT> d_r_work_buf;
        DeviceBuffer<SolveT> d_w_work_buf;
        DeviceBuffer<SolveT> d_z_work_buf;
        if constexpr (std::is_same_v<SolveT, double>)
        {
            d_r_work = reinterpret_cast<SolveT *>(d_r.get()); // alias
            d_z_work = reinterpret_cast<SolveT *>(d_z.get()); // alias
            d_w_work_buf.alloc(n);
            d_w_work = d_w_work_buf.get();
        }
        else
        {
            d_r_work_buf.alloc(n);
            d_w_work_buf.alloc(n);
            d_z_work_buf.alloc(n);
            d_r_work = d_r_work_buf.get();
            d_w_work = d_w_work_buf.get();
            d_z_work = d_z_work_buf.get();
        }

        cudaMemcpy(d_r.get(), d_b.get(), n * sizeof(double), cudaMemcpyDeviceToDevice);

        double nrmr0 = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_r.get(), 1, &nrmr0));
        double tol = 1e-6;
        double rho = 0.0, rhoOld = 0.0;
        int maxIters = 1000;
        iterations = 0;

        double sptrsv_total_ms = 0.0;
        int sptrsv_timed_iters = 0;
        CudaEvent sptrsv_start;
        CudaEvent sptrsv_stop;

        for (int k = 1; k <= maxIters; k++)
        {
            // (1) z = M^-1 r
            //   L * w = r
            //   L^T * z = w
            {
                if constexpr (!std::is_same_v<SolveT, double>)
                {
                    cast_vec<SolveT, double><<<grid, block>>>(n, d_r.get(), d_r_work);
                    CUDA_CHECK(cudaGetLastError());
                }

                CUDA_CHECK(cudaEventRecord(sptrsv_start, stream));

                int rc1 = SpTRSV_solve_levelsets_device<int, SolveT>(
                    n,
                    d_csrRowPtrL.get(),
                    d_csrColIndL.get(),
                    d_valL.get(),
                    d_r_work,
                    d_w_work,
                    FillMode::LOWER,
                    /*unit_diag=*/false,
                    d_levelsets_L,
                    stream);

                if (rc1 != 0)
                {
                    CUDA_CHECK(cudaEventRecord(sptrsv_stop, stream));
                    float iter_ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&iter_ms, sptrsv_start, sptrsv_stop));
                    sptrsv_total_ms += static_cast<double>(iter_ms);
                    sptrsv_timed_iters += 1;
                    std::cerr << "ERROR: SpTRSV(L) failed with code " << rc1 << " at iter " << k << "\n";
                    iterations = k;
                    finalRes = std::numeric_limits<double>::infinity();
                    break;
                }

                int rc2 = SpTRSV_solve_levelsets_device<int, SolveT>(
                    n,
                    d_csrRowPtrLt.get(),
                    d_csrColIndLt.get(),
                    d_valLt.get(),
                    d_w_work,
                    d_z_work,
                    FillMode::UPPER,
                    /*unit_diag=*/false,
                    d_levelsets_Lt,
                    stream);

                CUDA_CHECK(cudaEventRecord(sptrsv_stop, stream));

                if (rc2 != 0)
                {
                    float iter_ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&iter_ms, sptrsv_start, sptrsv_stop));
                    sptrsv_total_ms += static_cast<double>(iter_ms);
                    sptrsv_timed_iters += 1;
                    std::cerr << "ERROR: SpTRSV(L^T) failed with code " << rc2 << " at iter " << k << "\n";
                    iterations = k;
                    finalRes = std::numeric_limits<double>::infinity();
                    break;
                }

                if constexpr (!std::is_same_v<SolveT, double>)
                {
                    cast_vec<double, SolveT><<<grid, block>>>(n, d_z_work, d_z.get());
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            // (2) rho = r^T z
            rhoOld = rho;
            CUBLAS_CHECK(cublasDdot(cublasHandle, n, d_r.get(), 1, d_z.get(), 1, &rho));
            float iter_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&iter_ms, sptrsv_start, sptrsv_stop));
            sptrsv_total_ms += static_cast<double>(iter_ms);
            sptrsv_timed_iters += 1;

            // (3) p update
            if (k == 1)
            {
                cudaMemcpy(d_p.get(), d_z.get(), n * sizeof(double), cudaMemcpyDeviceToDevice);
            }
            else
            {
                double beta = (rho / rhoOld);
                CUBLAS_CHECK(cublasDscal(cublasHandle, n, &beta, d_p.get(), 1));
                double alphaOne = 1.0;
                CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &alphaOne, d_z.get(), 1, d_p.get(), 1));
            }

            // (4) q = A p, where A is symmetric but stored as (L+D) only.
            // q = (L+D)*p + (L^T)*p - diag(A).*p
            {
                double alpha1 = 1.0;
                double beta0 = 0.0;
                double beta1 = 1.0;

                CUSPARSE_CHECK(cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta0, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf.get()));

                CUSPARSE_CHECK(cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha1, spMatA, vecP_dev,
                    &beta1, vecQ_dev,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    d_spmvBuf.get()));

                diag_sub_from_diag<<<grid, block>>>(n, d_diagA.get(), d_p.get(), d_q.get());
                CUDA_CHECK(cudaGetLastError());
            }

            // (5) alpha = rho / (p^T q)
            double denom = 0.0;
            CUBLAS_CHECK(cublasDdot(cublasHandle, n, d_p.get(), 1, d_q.get(), 1, &denom));

            if (denom <= 0.0 || std::isnan(denom) || std::isinf(denom))
            {
                std::cerr << "ERROR: denom invalid in iter " << k << ": " << denom << "\n";
                iterations = k;
                finalRes = std::numeric_limits<double>::infinity();
                break;
            }

            double alpha = rho / denom;

            // (6) x = x + alpha p
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &alpha, d_p.get(), 1, d_x.get(), 1));

            // (7) r = r - alpha q
            double negAlpha = -alpha;
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &negAlpha, d_q.get(), 1, d_r.get(), 1));

            // (8) check convergence
            ew_mul<<<grid, block>>>(n, d_D.get(), d_r.get(), d_Dr.get());

            double nrmDr = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_Dr.get(), 1, &nrmDr));

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
        {
            iterations = maxIters;
            // finalRes = nrmDr / bnorm;
        }

        double sptrsv_avg_ms = 0.0;
        if (sptrsv_timed_iters > 0)
            sptrsv_avg_ms = sptrsv_total_ms / static_cast<double>(sptrsv_timed_iters);

        h_x.resize(n);
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x.get(), n * sizeof(double), cudaMemcpyDeviceToHost));

        std::cout << "SpTRSV total time (ms): " << sptrsv_total_ms
                  << ", avg per iteration (ms): " << sptrsv_avg_ms << "\n";
    }

    template void pcg<double>(
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

    template void pcg<float>(
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

    template void pcg<half_float::half>(
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

} // namespace ichol::solver
