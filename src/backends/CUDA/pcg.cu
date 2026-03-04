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
#include "ichol/pcg.hpp"
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
    PCGResult pcg(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<T_L> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        const PCGParams &params)
    {
        CusparseHandle cusparseHandle;
        CublasHandle cublasHandle;

        const int n = static_cast<int>(h_csrRowPtrA.size()) - 1;
        const int nnzA = static_cast<int>(h_valA.size());
        const int nnzL = static_cast<int>(h_valL.size());

        // Determine A format. If params doesn't specify, assume Full if nnz is high.
        // For 2D Poisson (5pt), full has ~5n. L+D has ~3n.
        const bool A_is_full = (nnzA > 3 * n);

        constexpr bool L_is_fp64 = std::is_same<T_L, double>::value;
        constexpr bool L_is_fp32 = std::is_same<T_L, float>::value;
        using SolveT = std::conditional_t<L_is_fp64, double, std::conditional_t<L_is_fp32, float, __half>>;

        cudaStream_t stream = 0;
        PCGResult result{};
        const bool use_custom_precond = (params.custom_precond != nullptr && params.custom_precond->apply != nullptr);

        // --- A Matrix Buffers ---
        DeviceBuffer<int> d_csrRowPtrA(n + 1);
        DeviceBuffer<int> d_csrColIndA(nnzA);
        DeviceBuffer<double> d_valA(nnzA);
        DeviceBuffer<double> d_diagA;

        if (!A_is_full)
        {
            d_diagA.alloc(n);
            std::vector<double> h_diagA(n, 0.0);
            for (int i = 0; i < n; ++i)
            {
                bool found = false;
                for (int p = h_csrRowPtrA[i]; p < h_csrRowPtrA[i + 1]; ++p)
                {
                    if (h_csrColIndA[p] == i)
                    {
                        h_diagA[i] = h_valA[p];
                        found = true;
                        break;
                    }
                }
                if (!found)
                    h_diagA[i] = h_valA[h_csrRowPtrA[i + 1] - 1]; // Fallback
            }
            CUDA_CHECK(cudaMemcpy(d_diagA.get(), h_diagA.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMemcpy(d_csrRowPtrA.get(), h_csrRowPtrA.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndA.get(), h_csrColIndA.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valA.get(), h_valA.data(), nnzA * sizeof(double), cudaMemcpyHostToDevice));

        CusparseSpMat spMatA;
        spMatA.create(n, n, nnzA, d_csrRowPtrA.get(), d_csrColIndA.get(), d_valA.get());

        // --- Preconditioner Buffers (Factorized or Full) ---
        SpTRSVLevelsetsPlan sptrsv_plan_L, sptrsv_plan_Lt;
        DeviceBuffer<int> d_csrRowPtrL, d_csrColIndL, d_csrRowPtrLt, d_csrColIndLt;
        DeviceBuffer<SolveT> d_valL, d_valLt;
        CusparseSpMat spMatM; // Used if precond_full = true

        if (!use_custom_precond)
        {
            if (params.precond_full)
            {
                d_csrRowPtrL.alloc(n + 1);
                d_csrColIndL.alloc(nnzL);
                d_valL.alloc(nnzL);
                CUDA_CHECK(cudaMemcpy(d_csrRowPtrL.get(), h_csrRowPtrL.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_csrColIndL.get(), h_csrColIndL.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));

                std::vector<SolveT> h_valL_solve(nnzL);
                for (int i = 0; i < nnzL; ++i)
                    h_valL_solve[i] = static_cast<SolveT>(h_valL[i]);
                CUDA_CHECK(cudaMemcpy(d_valL.get(), h_valL_solve.data(), nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));

                // Assume precond_full is an approximate inverse; apply via SpMV
                spMatM.create(n, n, nnzL, d_csrRowPtrL.get(), d_csrColIndL.get(), (double *)d_valL.get());
            }
            else
            {
                // Factorized L*L^T Setup
                std::vector<SolveT> h_valL_solve(nnzL);
                for (int i = 0; i < nnzL; ++i)
                    h_valL_solve[i] = static_cast<SolveT>(h_valL[i]);

                std::vector<int> h_csrRowPtrLt, h_csrColIndLt;
                std::vector<SolveT> h_valLt_solve;
                build_csr_trans<SolveT>(n, h_csrRowPtrL, h_csrColIndL, h_valL_solve, h_csrRowPtrLt, h_csrColIndLt, h_valLt_solve);

                const auto levelsets_L = build_level_sets_csr_diag_last(n, h_csrRowPtrL, h_csrColIndL, false);
                const auto levelsets_Lt = build_level_sets_csr_diag_last(n, h_csrRowPtrLt, h_csrColIndLt, true);

                sptrsv_plan_L.init(levelsets_L, stream);
                sptrsv_plan_Lt.init(levelsets_Lt, stream);

                d_csrRowPtrL.alloc(n + 1);
                d_csrColIndL.alloc(nnzL);
                d_valL.alloc(nnzL);
                CUDA_CHECK(cudaMemcpy(d_csrRowPtrL.get(), h_csrRowPtrL.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_csrColIndL.get(), h_csrColIndL.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_valL.get(), h_valL_solve.data(), nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));

                d_csrRowPtrLt.alloc(n + 1);
                d_csrColIndLt.alloc(nnzL);
                d_valLt.alloc(nnzL);
                CUDA_CHECK(cudaMemcpy(d_csrRowPtrLt.get(), h_csrRowPtrLt.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_csrColIndLt.get(), h_csrColIndLt.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_valLt.get(), h_valLt_solve.data(), nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));
            }
        }

        // --- Vector Setup ---
        DeviceBuffer<double> d_x(n), d_b(n), d_p(n), d_q(n), d_r(n), d_z(n);
        if (static_cast<int>(h_x.size()) == n)
            CUDA_CHECK(cudaMemcpy(d_x.get(), h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        else
            CUDA_CHECK(cudaMemset(d_x.get(), 0, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        CusparseDnVec vecP_dev, vecQ_dev, vecR_dev, vecZ_dev;
        vecP_dev.create(n, d_p.get());
        vecQ_dev.create(n, d_q.get());
        vecR_dev.create(n, d_r.get());
        vecZ_dev.create(n, d_z.get());

        size_t spmvBufSize = 0;
        DeviceBuffer<char> d_spmvBuf;
        {
            size_t sNT = 0, sT = 0;
            double a1 = 1.0, b0 = 0.0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &a1, spMatA, vecP_dev, &b0, vecQ_dev, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &sNT));
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &a1, spMatA, vecP_dev, &b0, vecQ_dev, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &sT));
            spmvBufSize = std::max(sNT, sT);
            if (spmvBufSize > 0)
                d_spmvBuf.alloc(spmvBufSize);
        }

        SolveT *d_r_work = nullptr, *d_w_work = nullptr, *d_z_work = nullptr;
        DeviceBuffer<SolveT> d_r_work_buf, d_w_work_buf, d_z_work_buf;
        if (!use_custom_precond && !params.precond_full)
        {
            d_r_work_buf.alloc(n);
            d_w_work_buf.alloc(n);
            d_z_work_buf.alloc(n);
            d_r_work = d_r_work_buf.get();
            d_w_work = d_w_work_buf.get();
            d_z_work = d_z_work_buf.get();
        }

        CUDA_CHECK(cudaMemcpy(d_r.get(), d_b.get(), n * sizeof(double), cudaMemcpyDeviceToDevice));
        double rho = 0.0, bnorm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_b.get(), 1, &bnorm));
        if (bnorm == 0.0)
            bnorm = 1.0;

        double sptrsv_total_ms = 0.0;
        int sptrsv_timed_iters = 0;
        CudaEvent sptrsv_start, sptrsv_stop;

        // --- Main Loop ---
        for (int k = 1; k <= params.maxits; k++)
        {
            // (1) z = M^-1 r
            if (use_custom_precond)
            {
                params.custom_precond->apply(params.custom_precond->ctx, d_r.get(), d_z.get(), n, stream);
            }
            else if (params.precond_full)
            {
                double a1 = 1.0, b0 = 0.0;
                CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &a1, spMatM, vecR_dev, &b0, vecZ_dev, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf.get()));
            }
            else
            {
                cast_vec<SolveT, double><<<(n + 255) / 256, 256, 0, stream>>>(n, d_r.get(), d_r_work);
                CUDA_CHECK(cudaEventRecord(sptrsv_start, stream));
                sptrsv_plan_L.solve<int, SolveT>(n, d_csrRowPtrL.get(), d_csrColIndL.get(), d_valL.get(), d_r_work, d_w_work, false, stream);
                sptrsv_plan_Lt.solve<int, SolveT>(n, d_csrRowPtrLt.get(), d_csrColIndLt.get(), d_valLt.get(), d_w_work, d_z_work, false, stream);
                CUDA_CHECK(cudaEventRecord(sptrsv_stop, stream));
                cast_vec<double, SolveT><<<(n + 255) / 256, 256, 0, stream>>>(n, d_z_work, d_z.get());

                float iter_ms = 0.0f;
                CUDA_CHECK(cudaEventSynchronize(sptrsv_stop));
                CUDA_CHECK(cudaEventElapsedTime(&iter_ms, sptrsv_start, sptrsv_stop));
                sptrsv_total_ms += iter_ms;
                sptrsv_timed_iters++;
            }

            double rhoOld = rho;
            CUBLAS_CHECK(cublasDdot(cublasHandle, n, d_r.get(), 1, d_z.get(), 1, &rho));

            // (3) p update
            if (k == 1)
                CUDA_CHECK(cudaMemcpy(d_p.get(), d_z.get(), n * sizeof(double), cudaMemcpyDeviceToDevice));
            else
            {
                double beta = rho / rhoOld;
                CUBLAS_CHECK(cublasDscal(cublasHandle, n, &beta, d_p.get(), 1));
                double a1 = 1.0;
                CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &a1, d_z.get(), 1, d_p.get(), 1));
            }

            // (4) q = A p
            double alpha1 = 1.0, beta0 = 0.0;
            CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha1, spMatA, vecP_dev, &beta0, vecQ_dev, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf.get()));

            if (!A_is_full)
            {
                double beta1 = 1.0;
                CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &alpha1, spMatA, vecP_dev, &beta1, vecQ_dev, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf.get()));
                diag_sub_from_diag<<<(n + 255) / 256, 256, 0, stream>>>(n, d_diagA.get(), d_p.get(), d_q.get());
            }

            // (5) alpha = rho / (p^T q)
            double denom = 0.0;
            CUBLAS_CHECK(cublasDdot(cublasHandle, n, d_p.get(), 1, d_q.get(), 1, &denom));
            double alpha = rho / denom;

            // (6) x = x + alpha p; (7) r = r - alpha q
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &alpha, d_p.get(), 1, d_x.get(), 1));
            double negAlpha = -alpha;
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &negAlpha, d_q.get(), 1, d_r.get(), 1));

            // (8) Convergence
            double res_norm = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_r.get(), 1, &res_norm));
            if (res_norm <= params.tol * bnorm)
            {
                result.iterations = k;
                result.finalRes = res_norm;
                break;
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_x.resize(n);
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x.get(), n * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }
    
    template PCGResult pcg<double>(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<double> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        const PCGParams &params);

    template PCGResult pcg<float>(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<float> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        const PCGParams &params);

    template PCGResult pcg<half_float::half>(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<half_float::half> &h_valL,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const std::vector<double> &h_D,
        const PCGParams &params);

} // namespace ichol::solver
