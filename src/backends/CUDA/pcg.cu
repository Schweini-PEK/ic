#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ichol/half.hpp"
#include "ichol/pcg.hpp"
#include "factor/supernodal_solve.hpp"
#include "factor/supernodal_solve_cuda.cuh"
#include "factor/supernodal_storage.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "solve/sptrsv/cuda/sptrsv_level.cuh"

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

__global__ void ew_mul(int n, const double *a, const double *b, double *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * b[i];
}

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

    for (int i = 0; i < n; ++i)
    {
        const int s = row_ptr[i];
        const int e = row_ptr[i + 1];
        const int end = e - 1;

        for (int p = s; p < end; ++p)
        {
            const int j = col_ind[p];
            ++row_ptr_T[j + 1];
        }

        ++row_ptr_T[i + 1];
    }

    for (int r = 0; r < n; ++r)
        row_ptr_T[r + 1] += row_ptr_T[r];

    std::vector<int> next(n);
    for (int r = 0; r < n; ++r)
        next[r] = row_ptr_T[r];

    for (int i = 0; i < n; ++i)
    {
        const int s = row_ptr[i];
        const int e = row_ptr[i + 1];
        const int end = e - 1;

        for (int p = s; p < end; ++p)
        {
            const int j = col_ind[p];
            const int dst = next[j]++;
            col_ind_T[dst] = i;
            val_T[dst] = val[p];
        }

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
        for (int p = row_ptr[i]; p < row_ptr[i + 1] - 1; ++p)
            best = std::max(best, level_of[col_ind[p]]);

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

static bool csr_has_upper_triangle_entries(
    int n,
    const std::vector<int> &row_ptr,
    const std::vector<int> &col_ind)
{
    for (int i = 0; i < n; ++i)
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
        {
            if (col_ind[p] > i)
                return true;
        }
    }
    return false;
}

struct SupernodeCandidateMetrics
{
    bool enabled = false;
    double relaxed_extra = 0.0;
    int relaxed_max_width = 0;
    int num_supernodes = 0;
    int scalar_num_levels = 0;
    int supernode_num_levels = 0;
    double avg_width = 0.0;
    double block_density = 0.0;
    double level_compression_rate = 0.0;
};

static SupernodeCandidateMetrics analyze_supernode_candidates_from_sym_l(
    int n,
    const std::vector<int> &row_ptr,
    const std::vector<int> &col_ind,
    const ichol::symbolic::SuperSym &sym)
{
    SupernodeCandidateMetrics m;
    m.enabled = true;

    if (n <= 0)
        return m;

    const auto scalar_levelsets = build_level_sets_csr_diag_last(n, row_ptr, col_ind, false);
    m.scalar_num_levels = std::max(0, (int)scalar_levelsets.level_ptr.size() - 1);

    m.num_supernodes = std::max(0, (int)sym.super.size() - 1);
    m.avg_width = (m.num_supernodes > 0) ? (double)n / (double)m.num_supernodes : 0.0;

    const long long dense_slots = sym.px.empty() ? 0LL : (long long)sym.px.back();
    const long long actual_nnz = (long long)col_ind.size();
    m.block_density = (dense_slots > 0) ? (double)actual_nnz / (double)dense_slots : 0.0;

    auto plan = ichol::symbolic::ll_plan_from_sym(sym, n);
    m.supernode_num_levels = (int)plan.buckets.size();
    m.level_compression_rate =
        (m.scalar_num_levels > 0)
            ? (1.0 - (double)m.supernode_num_levels / (double)m.scalar_num_levels)
            : 0.0;

    return m;
}

namespace
{
    template <typename T_L>
    using SolveType = std::conditional_t<std::is_same_v<T_L, double>, double,
                                         std::conditional_t<std::is_same_v<T_L, float>, float, __half>>;

    struct FullPrecondContext
    {
        CusparseSpMat spMatM;
        DeviceBuffer<int> d_csrRowPtrM;
        DeviceBuffer<int> d_csrColIndM;
        DeviceBuffer<double> d_valM;
    };

    template <typename SolveT>
    struct LevelPrecondContext
    {
        SpTRSVLevelsetsPlan sptrsv_plan_L;
        SpTRSVLevelsetsPlan sptrsv_plan_Lt;
        DeviceBuffer<int> d_csrRowPtrL;
        DeviceBuffer<int> d_csrColIndL;
        DeviceBuffer<int> d_csrRowPtrLt;
        DeviceBuffer<int> d_csrColIndLt;
        DeviceBuffer<SolveT> d_valL;
        DeviceBuffer<SolveT> d_valLt;
    };

    template <typename SolveT>
    struct SupernodePrecondContext
    {
        SupernodeCandidateMetrics metrics;
        std::vector<int> super;
        ichol::symbolic::SuperSym sym;
        ichol::symbolic::SupernodalLLPlan plan;
        std::vector<double> packed_values_fp64;
        std::vector<SolveT> packed_values;
        std::vector<std::vector<int>> solve_buckets;
        std::vector<int> solve_bucket_ptr;
        std::vector<int> solve_bucket_nodes;
        int solve_num_levels = 0;
        int solve_max_bucket_size = 0;
        std::vector<int> lt_row_ptr;
        std::vector<int> lt_col_ind;

        DeviceBuffer<int> d_super;
        DeviceBuffer<int> d_pi;
        DeviceBuffer<int> d_px;
        DeviceBuffer<int> d_s;
        DeviceBuffer<SolveT> d_packed_values;
        DeviceBuffer<int> d_solve_bucket_ptr;
        DeviceBuffer<int> d_solve_bucket_nodes;
        DeviceBuffer<int> d_lt_row_ptr;
        DeviceBuffer<int> d_lt_col_ind;

        bool solve_implemented = false;
    };

    constexpr double kFixedRelaxedSupernodeExtra = 0.3;
    constexpr int kFixedRelaxedSupernodeMaxWidth = 4;

    static void flatten_supernode_solve_buckets(
        const std::vector<std::vector<int>> &buckets,
        std::vector<int> &bucket_ptr,
        std::vector<int> &bucket_nodes,
        int &max_bucket_size)
    {
        bucket_ptr.clear();
        bucket_nodes.clear();
        bucket_ptr.reserve(buckets.size() + 1);
        bucket_ptr.push_back(0);
        max_bucket_size = 0;

        for (const auto &bucket : buckets)
        {
            if (bucket_nodes.size() + bucket.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("flatten_supernode_solve_buckets: too many bucket entries");

            max_bucket_size = std::max(max_bucket_size, static_cast<int>(bucket.size()));
            bucket_nodes.insert(bucket_nodes.end(), bucket.begin(), bucket.end());
            bucket_ptr.push_back(static_cast<int>(bucket_nodes.size()));
        }
    }

    template <typename SolveT>
    struct PcgCoreSupernodeRuntime
    {
        std::vector<double> h_r;
        std::vector<double> h_z;
        DeviceBuffer<SolveT> d_r_work_buf;
        DeviceBuffer<SolveT> d_w_work_buf;
        DeviceBuffer<SolveT> d_z_work_buf;
        DeviceBuffer<int> d_status;
        int h_status = 0;
        double solve_total_ms = 0.0;
        int solve_timed_iters = 0;
        CudaEvent solve_start;
        CudaEvent solve_stop;
    };

    template <typename SolveT>
    struct PcgCoreLevelRuntime
    {
        SolveT *d_r_work = nullptr;
        SolveT *d_w_work = nullptr;
        SolveT *d_z_work = nullptr;
        DeviceBuffer<SolveT> d_r_work_buf;
        DeviceBuffer<SolveT> d_w_work_buf;
        DeviceBuffer<SolveT> d_z_work_buf;
        double sptrsv_total_ms = 0.0;
        int sptrsv_timed_iters = 0;
        CudaEvent sptrsv_start;
        CudaEvent sptrsv_stop;
    };

    static void build_full_precond(
        FullPrecondContext &ctx,
        int n,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<double> &h_valL)
    {
        const int nnzL = static_cast<int>(h_valL.size());
        ctx.d_csrRowPtrM.alloc(n + 1);
        ctx.d_csrColIndM.alloc(nnzL);
        ctx.d_valM.alloc(nnzL);

        CUDA_CHECK(cudaMemcpy(ctx.d_csrRowPtrM.get(), h_csrRowPtrL.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_csrColIndM.get(), h_csrColIndL.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_valM.get(), h_valL.data(), nnzL * sizeof(double), cudaMemcpyHostToDevice));
        ctx.spMatM.create(n, n, nnzL, ctx.d_csrRowPtrM.get(), ctx.d_csrColIndM.get(), ctx.d_valM.get());
    }

    template <typename T_L, typename SolveT>
    static void build_level_precond(
        LevelPrecondContext<SolveT> &ctx,
        int n,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<T_L> &h_valL,
        cudaStream_t stream)
    {
        const int nnzL = static_cast<int>(h_valL.size());
        std::vector<SolveT> h_valL_solve(nnzL);
        for (int i = 0; i < nnzL; ++i)
            h_valL_solve[i] = static_cast<SolveT>(h_valL[i]);

        std::vector<int> h_csrRowPtrLt, h_csrColIndLt;
        std::vector<SolveT> h_valLt_solve;
        build_csr_trans<SolveT>(n, h_csrRowPtrL, h_csrColIndL, h_valL_solve, h_csrRowPtrLt, h_csrColIndLt, h_valLt_solve);

        const auto levelsets_L = build_level_sets_csr_diag_last(n, h_csrRowPtrL, h_csrColIndL, false);
        const auto levelsets_Lt = build_level_sets_csr_diag_last(n, h_csrRowPtrLt, h_csrColIndLt, true);

        ctx.sptrsv_plan_L.init(levelsets_L, stream);
        ctx.sptrsv_plan_Lt.init(levelsets_Lt, stream);

        ctx.d_csrRowPtrL.alloc(n + 1);
        ctx.d_csrColIndL.alloc(nnzL);
        ctx.d_valL.alloc(nnzL);
        CUDA_CHECK(cudaMemcpy(ctx.d_csrRowPtrL.get(), h_csrRowPtrL.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_csrColIndL.get(), h_csrColIndL.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_valL.get(), h_valL_solve.data(), nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));

        ctx.d_csrRowPtrLt.alloc(n + 1);
        ctx.d_csrColIndLt.alloc(nnzL);
        ctx.d_valLt.alloc(nnzL);
        CUDA_CHECK(cudaMemcpy(ctx.d_csrRowPtrLt.get(), h_csrRowPtrLt.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_csrColIndLt.get(), h_csrColIndLt.data(), nnzL * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_valLt.get(), h_valLt_solve.data(), nnzL * sizeof(SolveT), cudaMemcpyHostToDevice));
    }

    template <typename T_L, typename SolveT>
    static void build_supernode_precond(
        SupernodePrecondContext<SolveT> &ctx,
        int n,
        const std::vector<int> &h_csrRowPtrL,
        const std::vector<int> &h_csrColIndL,
        const std::vector<T_L> &h_valL)
    {
        ctx.sym = ichol::supernodal::build_relaxed_super_sym_from_csr_l(
            n, h_csrRowPtrL, h_csrColIndL,
            kFixedRelaxedSupernodeMaxWidth,
            kFixedRelaxedSupernodeExtra);
        ctx.super = ctx.sym.super;
        ctx.metrics = analyze_supernode_candidates_from_sym_l(n, h_csrRowPtrL, h_csrColIndL, ctx.sym);
        ctx.metrics.relaxed_extra = kFixedRelaxedSupernodeExtra;
        ctx.metrics.relaxed_max_width = kFixedRelaxedSupernodeMaxWidth;
        ctx.plan = ichol::symbolic::ll_plan_from_sym(ctx.sym, n);
        ctx.solve_buckets = ichol::supernodal::build_forward_solve_buckets_from_sym(ctx.sym);
        flatten_supernode_solve_buckets(
            ctx.solve_buckets,
            ctx.solve_bucket_ptr,
            ctx.solve_bucket_nodes,
            ctx.solve_max_bucket_size);
        ctx.solve_num_levels = static_cast<int>(ctx.solve_buckets.size());
        ctx.packed_values_fp64 = ichol::supernodal::pack_supernode_values_from_csr_l<double>(
            n, h_csrRowPtrL, h_csrColIndL, h_valL, ctx.sym);
        ctx.packed_values = ichol::supernodal::pack_supernode_values_from_csr_l<SolveT>(
            n, h_csrRowPtrL, h_csrColIndL, h_valL, ctx.sym);

        std::vector<int> dummy_vals(h_csrColIndL.size(), 0);
        std::vector<int> dummy_vals_t;
        build_csr_trans<int>(n, h_csrRowPtrL, h_csrColIndL, dummy_vals, ctx.lt_row_ptr, ctx.lt_col_ind, dummy_vals_t);

        ctx.d_super.alloc(ctx.sym.super.size());
        ctx.d_pi.alloc(ctx.sym.pi.size());
        ctx.d_px.alloc(ctx.sym.px.size());
        if (!ctx.sym.s.empty())
            ctx.d_s.alloc(ctx.sym.s.size());
        if (!ctx.packed_values.empty())
            ctx.d_packed_values.alloc(ctx.packed_values.size());
        if (!ctx.solve_bucket_ptr.empty())
            ctx.d_solve_bucket_ptr.alloc(ctx.solve_bucket_ptr.size());
        if (!ctx.solve_bucket_nodes.empty())
            ctx.d_solve_bucket_nodes.alloc(ctx.solve_bucket_nodes.size());
        ctx.d_lt_row_ptr.alloc(ctx.lt_row_ptr.size());
        if (!ctx.lt_col_ind.empty())
            ctx.d_lt_col_ind.alloc(ctx.lt_col_ind.size());

        CUDA_CHECK(cudaMemcpy(ctx.d_super.get(), ctx.sym.super.data(), ctx.sym.super.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_pi.get(), ctx.sym.pi.data(), ctx.sym.pi.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_px.get(), ctx.sym.px.data(), ctx.sym.px.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (!ctx.sym.s.empty())
            CUDA_CHECK(cudaMemcpy(ctx.d_s.get(), ctx.sym.s.data(), ctx.sym.s.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (!ctx.packed_values.empty())
            CUDA_CHECK(cudaMemcpy(ctx.d_packed_values.get(), ctx.packed_values.data(), ctx.packed_values.size() * sizeof(SolveT), cudaMemcpyHostToDevice));
        if (!ctx.solve_bucket_ptr.empty())
            CUDA_CHECK(cudaMemcpy(ctx.d_solve_bucket_ptr.get(), ctx.solve_bucket_ptr.data(), ctx.solve_bucket_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (!ctx.solve_bucket_nodes.empty())
            CUDA_CHECK(cudaMemcpy(ctx.d_solve_bucket_nodes.get(), ctx.solve_bucket_nodes.data(), ctx.solve_bucket_nodes.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_lt_row_ptr.get(), ctx.lt_row_ptr.data(), ctx.lt_row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (!ctx.lt_col_ind.empty())
            CUDA_CHECK(cudaMemcpy(ctx.d_lt_col_ind.get(), ctx.lt_col_ind.data(), ctx.lt_col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));

        ctx.solve_implemented = true;
    }

    template <typename SolveT>
    static void print_level_runtime_metrics(
        const LevelPrecondContext<SolveT> &level_precond,
        const PcgCoreLevelRuntime<SolveT> &runtime,
        const ichol::solver::PCGResult &result,
        bool sptrsv_debug)
    {
        if (sptrsv_debug)
        {
            std::cout << "[debug--level]:\n";
            std::cout << "L: num_levels=" << level_precond.sptrsv_plan_L.ls.num_levels
                      << ", max_level_size=" << level_precond.sptrsv_plan_L.ls.max_level_size << "\n";
            std::cout << "Lt: num_levels=" << level_precond.sptrsv_plan_Lt.ls.num_levels
                      << ", max_level_size=" << level_precond.sptrsv_plan_Lt.ls.max_level_size << "\n";

            std::cout << "[debug--PCG]:\n";
            std::cout << "PCG iterations=" << result.iterations
                      << ", finalRes=" << result.finalRes << "\n";

            std::cout << "[debug--SpTRSV]:\n";
            std::cout << "sptrsv_total_ms=" << runtime.sptrsv_total_ms
                      << ", sptrsv_timed_iters=" << runtime.sptrsv_timed_iters;

            if (runtime.sptrsv_timed_iters > 0)
                std::cout << ", avg_sptrsv_ms=" << (runtime.sptrsv_total_ms / runtime.sptrsv_timed_iters);

            std::cout << "\n";
        }
    }

    template <typename SolveT>
    static void print_supernode_runtime_metrics(
        const SupernodePrecondContext<SolveT> &supernode_precond,
        const PcgCoreSupernodeRuntime<SolveT> &runtime,
        const ichol::solver::PCGResult &result,
        bool sptrsv_debug)
    {
        if (sptrsv_debug)
        {
            std::cout << "[debug--supernode-runtime]:\n";
            std::cout << "PCG iterations=" << result.iterations
                      << ", finalRes=" << result.finalRes << "\n";
            std::cout << "num_supernodes=" << supernode_precond.metrics.num_supernodes
                      << ", packed_dense_slots=" << supernode_precond.packed_values.size()
                      << ", solve_dependency_levels=" << supernode_precond.solve_num_levels
                      << ", max_solve_bucket_size=" << supernode_precond.solve_max_bucket_size << "\n";
            std::cout << "supernode_solve_total_ms=" << runtime.solve_total_ms
                      << ", supernode_solve_timed_iters=" << runtime.solve_timed_iters;
            if (runtime.solve_timed_iters > 0)
                std::cout << ", avg_supernode_solve_ms=" << (runtime.solve_total_ms / runtime.solve_timed_iters);
            std::cout << "\n";

            std::cout << "[debug--SpTRSV]:\n";
            std::cout << "sptrsv_total_ms=" << runtime.solve_total_ms
                      << ", sptrsv_timed_iters=" << runtime.solve_timed_iters;
            if (runtime.solve_timed_iters > 0)
                std::cout << ", avg_sptrsv_ms=" << (runtime.solve_total_ms / runtime.solve_timed_iters);
            std::cout << "\n";
        }
    }

    template <typename T_L>
    static ichol::solver::PCGResult pcg_core(
        const std::vector<int> &h_csrRowPtrA,
        const std::vector<int> &h_csrColIndA,
        const std::vector<double> &h_valA,
        const std::vector<double> &h_b,
        std::vector<double> &h_x,
        const ichol::solver::PCGParams &params,
        FullPrecondContext *full_precond,
        LevelPrecondContext<SolveType<T_L>> *level_precond,
        SupernodePrecondContext<SolveType<T_L>> *supernode_precond)
    {
        constexpr bool L_is_fp64 = std::is_same<T_L, double>::value;
        constexpr bool L_is_fp32 = std::is_same<T_L, float>::value;
        using SolveT = std::conditional_t<L_is_fp64, double, std::conditional_t<L_is_fp32, float, __half>>;
        const bool sptrsv_debug = true;

        CusparseHandle cusparseHandle;
        CublasHandle cublasHandle;

        const int n = static_cast<int>(h_csrRowPtrA.size()) - 1;
        const int nnzA = static_cast<int>(h_valA.size());
        const bool A_is_full = csr_has_upper_triangle_entries(n, h_csrRowPtrA, h_csrColIndA);
        const bool use_custom_precond = (params.custom_precond != nullptr && params.custom_precond->apply != nullptr);

        cudaStream_t stream = 0;
        ichol::solver::PCGResult result{};

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
                    h_diagA[i] = h_valA[h_csrRowPtrA[i + 1] - 1];
            }
            CUDA_CHECK(cudaMemcpy(d_diagA.get(), h_diagA.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMemcpy(d_csrRowPtrA.get(), h_csrRowPtrA.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csrColIndA.get(), h_csrColIndA.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valA.get(), h_valA.data(), nnzA * sizeof(double), cudaMemcpyHostToDevice));

        CusparseSpMat spMatA;
        spMatA.create(n, n, nnzA, d_csrRowPtrA.get(), d_csrColIndA.get(), d_valA.get());

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

        PcgCoreLevelRuntime<SolveT> level_runtime;
        PcgCoreSupernodeRuntime<SolveT> supernode_runtime;
        if (!use_custom_precond && !params.precond_full && level_precond != nullptr)
        {
            level_runtime.d_r_work_buf.alloc(n);
            level_runtime.d_w_work_buf.alloc(n);
            level_runtime.d_z_work_buf.alloc(n);
            level_runtime.d_r_work = level_runtime.d_r_work_buf.get();
            level_runtime.d_w_work = level_runtime.d_w_work_buf.get();
            level_runtime.d_z_work = level_runtime.d_z_work_buf.get();
        }
        if (!use_custom_precond && !params.precond_full && supernode_precond != nullptr)
        {
            supernode_runtime.h_r.resize((size_t)n);
            supernode_runtime.h_z.resize((size_t)n);
            if constexpr (std::is_same_v<SolveT, double> || std::is_same_v<SolveT, float>)
            {
                supernode_runtime.d_r_work_buf.alloc(n);
                supernode_runtime.d_w_work_buf.alloc(n);
                supernode_runtime.d_z_work_buf.alloc(n);
                supernode_runtime.d_status.alloc(1);
            }
        }

        CUDA_CHECK(cudaMemcpy(d_r.get(), d_b.get(), n * sizeof(double), cudaMemcpyDeviceToDevice));
        double rho = 0.0, bnorm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_b.get(), 1, &bnorm));
        if (bnorm == 0.0)
            bnorm = 1.0;

        double last_res_norm = bnorm;
        int last_completed_iter = 0;

        for (int k = 1; k <= params.maxits; k++)
        {
            if (use_custom_precond)
            {
                params.custom_precond->apply(params.custom_precond->ctx, d_r.get(), d_z.get(), n, stream);
            }
            else if (params.precond_full && full_precond != nullptr)
            {
                double a1 = 1.0, b0 = 0.0;
                CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &a1, full_precond->spMatM, vecR_dev, &b0, vecZ_dev, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf.get()));
            }
            else
            {
                if (level_precond != nullptr)
                {
                    cast_vec<SolveT, double><<<(n + 255) / 256, 256, 0, stream>>>(n, d_r.get(), level_runtime.d_r_work);
                    CUDA_CHECK(cudaEventRecord(level_runtime.sptrsv_start, stream));

                    level_precond->sptrsv_plan_L.template solve<int, SolveT>(
                        n, level_precond->d_csrRowPtrL.get(), level_precond->d_csrColIndL.get(), level_precond->d_valL.get(),
                        level_runtime.d_r_work, level_runtime.d_w_work, false, stream, sptrsv_debug);
                    level_precond->sptrsv_plan_Lt.template solve<int, SolveT>(
                        n, level_precond->d_csrRowPtrLt.get(), level_precond->d_csrColIndLt.get(), level_precond->d_valLt.get(),
                        level_runtime.d_w_work, level_runtime.d_z_work, false, stream, sptrsv_debug);

                    CUDA_CHECK(cudaEventRecord(level_runtime.sptrsv_stop, stream));
                    cast_vec<double, SolveT><<<(n + 255) / 256, 256, 0, stream>>>(n, level_runtime.d_z_work, d_z.get());

                    float iter_ms = 0.0f;
                    CUDA_CHECK(cudaEventSynchronize(level_runtime.sptrsv_stop));
                    CUDA_CHECK(cudaEventElapsedTime(&iter_ms, level_runtime.sptrsv_start, level_runtime.sptrsv_stop));
                    level_runtime.sptrsv_total_ms += iter_ms;
                    level_runtime.sptrsv_timed_iters++;
                }
                else if (supernode_precond != nullptr)
                {
                    if constexpr (std::is_same_v<SolveT, double> || std::is_same_v<SolveT, float>)
                    {
                        cast_vec<SolveT, double><<<(n + 255) / 256, 256, 0, stream>>>(
                            n, d_r.get(), supernode_runtime.d_r_work_buf.get());

                        CUDA_CHECK(cudaEventRecord(supernode_runtime.solve_start, stream));
                        const bool forward_persistent = ichol::supernodal::cuda_reference::solve_lower_device_persistent<SolveT>(
                            n,
                            supernode_precond->solve_num_levels,
                            supernode_precond->solve_max_bucket_size,
                            supernode_precond->d_solve_bucket_ptr.get(),
                            supernode_precond->d_solve_bucket_nodes.get(),
                            supernode_precond->d_super.get(),
                            supernode_precond->d_pi.get(),
                            supernode_precond->d_px.get(),
                            supernode_precond->d_s.get(),
                            supernode_precond->d_packed_values.get(),
                            supernode_runtime.d_r_work_buf.get(),
                            supernode_runtime.d_r_work_buf.get(),
                            supernode_runtime.d_w_work_buf.get(),
                            supernode_runtime.d_status.get(),
                            stream,
                            true);
                        if (!forward_persistent)
                        {
                            ichol::supernodal::cuda_reference::solve_lower_device<SolveT>(
                                n,
                                supernode_precond->solve_bucket_ptr,
                                supernode_precond->d_solve_bucket_nodes.get(),
                                supernode_precond->d_super.get(),
                                supernode_precond->d_pi.get(),
                                supernode_precond->d_px.get(),
                                supernode_precond->d_s.get(),
                                supernode_precond->d_packed_values.get(),
                                supernode_runtime.d_r_work_buf.get(),
                                supernode_runtime.d_r_work_buf.get(),
                                supernode_runtime.d_w_work_buf.get(),
                                supernode_runtime.d_status.get(),
                                stream,
                                true);
                        }

                        const bool backward_persistent = ichol::supernodal::cuda_reference::solve_lower_transpose_device_persistent<SolveT>(
                            n,
                            supernode_precond->solve_num_levels,
                            supernode_precond->solve_max_bucket_size,
                            supernode_precond->d_solve_bucket_ptr.get(),
                            supernode_precond->d_solve_bucket_nodes.get(),
                            supernode_precond->d_super.get(),
                            supernode_precond->d_pi.get(),
                            supernode_precond->d_px.get(),
                            supernode_precond->d_s.get(),
                            supernode_precond->d_packed_values.get(),
                            supernode_runtime.d_w_work_buf.get(),
                            supernode_runtime.d_z_work_buf.get(),
                            supernode_runtime.d_status.get(),
                            stream,
                            false);
                        if (!backward_persistent)
                        {
                            ichol::supernodal::cuda_reference::solve_lower_transpose_device<SolveT>(
                                n,
                                supernode_precond->solve_bucket_ptr,
                                supernode_precond->d_solve_bucket_nodes.get(),
                                supernode_precond->d_super.get(),
                                supernode_precond->d_pi.get(),
                                supernode_precond->d_px.get(),
                                supernode_precond->d_s.get(),
                                supernode_precond->d_packed_values.get(),
                                supernode_runtime.d_w_work_buf.get(),
                                supernode_runtime.d_z_work_buf.get(),
                                supernode_runtime.d_status.get(),
                                stream,
                                false);
                        }
                        CUDA_CHECK(cudaMemcpyAsync(&supernode_runtime.h_status,
                                                   supernode_runtime.d_status.get(),
                                                   sizeof(int),
                                                   cudaMemcpyDeviceToHost,
                                                   stream));
                        CUDA_CHECK(cudaEventRecord(supernode_runtime.solve_stop, stream));
                        CUDA_CHECK(cudaEventSynchronize(supernode_runtime.solve_stop));
                        if (supernode_runtime.h_status != 0)
                            throw std::runtime_error("pcg_core: GPU supernode solve encountered a zero diagonal");

                        float iter_ms = 0.0f;
                        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, supernode_runtime.solve_start, supernode_runtime.solve_stop));
                        supernode_runtime.solve_total_ms += iter_ms;
                        supernode_runtime.solve_timed_iters++;

                        cast_vec<double, SolveT><<<(n + 255) / 256, 256, 0, stream>>>(
                            n, supernode_runtime.d_z_work_buf.get(), d_z.get());
                    }
                    else
                    {
                        const auto t0 = std::chrono::steady_clock::now();
                        CUDA_CHECK(cudaMemcpy(supernode_runtime.h_r.data(), d_r.get(), n * sizeof(double), cudaMemcpyDeviceToHost));
                        auto h_w_fp64 = ichol::supernodal::solve_lower_supernodal_bucketed(
                            supernode_precond->plan, supernode_precond->packed_values_fp64, supernode_runtime.h_r,
                            supernode_precond->solve_buckets);
                        supernode_runtime.h_z = ichol::supernodal::solve_lower_transpose_supernodal_bucketed(
                            supernode_precond->plan, supernode_precond->packed_values_fp64, h_w_fp64,
                            supernode_precond->solve_buckets);

                        CUDA_CHECK(cudaMemcpy(d_z.get(), supernode_runtime.h_z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
                        const auto t1 = std::chrono::steady_clock::now();
                        const double iter_ms =
                            std::chrono::duration<double, std::milli>(t1 - t0).count();
                        supernode_runtime.solve_total_ms += iter_ms;
                        supernode_runtime.solve_timed_iters++;
                    }
                }
                else
                {
                    throw std::runtime_error("pcg_core: no valid preconditioner route");
                }
            }


            double rhoOld = rho;
            CUBLAS_CHECK(cublasDdot(cublasHandle, n, d_r.get(), 1, d_z.get(), 1, &rho));

            if (k == 1)
                CUDA_CHECK(cudaMemcpy(d_p.get(), d_z.get(), n * sizeof(double), cudaMemcpyDeviceToDevice));
            else
            {
                double beta = rho / rhoOld;
                CUBLAS_CHECK(cublasDscal(cublasHandle, n, &beta, d_p.get(), 1));
                double a1 = 1.0;
                CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &a1, d_z.get(), 1, d_p.get(), 1));
            }

            double alpha1 = 1.0, beta0 = 0.0;
            CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha1, spMatA, vecP_dev, &beta0, vecQ_dev, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf.get()));

            if (!A_is_full)
            {
                double beta1 = 1.0;
                CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &alpha1, spMatA, vecP_dev, &beta1, vecQ_dev, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmvBuf.get()));
                diag_sub_from_diag<<<(n + 255) / 256, 256, 0, stream>>>(n, d_diagA.get(), d_p.get(), d_q.get());
            }

            double denom = 0.0;
            CUBLAS_CHECK(cublasDdot(cublasHandle, n, d_p.get(), 1, d_q.get(), 1, &denom));
            if (!std::isfinite(rho) || !std::isfinite(denom) || denom == 0.0)
            {
                result.iterations = last_completed_iter;
                result.finalRes = std::numeric_limits<double>::infinity();
                break;
            }
            double alpha = rho / denom;

            CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &alpha, d_p.get(), 1, d_x.get(), 1));
            double negAlpha = -alpha;
            CUBLAS_CHECK(cublasDaxpy(cublasHandle, n, &negAlpha, d_q.get(), 1, d_r.get(), 1));

            double res_norm = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_r.get(), 1, &res_norm));
            last_res_norm = res_norm;
            last_completed_iter = k;
            if (res_norm <= params.tol * bnorm)
            {
                result.iterations = k;
                result.finalRes = res_norm;
                break;
            }
        }

        if (result.iterations == 0 && !std::isinf(result.finalRes))
        {
            result.iterations = last_completed_iter;
            result.finalRes = last_res_norm;
        }

        if (level_precond != nullptr)
            print_level_runtime_metrics(*level_precond, level_runtime, result, sptrsv_debug);
        if (supernode_precond != nullptr)
            print_supernode_runtime_metrics(*supernode_precond, supernode_runtime, result, sptrsv_debug);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_x.resize(n);
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x.get(), n * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }
} // namespace

namespace ichol::solver
{
    template <typename T_L>
    PCGResult pcg_level(
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
        (void)h_D;
        if (params.custom_precond != nullptr && params.custom_precond->apply != nullptr)
            return pcg_core<T_L>(h_csrRowPtrA, h_csrColIndA, h_valA, h_b, h_x, params, nullptr, nullptr, nullptr);

        if (params.precond_full)
        {
            std::vector<double> h_valL_fp64(h_valL.size());
            for (size_t i = 0; i < h_valL.size(); ++i)
                h_valL_fp64[i] = static_cast<double>(h_valL[i]);
            FullPrecondContext full_precond;
            build_full_precond(full_precond, (int)h_csrRowPtrL.size() - 1, h_csrRowPtrL, h_csrColIndL, h_valL_fp64);
            return pcg_core<T_L>(h_csrRowPtrA, h_csrColIndA, h_valA, h_b, h_x, params, &full_precond, nullptr, nullptr);
        }

        LevelPrecondContext<SolveType<T_L>> level_precond;
        build_level_precond<T_L, SolveType<T_L>>(level_precond, (int)h_csrRowPtrL.size() - 1, h_csrRowPtrL, h_csrColIndL, h_valL, 0);
        return pcg_core<T_L>(h_csrRowPtrA, h_csrColIndA, h_valA, h_b, h_x, params, nullptr, &level_precond, nullptr);
    }

    template <typename T_L>
    PCGResult pcg_super(
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
        (void)h_D;
        SupernodePrecondContext<SolveType<T_L>> supernode_precond;
        build_supernode_precond<T_L, SolveType<T_L>>(supernode_precond, (int)h_csrRowPtrL.size() - 1, h_csrRowPtrL, h_csrColIndL, h_valL);
        std::cout << "[debug--supernode]:\n"
                  << " amalgamation=relaxed_" << supernode_precond.metrics.relaxed_extra
                  << ", max_width=" << supernode_precond.metrics.relaxed_max_width
                  << ","
                  << " num_supernodes=" << supernode_precond.metrics.num_supernodes
                  << ", avg_supernode_width=" << supernode_precond.metrics.avg_width
                  << ", supernode_block_density=" << supernode_precond.metrics.block_density
                  << ", level_compression_rate=" << supernode_precond.metrics.level_compression_rate
                  << ", scalar_levels=" << supernode_precond.metrics.scalar_num_levels
                  << ", supernode_levels=" << supernode_precond.metrics.supernode_num_levels
                  << ", solve_dependency_levels=" << supernode_precond.solve_num_levels
                  << ", max_solve_bucket_size=" << supernode_precond.solve_max_bucket_size
                  << ", packed_dense_slots=" << supernode_precond.packed_values.size()
                  << "\n";

        return pcg_core<T_L>(h_csrRowPtrA, h_csrColIndA, h_valA, h_b, h_x, params, nullptr, nullptr, &supernode_precond);
    }

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
        if (params.custom_precond != nullptr || params.precond_full)
            return pcg_level<T_L>(h_csrRowPtrA, h_csrColIndA, h_valA, h_csrRowPtrL, h_csrColIndL, h_valL, h_b, h_x, h_D, params);

        switch (params.factorized_precond_policy)
        {
        case FactorizedPrecondSolvePolicy::LevelScheduling:
            return pcg_level<T_L>(h_csrRowPtrA, h_csrColIndA, h_valA, h_csrRowPtrL, h_csrColIndL, h_valL, h_b, h_x, h_D, params);
        case FactorizedPrecondSolvePolicy::Supernode:
            return pcg_super<T_L>(h_csrRowPtrA, h_csrColIndA, h_valA, h_csrRowPtrL, h_csrColIndL, h_valL, h_b, h_x, h_D, params);
        default:
            throw std::runtime_error("pcg: unknown factorized preconditioner policy");
        }
    }

    template PCGResult pcg_level<double>(
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

    template PCGResult pcg_level<float>(
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

    template PCGResult pcg_level<half_float::half>(
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

    template PCGResult pcg_super<double>(
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

    template PCGResult pcg_super<float>(
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

    template PCGResult pcg_super<half_float::half>(
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
