#include <mkl.h>
#include <mkl_lapacke.h>
#include <mkl_spblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ichol/pcg.hpp"
#include "ichol/preconditioner.hpp"
#include "ichol/subdomain_preconditioner_mkl.hpp"
#include "backends/CUDA/mpcg_debug.hpp"

namespace
{
using Clock = std::chrono::steady_clock;

#define MKL_SPARSE_CHECK(call)                                                             \
    do                                                                                     \
    {                                                                                      \
        sparse_status_t _status = (call);                                                  \
        if (_status != SPARSE_STATUS_SUCCESS)                                              \
            throw std::runtime_error(std::string("MKL sparse error at line ") +            \
                                     std::to_string(__LINE__) + " status=" +               \
                                     std::to_string(static_cast<int>(_status)));            \
    } while (0)

#define MKL_LAPACK_CHECK(info, what)                                                       \
    do                                                                                     \
    {                                                                                      \
        if ((info) < 0)                                                                    \
            throw std::runtime_error(std::string(what) + " invalid argument " +            \
                                     std::to_string(-(info)));                             \
    } while (0)

struct ProfilePhase
{
    double total_ms = 0.0;
};

struct PrecisionMap
{
    ichol::solver::ComputePrecision io_prec;
    ichol::solver::ComputePrecision acc_prec;
    bool use_fp64 = true;
};

struct SpmmMap
{
    ichol::solver::ComputePrecision io_prec;
    ichol::solver::ComputePrecision acc_prec;
    bool use_fp64 = true;
};

struct StorageMap
{
    ichol::solver::ComputePrecision storage_prec;
    bool use_fp64 = true;
};

static double elapsed_ms(const Clock::time_point &start, const Clock::time_point &stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

static ichol::solver::ComputePrecision normalize_dense_precision(ichol::solver::ComputePrecision prec)
{
    using Prec = ichol::solver::ComputePrecision;
    switch (prec)
    {
    case Prec::FP64:
        return Prec::FP64;
    case Prec::FP32:
    case Prec::TF32:
    case Prec::FP16:
    case Prec::BF16:
        // CPU oneMKL path keeps non-fp64 dense work in fp32.
        return Prec::FP32;
    default:
        throw std::runtime_error("mpcg_mkl: unsupported precision mode");
    }
}

static ichol::solver::ComputePrecision normalize_precond_precision(ichol::solver::ComputePrecision prec)
{
    using Prec = ichol::solver::ComputePrecision;
    switch (prec)
    {
    case Prec::FP64:
        return Prec::FP64;
    case Prec::FP32:
    case Prec::TF32:
        return Prec::FP32;
    default:
        throw std::runtime_error("mpcg_mkl: CPU preconditioner path supports FP64 and FP32 only");
    }
}

static ichol::solver::ComputePrecision resolve_accum_precision(
    ichol::solver::ComputePrecision input_prec,
    ichol::solver::ComputePrecision requested_acc)
{
    using Prec = ichol::solver::ComputePrecision;
    const Prec io_prec = normalize_dense_precision(input_prec);
    const Prec acc_prec = normalize_dense_precision(requested_acc);
    if (io_prec == Prec::FP64)
        return Prec::FP64;
    (void)acc_prec;
    return Prec::FP32;
}

static PrecisionMap get_precision_map(ichol::solver::ComputePrecision prec, ichol::solver::ComputePrecision acc)
{
    PrecisionMap m{};
    m.io_prec = normalize_dense_precision(prec);
    m.acc_prec = resolve_accum_precision(prec, acc);
    m.use_fp64 = (m.acc_prec == ichol::solver::ComputePrecision::FP64);
    return m;
}

static SpmmMap get_spmm_map(ichol::solver::ComputePrecision prec, ichol::solver::ComputePrecision acc)
{
    SpmmMap m{};
    m.io_prec = normalize_dense_precision(prec);
    m.acc_prec = resolve_accum_precision(prec, acc);
    m.use_fp64 = (m.io_prec == ichol::solver::ComputePrecision::FP64);
    return m;
}

static StorageMap get_storage_map(ichol::solver::ComputePrecision prec)
{
    StorageMap m{};
    m.storage_prec = normalize_dense_precision(prec);
    m.use_fp64 = (m.storage_prec == ichol::solver::ComputePrecision::FP64);
    return m;
}

static StorageMap get_precond_map(ichol::solver::ComputePrecision prec)
{
    StorageMap m{};
    m.storage_prec = normalize_precond_precision(prec);
    m.use_fp64 = (m.storage_prec == ichol::solver::ComputePrecision::FP64);
    return m;
}

static double get_safe_rcond(ichol::solver::ComputePrecision prec, double base_rcond)
{
    using Prec = ichol::solver::ComputePrecision;
    if (prec == Prec::FP64)
        return base_rcond;
    return std::max(base_rcond, 1e-6);
}

static int flatten_local_3d(int x, int y, int z, int w, int h)
{
    return x + y * w + z * (w * h);
}

static void unflatten_global_3d(int gi, int gw, int gh, int &x, int &y, int &z)
{
    const int plane = gw * gh;
    z = gi / plane;
    const int rem = gi - z * plane;
    y = rem / gw;
    x = rem - y * gw;
}

static int local_from_global_host(
    int gj,
    const ichol::precond::GridShape &global,
    const ichol::precond::SubdomainRegion &reg,
    int lw,
    int lh)
{
    int x = 0;
    int y = 0;
    int z = 0;
    unflatten_global_3d(gj, global.w, global.h, x, y, z);
    if (x < reg.x0 || x >= reg.x1 || y < reg.y0 || y >= reg.y1 || z < reg.z0 || z >= reg.z1)
        return -1;
    return flatten_local_3d(x - reg.x0, y - reg.y0, z - reg.z0, lw, lh);
}

static ichol::matrix::CsrMatrix<double> extract_lower_subdomain_csr(
    const ichol::matrix::CsrMatrix<double> &A,
    const ichol::precond::GridShape &global,
    const ichol::precond::SubdomainRegion &reg)
{
    const int lw = reg.x1 - reg.x0;
    const int lh = reg.y1 - reg.y0;
    const int ld = reg.z1 - reg.z0;
    const int nsub = lw * lh * ld;

    ichol::matrix::CsrMatrix<double> sub;
    sub.num_rows = nsub;
    sub.num_cols = nsub;
    sub.row_ptr.resize(static_cast<size_t>(nsub) + 1, 0);

    std::vector<int> cols;
    std::vector<double> vals;
    for (int li = 0; li < nsub; ++li)
    {
        const int plane = lw * lh;
        const int lz = li / plane;
        const int rem = li - lz * plane;
        const int ly = rem / lw;
        const int lx = rem - ly * lw;
        const int gi = (reg.x0 + lx) + (reg.y0 + ly) * global.w + (reg.z0 + lz) * (global.w * global.h);

        std::vector<std::pair<int, double>> row_entries;
        row_entries.reserve(static_cast<size_t>(A.row_ptr[gi + 1] - A.row_ptr[gi]));
        for (int kk = A.row_ptr[gi]; kk < A.row_ptr[gi + 1]; ++kk)
        {
            const int lj = local_from_global_host(A.col_ind[kk], global, reg, lw, lh);
            if (lj < 0 || lj > li)
                continue;
            row_entries.push_back({lj, A.values[kk]});
        }

        std::sort(row_entries.begin(), row_entries.end(),
                  [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        int diag_pos = -1;
        for (int i = 0; i < static_cast<int>(row_entries.size()); ++i)
        {
            if (row_entries[static_cast<size_t>(i)].first == li)
            {
                diag_pos = i;
                break;
            }
        }
        if (diag_pos < 0)
            throw std::runtime_error("extract_lower_subdomain_csr: missing diagonal entry");

        for (int i = 0; i < static_cast<int>(row_entries.size()); ++i)
        {
            if (i == diag_pos)
                continue;
            cols.push_back(row_entries[static_cast<size_t>(i)].first);
            vals.push_back(row_entries[static_cast<size_t>(i)].second);
        }
        cols.push_back(li);
        vals.push_back(row_entries[static_cast<size_t>(diag_pos)].second);
        sub.row_ptr[static_cast<size_t>(li) + 1] = static_cast<int>(cols.size());
    }

    sub.col_ind = std::move(cols);
    sub.values = std::move(vals);
    sub.nnz = static_cast<int>(sub.values.size());
    return sub;
}

static void build_csr_transpose(
    int n,
    const std::vector<int> &row_ptr,
    const std::vector<int> &col_ind,
    const std::vector<double> &val,
    std::vector<int> &row_ptr_t,
    std::vector<int> &col_ind_t,
    std::vector<double> &val_t)
{
    const int nnz = static_cast<int>(val.size());
    row_ptr_t.assign(static_cast<size_t>(n) + 1, 0);
    col_ind_t.assign(static_cast<size_t>(nnz), 0);
    val_t.assign(static_cast<size_t>(nnz), 0.0);

    for (int i = 0; i < nnz; ++i)
        ++row_ptr_t[static_cast<size_t>(col_ind[i]) + 1];
    for (int i = 0; i < n; ++i)
        row_ptr_t[static_cast<size_t>(i) + 1] += row_ptr_t[static_cast<size_t>(i)];

    std::vector<int> next = row_ptr_t;
    for (int i = 0; i < n; ++i)
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
        {
            const int j = col_ind[p];
            const int dst = next[static_cast<size_t>(j)]++;
            col_ind_t[static_cast<size_t>(dst)] = i;
            val_t[static_cast<size_t>(dst)] = val[static_cast<size_t>(p)];
        }
    }
}

static std::vector<int> build_subdomain_gidx_host(
    int lw,
    int lh,
    int ld,
    int gw,
    int gh,
    int x0,
    int y0,
    int z0)
{
    const int nsub = lw * lh * ld;
    std::vector<int> gidx(static_cast<size_t>(nsub), 0);
    for (int li = 0; li < nsub; ++li)
    {
        const int plane = lw * lh;
        const int lz = li / plane;
        const int rem = li - lz * plane;
        const int ly = rem / lw;
        const int lx = rem - ly * lw;
        const int gx = x0 + lx;
        const int gy = y0 + ly;
        const int gz = z0 + lz;
        gidx[static_cast<size_t>(li)] = gx + gy * gw + gz * (gw * gh);
    }
    return gidx;
}

struct HostIc0Factorization
{
    std::vector<int> row_ptr_l;
    std::vector<int> col_ind_l;
    std::vector<double> val_l;
    std::vector<int> row_ptr_lt;
    std::vector<int> col_ind_lt;
    std::vector<double> val_lt;
};

static HostIc0Factorization factorize_subdomain_ic0_host(const ichol::matrix::CsrMatrix<double> &A_sub)
{
    HostIc0Factorization out;
    out.row_ptr_l = A_sub.row_ptr;
    out.col_ind_l = A_sub.col_ind;
    out.val_l = A_sub.values;

    const int n = A_sub.num_rows;
    std::vector<int> marker(static_cast<size_t>(n), -1);

    for (int i = 0; i < n; ++i)
    {
        const int row_begin = out.row_ptr_l[static_cast<size_t>(i)];
        const int row_end = out.row_ptr_l[static_cast<size_t>(i + 1)];
        const int diag_pos = row_end - 1;
        if (diag_pos < row_begin || out.col_ind_l[static_cast<size_t>(diag_pos)] != i)
            throw std::runtime_error("factorize_subdomain_ic0_host: expected diagonal last in row");

        for (int p = row_begin; p < diag_pos; ++p)
            marker[static_cast<size_t>(out.col_ind_l[static_cast<size_t>(p)])] = p;

        for (int p = row_begin; p < diag_pos; ++p)
        {
            const int j = out.col_ind_l[static_cast<size_t>(p)];
            double sum = out.val_l[static_cast<size_t>(p)];
            const int j_begin = out.row_ptr_l[static_cast<size_t>(j)];
            const int j_diag = out.row_ptr_l[static_cast<size_t>(j + 1)] - 1;
            for (int q = j_begin; q < j_diag; ++q)
            {
                const int k = out.col_ind_l[static_cast<size_t>(q)];
                const int pk = marker[static_cast<size_t>(k)];
                if (pk >= row_begin)
                    sum -= out.val_l[static_cast<size_t>(pk)] * out.val_l[static_cast<size_t>(q)];
            }
            const double ljj = out.val_l[static_cast<size_t>(j_diag)];
            if (!(ljj > 0.0))
                throw std::runtime_error("factorize_subdomain_ic0_host: non-positive pivot in previous row");
            out.val_l[static_cast<size_t>(p)] = sum / ljj;
        }

        double diag = out.val_l[static_cast<size_t>(diag_pos)];
        for (int p = row_begin; p < diag_pos; ++p)
        {
            const double lij = out.val_l[static_cast<size_t>(p)];
            diag -= lij * lij;
        }
        if (!(diag > 0.0))
            throw std::runtime_error("factorize_subdomain_ic0_host: non-positive pivot during IC(0)");
        out.val_l[static_cast<size_t>(diag_pos)] = std::sqrt(diag);

        for (int p = row_begin; p < diag_pos; ++p)
            marker[static_cast<size_t>(out.col_ind_l[static_cast<size_t>(p)])] = -1;
    }

    build_csr_transpose(n, out.row_ptr_l, out.col_ind_l, out.val_l,
                        out.row_ptr_lt, out.col_ind_lt, out.val_lt);
    return out;
}

struct HostStorageBuffer
{
    StorageMap map{};
    std::vector<double> fp64;
    std::vector<float> fp32;

    void resize(size_t count, const StorageMap &storage_map)
    {
        map = storage_map;
        if (map.use_fp64)
        {
            fp64.assign(count, 0.0);
            fp32.clear();
        }
        else
        {
            fp32.assign(count, 0.0f);
            fp64.clear();
        }
    }

    bool empty() const
    {
        return fp64.empty() && fp32.empty();
    }

    void store(const double *src, size_t count, size_t offset = 0)
    {
        if (map.use_fp64)
        {
            std::copy(src, src + count, fp64.begin() + static_cast<std::ptrdiff_t>(offset));
            return;
        }
        auto *dst = fp32.data() + static_cast<std::ptrdiff_t>(offset);
        for (size_t i = 0; i < count; ++i)
            dst[i] = static_cast<float>(src[i]);
    }

    void load(double *dst, size_t count, size_t offset = 0) const
    {
        if (map.use_fp64)
        {
            std::copy(fp64.begin() + static_cast<std::ptrdiff_t>(offset),
                      fp64.begin() + static_cast<std::ptrdiff_t>(offset + count),
                      dst);
            return;
        }
        const auto *src = fp32.data() + static_cast<std::ptrdiff_t>(offset);
        for (size_t i = 0; i < count; ++i)
            dst[i] = static_cast<double>(src[i]);
    }

    const float *fp32_ptr(size_t offset = 0) const
    {
        return fp32.empty() ? nullptr : fp32.data() + static_cast<std::ptrdiff_t>(offset);
    }

    float *fp32_ptr(size_t offset = 0)
    {
        return fp32.empty() ? nullptr : fp32.data() + static_cast<std::ptrdiff_t>(offset);
    }
};

static void sync_block_to_storage(
    const std::vector<double> &src,
    HostStorageBuffer *dst,
    const StorageMap &storage_map)
{
    if (!dst || storage_map.use_fp64)
        return;
    dst->store(src.data(), src.size());
}

static void sync_block_from_storage(
    const HostStorageBuffer *src,
    std::vector<double> &dst,
    const StorageMap &storage_map)
{
    if (!src || storage_map.use_fp64)
        return;
    src->load(dst.data(), dst.size());
}

struct MklSparseMatrix
{
    sparse_matrix_t handle = nullptr;

    ~MklSparseMatrix()
    {
        if (handle)
            mkl_sparse_destroy(handle);
    }
};

template <typename T>
static void create_csr_matrix(
    MklSparseMatrix &mat,
    int n,
    const std::vector<MKL_INT> &rows_start,
    const std::vector<MKL_INT> &rows_end,
    const std::vector<MKL_INT> &cols,
    std::vector<T> &values)
{
    if constexpr (std::is_same_v<T, double>)
    {
        MKL_SPARSE_CHECK(mkl_sparse_d_create_csr(
            &mat.handle, SPARSE_INDEX_BASE_ZERO, n, n,
            const_cast<MKL_INT *>(rows_start.data()),
            const_cast<MKL_INT *>(rows_end.data()),
            const_cast<MKL_INT *>(cols.data()),
            values.data()));
    }
    else
    {
        MKL_SPARSE_CHECK(mkl_sparse_s_create_csr(
            &mat.handle, SPARSE_INDEX_BASE_ZERO, n, n,
            const_cast<MKL_INT *>(rows_start.data()),
            const_cast<MKL_INT *>(rows_end.data()),
            const_cast<MKL_INT *>(cols.data()),
            values.data()));
    }
    MKL_SPARSE_CHECK(mkl_sparse_optimize(mat.handle));
}

template <typename T>
struct HostSubdomainTriangularSolveVariant
{
    std::vector<T> rhs;
    std::vector<T> y;
    std::vector<T> x;
    std::vector<T> val_l;
    std::vector<T> val_lt;

    std::vector<MKL_INT> row_start_l;
    std::vector<MKL_INT> row_end_l;
    std::vector<MKL_INT> col_ind_l;
    std::vector<MKL_INT> row_start_lt;
    std::vector<MKL_INT> row_end_lt;
    std::vector<MKL_INT> col_ind_lt;

    MklSparseMatrix mat_l;
    MklSparseMatrix mat_lt;
    matrix_descr descr_l{};
    matrix_descr descr_lt{};
};

static void fill_mkl_row_bounds(
    const std::vector<int> &row_ptr,
    std::vector<MKL_INT> &row_start,
    std::vector<MKL_INT> &row_end)
{
    const int n = static_cast<int>(row_ptr.size()) - 1;
    row_start.resize(static_cast<size_t>(n));
    row_end.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        row_start[static_cast<size_t>(i)] = static_cast<MKL_INT>(row_ptr[static_cast<size_t>(i)]);
        row_end[static_cast<size_t>(i)] = static_cast<MKL_INT>(row_ptr[static_cast<size_t>(i + 1)]);
    }
}

template <typename T>
static void init_subdomain_variant(
    HostSubdomainTriangularSolveVariant<T> &variant,
    int nsub,
    const HostIc0Factorization &factor)
{
    variant.rhs.assign(static_cast<size_t>(nsub), T{});
    variant.y.assign(static_cast<size_t>(nsub), T{});
    variant.x.assign(static_cast<size_t>(nsub), T{});
    variant.col_ind_l.assign(factor.col_ind_l.begin(), factor.col_ind_l.end());
    variant.col_ind_lt.assign(factor.col_ind_lt.begin(), factor.col_ind_lt.end());
    fill_mkl_row_bounds(factor.row_ptr_l, variant.row_start_l, variant.row_end_l);
    fill_mkl_row_bounds(factor.row_ptr_lt, variant.row_start_lt, variant.row_end_lt);
    variant.val_l.resize(factor.val_l.size());
    variant.val_lt.resize(factor.val_lt.size());
    for (size_t i = 0; i < factor.val_l.size(); ++i)
        variant.val_l[i] = static_cast<T>(factor.val_l[i]);
    for (size_t i = 0; i < factor.val_lt.size(); ++i)
        variant.val_lt[i] = static_cast<T>(factor.val_lt[i]);

    create_csr_matrix(variant.mat_l, nsub, variant.row_start_l, variant.row_end_l, variant.col_ind_l, variant.val_l);
    create_csr_matrix(variant.mat_lt, nsub, variant.row_start_lt, variant.row_end_lt, variant.col_ind_lt, variant.val_lt);

    variant.descr_l.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    variant.descr_l.mode = SPARSE_FILL_MODE_LOWER;
    variant.descr_l.diag = SPARSE_DIAG_NON_UNIT;

    variant.descr_lt.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    variant.descr_lt.mode = SPARSE_FILL_MODE_UPPER;
    variant.descr_lt.diag = SPARSE_DIAG_NON_UNIT;

    MKL_SPARSE_CHECK(mkl_sparse_set_sv_hint(
        variant.mat_l.handle,
        SPARSE_OPERATION_NON_TRANSPOSE,
        variant.descr_l,
        128));
    MKL_SPARSE_CHECK(mkl_sparse_set_sv_hint(
        variant.mat_lt.handle,
        SPARSE_OPERATION_NON_TRANSPOSE,
        variant.descr_lt,
        128));
    MKL_SPARSE_CHECK(mkl_sparse_optimize(variant.mat_l.handle));
    MKL_SPARSE_CHECK(mkl_sparse_optimize(variant.mat_lt.handle));
}

static void cast_d2f(const double *src, float *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = static_cast<float>(src[i]);
}

static void cast_f2d(const float *src, double *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = static_cast<double>(src[i]);
}

static void build_znew_columns_host(
    const std::vector<ichol::precond::PrecondApply> &preconds,
    const std::vector<double> &r,
    std::vector<float> &r_precond,
    std::vector<double> &Znew,
    std::vector<float> &Znew_precond,
    int n,
    const StorageMap &precond_map)
{
    const int k = static_cast<int>(preconds.size());
    if (precond_map.use_fp64)
    {
        for (int t = 0; t < k; ++t)
        {
            double *z_col = Znew.data() + static_cast<size_t>(t) * static_cast<size_t>(n);
            std::fill(z_col, z_col + n, 0.0);
            if (!preconds[t].apply)
                throw std::runtime_error("mpcg_mkl: null preconditioner callback");
            // The callback signature is still CUDA-shaped; the CPU backend passes
            // host pointers and a null stream, so callers need CPU-capable apply().
            preconds[t].apply(preconds[t].ctx, r.data(), z_col, n,
                              precond_map.storage_prec, nullptr);
        }
        return;
    }

    cast_d2f(r.data(), r_precond.data(), static_cast<size_t>(n));
    for (int t = 0; t < k; ++t)
    {
        float *z_col = Znew_precond.data() + static_cast<size_t>(t) * static_cast<size_t>(n);
        std::fill(z_col, z_col + n, 0.0f);
        if (!preconds[t].apply)
            throw std::runtime_error("mpcg_mkl: null preconditioner callback");
        preconds[t].apply(preconds[t].ctx, r_precond.data(), z_col, n,
                          precond_map.storage_prec, nullptr);
    }
    cast_f2d(Znew_precond.data(), Znew.data(), static_cast<size_t>(n) * static_cast<size_t>(k));
}

static void column_norms(const std::vector<double> &X, int n, int k, std::vector<double> &out)
{
    out.assign(static_cast<size_t>(k), 0.0);
    for (int col = 0; col < k; ++col)
    {
        const double *x = X.data() + static_cast<size_t>(col) * static_cast<size_t>(n);
        out[static_cast<size_t>(col)] = cblas_dnrm2(n, x, 1);
    }
}

static void scale_columns_from_norms(std::vector<double> &X, int n, int k, const std::vector<double> &norms, double floor)
{
    for (int col = 0; col < k; ++col)
    {
        double *x = X.data() + static_cast<size_t>(col) * static_cast<size_t>(n);
        const double norm = norms[static_cast<size_t>(col)];
        if (norm > floor)
        {
            const double scale = 1.0 / norm;
            cblas_dscal(n, scale, x, 1);
        }
        else
        {
            std::fill(x, x + n, 0.0);
        }
    }
}

static void column_dots(
    const std::vector<double> &X,
    const std::vector<double> &Y,
    int n,
    int k,
    std::vector<double> &out)
{
    out.assign(static_cast<size_t>(k), 0.0);
    for (int col = 0; col < k; ++col)
    {
        const double *x = X.data() + static_cast<size_t>(col) * static_cast<size_t>(n);
        const double *y = Y.data() + static_cast<size_t>(col) * static_cast<size_t>(n);
        out[static_cast<size_t>(col)] = cblas_ddot(n, x, 1, y, 1);
    }
}

static bool should_reproject(const std::vector<double> &z_dots,
                             const std::vector<double> &p_dots,
                             double ratio_sq)
{
    double z_sum = 0.0;
    double p_sum = 0.0;
    for (size_t i = 0; i < z_dots.size(); ++i)
    {
        z_sum += std::max(0.0, z_dots[i]);
        p_sum += std::max(0.0, p_dots[i]);
    }
    return z_sum > 0.0 && p_sum < ratio_sq * z_sum;
}

static void build_col_scale_from_gdiag(const std::vector<double> &G, std::vector<double> &col_scale, int k, double diag_floor)
{
    col_scale.assign(static_cast<size_t>(k), 0.0);
    for (int i = 0; i < k; ++i)
    {
        const double gii = G[static_cast<size_t>(i) + static_cast<size_t>(i) * static_cast<size_t>(k)];
        if (std::isfinite(gii) && gii > diag_floor)
            col_scale[static_cast<size_t>(i)] = 1.0 / std::sqrt(gii);
    }
}

static void fuse_scale_p_w(
    std::vector<double> &P,
    std::vector<double> &W,
    const std::vector<double> &col_scale,
    int n,
    int k)
{
    for (int col = 0; col < k; ++col)
    {
        const double scale = col_scale[static_cast<size_t>(col)];
        if (scale == 1.0)
            continue;
        cblas_dscal(n, scale, P.data() + static_cast<size_t>(col) * static_cast<size_t>(n), 1);
        cblas_dscal(n, scale, W.data() + static_cast<size_t>(col) * static_cast<size_t>(n), 1);
    }
}

static void congruence_scale_gram(std::vector<double> &G, const std::vector<double> &col_scale, int k)
{
    for (int col = 0; col < k; ++col)
        for (int row = 0; row < k; ++row)
            G[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(k)] *=
                col_scale[static_cast<size_t>(row)] * col_scale[static_cast<size_t>(col)];
}

static void set_identity(std::vector<double> &A, int k)
{
    A.assign(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    for (int i = 0; i < k; ++i)
        A[static_cast<size_t>(i) + static_cast<size_t>(i) * static_cast<size_t>(k)] = 1.0;
}

static void gemm_wt_p_to_c(
    const std::vector<double> &W,
    const std::vector<double> &P,
    int n,
    int k,
    const PrecisionMap &g_map,
    std::vector<float> &W_low,
    std::vector<float> &P_low,
    std::vector<float> &C_low,
    std::vector<double> &C)
{
    if (g_map.use_fp64)
    {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    k, k, n,
                    1.0, W.data(), n,
                    P.data(), n,
                    0.0, C.data(), k);
        return;
    }

    W_low.resize(W.size());
    P_low.resize(P.size());
    C_low.assign(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0f);
    cast_d2f(W.data(), W_low.data(), W.size());
    cast_d2f(P.data(), P_low.data(), P.size());
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                k, k, n,
                1.0f, W_low.data(), n,
                P_low.data(), n,
                0.0f, C_low.data(), k);
    cast_f2d(C_low.data(), C.data(), C.size());
}

static void pinv_svd_cpu(
    const double *G,
    int k,
    const double *B,
    int nrhs,
    double *X,
    double rcond)
{
    std::vector<double> G_copy(static_cast<size_t>(k) * static_cast<size_t>(k));
    std::vector<double> U(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    std::vector<double> VT(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    std::vector<double> S(static_cast<size_t>(k), 0.0);
    std::vector<double> superb(static_cast<size_t>(std::max(0, k - 1)), 0.0);
    std::vector<double> T1(static_cast<size_t>(k) * static_cast<size_t>(nrhs), 0.0);

    for (int col = 0; col < k; ++col)
        for (int row = 0; row < k; ++row)
            G_copy[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(k)] =
                0.5 * (G[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(k)] +
                       G[static_cast<size_t>(col) + static_cast<size_t>(row) * static_cast<size_t>(k)]);

    const int info = LAPACKE_dgesvd(
        LAPACK_COL_MAJOR, 'A', 'A',
        k, k,
        G_copy.data(), k,
        S.data(),
        U.data(), k,
        VT.data(), k,
        superb.data());
    MKL_LAPACK_CHECK(info, "LAPACKE_dgesvd");
    if (info > 0)
        throw std::runtime_error("mpcg_mkl: SVD failed to converge");

    const double s_max = *std::max_element(S.begin(), S.end());
    if (s_max == 0.0)
    {
        std::fill(X, X + static_cast<size_t>(k) * static_cast<size_t>(nrhs), 0.0);
        return;
    }

    const double threshold = std::max({rcond * s_max, 1e-15 * s_max, 1e-15});
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                k, nrhs, k,
                1.0, U.data(), k,
                B, k,
                0.0, T1.data(), k);
    for (int i = 0; i < k; ++i)
    {
        const double inv_sigma = (S[static_cast<size_t>(i)] > threshold) ? (1.0 / S[static_cast<size_t>(i)]) : 0.0;
        cblas_dscal(nrhs, inv_sigma, T1.data() + static_cast<size_t>(i), k);
    }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                k, nrhs, k,
                1.0, VT.data(), k,
                T1.data(), k,
                0.0, X, k);
}

static void chol_solve_cpu(
    const double *G,
    int k,
    const double *B,
    int nrhs,
    double *X,
    double rcond,
    bool use_svd_fallback)
{
    std::vector<double> G_copy(static_cast<size_t>(k) * static_cast<size_t>(k));
    for (int col = 0; col < k; ++col)
        for (int row = 0; row < k; ++row)
            G_copy[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(k)] =
                0.5 * (G[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(k)] +
                       G[static_cast<size_t>(col) + static_cast<size_t>(row) * static_cast<size_t>(k)]);

    std::copy(B, B + static_cast<size_t>(k) * static_cast<size_t>(nrhs), X);
    const int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', k, G_copy.data(), k);
    MKL_LAPACK_CHECK(info, "LAPACKE_dpotrf");
    if (info > 0)
    {
        if (!use_svd_fallback)
            throw std::runtime_error("mpcg_mkl: Cholesky solve failed");
        pinv_svd_cpu(G, k, B, nrhs, X, rcond);
        return;
    }

    const int info_solve = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', k, nrhs, G_copy.data(), k, X, k);
    MKL_LAPACK_CHECK(info_solve, "LAPACKE_dpotrs");
}
} // namespace

namespace ichol::precond
{
struct SubdomainIncompleteCholeskyMklContext
{
    int nsub = 0;
    std::vector<int> gidx;
    ichol::solver::ComputePrecision storage_prec = ichol::solver::ComputePrecision::FP64;
    HostSubdomainTriangularSolveVariant<double> fp64;
    HostSubdomainTriangularSolveVariant<float> fp32;
};

template <typename T>
static void apply_ic_mkl_impl(
    HostSubdomainTriangularSolveVariant<T> &variant,
    const std::vector<int> &gidx,
    const T *r,
    T *z)
{
    const int nsub = static_cast<int>(gidx.size());
    const T alpha = static_cast<T>(1);
    for (int i = 0; i < nsub; ++i)
        variant.rhs[static_cast<size_t>(i)] = r[gidx[static_cast<size_t>(i)]];

    if constexpr (std::is_same_v<T, double>)
    {
        MKL_SPARSE_CHECK(mkl_sparse_d_trsv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            variant.mat_l.handle,
            variant.descr_l,
            variant.rhs.data(),
            variant.y.data()));
        MKL_SPARSE_CHECK(mkl_sparse_d_trsv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            variant.mat_lt.handle,
            variant.descr_lt,
            variant.y.data(),
            variant.x.data()));
    }
    else
    {
        MKL_SPARSE_CHECK(mkl_sparse_s_trsv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            variant.mat_l.handle,
            variant.descr_l,
            variant.rhs.data(),
            variant.y.data()));
        MKL_SPARSE_CHECK(mkl_sparse_s_trsv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            variant.mat_lt.handle,
            variant.descr_lt,
            variant.y.data(),
            variant.x.data()));
    }

    for (int i = 0; i < nsub; ++i)
        z[gidx[static_cast<size_t>(i)]] = variant.x[static_cast<size_t>(i)];
}

SubdomainIncompleteCholeskyMklContext *create_subdomain_incomplete_cholesky_mkl_context(
    const ichol::matrix::CsrMatrix<double> &A,
    const GridShape &global_shape,
    const SubdomainRegion &region,
    const SubdomainPreconditionerOptions &options)
{
    auto *ctx = new SubdomainIncompleteCholeskyMklContext();
    try
    {
        const int lw = region.x1 - region.x0;
        const int lh = region.y1 - region.y0;
        const int ld = region.z1 - region.z0;
        ctx->nsub = lw * lh * ld;
        if (ctx->nsub <= 0)
            throw std::runtime_error("create_subdomain_incomplete_cholesky_mkl_context: empty subdomain");

        const auto A_sub = extract_lower_subdomain_csr(A, global_shape, region);
        const auto factor = factorize_subdomain_ic0_host(A_sub);
        ctx->gidx = build_subdomain_gidx_host(lw, lh, ld, global_shape.w, global_shape.h,
                                              region.x0, region.y0, region.z0);
        ctx->storage_prec = normalize_precond_precision(options.precision);

        switch (ctx->storage_prec)
        {
        case ichol::solver::ComputePrecision::FP64:
            init_subdomain_variant(ctx->fp64, ctx->nsub, factor);
            break;
        case ichol::solver::ComputePrecision::FP32:
            init_subdomain_variant(ctx->fp32, ctx->nsub, factor);
            break;
        default:
            throw std::runtime_error("create_subdomain_incomplete_cholesky_mkl_context: unsupported precision");
        }
        return ctx;
    }
    catch (...)
    {
        delete ctx;
        throw;
    }
}

std::vector<SubdomainIncompleteCholeskyMklContext *> create_subdomain_incomplete_cholesky_mkl_contexts_parallel(
    const ichol::matrix::CsrMatrix<double> &A,
    const GridShape &global_shape,
    const std::vector<SubdomainRegion> &regions,
    const SubdomainPreconditionerOptions &options)
{
    std::vector<std::future<SubdomainIncompleteCholeskyMklContext *>> futures;
    futures.reserve(regions.size());
    for (std::size_t region_index = 0; region_index < regions.size(); ++region_index)
    {
        const auto region = regions[region_index];
        auto region_options = options;
        region_options.debug_subdomain_index = static_cast<int>(region_index);
        futures.emplace_back(std::async(std::launch::async, [&A, global_shape, region, region_options]()
                                        { return create_subdomain_incomplete_cholesky_mkl_context(A, global_shape, region, region_options); }));
    }

    std::vector<SubdomainIncompleteCholeskyMklContext *> contexts;
    contexts.reserve(regions.size());
    try
    {
        for (auto &future : futures)
            contexts.push_back(future.get());
    }
    catch (...)
    {
        for (auto *ctx : contexts)
            destroy_subdomain_incomplete_cholesky_mkl_context(ctx);
        throw;
    }
    return contexts;
}

void destroy_subdomain_incomplete_cholesky_mkl_context(SubdomainIncompleteCholeskyMklContext *ctx)
{
    delete ctx;
}

void apply_subdomain_incomplete_cholesky_mkl(
    void *vctx,
    const void *r,
    void *z,
    int /*N*/,
    ichol::solver::ComputePrecision prec,
    cudaStream_t /*stream*/)
{
    auto *ctx = reinterpret_cast<SubdomainIncompleteCholeskyMklContext *>(vctx);
    const auto requested_prec = normalize_precond_precision(prec);
    if (requested_prec != ctx->storage_prec)
        throw std::runtime_error("apply_subdomain_incomplete_cholesky_mkl: requested precision does not match stored IC(0) precision");

    switch (ctx->storage_prec)
    {
    case ichol::solver::ComputePrecision::FP64:
        apply_ic_mkl_impl(ctx->fp64, ctx->gidx,
                          static_cast<const double *>(r),
                          static_cast<double *>(z));
        break;
    case ichol::solver::ComputePrecision::FP32:
        apply_ic_mkl_impl(ctx->fp32, ctx->gidx,
                          static_cast<const float *>(r),
                          static_cast<float *>(z));
        break;
    default:
        throw std::runtime_error("apply_subdomain_incomplete_cholesky_mkl: unsupported precision");
    }
}
} // namespace ichol::precond

namespace
{

template <typename T_L>
static ichol::solver::PCGResult mpcg_impl(
    const std::vector<int> &h_csrRowPtrA,
    const std::vector<int> &h_csrColIndA,
    const std::vector<double> &h_valA,
    const std::vector<ichol::precond::PrecondApply> &preconds,
    const std::vector<double> &h_b,
    std::vector<double> &h_x,
    const ichol::solver::PCGParams &params)
{
    using namespace ichol::solver;

    const char *solver_label = "MPCG-MKL";
    const int n = static_cast<int>(h_b.size());
    const int k = static_cast<int>(preconds.size());
    const int m = (params.restart <= 0) ? params.maxits : params.restart;
    if (m <= 0)
        throw std::runtime_error("mpcg_mkl: restart/maxits produced empty history window");

    const bool profile_enabled = params.verbose;
    const auto total_wall_start = Clock::now();
    double warmup_wall_ms = 0.0;

    ProfilePhase phase_precond{};
    ProfilePhase phase_ortho{};
    ProfilePhase phase_spmm{};
    ProfilePhase phase_dense{};
    ProfilePhase phase_reset{};

    PrecisionMap g_map = get_precision_map(params.prec_gemm, params.prec_acc);
    SpmmMap spmm_map = get_spmm_map(params.prec_spmm, params.prec_acc);
    StorageMap precond_map = get_precond_map(params.prec_precond);
    StorageMap Znew_store_map = get_storage_map(params.store_Znew);
    StorageMap Pnew_store_map = get_storage_map(params.store_Pnew);
    StorageMap Wnew_store_map = get_storage_map(params.store_Wnew);
    StorageMap P_hist_map = get_storage_map(params.store_P_hist);
    StorageMap W_hist_map = get_storage_map(params.store_W_hist);

    const bool use_lowp_history = (!P_hist_map.use_fp64) || (!W_hist_map.use_fp64);
    const bool enable_anorm_reprojection =
        use_lowp_history && params.projection_anorm_drop_tol > 0.0;
    const double anorm_drop_tol_sq =
        params.projection_anorm_drop_tol * params.projection_anorm_drop_tol;

    const size_t nk = static_cast<size_t>(n) * static_cast<size_t>(k);

    std::vector<MKL_INT> row_start(static_cast<size_t>(n));
    std::vector<MKL_INT> row_end(static_cast<size_t>(n));
    std::vector<MKL_INT> col_ind(static_cast<size_t>(h_csrColIndA.size()));
    for (int i = 0; i < n; ++i)
    {
        row_start[static_cast<size_t>(i)] = static_cast<MKL_INT>(h_csrRowPtrA[static_cast<size_t>(i)]);
        row_end[static_cast<size_t>(i)] = static_cast<MKL_INT>(h_csrRowPtrA[static_cast<size_t>(i + 1)]);
    }
    for (size_t i = 0; i < h_csrColIndA.size(); ++i)
        col_ind[i] = static_cast<MKL_INT>(h_csrColIndA[i]);

    std::vector<double> valA64 = h_valA;
    MklSparseMatrix matA64;
    create_csr_matrix(matA64, n, row_start, row_end, col_ind, valA64);

    std::vector<float> valA32;
    MklSparseMatrix matA32;
    if (!spmm_map.use_fp64)
    {
        valA32.resize(h_valA.size());
        cast_d2f(h_valA.data(), valA32.data(), h_valA.size());
        create_csr_matrix(matA32, n, row_start, row_end, col_ind, valA32);
    }

    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    std::vector<double> x = h_x;
    std::vector<double> b = h_b;
    std::vector<double> r(static_cast<size_t>(n), 0.0);
    std::vector<double> tmp(static_cast<size_t>(n), 0.0);

    std::vector<double> Znew(nk, 0.0);
    std::vector<double> Pnew(nk, 0.0);
    std::vector<double> Wnew(nk, 0.0);
    std::vector<double> Wz(enable_anorm_reprojection ? nk : 0, 0.0);
    std::vector<float> r_precond(!precond_map.use_fp64 ? static_cast<size_t>(n) : 0, 0.0f);
    std::vector<float> Znew_precond(!precond_map.use_fp64 ? nk : 0, 0.0f);

    HostStorageBuffer Znew_store;
    HostStorageBuffer Pnew_store;
    HostStorageBuffer Wnew_store;
    if (!Znew_store_map.use_fp64)
        Znew_store.resize(nk, Znew_store_map);
    if (!Pnew_store_map.use_fp64)
        Pnew_store.resize(nk, Pnew_store_map);
    if (!Wnew_store_map.use_fp64)
        Wnew_store.resize(nk, Wnew_store_map);

    HostStorageBuffer P_hist;
    HostStorageBuffer W_hist;
    P_hist.resize(static_cast<size_t>(m) * nk, P_hist_map);
    W_hist.resize(static_cast<size_t>(m) * nk, W_hist_map);

    std::vector<double> Pj_tmp(!P_hist_map.use_fp64 ? nk : 0, 0.0);
    std::vector<double> Wj_tmp(!W_hist_map.use_fp64 ? nk : 0, 0.0);

    std::vector<double> G_hist(static_cast<size_t>(m) * static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    std::vector<double> Ginv_hist(static_cast<size_t>(m) * static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    std::vector<double> hist_C(static_cast<size_t>(m) * static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    std::vector<double> hist_Y(static_cast<size_t>(m) * static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    std::vector<double> Gnew(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    std::vector<double> rhs(static_cast<size_t>(k), 0.0);
    std::vector<double> alpha(static_cast<size_t>(k), 0.0);
    std::vector<double> col_scale(static_cast<size_t>(k), 0.0);
    std::vector<double> col_norms(static_cast<size_t>(k), 0.0);
    std::vector<double> z_anorm_cols(static_cast<size_t>(k), 0.0);
    std::vector<double> p_anorm_cols(static_cast<size_t>(k), 0.0);
    std::vector<double> eye;
    set_identity(eye, k);

    std::vector<float> spmm_in_tmp(!spmm_map.use_fp64 ? nk : 0, 0.0f);
    std::vector<float> spmm_out_tmp(!spmm_map.use_fp64 ? nk : 0, 0.0f);
    std::vector<float> Pnew_low(!g_map.use_fp64 ? nk : 0, 0.0f);
    std::vector<float> Wnew_low(!g_map.use_fp64 ? nk : 0, 0.0f);
    std::vector<float> Wj_low(!g_map.use_fp64 ? nk : 0, 0.0f);
    std::vector<float> C_gemm(!g_map.use_fp64 ? static_cast<size_t>(k) * static_cast<size_t>(k) : 0, 0.0f);

    if (params.verbose)
    {
        const size_t P_hist_bytes = P_hist_map.use_fp64 ? static_cast<size_t>(m) * nk * sizeof(double)
                                                        : static_cast<size_t>(m) * nk * sizeof(float);
        const size_t W_hist_bytes = W_hist_map.use_fp64 ? static_cast<size_t>(m) * nk * sizeof(double)
                                                        : static_cast<size_t>(m) * nk * sizeof(float);
        std::fprintf(stderr,
                     "[%s history] m=%d nk=%zu P_hist=%.3f MiB W_hist=%.3f MiB total=%.3f MiB\n",
                     solver_label,
                     m, nk,
                     static_cast<double>(P_hist_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(W_hist_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(P_hist_bytes + W_hist_bytes) / (1024.0 * 1024.0));
    }

    auto run_spmv = [&](const std::vector<double> &src, std::vector<double> &dst)
    {
        MKL_SPARSE_CHECK(mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            1.0, matA64.handle, descr,
            src.data(),
            0.0, dst.data()));
    };

    auto run_spmm = [&](const std::vector<double> &src_fp64,
                        HostStorageBuffer *src_store,
                        const StorageMap &src_store_map,
                        std::vector<double> &dst_fp64,
                        HostStorageBuffer *dst_store,
                        const StorageMap &dst_store_map)
    {
        if (spmm_map.use_fp64)
        {
            MKL_SPARSE_CHECK(mkl_sparse_d_mm(
                SPARSE_OPERATION_NON_TRANSPOSE,
                1.0, matA64.handle, descr,
                SPARSE_LAYOUT_COLUMN_MAJOR,
                src_fp64.data(), k, n,
                0.0, dst_fp64.data(), n));
            if (dst_store && !dst_store_map.use_fp64)
                dst_store->store(dst_fp64.data(), nk);
            return;
        }

        const float *src_ptr = nullptr;
        if (src_store && !src_store_map.use_fp64)
        {
            src_store->store(src_fp64.data(), nk);
            src_ptr = src_store->fp32_ptr();
        }
        else
        {
            cast_d2f(src_fp64.data(), spmm_in_tmp.data(), nk);
            src_ptr = spmm_in_tmp.data();
        }

        float *dst_ptr = nullptr;
        if (dst_store && !dst_store_map.use_fp64)
            dst_ptr = dst_store->fp32_ptr();
        else
            dst_ptr = spmm_out_tmp.data();

        // oneMKL CPU sparse BLAS does not expose the CUDA-style explicit temp
        // workspace path; optimize() is the closest equivalent.
        MKL_SPARSE_CHECK(mkl_sparse_s_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            1.0f, matA32.handle, descr,
            SPARSE_LAYOUT_COLUMN_MAJOR,
            src_ptr, k, n,
            0.0f, dst_ptr, n));
        cast_f2d(dst_ptr, dst_fp64.data(), nk);
        if (dst_store && !dst_store_map.use_fp64 && dst_ptr != dst_store->fp32_ptr())
            dst_store->store(dst_fp64.data(), nk);
    };

    run_spmv(x, tmp);
    r = b;
    cblas_daxpy(n, -1.0, tmp.data(), 1, r.data(), 1);

    double bnorm = cblas_dnrm2(n, b.data(), 1);
    if (bnorm == 0.0)
        bnorm = 1.0;
    double current_res_norm = cblas_dnrm2(n, r.data(), 1);

    PCGResult result{};
    result.relResiduals.reserve(static_cast<size_t>(params.maxits) + 1);
    result.relResiduals.push_back(current_res_norm / bnorm);
    bool converged = false;

    double hist_rcond = get_safe_rcond(P_hist_map.storage_prec, params.rcond_base);
    hist_rcond = std::max(hist_rcond, 1e-15);
    std::vector<int> hist_slots;
    hist_slots.reserve(static_cast<size_t>(m));

    int reset_iter = P_hist_map.use_fp64 ? 50 : 10;

    auto solve_kk = [&](const double *G, int nrhs, const double *B, double *X)
    {
        if (params.use_svd)
            pinv_svd_cpu(G, k, B, nrhs, X, hist_rcond);
        else
            chol_solve_cpu(G, k, B, nrhs, X, hist_rcond, true);
    };

    constexpr int warmup_spmm_iters = 2;
    constexpr int warmup_gemm_iters = 2;
    const auto warmup_wall_start = Clock::now();
    if (k > 0)
    {
        build_znew_columns_host(preconds, r, r_precond, Znew, Znew_precond, n, precond_map);
        Pnew = Znew;
        sync_block_to_storage(Pnew, Pnew_store.empty() ? nullptr : &Pnew_store, Pnew_store_map);

        for (int i = 0; i < warmup_spmm_iters; ++i)
            run_spmm(Pnew, Pnew_store.empty() ? nullptr : &Pnew_store, Pnew_store_map,
                     Wnew, Wnew_store.empty() ? nullptr : &Wnew_store, Wnew_store_map);

        for (int i = 0; i < warmup_gemm_iters; ++i)
            gemm_wt_p_to_c(Wnew, Pnew, n, k, g_map, Wnew_low, Pnew_low, C_gemm, Gnew);
    }
    warmup_wall_ms = elapsed_ms(warmup_wall_start, Clock::now());

    const auto iter_wall_start = Clock::now();

    for (int iter = 0; iter < params.maxits; ++iter)
    {
        if (current_res_norm <= params.tol * bnorm)
        {
            result.iterations = iter;
            result.finalRes = current_res_norm;
            converged = true;
            break;
        }

        {
            const auto t0 = Clock::now();
            build_znew_columns_host(preconds, r, r_precond, Znew, Znew_precond, n, precond_map);
            sync_block_to_storage(Znew, Znew_store.empty() ? nullptr : &Znew_store, Znew_store_map);
            Pnew = Znew;
            phase_precond.total_ms += elapsed_ms(t0, Clock::now());
        }

        const int hist_count = std::min(iter, m);
        hist_slots.clear();
        for (int jj = iter - hist_count; jj < iter; ++jj)
            hist_slots.push_back(jj % m);

        auto project_against_history = [&]()
        {
            if (hist_count == 0)
                return;

            for (int hist_idx = 0; hist_idx < hist_count; ++hist_idx)
            {
                const int slot = hist_slots[static_cast<size_t>(hist_idx)];
                const size_t hist_off = static_cast<size_t>(slot) * nk;
                const double *Pj = nullptr;
                const double *Wj = nullptr;
                std::vector<double> *Pj_owner = nullptr;
                std::vector<double> *Wj_owner = nullptr;

                if (P_hist_map.use_fp64)
                {
                    Pj = P_hist.fp64.data() + static_cast<std::ptrdiff_t>(hist_off);
                }
                else
                {
                    P_hist.load(Pj_tmp.data(), nk, hist_off);
                    Pj_owner = &Pj_tmp;
                    Pj = Pj_owner->data();
                }

                if (W_hist_map.use_fp64)
                {
                    Wj = W_hist.fp64.data() + static_cast<std::ptrdiff_t>(hist_off);
                }
                else
                {
                    W_hist.load(Wj_tmp.data(), nk, hist_off);
                    Wj_owner = &Wj_tmp;
                    Wj = Wj_owner->data();
                }

                double *Cj = hist_C.data() + static_cast<size_t>(hist_idx) * static_cast<size_t>(k) * static_cast<size_t>(k);
                double *Yj = hist_Y.data() + static_cast<size_t>(hist_idx) * static_cast<size_t>(k) * static_cast<size_t>(k);
                const double *Ginvj = Ginv_hist.data() + static_cast<size_t>(slot) * static_cast<size_t>(k) * static_cast<size_t>(k);

                std::vector<double> Wj_vec(Wj, Wj + nk);
                std::vector<double> Cj_vec(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
                gemm_wt_p_to_c(Wj_vec, Pnew, n, k, g_map, Wj_low, Pnew_low, C_gemm, Cj_vec);
                std::copy(Cj_vec.begin(), Cj_vec.end(), Cj);

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            k, k, k,
                            1.0, Ginvj, k,
                            Cj, k,
                            0.0, Yj, k);

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            n, k, k,
                            -1.0, Pj, n,
                            Yj, k,
                            1.0, Pnew.data(), n);
            }
        };

        {
            const auto t0 = Clock::now();
            project_against_history();
            if (enable_anorm_reprojection && hist_count > 0)
            {
                run_spmm(Znew, Znew_store.empty() ? nullptr : &Znew_store, Znew_store_map,
                         Wz, nullptr, get_storage_map(ichol::solver::ComputePrecision::FP64));
                run_spmm(Pnew, Pnew_store.empty() ? nullptr : &Pnew_store, Pnew_store_map,
                         Wnew, Wnew_store.empty() ? nullptr : &Wnew_store, Wnew_store_map);

                column_dots(Znew, Wz, n, k, z_anorm_cols);
                column_dots(Pnew, Wnew, n, k, p_anorm_cols);
                if (should_reproject(z_anorm_cols, p_anorm_cols, anorm_drop_tol_sq))
                    project_against_history();
            }

            column_norms(Pnew, n, k, col_norms);
            scale_columns_from_norms(Pnew, n, k, col_norms, 1e-30);
            sync_block_to_storage(Pnew, Pnew_store.empty() ? nullptr : &Pnew_store, Pnew_store_map);
            phase_ortho.total_ms += elapsed_ms(t0, Clock::now());
        }

        {
            const auto t0 = Clock::now();
            run_spmm(Pnew, Pnew_store.empty() ? nullptr : &Pnew_store, Pnew_store_map,
                     Wnew, Wnew_store.empty() ? nullptr : &Wnew_store, Wnew_store_map);
            phase_spmm.total_ms += elapsed_ms(t0, Clock::now());
        }

        {
            const auto t0 = Clock::now();
            gemm_wt_p_to_c(Wnew, Pnew, n, k, g_map, Wnew_low, Pnew_low, C_gemm, Gnew);

            build_col_scale_from_gdiag(Gnew, col_scale, k, 1e-30);
            fuse_scale_p_w(Pnew, Wnew, col_scale, n, k);
            congruence_scale_gram(Gnew, col_scale, k);
            sync_block_to_storage(Pnew, Pnew_store.empty() ? nullptr : &Pnew_store, Pnew_store_map);
            sync_block_to_storage(Wnew, Wnew_store.empty() ? nullptr : &Wnew_store, Wnew_store_map);

            cblas_dgemv(CblasColMajor, CblasTrans,
                        n, k,
                        1.0, Pnew.data(), n,
                        r.data(), 1,
                        0.0, rhs.data(), 1);

            solve_kk(Gnew.data(), 1, rhs.data(), alpha.data());

            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        n, k,
                        1.0, Pnew.data(), n,
                        alpha.data(), 1,
                        1.0, x.data(), 1);

            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        n, k,
                        -1.0, Wnew.data(), n,
                        alpha.data(), 1,
                        1.0, r.data(), 1);

            if ((iter + 1) % reset_iter == 0)
            {
                const auto reset_t0 = Clock::now();
                run_spmv(x, tmp);
                r = b;
                cblas_daxpy(n, -1.0, tmp.data(), 1, r.data(), 1);
                phase_reset.total_ms += elapsed_ms(reset_t0, Clock::now());
            }

            const int slot = iter % m;
            const size_t hist_off = static_cast<size_t>(slot) * nk;
            P_hist.store(Pnew.data(), nk, hist_off);
            W_hist.store(Wnew.data(), nk, hist_off);

            std::copy(Gnew.begin(), Gnew.end(),
                      G_hist.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(slot) * static_cast<size_t>(k) * static_cast<size_t>(k)));
            solve_kk(Gnew.data(), k, eye.data(),
                     Ginv_hist.data() + static_cast<std::ptrdiff_t>(static_cast<size_t>(slot) * static_cast<size_t>(k) * static_cast<size_t>(k)));

            current_res_norm = cblas_dnrm2(n, r.data(), 1);
            phase_dense.total_ms += elapsed_ms(t0, Clock::now());
        }

        result.iterations = iter + 1;
        result.relResiduals.push_back(current_res_norm / bnorm);
    }

    const auto iter_wall_end = Clock::now();
    if (!converged)
        result.finalRes = current_res_norm;

    const auto finalize_wall_start = Clock::now();
    h_x = x;
    const auto finalize_wall_end = Clock::now();

    const double total_wall_ms = std::max(0.0, elapsed_ms(total_wall_start, finalize_wall_end) - warmup_wall_ms);
    const double setup_wall_ms = std::max(0.0, elapsed_ms(total_wall_start, iter_wall_start) - warmup_wall_ms);
    const double iter_wall_ms = elapsed_ms(iter_wall_start, iter_wall_end);
    const double finalize_wall_ms = elapsed_ms(finalize_wall_start, finalize_wall_end);
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
                     "[%s profile] n=%d k=%d iters=%d total=%.3fms setup=%.3fms iter=%.3fms finalize=%.3fms warmup=%.3fms(excluded)\n",
                     solver_label, n, k, result.iterations, total_wall_ms, setup_wall_ms, iter_wall_ms, finalize_wall_ms, warmup_wall_ms);
        std::fprintf(stderr,
                     "[%s profile] precond=%.3fms ortho=%.3fms spmm=%.3fms dense=%.3fms reset=%.3fms other_iter=%.3fms\n",
                     solver_label, phase_precond.total_ms, phase_ortho.total_ms, phase_spmm.total_ms,
                     dense_exclusive_ms, phase_reset.total_ms, other_iter_ms);
    }

    return result;
}
} // namespace

namespace ichol::solver
{
template <typename T_L>
PCGResult mpcg_cpu_baseline(
    const std::vector<int> &h_csrRowPtrA,
    const std::vector<int> &h_csrColIndA,
    const std::vector<double> &h_valA,
    const std::vector<ichol::precond::PrecondApply> &preconds,
    const std::vector<double> &h_b,
    std::vector<double> &h_x,
    const PCGParams &params)
{
    return mpcg_impl<T_L>(
        h_csrRowPtrA,
        h_csrColIndA,
        h_valA,
        preconds,
        h_b,
        h_x,
        params);
}

#ifdef ICHOL_USE_MKL_MPCG
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
    return mpcg_cpu_baseline<T_L>(
        h_csrRowPtrA,
        h_csrColIndA,
        h_valA,
        preconds,
        h_b,
        h_x,
        params);
}
#endif

#ifdef ICHOL_USE_MKL_MPCG
namespace debug
{
std::vector<double> build_mpcg_znew_columns(
    const std::vector<ichol::precond::PrecondApply> &preconds,
    const std::vector<double> &h_r,
    const ZnewBuildOptions &options)
{
    const int n = static_cast<int>(h_r.size());
    const int k = static_cast<int>(preconds.size());
    std::vector<double> Znew(static_cast<size_t>(n) * static_cast<size_t>(k), 0.0);
    std::vector<float> r_precond;
    std::vector<float> Znew_precond;
    const StorageMap precond_map = get_precond_map(options.prec_precond);
    if (!precond_map.use_fp64)
    {
        r_precond.resize(static_cast<size_t>(n));
        Znew_precond.resize(static_cast<size_t>(n) * static_cast<size_t>(k));
    }
    build_znew_columns_host(preconds, h_r, r_precond, Znew, Znew_precond, n, precond_map);
    return Znew;
}
} // namespace debug
#endif

template PCGResult mpcg_cpu_baseline<double>(
    const std::vector<int> &,
    const std::vector<int> &,
    const std::vector<double> &,
    const std::vector<ichol::precond::PrecondApply> &,
    const std::vector<double> &,
    std::vector<double> &,
    const PCGParams &);

#ifdef ICHOL_USE_MKL_MPCG
template PCGResult mpcg<double>(
    const std::vector<int> &,
    const std::vector<int> &,
    const std::vector<double> &,
    const std::vector<ichol::precond::PrecondApply> &,
    const std::vector<double> &,
    std::vector<double> &,
    const PCGParams &);
#endif
} // namespace ichol::solver
