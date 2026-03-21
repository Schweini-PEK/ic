#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "factor/numerical/factorize.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "backends/CUDA/mpcg_debug.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "ichol/preconditioner.hpp"
#include "ichol/subdomain_preconditioner_gpu.hpp"
#include "unit/test_utils.hpp"

#ifndef ICHOL_HAVE_CUDSS
#define ICHOL_HAVE_CUDSS 0
#endif

namespace
{
constexpr int kProblemN = 32;
constexpr int kSubdomainExtent = 16;
constexpr double kExactResidualTol = 1e-10;
constexpr double kExactSolutionTol = 1e-10;
constexpr double kIcOperatorResidualTol = 1e-12;
constexpr double kIcSolutionTol = 1e-12;
constexpr double kSpaiMaxExactResidual = 0.5;
constexpr double kSpaiMaxSolutionError = 0.5;

void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

void cusparse_check(cusparseStatus_t status, const char *what)
{
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed with cuSPARSE status " + std::to_string(static_cast<int>(status)));
}

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t size)
        : size_(size)
    {
        if (size_ > 0)
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&ptr_), size_ * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr)
            cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this == &other)
            return *this;
        if (ptr_ != nullptr)
            cudaFree(ptr_);
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }
    std::size_t size() const { return size_; }

    void copy_from_host(const std::vector<T> &host)
    {
        if (host.size() != size_)
            throw std::runtime_error("DeviceBuffer::copy_from_host size mismatch");
        if (size_ > 0)
            cuda_check(cudaMemcpy(ptr_, host.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
    }

    std::vector<T> copy_to_host() const
    {
        std::vector<T> host(size_);
        if (size_ > 0)
            cuda_check(cudaMemcpy(host.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
        return host;
    }

    void zero()
    {
        if (size_ > 0)
            cuda_check(cudaMemset(ptr_, 0, size_ * sizeof(T)), "cudaMemset");
    }

private:
    T *ptr_ = nullptr;
    std::size_t size_ = 0;
};

struct SubdomainContextDeleter
{
    void operator()(ichol::precond::SubdomainPreconditionerContext *ctx) const
    {
        if (ctx != nullptr)
            ichol::precond::destroy_subdomain_preconditioner_context(ctx);
    }
};

using SubdomainContextPtr = std::unique_ptr<ichol::precond::SubdomainPreconditionerContext, SubdomainContextDeleter>;

struct ProblemSetup
{
    ichol::matrix::CsrMatrix<double> A;
    std::vector<double> rhs;
    ichol::precond::GridShape global_shape{kProblemN, kProblemN, kProblemN};
    ichol::precond::SubdomainSize subdomain_size{kSubdomainExtent, kSubdomainExtent, kSubdomainExtent};
    std::vector<ichol::precond::SubdomainRegion> regions;
};

struct CsrFactor
{
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values;
};

struct LegacyIc0Factorization
{
    std::vector<int> row_ptr_l;
    std::vector<int> col_ind_l;
    std::vector<double> val_l;
    std::vector<int> row_ptr_lt;
    std::vector<int> col_ind_lt;
    std::vector<double> val_lt;
};

ichol::matrix::CsrMatrix<double> exact_local_factor(ichol::matrix::CsrMatrix<double> A_sub);

void apply_unit_col_prescaling_system(ichol::matrix::CsrMatrix<double> &A,
                                      std::vector<double> &b)
{
    const auto D = ichol::numeric::scale_diag_sqrt(A);
    ichol::numeric::apply_prescaling(A, D);
    ichol::numeric::apply_rhs_prescaling(b, D);
}

bool cuda_device_available()
{
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice)
        return false;
    cuda_check(err, "cudaGetDeviceCount");
    return count > 0;
}

int flatten_local_3d(int x, int y, int z, int w, int h)
{
    return x + y * w + z * (w * h);
}

void unflatten_global_3d(int gi, int gw, int gh, int &x, int &y, int &z)
{
    const int plane = gw * gh;
    z = gi / plane;
    const int rem = gi - z * plane;
    y = rem / gw;
    x = rem - y * gw;
}

int local_from_global(int gj,
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

std::vector<int> build_subdomain_gidx_host(const ichol::precond::GridShape &global,
                                           const ichol::precond::SubdomainRegion &reg)
{
    const int lw = reg.x1 - reg.x0;
    const int lh = reg.y1 - reg.y0;
    const int ld = reg.z1 - reg.z0;
    const int nsub = lw * lh * ld;
    const int plane = lw * lh;

    std::vector<int> gidx(static_cast<std::size_t>(nsub), 0);
    for (int li = 0; li < nsub; ++li)
    {
        const int lz = li / plane;
        const int rem = li - lz * plane;
        const int ly = rem / lw;
        const int lx = rem - ly * lw;
        gidx[static_cast<std::size_t>(li)] =
            (reg.x0 + lx) + (reg.y0 + ly) * global.w + (reg.z0 + lz) * (global.w * global.h);
    }
    return gidx;
}

ProblemSetup build_problem()
{
    ProblemSetup setup;
    setup.A = ichol::io::gen_3dpoi<double>(kProblemN);
    setup.rhs.resize(static_cast<std::size_t>(setup.A.num_rows), 0.0);
    for (int i = 0; i < setup.A.num_rows; ++i)
        setup.rhs[static_cast<std::size_t>(i)] = 1.0 + static_cast<double>(i % 7);
    apply_unit_col_prescaling_system(setup.A, setup.rhs);
    setup.regions = ichol::precond::partition_subdomains(setup.global_shape, setup.subdomain_size);
    if (setup.regions.empty())
        throw std::runtime_error("build_problem: partition_subdomains returned no regions");
    return setup;
}

std::vector<SubdomainContextPtr> create_contexts(
    const ProblemSetup &setup,
    ichol::precond::SubdomainPreconditionerKind kind)
{
    ichol::precond::SubdomainPreconditionerOptions options;
    options.kind = kind;
    options.ic_level_k = 0;
    options.precision = ichol::solver::ComputePrecision::FP64;

    auto raw_contexts = ichol::precond::create_subdomain_preconditioner_contexts_parallel(
        setup.A, setup.global_shape, setup.regions, options);

    std::vector<SubdomainContextPtr> contexts;
    contexts.reserve(raw_contexts.size());
    for (auto *ctx : raw_contexts)
        contexts.emplace_back(ctx);
    return contexts;
}

std::vector<double> gather_subvector(const std::vector<double> &global,
                                     const std::vector<int> &gidx)
{
    std::vector<double> local(gidx.size(), 0.0);
    for (std::size_t i = 0; i < gidx.size(); ++i)
        local[i] = global[static_cast<std::size_t>(gidx[i])];
    return local;
}

std::vector<double> scatter_subvector(int n,
                                      const std::vector<int> &gidx,
                                      const std::vector<double> &local)
{
    std::vector<double> global(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < gidx.size(); ++i)
        global[static_cast<std::size_t>(gidx[i])] = local[i];
    return global;
}

ichol::matrix::CsrMatrix<double> extract_lower_subdomain_csr(
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
    sub.row_ptr.resize(static_cast<std::size_t>(nsub) + 1, 0);

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
        row_entries.reserve(static_cast<std::size_t>(A.row_ptr[gi + 1] - A.row_ptr[gi]));
        for (int kk = A.row_ptr[gi]; kk < A.row_ptr[gi + 1]; ++kk)
        {
            const int lj = local_from_global(A.col_ind[kk], global, reg, lw, lh);
            if (lj < 0 || lj > li)
                continue;
            row_entries.push_back({lj, A.values[kk]});
        }

        std::sort(row_entries.begin(), row_entries.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        int diag_pos = -1;
        for (int i = 0; i < static_cast<int>(row_entries.size()); ++i)
        {
            if (row_entries[static_cast<std::size_t>(i)].first == li)
            {
                diag_pos = i;
                break;
            }
        }
        if (diag_pos < 0)
            throw std::runtime_error("extract_lower_subdomain_csr: missing diagonal");

        for (int i = 0; i < static_cast<int>(row_entries.size()); ++i)
        {
            if (i == diag_pos)
                continue;
            cols.push_back(row_entries[static_cast<std::size_t>(i)].first);
            vals.push_back(row_entries[static_cast<std::size_t>(i)].second);
        }
        cols.push_back(li);
        vals.push_back(row_entries[static_cast<std::size_t>(diag_pos)].second);
        sub.row_ptr[static_cast<std::size_t>(li) + 1] = static_cast<int>(cols.size());
    }

    sub.col_ind = std::move(cols);
    sub.values = std::move(vals);
    sub.nnz = static_cast<int>(sub.values.size());
    return sub;
}

CsrFactor transpose_csr(int n,
                        const std::vector<int> &row_ptr,
                        const std::vector<int> &col_ind,
                        const std::vector<double> &values)
{
    CsrFactor transposed;
    const int nnz = static_cast<int>(values.size());
    transposed.row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    transposed.col_ind.assign(static_cast<std::size_t>(nnz), 0);
    transposed.values.assign(static_cast<std::size_t>(nnz), 0.0);

    for (int p = 0; p < nnz; ++p)
        ++transposed.row_ptr[static_cast<std::size_t>(col_ind[p]) + 1];

    for (int i = 0; i < n; ++i)
        transposed.row_ptr[static_cast<std::size_t>(i) + 1] += transposed.row_ptr[static_cast<std::size_t>(i)];

    std::vector<int> next = transposed.row_ptr;
    for (int i = 0; i < n; ++i)
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
        {
            const int dst = next[static_cast<std::size_t>(col_ind[p])]++;
            transposed.col_ind[static_cast<std::size_t>(dst)] = i;
            transposed.values[static_cast<std::size_t>(dst)] = values[static_cast<std::size_t>(p)];
        }
    }
    return transposed;
}

double l2_norm(const std::vector<double> &v)
{
    return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
}

double relative_error(const std::vector<double> &actual,
                      const std::vector<double> &expected)
{
    if (actual.size() != expected.size())
        throw std::runtime_error("relative_error: size mismatch");

    std::vector<double> diff(actual.size(), 0.0);
    for (std::size_t i = 0; i < actual.size(); ++i)
        diff[i] = actual[i] - expected[i];

    const double denom = l2_norm(expected);
    const double numer = l2_norm(diff);
    return numer / (denom > 0.0 ? denom : 1.0);
}

std::vector<double> symm_lower_csr_matvec(const ichol::matrix::CsrMatrix<double> &A,
                                          const std::vector<double> &x)
{
    if (x.size() != static_cast<std::size_t>(A.num_cols))
        throw std::runtime_error("symm_lower_csr_matvec: size mismatch");

    std::vector<double> y(static_cast<std::size_t>(A.num_rows), 0.0);
    for (int i = 0; i < A.num_rows; ++i)
    {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
        {
            const int j = A.col_ind[p];
            const double aij = A.values[static_cast<std::size_t>(p)];
            y[static_cast<std::size_t>(i)] += aij * x[static_cast<std::size_t>(j)];
            if (j != i)
                y[static_cast<std::size_t>(j)] += aij * x[static_cast<std::size_t>(i)];
        }
    }
    return y;
}

std::vector<double> csr_matvec(const std::vector<int> &row_ptr,
                               const std::vector<int> &col_ind,
                               const std::vector<double> &values,
                               const std::vector<double> &x)
{
    const int n = static_cast<int>(row_ptr.size()) - 1;
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;
        for (int p = row_ptr[static_cast<std::size_t>(i)]; p < row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
            sum += values[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(col_ind[static_cast<std::size_t>(p)])];
        y[static_cast<std::size_t>(i)] = sum;
    }
    return y;
}

std::vector<double> solve_lower_triangular(const std::vector<int> &row_ptr,
                                           const std::vector<int> &col_ind,
                                           const std::vector<double> &values,
                                           const std::vector<double> &rhs)
{
    const int n = static_cast<int>(row_ptr.size()) - 1;
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
    {
        double sum = rhs[static_cast<std::size_t>(i)];
        const int row_begin = row_ptr[static_cast<std::size_t>(i)];
        const int row_end = row_ptr[static_cast<std::size_t>(i) + 1];
        if (row_begin >= row_end)
            throw std::runtime_error("solve_lower_triangular: empty row");

        const int diag_pos = row_end - 1;
        if (col_ind[static_cast<std::size_t>(diag_pos)] != i)
            throw std::runtime_error("solve_lower_triangular: diagonal not stored last");
        for (int p = row_begin; p < diag_pos; ++p)
            sum -= values[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(col_ind[static_cast<std::size_t>(p)])];

        const double diag = values[static_cast<std::size_t>(diag_pos)];
        if (diag == 0.0)
            throw std::runtime_error("solve_lower_triangular: zero diagonal");
        x[static_cast<std::size_t>(i)] = sum / diag;
    }
    return x;
}

std::vector<double> solve_upper_triangular(const std::vector<int> &row_ptr,
                                           const std::vector<int> &col_ind,
                                           const std::vector<double> &values,
                                           const std::vector<double> &rhs)
{
    const int n = static_cast<int>(row_ptr.size()) - 1;
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i)
    {
        double sum = rhs[static_cast<std::size_t>(i)];
        double diag = 0.0;
        bool have_diag = false;
        for (int p = row_ptr[static_cast<std::size_t>(i)]; p < row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
        {
            const int j = col_ind[static_cast<std::size_t>(p)];
            const double val = values[static_cast<std::size_t>(p)];
            if (j == i)
            {
                diag = val;
                have_diag = true;
            }
            else
            {
                if (j <= i)
                    throw std::runtime_error("solve_upper_triangular: encountered non-upper entry");
                sum -= val * x[static_cast<std::size_t>(j)];
            }
        }
        if (!have_diag)
            throw std::runtime_error("solve_upper_triangular: missing diagonal");
        if (diag == 0.0)
            throw std::runtime_error("solve_upper_triangular: zero diagonal");
        x[static_cast<std::size_t>(i)] = sum / diag;
    }
    return x;
}

std::vector<double> solve_with_factor(const std::vector<int> &row_ptr_l,
                                      const std::vector<int> &col_ind_l,
                                      const std::vector<double> &val_l,
                                      const std::vector<int> &row_ptr_lt,
                                      const std::vector<int> &col_ind_lt,
                                      const std::vector<double> &val_lt,
                                      const std::vector<double> &rhs)
{
    const auto y = solve_lower_triangular(row_ptr_l, col_ind_l, val_l, rhs);
    return solve_upper_triangular(row_ptr_lt, col_ind_lt, val_lt, y);
}

std::vector<double> solve_with_factor(const ichol::matrix::CsrMatrix<double> &L,
                                      const std::vector<double> &rhs)
{
    const auto Lt = transpose_csr(L.num_rows, L.row_ptr, L.col_ind, L.values);
    return solve_with_factor(L.row_ptr, L.col_ind, L.values, Lt.row_ptr, Lt.col_ind, Lt.values, rhs);
}

std::vector<double> extract_diagonal(const ichol::matrix::CsrMatrix<double> &A)
{
    std::vector<double> diag(static_cast<std::size_t>(A.num_rows), 0.0);
    for (int i = 0; i < A.num_rows; ++i)
    {
        const int row_begin = A.row_ptr[static_cast<std::size_t>(i)];
        const int row_end = A.row_ptr[static_cast<std::size_t>(i) + 1];
        if (row_begin >= row_end)
            throw std::runtime_error("extract_diagonal: empty row");

        const int diag_pos = row_end - 1;
        if (A.col_ind[static_cast<std::size_t>(diag_pos)] != i)
            throw std::runtime_error("extract_diagonal: diagonal not stored last");

        diag[static_cast<std::size_t>(i)] = A.values[static_cast<std::size_t>(diag_pos)];
    }
    return diag;
}

std::vector<double> apply_jacobi(const std::vector<double> &diag,
                                 const std::vector<double> &rhs)
{
    if (diag.size() != rhs.size())
        throw std::runtime_error("apply_jacobi: size mismatch");

    std::vector<double> x(rhs.size(), 0.0);
    for (std::size_t i = 0; i < rhs.size(); ++i)
    {
        if (diag[i] == 0.0)
            throw std::runtime_error("apply_jacobi: zero diagonal");
        x[i] = rhs[i] / diag[i];
    }
    return x;
}

double relative_residual(const std::vector<double> &lhs,
                         const std::vector<double> &rhs)
{
    if (lhs.size() != rhs.size())
        throw std::runtime_error("relative_residual: size mismatch");

    std::vector<double> diff(lhs.size(), 0.0);
    for (std::size_t i = 0; i < lhs.size(); ++i)
        diff[i] = lhs[i] - rhs[i];

    const double denom = l2_norm(rhs);
    return l2_norm(diff) / (denom > 0.0 ? denom : 1.0);
}

void assert_local_matrix_invariants(const ichol::matrix::CsrMatrix<double> &A_sub)
{
    ASSERT_EQ(A_sub.num_rows, A_sub.num_cols);
    ASSERT_EQ(A_sub.row_ptr.size(), static_cast<std::size_t>(A_sub.num_rows) + 1);
    ASSERT_EQ(A_sub.nnz, static_cast<int>(A_sub.values.size()));
    ASSERT_EQ(A_sub.nnz, static_cast<int>(A_sub.col_ind.size()));
    ichol::testutil::assert_lower_only_csr(A_sub.row_ptr, A_sub.col_ind);
    ichol::testutil::assert_diag_last_csr(A_sub.row_ptr, A_sub.col_ind);
    ichol::testutil::assert_diag_positive_csr(A_sub.row_ptr, A_sub.col_ind, A_sub.values);
    ichol::testutil::assert_cols_sorted_unique(A_sub.row_ptr, A_sub.col_ind);

    for (int i = 0; i < A_sub.num_rows; ++i)
    {
        for (int p = A_sub.row_ptr[static_cast<std::size_t>(i)]; p < A_sub.row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
        {
            const int j = A_sub.col_ind[static_cast<std::size_t>(p)];
            ASSERT_GE(j, 0);
            ASSERT_LT(j, A_sub.num_cols);
        }
    }
}

void assert_factor_invariants(const std::vector<int> &row_ptr,
                              const std::vector<int> &col_ind,
                              const std::vector<double> &values,
                              bool expect_lower)
{
    const int n = static_cast<int>(row_ptr.size()) - 1;
    ASSERT_GE(n, 0);
    ASSERT_EQ(col_ind.size(), values.size());
    ASSERT_EQ(row_ptr.back(), static_cast<int>(values.size()));
    for (int i = 0; i < n; ++i)
    {
        ASSERT_LE(row_ptr[static_cast<std::size_t>(i)], row_ptr[static_cast<std::size_t>(i) + 1]);
        bool have_diag = false;
        for (int p = row_ptr[static_cast<std::size_t>(i)]; p < row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
        {
            const int j = col_ind[static_cast<std::size_t>(p)];
            ASSERT_GE(j, 0);
            ASSERT_LT(j, n);
            ASSERT_TRUE(std::isfinite(values[static_cast<std::size_t>(p)]));
            if (expect_lower)
                ASSERT_LE(j, i);
            else
                ASSERT_GE(j, i);
            have_diag = have_diag || (j == i);
        }
        ASSERT_TRUE(have_diag);
    }
}

ichol::matrix::CsrMatrix<double> exact_local_factor(ichol::matrix::CsrMatrix<double> A_sub)
{
    ichol::SymbolicOptions sym_opts;
    sym_opts.ordering = ichol::Ordering::Identity;
    sym_opts.level_k = -1;

    auto sym_plan = ichol::symbolic::ic_analyze(A_sub, sym_opts);

    ichol::IncompleteCholeskyOptions ic_opts;
    ic_opts.scaling = ichol::Scaling::None;
    ic_opts.pivot_shift_strategy = ichol::PivotShiftStrategy::None;
    ic_opts.algorithm = ichol::FactorizationAlgorithm::ICKDT;
    ic_opts.max_restarts = 1;
    ic_opts.verbose = false;
    ic_opts.lfil = A_sub.num_rows;
    ic_opts.drop_tol = 0.0;

    ichol::numeric::NumericPlan num_plan;
    return ichol::numeric::incomplete_cholesky_preconditioner<double>(A_sub, sym_plan, num_plan, ic_opts);
}

LegacyIc0Factorization factorize_subdomain_ic0_reference(const ichol::matrix::CsrMatrix<double> &A_sub)
{
    LegacyIc0Factorization out;
    out.row_ptr_l = A_sub.row_ptr;
    out.col_ind_l = A_sub.col_ind;
    out.val_l = A_sub.values;

    cusparseHandle_t handle = nullptr;
    cusparseMatDescr_t descr = nullptr;
    csric02Info_t info = nullptr;
    int *d_row_ptr = nullptr;
    int *d_col_ind = nullptr;
    double *d_val = nullptr;
    void *d_buf = nullptr;

    try
    {
        const int n = A_sub.num_rows;
        const int nnz = A_sub.nnz;

        cusparse_check(cusparseCreate(&handle), "cusparseCreate");
        cusparse_check(cusparseCreateMatDescr(&descr), "cusparseCreateMatDescr");
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparse_check(cusparseCreateCsric02Info(&info), "cusparseCreateCsric02Info");

        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_row_ptr), static_cast<std::size_t>(n + 1) * sizeof(int)), "cudaMalloc(d_row_ptr)");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_col_ind), static_cast<std::size_t>(nnz) * sizeof(int)), "cudaMalloc(d_col_ind)");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_val), static_cast<std::size_t>(nnz) * sizeof(double)), "cudaMalloc(d_val)");
        cuda_check(cudaMemcpy(d_row_ptr, out.row_ptr_l.data(), static_cast<std::size_t>(n + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_row_ptr)");
        cuda_check(cudaMemcpy(d_col_ind, out.col_ind_l.data(), static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_col_ind)");
        cuda_check(cudaMemcpy(d_val, out.val_l.data(), static_cast<std::size_t>(nnz) * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(d_val)");

        int buffer_size = 0;
        cusparse_check(cusparseDcsric02_bufferSize(handle, n, nnz, descr, d_val, d_row_ptr, d_col_ind, info, &buffer_size),
                       "cusparseDcsric02_bufferSize");
        if (buffer_size > 0)
            cuda_check(cudaMalloc(&d_buf, static_cast<std::size_t>(buffer_size)), "cudaMalloc(d_buf)");

        cusparse_check(cusparseDcsric02_analysis(handle, n, nnz, descr, d_val, d_row_ptr, d_col_ind, info,
                                                 CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_buf),
                       "cusparseDcsric02_analysis");

        int pivot = -1;
        const cusparseStatus_t structural_status = cusparseXcsric02_zeroPivot(handle, info, &pivot);
        if (structural_status == CUSPARSE_STATUS_ZERO_PIVOT)
            throw std::runtime_error("factorize_subdomain_ic0_reference: structural zero pivot at row " + std::to_string(pivot));
        cusparse_check(structural_status, "cusparseXcsric02_zeroPivot(structural)");

        cusparse_check(cusparseDcsric02(handle, n, nnz, descr, d_val, d_row_ptr, d_col_ind, info,
                                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_buf),
                       "cusparseDcsric02");
        const cusparseStatus_t numeric_status = cusparseXcsric02_zeroPivot(handle, info, &pivot);
        if (numeric_status == CUSPARSE_STATUS_ZERO_PIVOT)
            throw std::runtime_error("factorize_subdomain_ic0_reference: numeric zero pivot at row " + std::to_string(pivot));
        cusparse_check(numeric_status, "cusparseXcsric02_zeroPivot(numeric)");

        cuda_check(cudaMemcpy(out.val_l.data(), d_val, static_cast<std::size_t>(nnz) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(val_l)");

        const auto Lt = transpose_csr(n, out.row_ptr_l, out.col_ind_l, out.val_l);
        out.row_ptr_lt = Lt.row_ptr;
        out.col_ind_lt = Lt.col_ind;
        out.val_lt = Lt.values;
    }
    catch (...)
    {
        cudaFree(d_buf);
        cudaFree(d_val);
        cudaFree(d_col_ind);
        cudaFree(d_row_ptr);
        if (info != nullptr)
            cusparseDestroyCsric02Info(info);
        if (descr != nullptr)
            cusparseDestroyMatDescr(descr);
        if (handle != nullptr)
            cusparseDestroy(handle);
        throw;
    }

    cudaFree(d_buf);
    cudaFree(d_val);
    cudaFree(d_col_ind);
    cudaFree(d_row_ptr);
    cusparseDestroyCsric02Info(info);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    return out;
}

std::vector<double> apply_subdomain_context(
    ichol::precond::SubdomainPreconditionerContext *ctx,
    const std::vector<double> &rhs,
    int n)
{
    DeviceBuffer<double> d_rhs(rhs.size());
    DeviceBuffer<double> d_z(rhs.size());
    d_rhs.copy_from_host(rhs);
    d_z.zero();
    ichol::precond::apply_subdomain_preconditioner(
        ctx,
        d_rhs.data(),
        d_z.data(),
        n,
        ichol::solver::ComputePrecision::FP64,
        0);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return d_z.copy_to_host();
}

void assert_zero_outside_support(const std::vector<double> &global,
                                 const std::vector<int> &gidx)
{
    std::vector<unsigned char> in_support(global.size(), 0);
    for (int gi : gidx)
        in_support[static_cast<std::size_t>(gi)] = 1;

    for (std::size_t i = 0; i < global.size(); ++i)
    {
        if (in_support[i] == 0)
            ASSERT_DOUBLE_EQ(global[i], 0.0);
    }
}

std::vector<ichol::precond::PrecondApply> make_preconds(
    const std::vector<SubdomainContextPtr> &contexts,
    const std::vector<std::size_t> &indices = {})
{
    std::vector<ichol::precond::PrecondApply> preconds;
    if (indices.empty())
    {
        preconds.reserve(contexts.size());
        for (const auto &ctx : contexts)
            preconds.push_back({&ichol::precond::apply_subdomain_exact_spsv, ctx.get()});
        return preconds;
    }

    preconds.reserve(indices.size());
    for (const std::size_t index : indices)
        preconds.push_back({&ichol::precond::apply_subdomain_exact_spsv, contexts[index].get()});
    return preconds;
}

std::vector<double> extract_column(const std::vector<double> &columns,
                                   int n,
                                   std::size_t column_index)
{
    const auto begin = columns.begin() + static_cast<std::ptrdiff_t>(column_index) * n;
    return std::vector<double>(begin, begin + n);
}

double outside_support_norm(const std::vector<double> &global,
                            const std::vector<int> &gidx)
{
    std::vector<unsigned char> in_support(global.size(), 0);
    for (const int gi : gidx)
        in_support[static_cast<std::size_t>(gi)] = 1;

    double accum = 0.0;
    for (std::size_t i = 0; i < global.size(); ++i)
    {
        if (in_support[i] == 0)
            accum += global[i] * global[i];
    }
    return std::sqrt(accum);
}

std::vector<double> build_reference_exact_global_column(
    const ProblemSetup &setup,
    const ichol::precond::SubdomainRegion &reg)
{
    auto A_sub = extract_lower_subdomain_csr(setup.A, setup.global_shape, reg);
    const auto L_exact = exact_local_factor(A_sub);
    const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
    const auto b_local = gather_subvector(setup.rhs, gidx);
    const auto x_ref = solve_with_factor(L_exact, b_local);
    return scatter_subvector(setup.A.num_rows, gidx, x_ref);
}
} // namespace

TEST(DDPrecond, ExactSubdomainApplyCorrectness)
{
    if (!cuda_device_available())
        GTEST_SKIP() << "CUDA device unavailable";

#if !ICHOL_HAVE_CUDSS
    GTEST_SKIP() << "Exact subdomain backend requires cuDSS";
#else
    const ProblemSetup setup = build_problem();
    ASSERT_EQ(setup.A.num_rows, kProblemN * kProblemN * kProblemN);
    ASSERT_EQ(setup.regions.size(), 8u);

    const auto contexts = create_contexts(setup, ichol::precond::SubdomainPreconditionerKind::ExactCholesky);
    ASSERT_EQ(contexts.size(), setup.regions.size());

    double max_rel_residual = 0.0;
    double max_rel_solution_diff = 0.0;
    double max_rel_scatter_diff = 0.0;

    for (std::size_t subdomain = 0; subdomain < setup.regions.size(); ++subdomain)
    {
        SCOPED_TRACE("subdomain=" + std::to_string(subdomain));

        const auto &reg = setup.regions[subdomain];
        const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
        auto A_sub = extract_lower_subdomain_csr(setup.A, setup.global_shape, reg);
        assert_local_matrix_invariants(A_sub);

        const auto b_local = gather_subvector(setup.rhs, gidx);
        const auto z_global = apply_subdomain_context(contexts[subdomain].get(), setup.rhs, setup.A.num_rows);
        const auto x_backend = gather_subvector(z_global, gidx);

        for (double value : x_backend)
            ASSERT_TRUE(std::isfinite(value));

        const auto L_exact = exact_local_factor(A_sub);
        assert_factor_invariants(L_exact.row_ptr, L_exact.col_ind, L_exact.values, true);
        const auto x_ref = solve_with_factor(L_exact, b_local);
        const auto expected_global = scatter_subvector(setup.A.num_rows, gidx, x_ref);

        const auto Ax = symm_lower_csr_matvec(A_sub, x_backend);
        const double rel_residual = relative_residual(Ax, b_local);
        const double rel_solution_diff = relative_error(x_backend, x_ref);
        const double rel_scatter_diff = relative_error(z_global, expected_global);

        max_rel_residual = std::max(max_rel_residual, rel_residual);
        max_rel_solution_diff = std::max(max_rel_solution_diff, rel_solution_diff);
        max_rel_scatter_diff = std::max(max_rel_scatter_diff, rel_scatter_diff);

        assert_zero_outside_support(z_global, gidx);
        EXPECT_LT(rel_residual, kExactResidualTol);
        EXPECT_LT(rel_solution_diff, kExactSolutionTol);
        EXPECT_LT(rel_scatter_diff, kExactSolutionTol);
    }

    std::cout << "[DDPrecond.Exact] max_rel_residual=" << max_rel_residual
              << " max_rel_solution_diff=" << max_rel_solution_diff
              << " max_rel_scatter_diff=" << max_rel_scatter_diff << "\n";
#endif
}

TEST(DDPrecond, CuDSSExactColumnsMatchReferenceBeforeOrthogonalization)
{
    if (!cuda_device_available())
        GTEST_SKIP() << "CUDA device unavailable";

#if !ICHOL_HAVE_CUDSS
    GTEST_SKIP() << "Exact subdomain backend requires cuDSS";
#else
    const ProblemSetup setup = build_problem();
    const auto contexts = create_contexts(setup, ichol::precond::SubdomainPreconditionerKind::ExactCholesky);
    const auto preconds = make_preconds(contexts);

    ichol::solver::debug::ZnewBuildOptions options;
    options.prec_precond = ichol::solver::ComputePrecision::FP64;
    const auto znew = ichol::solver::debug::build_mpcg_znew_columns(preconds, setup.rhs, options);

    double max_rel_column_error = 0.0;
    double max_inside_support_error = 0.0;
    double max_outside_support_norm = 0.0;

    for (std::size_t subdomain = 0; subdomain < setup.regions.size(); ++subdomain)
    {
        SCOPED_TRACE("subdomain=" + std::to_string(subdomain));
        const auto gidx = build_subdomain_gidx_host(setup.global_shape, setup.regions[subdomain]);
        const auto actual = extract_column(znew, setup.A.num_rows, subdomain);
        const auto expected = build_reference_exact_global_column(setup, setup.regions[subdomain]);
        const auto actual_local = gather_subvector(actual, gidx);
        const auto expected_local = gather_subvector(expected, gidx);

        const double rel_column_error = relative_error(actual, expected);
        const double inside_support_error = relative_error(actual_local, expected_local);
        const double support_leak_norm = outside_support_norm(actual, gidx);

        max_rel_column_error = std::max(max_rel_column_error, rel_column_error);
        max_inside_support_error = std::max(max_inside_support_error, inside_support_error);
        max_outside_support_norm = std::max(max_outside_support_norm, support_leak_norm);

        EXPECT_LT(rel_column_error, kExactSolutionTol);
        EXPECT_LT(inside_support_error, kExactSolutionTol);
        EXPECT_LT(support_leak_norm, 1e-14);
    }

    std::cout << "[DDPrecond.CuDSSColumns] max_rel_column_error=" << max_rel_column_error
              << " max_inside_support_error=" << max_inside_support_error
              << " max_outside_support_norm=" << max_outside_support_norm << "\n";
#endif
}

TEST(DDPrecond, CuDSSExactColumnWritesDoNotOverwriteOtherColumns)
{
    if (!cuda_device_available())
        GTEST_SKIP() << "CUDA device unavailable";

#if !ICHOL_HAVE_CUDSS
    GTEST_SKIP() << "Exact subdomain backend requires cuDSS";
#else
    const ProblemSetup setup = build_problem();
    const auto contexts = create_contexts(setup, ichol::precond::SubdomainPreconditionerKind::ExactCholesky);

    ichol::solver::debug::ZnewBuildOptions options;
    options.prec_precond = ichol::solver::ComputePrecision::FP64;

    const auto z0 = ichol::solver::debug::build_mpcg_znew_columns(
        make_preconds(contexts, {0}), setup.rhs, options);
    const auto z01 = ichol::solver::debug::build_mpcg_znew_columns(
        make_preconds(contexts, {0, 1}), setup.rhs, options);
    const auto z012 = ichol::solver::debug::build_mpcg_znew_columns(
        make_preconds(contexts, {0, 1, 2}), setup.rhs, options);

    const auto col0_alone = extract_column(z0, setup.A.num_rows, 0);
    const auto col0_pair = extract_column(z01, setup.A.num_rows, 0);
    const auto col1_pair = extract_column(z01, setup.A.num_rows, 1);
    const auto col0_triplet = extract_column(z012, setup.A.num_rows, 0);
    const auto col1_triplet = extract_column(z012, setup.A.num_rows, 1);

    const double rel_col0_pair = relative_error(col0_pair, col0_alone);
    const double rel_col0_triplet = relative_error(col0_triplet, col0_alone);
    const double rel_col1_triplet = relative_error(col1_triplet, col1_pair);

    EXPECT_LT(rel_col0_pair, kExactSolutionTol);
    EXPECT_LT(rel_col0_triplet, kExactSolutionTol);
    EXPECT_LT(rel_col1_triplet, kExactSolutionTol);

    std::cout << "[DDPrecond.CuDSSOverwrite] rel_col0_pair=" << rel_col0_pair
              << " rel_col0_triplet=" << rel_col0_triplet
              << " rel_col1_triplet=" << rel_col1_triplet << "\n";
#endif
}

TEST(DDPrecond, CuDSSExactSerialAndProductionZnewMatch)
{
    if (!cuda_device_available())
        GTEST_SKIP() << "CUDA device unavailable";

#if !ICHOL_HAVE_CUDSS
    GTEST_SKIP() << "Exact subdomain backend requires cuDSS";
#else
    const ProblemSetup setup = build_problem();
    const auto contexts = create_contexts(setup, ichol::precond::SubdomainPreconditionerKind::ExactCholesky);
    const auto preconds = make_preconds(contexts);

    ichol::solver::debug::ZnewBuildOptions production;
    production.prec_precond = ichol::solver::ComputePrecision::FP64;

    ichol::solver::debug::ZnewBuildOptions serial = production;
    serial.serial = true;
    serial.sync_after_each_apply = true;

    const auto z_serial = ichol::solver::debug::build_mpcg_znew_columns(preconds, setup.rhs, serial);
    const auto z_parallel = ichol::solver::debug::build_mpcg_znew_columns(preconds, setup.rhs, production);

    double max_rel_diff = 0.0;
    double max_serial_ref = 0.0;
    double max_parallel_ref = 0.0;
    for (std::size_t subdomain = 0; subdomain < setup.regions.size(); ++subdomain)
    {
        const auto parallel_col = extract_column(z_parallel, setup.A.num_rows, subdomain);
        const auto serial_col = extract_column(z_serial, setup.A.num_rows, subdomain);
        const auto reference_col = build_reference_exact_global_column(setup, setup.regions[subdomain]);
        const double rel_diff = relative_error(parallel_col, serial_col);
        const double rel_serial_ref = relative_error(serial_col, reference_col);
        const double rel_parallel_ref = relative_error(parallel_col, reference_col);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
        max_serial_ref = std::max(max_serial_ref, rel_serial_ref);
        max_parallel_ref = std::max(max_parallel_ref, rel_parallel_ref);
        EXPECT_LT(rel_diff, kExactSolutionTol);
    }

    std::cout << "[DDPrecond.CuDSSSerialVsProduction] max_rel_diff=" << max_rel_diff
              << " max_serial_ref=" << max_serial_ref
              << " max_parallel_ref=" << max_parallel_ref << "\n";
#endif
}

TEST(DDPrecond, ICSubdomainApplyCorrectness)
{
    if (!cuda_device_available())
        GTEST_SKIP() << "CUDA device unavailable";

    const ProblemSetup setup = build_problem();
    ASSERT_EQ(setup.A.num_rows, kProblemN * kProblemN * kProblemN);
    ASSERT_EQ(setup.regions.size(), 8u);

    std::vector<SubdomainContextPtr> contexts;
    try
    {
        contexts = create_contexts(setup, ichol::precond::SubdomainPreconditionerKind::IncompleteCholesky);
    }
    catch (const std::exception &e)
    {
        FAIL() << "IC context creation failed: " << e.what();
    }
    ASSERT_EQ(contexts.size(), setup.regions.size());

    double max_rel_prec_residual = 0.0;
    double max_rel_solution_diff = 0.0;
    double max_weak_exact_residual = 0.0;

    for (std::size_t subdomain = 0; subdomain < setup.regions.size(); ++subdomain)
    {
        SCOPED_TRACE("subdomain=" + std::to_string(subdomain));

        const auto &reg = setup.regions[subdomain];
        const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
        const auto A_sub = extract_lower_subdomain_csr(setup.A, setup.global_shape, reg);
        assert_local_matrix_invariants(A_sub);

        LegacyIc0Factorization factor;
        try
        {
            factor = factorize_subdomain_ic0_reference(A_sub);
        }
        catch (const std::exception &e)
        {
            FAIL() << "Reference IC(0) reconstruction failed for subdomain " << subdomain << ": " << e.what();
        }
        ASSERT_EQ(factor.row_ptr_l, A_sub.row_ptr);
        ASSERT_EQ(factor.col_ind_l, A_sub.col_ind);
        ASSERT_EQ(factor.row_ptr_l.size(), static_cast<std::size_t>(A_sub.num_rows) + 1);
        ASSERT_EQ(factor.row_ptr_lt.size(), static_cast<std::size_t>(A_sub.num_rows) + 1);
        assert_factor_invariants(factor.row_ptr_l, factor.col_ind_l, factor.val_l, true);
        assert_factor_invariants(factor.row_ptr_lt, factor.col_ind_lt, factor.val_lt, false);

        const auto b_local = gather_subvector(setup.rhs, gidx);
        std::vector<double> z_global;
        try
        {
            z_global = apply_subdomain_context(contexts[subdomain].get(), setup.rhs, setup.A.num_rows);
        }
        catch (const std::exception &e)
        {
            FAIL() << "IC apply failed for subdomain " << subdomain << ": " << e.what();
        }
        const auto x_backend = gather_subvector(z_global, gidx);
        const auto x_ref = solve_with_factor(
            factor.row_ptr_l, factor.col_ind_l, factor.val_l,
            factor.row_ptr_lt, factor.col_ind_lt, factor.val_lt,
            b_local);

        for (double value : x_backend)
            ASSERT_TRUE(std::isfinite(value));

        const auto lt_x = csr_matvec(factor.row_ptr_lt, factor.col_ind_lt, factor.val_lt, x_backend);
        const auto llt_x = csr_matvec(factor.row_ptr_l, factor.col_ind_l, factor.val_l, lt_x);
        const auto weak_exact = symm_lower_csr_matvec(A_sub, x_backend);

        const double rel_prec_residual = relative_residual(llt_x, b_local);
        const double rel_solution_diff = relative_error(x_backend, x_ref);
        const double weak_exact_residual = relative_residual(weak_exact, b_local);

        max_rel_prec_residual = std::max(max_rel_prec_residual, rel_prec_residual);
        max_rel_solution_diff = std::max(max_rel_solution_diff, rel_solution_diff);
        max_weak_exact_residual = std::max(max_weak_exact_residual, weak_exact_residual);

        assert_zero_outside_support(z_global, gidx);
        EXPECT_LT(rel_prec_residual, kIcOperatorResidualTol);
        EXPECT_LT(rel_solution_diff, kIcSolutionTol);
    }

    std::cout << "[DDPrecond.IC] max_rel_prec_residual=" << max_rel_prec_residual
              << " max_rel_solution_diff=" << max_rel_solution_diff
              << " max_weak_exact_residual=" << max_weak_exact_residual << "\n";
}

TEST(DDPrecond, SPAISubdomainApplyQuality)
{
    if (!cuda_device_available())
        GTEST_SKIP() << "CUDA device unavailable";

    const ProblemSetup setup = build_problem();
    ASSERT_EQ(setup.A.num_rows, kProblemN * kProblemN * kProblemN);
    ASSERT_EQ(setup.regions.size(), 8u);

    std::vector<SubdomainContextPtr> contexts;
    try
    {
        contexts = create_contexts(setup, ichol::precond::SubdomainPreconditionerKind::SPAI);
    }
    catch (const std::exception &e)
    {
        FAIL() << "SPAI context creation failed: " << e.what();
    }
    ASSERT_EQ(contexts.size(), setup.regions.size());

    double max_rel_exact_residual = 0.0;
    double max_rel_solution_diff = 0.0;
    double max_jacobi_rel_exact_residual = 0.0;
    double max_jacobi_rel_solution_diff = 0.0;

    for (std::size_t subdomain = 0; subdomain < setup.regions.size(); ++subdomain)
    {
        SCOPED_TRACE("subdomain=" + std::to_string(subdomain));

        const auto &reg = setup.regions[subdomain];
        const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
        const auto A_sub = extract_lower_subdomain_csr(setup.A, setup.global_shape, reg);
        assert_local_matrix_invariants(A_sub);

        const auto b_local = gather_subvector(setup.rhs, gidx);
        const auto diag = extract_diagonal(A_sub);
        const auto jacobi_x = apply_jacobi(diag, b_local);

        std::vector<double> z_global;
        try
        {
            z_global = apply_subdomain_context(contexts[subdomain].get(), setup.rhs, setup.A.num_rows);
        }
        catch (const std::exception &e)
        {
            FAIL() << "SPAI apply failed for subdomain " << subdomain << ": " << e.what();
        }

        const auto x_spai = gather_subvector(z_global, gidx);
        for (double value : x_spai)
            ASSERT_TRUE(std::isfinite(value));

        const auto L_exact = exact_local_factor(A_sub);
        const auto x_ref = solve_with_factor(L_exact, b_local);

        const auto Ax_spai = symm_lower_csr_matvec(A_sub, x_spai);
        const auto Ax_jacobi = symm_lower_csr_matvec(A_sub, jacobi_x);

        const double rel_exact_residual = relative_residual(Ax_spai, b_local);
        const double rel_solution_diff = relative_error(x_spai, x_ref);
        const double jacobi_rel_exact_residual = relative_residual(Ax_jacobi, b_local);
        const double jacobi_rel_solution_diff = relative_error(jacobi_x, x_ref);

        max_rel_exact_residual = std::max(max_rel_exact_residual, rel_exact_residual);
        max_rel_solution_diff = std::max(max_rel_solution_diff, rel_solution_diff);
        max_jacobi_rel_exact_residual = std::max(max_jacobi_rel_exact_residual, jacobi_rel_exact_residual);
        max_jacobi_rel_solution_diff = std::max(max_jacobi_rel_solution_diff, jacobi_rel_solution_diff);

        assert_zero_outside_support(z_global, gidx);
        EXPECT_LT(rel_exact_residual, jacobi_rel_exact_residual);
        EXPECT_LT(rel_solution_diff, jacobi_rel_solution_diff);
        EXPECT_LT(rel_exact_residual, kSpaiMaxExactResidual);
        EXPECT_LT(rel_solution_diff, kSpaiMaxSolutionError);
    }

    std::cout << "[DDPrecond.SPAI] max_rel_exact_residual=" << max_rel_exact_residual
              << " max_rel_solution_diff=" << max_rel_solution_diff
              << " max_jacobi_rel_exact_residual=" << max_jacobi_rel_exact_residual
              << " max_jacobi_rel_solution_diff=" << max_jacobi_rel_solution_diff << "\n";
}
