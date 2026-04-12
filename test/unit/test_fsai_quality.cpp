#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "factor/numerical/factorize.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/subdomain_preconditioner_gpu.hpp"

namespace
{
    void cuda_check(cudaError_t err, const char *what)
    {
        if (err != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
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

        T *data() { return ptr_; }
        const T *data() const { return ptr_; }

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

    void apply_unit_col_prescaling_system(ichol::matrix::CsrMatrix<double> &A,
                                          std::vector<double> &b)
    {
        const auto D = ichol::numeric::scale_diag_sqrt(A);
        ichol::numeric::apply_prescaling(A, D);
        ichol::numeric::apply_rhs_prescaling(b, D);
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

    ichol::matrix::CsrMatrix<double> extract_full_subdomain_csr(
        const ichol::matrix::CsrMatrix<double> &A,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg)
    {
        const auto A_lower = extract_lower_subdomain_csr(A, global, reg);
        const int n = A_lower.num_rows;

        std::vector<std::vector<std::pair<int, double>>> rows(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            for (int p = A_lower.row_ptr[static_cast<std::size_t>(i)];
                 p < A_lower.row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
            {
                const int j = A_lower.col_ind[static_cast<std::size_t>(p)];
                const double v = A_lower.values[static_cast<std::size_t>(p)];
                rows[static_cast<std::size_t>(i)].push_back({j, v});
                if (j != i)
                    rows[static_cast<std::size_t>(j)].push_back({i, v});
            }
        }

        ichol::matrix::CsrMatrix<double> A_full;
        A_full.num_rows = n;
        A_full.num_cols = n;
        A_full.row_ptr.resize(static_cast<std::size_t>(n) + 1, 0);

        for (int i = 0; i < n; ++i)
        {
            auto &row = rows[static_cast<std::size_t>(i)];
            std::sort(row.begin(), row.end(), [](const auto &a, const auto &b)
                      { return a.first < b.first; });
            for (const auto &[j, v] : row)
            {
                A_full.col_ind.push_back(j);
                A_full.values.push_back(v);
            }
            A_full.row_ptr[static_cast<std::size_t>(i) + 1] = static_cast<int>(A_full.col_ind.size());
        }
        A_full.nnz = static_cast<int>(A_full.values.size());
        return A_full;
    }

    std::vector<double> full_csr_to_dense(const ichol::matrix::CsrMatrix<double> &A)
    {
        std::vector<double> dense(static_cast<std::size_t>(A.num_rows) * static_cast<std::size_t>(A.num_cols), 0.0);
        for (int i = 0; i < A.num_rows; ++i)
        {
            for (int p = A.row_ptr[static_cast<std::size_t>(i)];
                 p < A.row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
            {
                dense[static_cast<std::size_t>(i) * static_cast<std::size_t>(A.num_cols) +
                      static_cast<std::size_t>(A.col_ind[static_cast<std::size_t>(p)])] =
                    A.values[static_cast<std::size_t>(p)];
            }
        }
        return dense;
    }

    double frobenius_norm_identity_minus_product(const std::vector<double> &M,
                                                 const std::vector<double> &A,
                                                 int n)
    {
        double accum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                double value = (i == j) ? 1.0 : 0.0;
                for (int k = 0; k < n; ++k)
                {
                    value -= M[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(k)] *
                             A[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
                }
                accum += value * value;
            }
        }
        return std::sqrt(accum);
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

    struct FsaiCase
    {
        std::string label;
        ichol::matrix::CsrMatrix<double> A;
        ichol::precond::GridShape global_shape;
        ichol::precond::SubdomainSize subdomain_size;
        std::vector<ichol::precond::SubdomainRegion> regions;
        int fsai_level_k = 1;
    };

    FsaiCase build_2d_case()
    {
        FsaiCase c;
        c.label = "2D";
        c.A = ichol::io::gen_2dpoi<double>(16, 1.0);
        c.global_shape = {16, 16, 1};
        c.subdomain_size = {8, 8, 1};
        std::vector<double> rhs(static_cast<std::size_t>(c.A.num_rows), 1.0);
        apply_unit_col_prescaling_system(c.A, rhs);
        c.regions = ichol::precond::partition_subdomains(c.global_shape, c.subdomain_size);
        return c;
    }

    FsaiCase build_3d_case()
    {
        FsaiCase c;
        c.label = "3D";
        c.A = ichol::io::gen_3dpoi<double>(8);
        c.global_shape = {8, 8, 8};
        c.subdomain_size = {4, 4, 4};
        std::vector<double> rhs(static_cast<std::size_t>(c.A.num_rows), 1.0);
        apply_unit_col_prescaling_system(c.A, rhs);
        c.regions = ichol::precond::partition_subdomains(c.global_shape, c.subdomain_size);
        return c;
    }

    void run_fsai_quality_case(const FsaiCase &test_case)
    {
        ichol::precond::SubdomainPreconditionerOptions options;
        options.kind = ichol::precond::SubdomainPreconditionerKind::FSAI;
        options.fsai_level_k = test_case.fsai_level_k;
        options.precision = ichol::solver::ComputePrecision::FP64;

        const auto build_t0 = std::chrono::high_resolution_clock::now();
        auto raw_contexts = ichol::precond::create_subdomain_preconditioner_contexts_parallel(
            test_case.A, test_case.global_shape, test_case.regions, options);
        const auto build_t1 = std::chrono::high_resolution_clock::now();

        std::vector<SubdomainContextPtr> contexts;
        contexts.reserve(raw_contexts.size());
        for (auto *ctx : raw_contexts)
            contexts.emplace_back(ctx);

        ASSERT_EQ(contexts.size(), test_case.regions.size());

        double max_norm = 0.0;
        double avg_norm = 0.0;
        double baseline_zero_norm = 0.0;

        for (std::size_t subdomain = 0; subdomain < test_case.regions.size(); ++subdomain)
        {
            SCOPED_TRACE(test_case.label + " subdomain=" + std::to_string(subdomain));

            const auto &reg = test_case.regions[subdomain];
            const auto gidx = build_subdomain_gidx_host(test_case.global_shape, reg);
            const auto A_full = extract_full_subdomain_csr(test_case.A, test_case.global_shape, reg);
            const auto A_dense = full_csr_to_dense(A_full);
            const int nsub = A_full.num_rows;

            std::vector<double> M_dense(static_cast<std::size_t>(nsub) * static_cast<std::size_t>(nsub), 0.0);
            for (int col = 0; col < nsub; ++col)
            {
                std::vector<double> e_local(static_cast<std::size_t>(nsub), 0.0);
                e_local[static_cast<std::size_t>(col)] = 1.0;
                const auto e_global = scatter_subvector(test_case.A.num_rows, gidx, e_local);
                const auto m_col_global = apply_subdomain_context(contexts[subdomain].get(), e_global, test_case.A.num_rows);
                const auto m_col_local = gather_subvector(m_col_global, gidx);
                for (int row = 0; row < nsub; ++row)
                    M_dense[static_cast<std::size_t>(row) * static_cast<std::size_t>(nsub) + static_cast<std::size_t>(col)] =
                        m_col_local[static_cast<std::size_t>(row)];
            }

            const double norm = frobenius_norm_identity_minus_product(M_dense, A_dense, nsub);
            ASSERT_TRUE(std::isfinite(norm));

            max_norm = std::max(max_norm, norm);
            avg_norm += norm;
            baseline_zero_norm += std::sqrt(static_cast<double>(nsub));
        }

        avg_norm /= static_cast<double>(test_case.regions.size());
        baseline_zero_norm /= static_cast<double>(test_case.regions.size());
        const double build_secs = std::chrono::duration<double>(build_t1 - build_t0).count();

        std::cout << "[FSAIQuality." << test_case.label << "]"
                  << " grid=(" << test_case.global_shape.w << "," << test_case.global_shape.h << "," << test_case.global_shape.d << ")"
                  << " sub=(" << test_case.subdomain_size.w << "," << test_case.subdomain_size.h << "," << test_case.subdomain_size.d << ")"
                  << " k=" << test_case.fsai_level_k
                  << " build_time_s=" << build_secs
                  << " avg_fro_norm_I_minus_MA=" << avg_norm
                  << " max_fro_norm_I_minus_MA=" << max_norm
                  << "\n";

        EXPECT_LT(avg_norm, baseline_zero_norm);
    }
} // namespace

TEST(FSAIQuality, Poisson2D)
{
    if (!cuda_device_available())
        GTEST_SKIP() << "CUDA device unavailable";

    run_fsai_quality_case(build_2d_case());
}

TEST(FSAIQuality, Poisson3D)
{
    if (!cuda_device_available())
        GTEST_SKIP() << "CUDA device unavailable";

    run_fsai_quality_case(build_3d_case());
}
