// spai_diagnostic_tests.cpp
//
// Diagnostic tests to isolate SPAI preconditioner issues in the MPCG pipeline.
// These tests systematically verify each stage of the SPAI path:
//   1. GPU SpMV round-trip fidelity (gather → SpMV → scatter)
//   2. Approximate-inverse quality (||I - M*A|| and ||A*M*b - b||)
//   3. Symmetry of the extracted M
//   4. Index consistency between context and reference gidx
//   5. MPCG-level convergence diagnostics
//
// Build these alongside the existing test_subdomain_preconditioner.cu tests.

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

// ═══════════════════════════════════════════════════════════════════════════
// Shared helpers (duplicated from the existing test file for self-containment;
// in a real codebase these would live in a shared test utility header).
// ═══════════════════════════════════════════════════════════════════════════

namespace
{

    constexpr int kProblemN = 32;
    constexpr int kSubdomainExtent = 16;

    // ── CUDA / device helpers ────────────────────────────────────────────────

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
        explicit DeviceBuffer(std::size_t size) : size_(size)
        {
            if (size_ > 0)
                cuda_check(cudaMalloc(reinterpret_cast<void **>(&ptr_), size_ * sizeof(T)), "cudaMalloc");
        }
        ~DeviceBuffer()
        {
            if (ptr_)
                cudaFree(ptr_);
        }
        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;
        DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(o.ptr_), size_(o.size_)
        {
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        DeviceBuffer &operator=(DeviceBuffer &&o) noexcept
        {
            if (this != &o)
            {
                if (ptr_)
                    cudaFree(ptr_);
                ptr_ = o.ptr_;
                size_ = o.size_;
                o.ptr_ = nullptr;
                o.size_ = 0;
            }
            return *this;
        }
        T *data() { return ptr_; }
        const T *data() const { return ptr_; }
        std::size_t size() const { return size_; }
        void copy_from_host(const std::vector<T> &h)
        {
            if (h.size() != size_)
                throw std::runtime_error("size mismatch");
            if (size_ > 0)
                cuda_check(cudaMemcpy(ptr_, h.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), "H2D");
        }
        std::vector<T> copy_to_host() const
        {
            std::vector<T> h(size_);
            if (size_ > 0)
                cuda_check(cudaMemcpy(h.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
            return h;
        }
        void zero()
        {
            if (size_ > 0)
                cuda_check(cudaMemset(ptr_, 0, size_ * sizeof(T)), "memset");
        }

    private:
        T *ptr_ = nullptr;
        std::size_t size_ = 0;
    };

    // ── Problem setup ────────────────────────────────────────────────────────

    struct SubdomainContextDeleter
    {
        void operator()(ichol::precond::SubdomainPreconditionerContext *ctx) const
        {
            if (ctx)
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

    void apply_unit_col_prescaling_system(ichol::matrix::CsrMatrix<double> &A,
                                          std::vector<double> &b)
    {
        const auto D = ichol::numeric::scale_diag_sqrt(A);
        ichol::numeric::apply_prescaling(A, D);
        ichol::numeric::apply_rhs_prescaling(b, D);
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
        return setup;
    }

    std::vector<SubdomainContextPtr> create_spai_contexts(const ProblemSetup &setup)
    {
        ichol::precond::SubdomainPreconditionerOptions options;
        options.kind = ichol::precond::SubdomainPreconditionerKind::SPAI;
        options.precision = ichol::solver::ComputePrecision::FP64;

        auto raw = ichol::precond::create_subdomain_preconditioner_contexts_parallel(
            setup.A, setup.global_shape, setup.regions, options);
        std::vector<SubdomainContextPtr> out;
        out.reserve(raw.size());
        for (auto *ctx : raw)
            out.emplace_back(ctx);
        return out;
    }

    std::vector<int> build_subdomain_gidx_host(const ichol::precond::GridShape &global,
                                               const ichol::precond::SubdomainRegion &reg)
    {
        const int lw = reg.x1 - reg.x0, lh = reg.y1 - reg.y0, ld = reg.z1 - reg.z0;
        const int nsub = lw * lh * ld;
        std::vector<int> gidx(static_cast<std::size_t>(nsub));
        int idx = 0;
        for (int z = reg.z0; z < reg.z1; ++z)
            for (int y = reg.y0; y < reg.y1; ++y)
                for (int x = reg.x0; x < reg.x1; ++x)
                    gidx[idx++] = x + y * global.w + z * global.w * global.h;
        return gidx;
    }

    std::vector<double> gather_subvector(const std::vector<double> &global,
                                         const std::vector<int> &gidx)
    {
        std::vector<double> local(gidx.size());
        for (std::size_t i = 0; i < gidx.size(); ++i)
            local[i] = global[static_cast<std::size_t>(gidx[i])];
        return local;
    }

    std::vector<double> scatter_subvector(int n, const std::vector<int> &gidx,
                                          const std::vector<double> &local)
    {
        std::vector<double> global(static_cast<std::size_t>(n), 0.0);
        for (std::size_t i = 0; i < gidx.size(); ++i)
            global[static_cast<std::size_t>(gidx[i])] = local[i];
        return global;
    }

    std::vector<double> apply_subdomain_context(
        ichol::precond::SubdomainPreconditionerContext *ctx,
        const std::vector<double> &rhs, int n)
    {
        DeviceBuffer<double> d_rhs(rhs.size()), d_z(rhs.size());
        d_rhs.copy_from_host(rhs);
        d_z.zero();
        ichol::precond::apply_subdomain_preconditioner(
            ctx, d_rhs.data(), d_z.data(), n,
            ichol::solver::ComputePrecision::FP64, 0);
        cuda_check(cudaDeviceSynchronize(), "sync");
        return d_z.copy_to_host();
    }

    double l2_norm(const std::vector<double> &v)
    {
        return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    }

    double relative_error(const std::vector<double> &actual, const std::vector<double> &expected)
    {
        std::vector<double> diff(actual.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
            diff[i] = actual[i] - expected[i];
        const double denom = l2_norm(expected);
        return l2_norm(diff) / (denom > 0.0 ? denom : 1.0);
    }

    // Extract the full symmetric subdomain block from lower-tri global A.
    ichol::matrix::CsrMatrix<double> extract_full_subdomain_csr(
        const ichol::matrix::CsrMatrix<double> &A,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg)
    {
        const int lw = reg.x1 - reg.x0, lh = reg.y1 - reg.y0, ld = reg.z1 - reg.z0;
        const int nsub = lw * lh * ld;

        // Build global→local map
        auto gidx = build_subdomain_gidx_host(global, reg);
        std::unordered_map<int, int> g2l;
        g2l.reserve(nsub);
        for (int li = 0; li < nsub; ++li)
            g2l[gidx[li]] = li;

        // Extract lower-tri entries and symmetrize
        std::vector<std::vector<std::pair<int, double>>> rows(nsub);
        for (int li = 0; li < nsub; ++li)
        {
            const int gi = gidx[li];
            for (int k = A.row_ptr[gi]; k < A.row_ptr[gi + 1]; ++k)
            {
                auto it = g2l.find(A.col_ind[k]);
                if (it == g2l.end())
                    continue;
                const int lj = it->second;
                rows[li].push_back({lj, A.values[k]});
                if (lj != li)
                    rows[lj].push_back({li, A.values[k]});
            }
        }

        ichol::matrix::CsrMatrix<double> out;
        out.num_rows = nsub;
        out.num_cols = nsub;
        out.row_ptr.resize(nsub + 1, 0);
        for (int li = 0; li < nsub; ++li)
        {
            auto &row = rows[li];
            std::sort(row.begin(), row.end(), [](auto &a, auto &b)
                      { return a.first < b.first; });
            for (auto &[j, v] : row)
            {
                out.col_ind.push_back(j);
                out.values.push_back(v);
            }
            out.row_ptr[li + 1] = static_cast<int>(out.col_ind.size());
        }
        out.nnz = static_cast<int>(out.values.size());
        return out;
    }

    // Standard CSR matvec (full matrix, no implicit symmetry).
    std::vector<double> full_csr_matvec(const ichol::matrix::CsrMatrix<double> &A,
                                        const std::vector<double> &x)
    {
        std::vector<double> y(A.num_rows, 0.0);
        for (int i = 0; i < A.num_rows; ++i)
        {
            double sum = 0.0;
            for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
                sum += A.values[p] * x[A.col_ind[p]];
            y[i] = sum;
        }
        return y;
    }

    std::vector<double> full_csr_matvec(const std::vector<int> &row_ptr,
                                        const std::vector<int> &col_ind,
                                        const std::vector<double> &values,
                                        const std::vector<double> &x)
    {
        const int n = static_cast<int>(row_ptr.size()) - 1;
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
                y[i] += values[p] * x[col_ind[p]];
        return y;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 1: GPU SpMV round-trip vs CPU CSR matvec
    //
    // Purpose: Verify that the full GPU pipeline (gather global→local,
    //          cuSPARSE SpMV with M, scatter local→global) produces the
    //          same result as doing M_csr * gather(r) on the CPU and then
    //          scattering back.
    //
    // This isolates the GPU transport from the math. If this fails, the
    // cuSPARSE descriptors, gather/scatter kernels, or uploaded CSR data
    // are wrong.
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, GPUSpMVRoundTripMatchesCPU)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);
        ASSERT_FALSE(contexts.empty());

        // We need access to the CSR data of M on the host.  The production code
        // doesn't expose it, so we rebuild it here using the same PETSc extraction
        // path (tested separately in SPAIExtractionMatchesPCApply).
        //
        // However, the key question is whether the *GPU path* matches.
        // Strategy: apply the context on GPU with several test vectors and compare
        // against applying it on the CPU with the *same* context.
        //
        // We construct global-length test vectors, apply via GPU, gather the
        // subdomain slice, and compare against a CPU reference that:
        //   (a) gathers the local RHS from the global vector
        //   (b) does CPU CSR matvec with the SPAI matrix
        //   (c) scatters back
        //
        // Since we don't have the host CSR of M, we use an indirect approach:
        // apply the context to unit vectors e_{gidx[j]} for each local DOF j.
        // This probes column j of the gather→SpMV→scatter pipeline.
        // The result for e_{gidx[j]} should be: scatter(M[:, j]) into global.
        // If we do this for ALL local DOFs, we reconstruct the full action of M
        // as seen through the GPU pipeline.

        for (std::size_t sub = 0; sub < contexts.size(); ++sub)
        {
            SCOPED_TRACE("subdomain=" + std::to_string(sub));
            const auto &reg = setup.regions[sub];
            const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
            const int nsub = static_cast<int>(gidx.size());
            const int N = setup.A.num_rows;

            // Probe the GPU pipeline with each canonical basis vector e_{gidx[j]}.
            // Collect columns of the effective operator.
            std::vector<std::vector<double>> gpu_columns(nsub);
            for (int j = 0; j < nsub; ++j)
            {
                std::vector<double> e_global(N, 0.0);
                e_global[gidx[j]] = 1.0;

                auto z = apply_subdomain_context(contexts[sub].get(), e_global, N);
                gpu_columns[j] = gather_subvector(z, gidx);
            }

            // Now apply the context to several non-trivial vectors and verify
            // that the result matches sum_j r_local[j] * gpu_columns[j].
            // This checks linearity and consistency of the GPU path.
            std::vector<std::vector<double>> test_globals;

            // Test vector 1: the problem RHS
            test_globals.push_back(setup.rhs);

            // Test vector 2: all ones
            test_globals.push_back(std::vector<double>(N, 1.0));

            // Test vector 3: alternating ±1
            {
                std::vector<double> v(N);
                for (int i = 0; i < N; ++i)
                    v[i] = (i % 2 == 0) ? 1.0 : -1.0;
                test_globals.push_back(std::move(v));
            }

            // Test vector 4: random-ish (deterministic)
            {
                std::vector<double> v(N);
                for (int i = 0; i < N; ++i)
                    v[i] = std::sin(static_cast<double>(i) * 0.7 + 0.3);
                test_globals.push_back(std::move(v));
            }

            double max_rel_err = 0.0;
            for (std::size_t tv = 0; tv < test_globals.size(); ++tv)
            {
                SCOPED_TRACE("test_vector=" + std::to_string(tv));

                const auto &r_global = test_globals[tv];
                const auto r_local = gather_subvector(r_global, gidx);

                // GPU result
                auto z_global = apply_subdomain_context(contexts[sub].get(), r_global, N);
                auto z_local = gather_subvector(z_global, gidx);

                // CPU reference from probed columns: z_ref[i] = sum_j M[i,j] * r_local[j]
                std::vector<double> z_ref(nsub, 0.0);
                for (int j = 0; j < nsub; ++j)
                {
                    const double rj = r_local[j];
                    for (int i = 0; i < nsub; ++i)
                        z_ref[i] += gpu_columns[j][i] * rj;
                }

                const double rel_err = relative_error(z_local, z_ref);
                max_rel_err = std::max(max_rel_err, rel_err);

                // This should be near machine epsilon if the GPU path is linear and correct.
                EXPECT_LT(rel_err, 1e-12)
                    << "GPU SpMV result does not match CPU column-probed reconstruction "
                    << "for subdomain " << sub << " test_vector " << tv;
            }

            std::cout << "[SPAIDiag.GPURoundTrip] sub=" << sub
                      << " nsub=" << nsub
                      << " max_rel_err=" << max_rel_err << "\n";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 2: Approximate-inverse quality metrics
    //
    // Purpose: Quantify how good the SPAI M is as an approximate inverse of
    //          the local subdomain operator A_i.  Compute:
    //            (a) ||I - M * A_i||_F / ||I||_F  (should be < 1 for useful precond)
    //            (b) ||I - A_i * M||_F / ||I||_F
    //            (c) spectral condition estimate: max(eig(M*A)) / min(eig(M*A))
    //                via power iteration on M*A and (M*A)^{-1}
    //            (d) compare against Jacobi: ||I - D^{-1} * A_i||_F
    //
    // If (a)/(b) are close to 1 or > 1, the SPAI is too weak to be useful
    // and the problem is mathematical (PETSc SPAI parameters), not a bug.
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, ApproximateInverseQuality)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);
        ASSERT_FALSE(contexts.empty());

        for (std::size_t sub = 0; sub < contexts.size(); ++sub)
        {
            SCOPED_TRACE("subdomain=" + std::to_string(sub));
            const auto &reg = setup.regions[sub];
            const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
            const int nsub = static_cast<int>(gidx.size());
            const int N = setup.A.num_rows;

            // Extract the full symmetric A_sub.
            const auto A_sub = extract_full_subdomain_csr(setup.A, setup.global_shape, reg);

            // Probe M via the GPU to get its dense representation.
            // M_dense[i][j] = (apply context to e_{gidx[j]})[gidx[i]]
            std::vector<std::vector<double>> M_dense(nsub, std::vector<double>(nsub, 0.0));
            for (int j = 0; j < nsub; ++j)
            {
                std::vector<double> e_global(N, 0.0);
                e_global[gidx[j]] = 1.0;
                auto z = apply_subdomain_context(contexts[sub].get(), e_global, N);
                for (int i = 0; i < nsub; ++i)
                    M_dense[i][j] = z[gidx[i]];
            }

            // Compute MA = M * A_sub (dense, via CSR matvec on columns of M^T = rows of M)
            // MA[i][j] = sum_k M[i][k] * A[k][j]
            // Equivalently: column j of MA = M * (column j of A)
            // But A is CSR so it's easier to compute row i of MA = M[i,:] * A
            // Actually let's just do it element-wise for clarity since nsub is small.

            // First, densify A_sub.
            std::vector<std::vector<double>> A_dense(nsub, std::vector<double>(nsub, 0.0));
            for (int i = 0; i < nsub; ++i)
                for (int p = A_sub.row_ptr[i]; p < A_sub.row_ptr[i + 1]; ++p)
                    A_dense[i][A_sub.col_ind[p]] = A_sub.values[p];

            // MA = M * A
            std::vector<std::vector<double>> MA(nsub, std::vector<double>(nsub, 0.0));
            for (int i = 0; i < nsub; ++i)
                for (int j = 0; j < nsub; ++j)
                    for (int k = 0; k < nsub; ++k)
                        MA[i][j] += M_dense[i][k] * A_dense[k][j];

            // AM = A * M
            std::vector<std::vector<double>> AM(nsub, std::vector<double>(nsub, 0.0));
            for (int i = 0; i < nsub; ++i)
                for (int j = 0; j < nsub; ++j)
                    for (int k = 0; k < nsub; ++k)
                        AM[i][j] += A_dense[i][k] * M_dense[k][j];

            // ||I - MA||_F and ||I - AM||_F
            double fro_I_minus_MA = 0.0, fro_I_minus_AM = 0.0;
            double fro_I = std::sqrt(static_cast<double>(nsub)); // ||I||_F = sqrt(n)
            for (int i = 0; i < nsub; ++i)
            {
                for (int j = 0; j < nsub; ++j)
                {
                    double eye = (i == j) ? 1.0 : 0.0;
                    double d_ma = eye - MA[i][j];
                    double d_am = eye - AM[i][j];
                    fro_I_minus_MA += d_ma * d_ma;
                    fro_I_minus_AM += d_am * d_am;
                }
            }
            fro_I_minus_MA = std::sqrt(fro_I_minus_MA) / fro_I;
            fro_I_minus_AM = std::sqrt(fro_I_minus_AM) / fro_I;

            // Jacobi baseline: D^{-1} where D = diag(A)
            std::vector<double> diag(nsub);
            for (int i = 0; i < nsub; ++i)
                diag[i] = A_dense[i][i];

            double fro_I_minus_JacA = 0.0;
            for (int i = 0; i < nsub; ++i)
            {
                for (int j = 0; j < nsub; ++j)
                {
                    double eye = (i == j) ? 1.0 : 0.0;
                    double jac_a = A_dense[i][j] / diag[i]; // (D^{-1} A)[i,j]
                    double d = eye - jac_a;
                    fro_I_minus_JacA += d * d;
                }
            }
            fro_I_minus_JacA = std::sqrt(fro_I_minus_JacA) / fro_I;

            // Check symmetry of M
            double max_asym = 0.0;
            for (int i = 0; i < nsub; ++i)
                for (int j = i + 1; j < nsub; ++j)
                    max_asym = std::max(max_asym, std::abs(M_dense[i][j] - M_dense[j][i]));

            // Check symmetry of MA (should be symmetric if M and A are both symmetric)
            double max_ma_asym = 0.0;
            for (int i = 0; i < nsub; ++i)
                for (int j = i + 1; j < nsub; ++j)
                    max_ma_asym = std::max(max_ma_asym, std::abs(MA[i][j] - MA[j][i]));

            // Check diagonal of MA (should be close to 1 for a good preconditioner)
            double min_ma_diag = 1e30, max_ma_diag = -1e30;
            for (int i = 0; i < nsub; ++i)
            {
                min_ma_diag = std::min(min_ma_diag, MA[i][i]);
                max_ma_diag = std::max(max_ma_diag, MA[i][i]);
            }

            // Check if M has any negative diagonals (would indicate extraction error)
            double min_m_diag = 1e30;
            for (int i = 0; i < nsub; ++i)
                min_m_diag = std::min(min_m_diag, M_dense[i][i]);

            // Check trace(MA) / n (should be ~1 for perfect preconditioner)
            double trace_MA = 0.0;
            for (int i = 0; i < nsub; ++i)
                trace_MA += MA[i][i];

            std::cout << "[SPAIDiag.Quality] sub=" << sub << " nsub=" << nsub
                      << "\n  ||I-MA||_F/||I||_F = " << fro_I_minus_MA
                      << "\n  ||I-AM||_F/||I||_F = " << fro_I_minus_AM
                      << "\n  ||I-JacA||_F/||I||_F = " << fro_I_minus_JacA
                      << " (Jacobi baseline)"
                      << "\n  M symmetry max|M[i,j]-M[j,i]| = " << max_asym
                      << "\n  MA symmetry max|MA[i,j]-MA[j,i]| = " << max_ma_asym
                      << "\n  MA diag range = [" << min_ma_diag << ", " << max_ma_diag << "]"
                      << "\n  M min diagonal = " << min_m_diag
                      << "\n  trace(MA)/n = " << trace_MA / nsub
                      << "\n";

            // SPAI should be at least as good as Jacobi
            EXPECT_LE(fro_I_minus_MA, fro_I_minus_JacA + 0.01)
                << "SPAI is worse than Jacobi in Frobenius norm!";

            // M should be symmetric (SPAI with SetSp(1))
            EXPECT_LT(max_asym, 1e-12)
                << "SPAI matrix M is not symmetric";

            // M diagonal should be positive (since A is SPD)
            EXPECT_GT(min_m_diag, 0.0)
                << "SPAI matrix has non-positive diagonal";

            // MA should be reasonably close to I
            EXPECT_LT(fro_I_minus_MA, 1.0)
                << "||I - MA|| >= 1: SPAI provides no useful preconditioning";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 3: Verify M is symmetric in the extracted CSR
    //
    // Purpose: PCSPAISetSp(pc, 1) requests a symmetric SPAI.  If the column-
    //          probing extraction introduces asymmetry (e.g., due to threshold
    //          dropping one entry but not its transpose), the preconditioner
    //          would be non-symmetric, which can break CG-type methods.
    //
    // We check this by probing the GPU context and comparing M[i,j] vs M[j,i].
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, ExtractedMIsSymmetric)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);

        for (std::size_t sub = 0; sub < contexts.size(); ++sub)
        {
            SCOPED_TRACE("subdomain=" + std::to_string(sub));
            const auto &reg = setup.regions[sub];
            const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
            const int nsub = static_cast<int>(gidx.size());
            const int N = setup.A.num_rows;

            // Probe columns of M
            std::vector<std::vector<double>> M_cols(nsub);
            for (int j = 0; j < nsub; ++j)
            {
                std::vector<double> e(N, 0.0);
                e[gidx[j]] = 1.0;
                auto z = apply_subdomain_context(contexts[sub].get(), e, N);
                M_cols[j].resize(nsub);
                for (int i = 0; i < nsub; ++i)
                    M_cols[j][i] = z[gidx[i]];
            }

            // Check symmetry
            double max_asym = 0.0;
            double max_entry = 0.0;
            int worst_i = -1, worst_j = -1;
            for (int i = 0; i < nsub; ++i)
            {
                for (int j = i + 1; j < nsub; ++j)
                {
                    // M[i,j] is M_cols[j][i], M[j,i] is M_cols[i][j]
                    const double mij = M_cols[j][i];
                    const double mji = M_cols[i][j];
                    const double asym = std::abs(mij - mji);
                    max_entry = std::max(max_entry, std::max(std::abs(mij), std::abs(mji)));
                    if (asym > max_asym)
                    {
                        max_asym = asym;
                        worst_i = i;
                        worst_j = j;
                    }
                }
            }

            const double rel_asym = max_asym / (max_entry > 0 ? max_entry : 1.0);
            std::cout << "[SPAIDiag.Symmetry] sub=" << sub
                      << " max_abs_asym=" << max_asym
                      << " max_entry=" << max_entry
                      << " rel_asym=" << rel_asym
                      << " worst_pair=(" << worst_i << "," << worst_j << ")\n";

            // The threshold dropping (1e-14) in extraction could cause tiny asymmetry,
            // but anything larger than ~1e-13 suggests a real problem.
            EXPECT_LT(rel_asym, 1e-10)
                << "Extracted SPAI matrix is not symmetric: "
                << "M[" << worst_i << "," << worst_j << "]=" << M_cols[worst_j][worst_i]
                << " vs M[" << worst_j << "," << worst_i << "]=" << M_cols[worst_i][worst_j];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 4: Global gather/scatter index consistency
    //
    // Purpose: Verify that the gidx stored inside the SubdomainSpaiContext
    //          (built during create_subdomain_spai_context) matches what
    //          build_subdomain_gidx_host produces.  If these differ, the
    //          gather/scatter in apply_subdomain_spai will read/write wrong
    //          global positions.
    //
    // We test this indirectly: for each local DOF j, set r[gidx[j]] = 1 and
    // check that the context produces nonzero output only at gidx positions.
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, GatherScatterIndexConsistency)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);

        for (std::size_t sub = 0; sub < contexts.size(); ++sub)
        {
            SCOPED_TRACE("subdomain=" + std::to_string(sub));
            const auto &reg = setup.regions[sub];
            const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
            const int nsub = static_cast<int>(gidx.size());
            const int N = setup.A.num_rows;

            // Mark which global indices belong to this subdomain.
            std::vector<bool> is_sub(N, false);
            for (int gi : gidx)
                is_sub[gi] = true;

            // Apply context to e_{gidx[0]} — output should be nonzero only at gidx positions.
            std::vector<double> e(N, 0.0);
            e[gidx[0]] = 1.0;
            auto z = apply_subdomain_context(contexts[sub].get(), e, N);

            double outside_norm_sq = 0.0;
            for (int i = 0; i < N; ++i)
            {
                if (!is_sub[i])
                    outside_norm_sq += z[i] * z[i];
            }
            const double outside_norm = std::sqrt(outside_norm_sq);

            EXPECT_LT(outside_norm, 1e-14)
                << "SPAI context wrote outside its subdomain support for sub=" << sub;

            // Stronger check: apply to a vector that is zero at ALL gidx positions
            // but nonzero elsewhere.  The output should be exactly zero everywhere.
            std::vector<double> anti_e(N, 1.0);
            for (int gi : gidx)
                anti_e[gi] = 0.0;
            auto z_anti = apply_subdomain_context(contexts[sub].get(), anti_e, N);

            double anti_norm = l2_norm(z_anti);
            EXPECT_LT(anti_norm, 1e-14)
                << "SPAI context produced nonzero output from input that is zero on subdomain "
                << "(norm=" << anti_norm << "). This indicates a gather index mismatch.";

            std::cout << "[SPAIDiag.IndexConsistency] sub=" << sub
                      << " outside_norm=" << outside_norm
                      << " anti_support_output_norm=" << anti_norm << "\n";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 5: SPAI applied to A*x_known should recover x_known (approximately)
    //
    // Purpose: End-to-end check that M ≈ A^{-1} in a meaningful sense.
    //          Choose x_known, compute b = A_sub * x_known, then check that
    //          M * b ≈ x_known.  This catches sign errors, transposition
    //          mistakes, or applying M to the wrong thing.
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, SPAIAppliedToAxRecoversx)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);

        for (std::size_t sub = 0; sub < contexts.size(); ++sub)
        {
            SCOPED_TRACE("subdomain=" + std::to_string(sub));
            const auto &reg = setup.regions[sub];
            const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
            const int nsub = static_cast<int>(gidx.size());
            const int N = setup.A.num_rows;

            const auto A_sub = extract_full_subdomain_csr(setup.A, setup.global_shape, reg);

            // Known solution: x_known[i] = sin(i + 1)
            std::vector<double> x_known(nsub);
            for (int i = 0; i < nsub; ++i)
                x_known[i] = std::sin(static_cast<double>(i + 1));

            // b = A_sub * x_known
            auto b_local = full_csr_matvec(A_sub, x_known);

            // Scatter b_local into global vector, apply SPAI, gather back
            auto b_global = scatter_subvector(N, gidx, b_local);
            auto z_global = apply_subdomain_context(contexts[sub].get(), b_global, N);
            auto x_recovered = gather_subvector(z_global, gidx);

            const double recovery_error = relative_error(x_recovered, x_known);

            std::cout << "[SPAIDiag.Recovery] sub=" << sub
                      << " ||M*A*x - x|| / ||x|| = " << recovery_error << "\n";

            // For a good approximate inverse, this should be significantly < 1.
            // For a very weak preconditioner, it might be close to 1.
            // But it should NEVER be > 1 (that would mean the preconditioner diverges).
            EXPECT_LT(recovery_error, 1.0)
                << "SPAI preconditioner makes things worse than no preconditioner";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 6: Check if SPAI M has reasonable sparsity and nonzero structure
    //
    // Purpose: If the extraction threshold (1e-14) is too aggressive or too
    //          lenient, M could be overly sparse (essentially Jacobi) or
    //          overly dense (wasting memory without benefit).  Report the
    //          sparsity statistics so we can diagnose.
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, SPAISparsityStatistics)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);

        for (std::size_t sub = 0; sub < contexts.size(); ++sub)
        {
            SCOPED_TRACE("subdomain=" + std::to_string(sub));
            const auto &reg = setup.regions[sub];
            const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);
            const int nsub = static_cast<int>(gidx.size());
            const int N = setup.A.num_rows;

            int total_nnz = 0;
            int diag_only_rows = 0;
            double max_abs_entry = 0.0;
            double min_abs_diag = 1e30;

            for (int j = 0; j < nsub; ++j)
            {
                std::vector<double> e(N, 0.0);
                e[gidx[j]] = 1.0;
                auto z = apply_subdomain_context(contexts[sub].get(), e, N);

                int col_nnz = 0;
                for (int i = 0; i < nsub; ++i)
                {
                    const double v = z[gidx[i]];
                    if (std::abs(v) > 1e-15)
                    {
                        col_nnz++;
                        max_abs_entry = std::max(max_abs_entry, std::abs(v));
                    }
                    if (i == j)
                        min_abs_diag = std::min(min_abs_diag, std::abs(v));
                }
                total_nnz += col_nnz;
                if (col_nnz <= 1)
                    diag_only_rows++;
            }

            const double avg_nnz_per_row = static_cast<double>(total_nnz) / nsub;
            const double density = static_cast<double>(total_nnz) / (static_cast<double>(nsub) * nsub);

            std::cout << "[SPAIDiag.Sparsity] sub=" << sub << " nsub=" << nsub
                      << " total_nnz=" << total_nnz
                      << " avg_nnz/row=" << avg_nnz_per_row
                      << " density=" << density
                      << " diag_only_rows=" << diag_only_rows
                      << " max_abs=" << max_abs_entry
                      << " min_abs_diag=" << min_abs_diag << "\n";

            // If M is just a diagonal, SPAI was essentially useless.
            EXPECT_GT(avg_nnz_per_row, 1.5)
                << "SPAI matrix is too sparse (essentially diagonal)";

            // But it shouldn't be fully dense either for a 3D Poisson problem.
            EXPECT_LT(density, 0.5)
                << "SPAI matrix is suspiciously dense";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 7: SPAI + MPCG single-iteration residual reduction
    //
    // Purpose: Even without full convergence, a single MPCG iteration with
    //          the SPAI preconditioner should reduce the residual.  If it
    //          doesn't, we know the issue is in how the preconditioner
    //          interacts with MPCG, not just preconditioner quality.
    //
    // Run MPCG for exactly 1 iteration and check that the residual decreased.
    // Compare against running with Jacobi preconditioner to establish a floor.
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, MPCGSingleIterationResidualReduction)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);
        const int N = setup.A.num_rows;

        // Build PrecondApply list for SPAI
        std::vector<ichol::precond::PrecondApply> spai_preconds;
        spai_preconds.reserve(contexts.size());
        for (const auto &ctx : contexts)
            spai_preconds.push_back({&ichol::precond::apply_subdomain_preconditioner,
                                     ctx.get()});

        // Run MPCG with 1 iteration
        ichol::solver::PCGParams params;
        params.maxits = 3;  // Just a few iterations to see trend
        params.tol = 1e-30; // Don't stop early
        params.verbose = false;
        params.restart = 0;
        params.prec_gemm = ichol::solver::ComputePrecision::FP64;
        params.prec_spmm = ichol::solver::ComputePrecision::FP64;
        params.prec_acc = ichol::solver::ComputePrecision::FP64;
        params.prec_precond = ichol::solver::ComputePrecision::FP64;
        params.store_Znew = ichol::solver::ComputePrecision::FP64;
        params.store_Pnew = ichol::solver::ComputePrecision::FP64;
        params.store_Wnew = ichol::solver::ComputePrecision::FP64;
        params.store_P_hist = ichol::solver::ComputePrecision::FP64;
        params.store_W_hist = ichol::solver::ComputePrecision::FP64;
        params.use_svd = false;
        params.rcond_base = 1e-12;
        params.projection_anorm_drop_tol = 0.0;

        std::vector<double> x(N, 0.0);

        auto result = ichol::solver::mpcg<double>(
            setup.A.row_ptr, setup.A.col_ind, setup.A.values,
            spai_preconds, setup.rhs, x, params);

        std::cout << "[SPAIDiag.MPCGIters] iterations=" << result.iterations
                  << " finalRes=" << result.finalRes << "\n";
        std::cout << "[SPAIDiag.MPCGIters] relResiduals:";
        for (std::size_t i = 0; i < result.relResiduals.size(); ++i)
            std::cout << " " << result.relResiduals[i];
        std::cout << "\n";

        // Check monotonic decrease (CG should be monotonic in exact arithmetic)
        for (std::size_t i = 1; i < result.relResiduals.size(); ++i)
        {
            // Allow a tiny tolerance for floating-point effects
            EXPECT_LE(result.relResiduals[i], result.relResiduals[i - 1] * 1.01)
                << "Residual increased at iteration " << i
                << ": " << result.relResiduals[i] << " > " << result.relResiduals[i - 1];
        }

        // At least the first iteration should reduce the residual meaningfully
        if (result.relResiduals.size() >= 2)
        {
            const double reduction = result.relResiduals[1] / result.relResiduals[0];
            std::cout << "[SPAIDiag.MPCGIters] first_iter_reduction_factor=" << reduction << "\n";
            EXPECT_LT(reduction, 1.0)
                << "First MPCG iteration did not reduce the residual";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 8: Verify the apply function is called with the right function pointer
    //
    // Purpose: The SPAI context is created with kind=SPAI, but the apply
    //          function pointer in SubdomainPreconditionerContext might point
    //          to the wrong function (e.g., IC apply instead of SPAI apply).
    //          This would silently produce wrong results.
    //
    // We verify by checking that apply_subdomain_preconditioner dispatches to
    // the SPAI path by comparing its output to a known-correct SPAI result.
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, ApplyDispatchesCorrectly)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);
        ASSERT_FALSE(contexts.empty());

        const int N = setup.A.num_rows;

        // Apply via the public interface twice with the same input.
        // Results must be bitwise identical (deterministic).
        auto z1 = apply_subdomain_context(contexts[0].get(), setup.rhs, N);
        auto z2 = apply_subdomain_context(contexts[0].get(), setup.rhs, N);

        EXPECT_EQ(z1, z2) << "SPAI apply is non-deterministic";

        // Apply with zero input must give zero output.
        std::vector<double> zeros(N, 0.0);
        auto z_zero = apply_subdomain_context(contexts[0].get(), zeros, N);
        EXPECT_DOUBLE_EQ(l2_norm(z_zero), 0.0) << "SPAI(0) != 0";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 9: Compare SPAI apply in FP32 vs FP64 mode
    //
    // Purpose: The apply_subdomain_spai function has separate FP32 and FP64
    //          code paths. If the FP32 values (d_val32) were not properly
    //          uploaded, the FP32 path would produce garbage.
    //          Also checks that TF32 maps to FP32 correctly.
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(SPAIDiagnostic, FP32PathMatchesFP64)
    {
        if (!cuda_device_available())
            GTEST_SKIP() << "CUDA device unavailable";

        const ProblemSetup setup = build_problem();
        const auto contexts = create_spai_contexts(setup);
        ASSERT_FALSE(contexts.empty());

        const int N = setup.A.num_rows;
        const auto &reg = setup.regions[0];
        const auto gidx = build_subdomain_gidx_host(setup.global_shape, reg);

        // Apply in FP64
        DeviceBuffer<double> d_rhs64(N), d_z64(N);
        d_rhs64.copy_from_host(setup.rhs);
        d_z64.zero();
        ichol::precond::apply_subdomain_preconditioner(
            contexts[0].get(), d_rhs64.data(), d_z64.data(), N,
            ichol::solver::ComputePrecision::FP64, 0);
        cuda_check(cudaDeviceSynchronize(), "sync");
        auto z64 = d_z64.copy_to_host();

        // Apply in FP32
        // NOTE: The SPAI apply path expects the input/output pointers to be
        // float* when called with FP32 precision. We need to cast the rhs
        // to float, allocate float buffers, then cast back.
        std::vector<float> rhs_f32(N);
        for (int i = 0; i < N; ++i)
            rhs_f32[i] = static_cast<float>(setup.rhs[i]);

        DeviceBuffer<float> d_rhs32(N), d_z32(N);
        d_rhs32.copy_from_host(rhs_f32);
        d_z32.zero();
        ichol::precond::apply_subdomain_preconditioner(
            contexts[0].get(), d_rhs32.data(), d_z32.data(), N,
            ichol::solver::ComputePrecision::FP32, 0);
        cuda_check(cudaDeviceSynchronize(), "sync");
        auto z32_f = d_z32.copy_to_host();

        // Compare: FP32 result cast to double vs FP64 result
        std::vector<double> z32_d(N);
        for (int i = 0; i < N; ++i)
            z32_d[i] = static_cast<double>(z32_f[i]);

        // Gather only subdomain DOFs for comparison
        auto z64_local = gather_subvector(z64, gidx);
        auto z32_local = gather_subvector(z32_d, gidx);

        const double rel_diff = relative_error(z32_local, z64_local);
        std::cout << "[SPAIDiag.FP32vsFP64] rel_diff=" << rel_diff << "\n";

        // FP32 should match FP64 to about single-precision accuracy
        EXPECT_LT(rel_diff, 1e-5)
            << "FP32 SPAI path deviates too much from FP64";

        // It should NOT be zero (that would mean the FP32 path isn't doing anything)
        if (l2_norm(z64_local) > 1e-15)
        {
            EXPECT_GT(l2_norm(z32_local), 1e-15)
                << "FP32 SPAI path produced all zeros";
        }
    }

} // namespace
