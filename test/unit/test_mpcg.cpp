// test/unit/test_mpcg.cpp
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <utility>
#include <future>
#include <string>
#include <fstream>
#include <filesystem>
#include <limits>
#include <mutex>
#include <cuda_runtime.h>
#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

#include "ichol/mtx_read.hpp"
#include "ichol/preconditioner.hpp"
#include "ichol/pcg.hpp"
#include "ichol/subdomain_exact_gpu.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/detail/numeric_plan.hpp"
#include "factor/numerical/factorize.hpp"
#include "unit/test_utils.hpp"

class MPCGTest : public ::testing::Test
{
public:
    static int n;
};

int MPCGTest::n = 24;

namespace
{
    struct PetscSession
    {
        ~PetscSession()
        {
            PetscBool initialized = PETSC_FALSE;
            PetscBool finalized = PETSC_FALSE;
            (void)PetscInitialized(&initialized);
            (void)PetscFinalized(&finalized);
            if (initialized && !finalized)
                (void)PetscFinalize();
        }
    };

    std::once_flag g_petsc_init_once;
    std::vector<std::string> g_petsc_arg_storage;
    std::vector<char *> g_petsc_argv;
    std::unique_ptr<PetscSession> g_petsc_session;

    constexpr const char *kPetscPoissonHelp =
        "PETSc 3D Poisson solve test on an nx-by-ny-by-nz structured grid.\n"
        "Options:\n"
        "  -nx <int> -ny <int> -nz <int>\n"
        "  -ksp_type <type>\n"
        "  -pc_type <type>\n"
        "Examples:\n"
        "  ./test_mpcg --gtest_filter=MPCG.PETSc_3D_Poisson_KSP "
        "-nx 32 -ny 32 -nz 32 -ksp_type cg -pc_type gamg\n"
        "  mpiexec -n 2 ./test_mpcg --gtest_filter=MPCG.PETSc_3D_Poisson_KSP "
        "-nx 32 -ny 32 -nz 32 -ksp_type cg -pc_type hypre -pc_hypre_type boomeramg\n";

    static void set_petsc_command_line(const std::vector<std::string> &args)
    {
        g_petsc_arg_storage = args;
        if (g_petsc_arg_storage.empty())
            g_petsc_arg_storage.emplace_back("test_mpcg");

        g_petsc_argv.clear();
        g_petsc_argv.reserve(g_petsc_arg_storage.size() + 1);
        for (auto &arg : g_petsc_arg_storage)
            g_petsc_argv.push_back(arg.data());
        g_petsc_argv.push_back(nullptr);
    }

    static void petsc_check(PetscErrorCode ierr, const char *what)
    {
        if (ierr)
            throw std::runtime_error(std::string("PETSc ") + what +
                                     ": error " + std::to_string(ierr));
    }

    static void set_default_petsc_device_types_if_available()
    {
        const char *mat_type = MATAIJ;
        const char *vec_type = nullptr;

#if defined(PETSC_HAVE_CUDA)
        mat_type = MATAIJCUSPARSE;
        vec_type = VECCUDA;
#elif defined(PETSC_HAVE_HIP)
        mat_type = MATAIJHIPSPARSE;
        vec_type = VECHIP;
#elif defined(PETSC_HAVE_KOKKOS)
        mat_type = MATAIJKOKKOS;
        vec_type = VECKOKKOS;
#endif

        PetscBool has_mat_type = PETSC_FALSE;
        PetscBool has_vec_type = PETSC_FALSE;
        petsc_check(PetscOptionsHasName(nullptr, nullptr, "-mat_type", &has_mat_type),
                    "PetscOptionsHasName(-mat_type)");
        petsc_check(PetscOptionsHasName(nullptr, nullptr, "-vec_type", &has_vec_type),
                    "PetscOptionsHasName(-vec_type)");
        if (!has_mat_type)
            petsc_check(PetscOptionsSetValue(nullptr, "-mat_type", mat_type),
                        "PetscOptionsSetValue(-mat_type)");
        if (vec_type && !has_vec_type)
            petsc_check(PetscOptionsSetValue(nullptr, "-vec_type", vec_type),
                        "PetscOptionsSetValue(-vec_type)");
    }

    static void ensure_petsc_initialized()
    {
        std::call_once(g_petsc_init_once, []()
                       {
                           PetscBool initialized = PETSC_FALSE;
                           petsc_check(PetscInitialized(&initialized), "PetscInitialized");
                           if (!initialized)
                           {
                               int argc = static_cast<int>(g_petsc_arg_storage.size());
                               char **argv = g_petsc_argv.empty() ? nullptr : g_petsc_argv.data();
                               petsc_check(PetscInitialize(&argc, &argv, nullptr, kPetscPoissonHelp),
                                           "PetscInitialize");
                               g_petsc_session = std::make_unique<PetscSession>();
                           }
                           set_default_petsc_device_types_if_available();
                       });
    }

    static void apply_unit_col_prescaling_system(ichol::matrix::CsrMatrix<double> &A,
                                                 std::vector<double> &b)
    {
        const auto D = ichol::numeric::scale_diag_sqrt(A);
        ichol::numeric::apply_prescaling(A, D);
        ichol::numeric::apply_rhs_prescaling(b, D);
    }

    static bool regions_overlap(const ichol::precond::SubdomainRegion &a,
                                const ichol::precond::SubdomainRegion &b)
    {
        const bool x_overlap = (a.x0 < b.x1) && (b.x0 < a.x1);
        const bool y_overlap = (a.y0 < b.y1) && (b.y0 < a.y1);
        const bool z_overlap = (a.z0 < b.z1) && (b.z0 < a.z1);
        return x_overlap && y_overlap && z_overlap;
    }

    static void unflatten_global_3d(int gi, int gw, int gh, int &x, int &y, int &z)
    {
        const int plane = gw * gh;
        z = gi / plane;
        const int rem = gi - z * plane;
        y = rem / gw;
        x = rem - y * gw;
    }

    static int flatten_local_3d(int x, int y, int z, int w, int h)
    {
        return x + y * w + z * (w * h);
    }

    static int local_from_global(
        int gj,
        const ichol::precond::GridShape &global,
        const ichol::precond::SubdomainRegion &reg,
        int lw,
        int lh)
    {
        int x = 0, y = 0, z = 0;
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
        sub.row_ptr.resize((size_t)nsub + 1, 0);

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
            row_entries.reserve((size_t)(A.row_ptr[gi + 1] - A.row_ptr[gi]));

            for (int kk = A.row_ptr[gi]; kk < A.row_ptr[gi + 1]; ++kk)
            {
                const int lj = local_from_global(A.col_ind[kk], global, reg, lw, lh);
                if (lj < 0 || lj > li)
                    continue;
                row_entries.push_back({lj, A.values[kk]});
            }

            std::sort(row_entries.begin(), row_entries.end(), [](const auto &u, const auto &v)
                      { return u.first < v.first; });

            int diag_pos = -1;
            for (int i = 0; i < (int)row_entries.size(); ++i)
            {
                if (row_entries[(size_t)i].first == li)
                {
                    diag_pos = i;
                    break;
                }
            }
            if (diag_pos < 0)
                throw std::runtime_error("extract_lower_subdomain_csr: missing diagonal");

            for (int i = 0; i < (int)row_entries.size(); ++i)
            {
                if (i == diag_pos)
                    continue;
                cols.push_back(row_entries[(size_t)i].first);
                vals.push_back(row_entries[(size_t)i].second);
            }
            cols.push_back(li);
            vals.push_back(row_entries[(size_t)diag_pos].second);
            sub.row_ptr[(size_t)li + 1] = (int)cols.size();
        }

        sub.col_ind = std::move(cols);
        sub.values = std::move(vals);
        sub.nnz = (int)sub.values.size();
        return sub;
    }

    static double exact_cholesky_residual_fro_norm(
        const ichol::matrix::CsrMatrix<double> &A_sub,
        const ichol::matrix::CsrMatrix<double> &L)
    {
        const int n = A_sub.num_rows;
        std::vector<double> A_dense((size_t)n * (size_t)n, 0.0);
        std::vector<double> L_dense((size_t)n * (size_t)n, 0.0);

        for (int i = 0; i < n; ++i)
        {
            for (int p = A_sub.row_ptr[i]; p < A_sub.row_ptr[i + 1]; ++p)
            {
                const int j = A_sub.col_ind[p];
                const double v = A_sub.values[p];
                A_dense[(size_t)i * (size_t)n + (size_t)j] = v;
                A_dense[(size_t)j * (size_t)n + (size_t)i] = v;
            }
            for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p)
            {
                const int j = L.col_ind[p];
                L_dense[(size_t)i * (size_t)n + (size_t)j] = L.values[p];
            }
        }

        double sum_sq = 0.0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                const int kmax = (i < j) ? i : j;
                double llt_ij = 0.0;
                for (int k = 0; k <= kmax; ++k)
                {
                    llt_ij += L_dense[(size_t)i * (size_t)n + (size_t)k] * L_dense[(size_t)j * (size_t)n + (size_t)k];
                }
                const double r = A_dense[(size_t)i * (size_t)n + (size_t)j] - llt_ij;
                sum_sq += r * r;
            }
        }
        return std::sqrt(sum_sq);
    }

    static std::filesystem::path ensure_test_artifact_dir()
    {
        const auto dir = std::filesystem::path("test_artifacts");
        std::filesystem::create_directories(dir);
        return dir;
    }

    static void write_residual_history_txt(
        const std::filesystem::path &txt_path,
        const std::vector<double> &pcg_rel_residuals,
        const std::vector<double> &mpcg_rel_residuals,
        const std::vector<double> &mpcg20_rel_residuals)
    {
        std::ofstream out(txt_path);
        if (!out)
            throw std::runtime_error("failed to open convergence txt for writing");

        out << "# iteration pcg_rel_residual mpcg_rel_residual mpcg20_rel_residual\n";
        const size_t nrows = std::max(
            pcg_rel_residuals.size(),
            std::max(mpcg_rel_residuals.size(), mpcg20_rel_residuals.size()));
        for (size_t i = 0; i < nrows; ++i)
        {
            out << i << " ";
            if (i < pcg_rel_residuals.size())
                out << pcg_rel_residuals[i];
            else
                out << "nan";
            out << " ";
            if (i < mpcg_rel_residuals.size())
                out << mpcg_rel_residuals[i];
            else
                out << "nan";
            out << " ";
            if (i < mpcg20_rel_residuals.size())
                out << mpcg20_rel_residuals[i];
            else
                out << "nan";
            out << "\n";
        }
    }

} // namespace

TEST_F(MPCGTest, 3D_Poisson)
{
    const int nloc = MPCGTest::n;
    ichol::matrix::CsrMatrix<double> A = ichol::io::gen_3dpoi<double>(nloc);

    std::vector<double> b(A.num_rows);
    {
        std::mt19937 rng(12345);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < A.num_rows; ++i)
        {
            b[i] = dist(rng);
        }
    }
    apply_unit_col_prescaling_system(A, b);

    std::vector<double> x(A.num_rows, 0.0);

    ichol::solver::PCGParams params;
    params.maxits = 500;
    params.tol = 1e-10;
    params.restart = 1; // truncated
    params.prec_gemm = ichol::solver::ComputePrecision::FP64;
    params.prec_spmm = ichol::solver::ComputePrecision::FP64;
    params.prec_precond = ichol::solver::ComputePrecision::FP64;

    params.store_P_hist = ichol::solver::ComputePrecision::FP64;
    params.store_W_hist = ichol::solver::ComputePrecision::FP64;

    std::vector<ichol::precond::ADIContext> ctxs;
    ctxs.push_back({nloc, ichol::precond::ADIDirection3D::X});
    ctxs.push_back({nloc, ichol::precond::ADIDirection3D::Y});
    ctxs.push_back({nloc, ichol::precond::ADIDirection3D::Z});

    std::vector<ichol::precond::PrecondApply> preconds;
    preconds.reserve(3);
    for (int t = 0; t < 3; ++t)
    {
        ichol::precond::PrecondApply P;
        P.apply = &ichol::precond::apply_adi3d_dir;
        P.ctx = static_cast<void *>(&ctxs[t]);
        preconds.push_back(P);
    }

    const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
    auto t0 = std::chrono::high_resolution_clock::now();

    ichol::solver::PCGResult result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds,
        b, x,
        params);

    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    EXPECT_GT(result.iterations, 0);
    EXPECT_LT(result.finalRes, params.tol * (bnorm > 0.0 ? bnorm : 1.0));

    std::cout << "[MPCG 3D_Poisson] n=" << nloc
              << " N=" << A.num_rows
              << " restart=" << params.restart
              << " iters=" << result.iterations
              << " finalRes=" << result.finalRes
              << " time=" << secs << "s\n";
}

TEST(MPCG, 2D_Poisson_Asymmetric)
{
    const int n = 512;
    const double epsilon = 0.5;
    auto A = ichol::io::gen_2dpoi<double>(n, epsilon);
    auto b = ichol::io::rhs_2d_poisson_manufactured(A, n);
    apply_unit_col_prescaling_system(A, b);
    std::vector<double> x(A.num_rows, 0.0);

    ichol::matrix::CsrMatrix<double> M1, M2;
    ichol::precond::generateADIPreconditioners(n, 1.0, M1, M2);

    ichol::solver::PCGParams params;
    params.maxits = 5000;
    params.tol = 1e-10;
    params.restart = 0;
    params.store_P_hist = ichol::solver::ComputePrecision::FP64;
    params.store_W_hist = ichol::solver::ComputePrecision::FP64;

    ichol::precond::ADI2DContext ctxX{n, ichol::precond::ADIDirection2D::X, 1.0};
    ichol::precond::ADI2DContext ctxY{n, ichol::precond::ADIDirection2D::Y, epsilon};

    std::vector<ichol::precond::PrecondApply> preconds(2);
    preconds[0] = {&ichol::precond::apply_adi2d_dir, &ctxX};
    preconds[1] = {&ichol::precond::apply_adi2d_dir, &ctxY};

    auto t_mpcg_start = std::chrono::high_resolution_clock::now();
    ichol::solver::PCGResult result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds,
        b, x,
        params);
    auto t_mpcg_end = std::chrono::high_resolution_clock::now();
    const double mpcg_secs = std::chrono::duration<double>(t_mpcg_end - t_mpcg_start).count();

    std::cout << "[MPCG 2D_Poisson_Asymmetric] n=" << n
              << " N=" << A.num_rows
              << " restart=" << params.restart
              << " iters=" << result.iterations
              << " finalRes=" << result.finalRes
              << " time=" << mpcg_secs << "s\n";

    std::vector<double> h_D(A.num_rows, 1.0); // No scaling in this test

    ichol::precond::PrecondApply adi_p;
    adi_p.apply = &ichol::precond::apply_adi2d_dir;
    adi_p.ctx = &ctxX;
    params.custom_precond = &adi_p;

    auto t_pcg_start = std::chrono::high_resolution_clock::now();
    ichol::solver::PCGResult pcg_result = ichol::solver::pcg<double>(
        A.row_ptr, A.col_ind, A.values,
        M1.row_ptr, M1.col_ind, M1.values,
        b, x, h_D, params);

    auto t_pcg_end = std::chrono::high_resolution_clock::now();
    const double pcg_secs = std::chrono::duration<double>(t_pcg_end - t_pcg_start).count();
    std::cout << "[PCG 2D_Poisson_Asymmetric] n=" << n
              << " N=" << A.num_rows
              << " restart=" << params.restart
              << " iters=" << pcg_result.iterations
              << " finalRes=" << pcg_result.finalRes
              << " time=" << pcg_secs << "s\n";
}

TEST(MPCG, 2D_Poisson_MixedPrecisionSmoke)
{
    const int n = 64;
    const double epsilon = 0.5;
    auto A = ichol::io::gen_2dpoi<double>(n, epsilon);
    auto b = ichol::io::rhs_2d_poisson_manufactured(A, n);
    apply_unit_col_prescaling_system(A, b);
    std::vector<double> x(A.num_rows, 0.0);

    ichol::solver::PCGParams params;
    params.maxits = 400;
    params.tol = 1e-6;
    params.restart = 1;
    params.prec_gemm = ichol::solver::ComputePrecision::FP64;
    params.prec_spmm = ichol::solver::ComputePrecision::FP32;
    params.prec_precond = ichol::solver::ComputePrecision::FP32;

    ichol::precond::ADI2DContext ctxX{n, ichol::precond::ADIDirection2D::X, 1.0};
    ichol::precond::ADI2DContext ctxY{n, ichol::precond::ADIDirection2D::Y, epsilon};

    std::vector<ichol::precond::PrecondApply> preconds(2);
    preconds[0] = {&ichol::precond::apply_adi2d_dir, &ctxX};
    preconds[1] = {&ichol::precond::apply_adi2d_dir, &ctxY};

    const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
    const auto result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds,
        b, x,
        params);

    EXPECT_GT(result.iterations, 0);
    EXPECT_TRUE(std::isfinite(result.finalRes));
    EXPECT_LT(result.finalRes, params.tol * (bnorm > 0.0 ? bnorm : 1.0));
}

TEST(MPCG, 2D_Poisson_DomainDecomposition)
{
    const int n = 100;
    auto A = ichol::io::gen_2dpoi<double>(n, 1.0);
    std::vector<double> b(A.num_rows);
    {
        std::mt19937 rng(12345);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < A.num_rows; ++i)
        {
            b[i] = dist(rng);
        }
    }
    apply_unit_col_prescaling_system(A, b);
    std::vector<double> x(A.num_rows, 0.0);

    const ichol::precond::GridShape global_shape{n, n, 1};
    const ichol::precond::SubdomainSize subdomain_size{n / 2, n, 1};
    const auto regions = ichol::precond::partition_subdomains(global_shape, subdomain_size);

    auto ctx_deleter = [](ichol::precond::SubdomainSpSVContext *p)
    {
        ichol::precond::destroy_subdomain_spsv_context(p);
    };

    ichol::precond::SubdomainPreconditionerOptions options;
    options.kind = ichol::precond::SubdomainPreconditionerKind::ExactCholesky;
    options.precision = ichol::solver::ComputePrecision::FP64;

    std::unique_ptr<ichol::precond::SubdomainSpSVContext, decltype(ctx_deleter)> ctx1(
        ichol::precond::create_subdomain_spsv_context(A, global_shape, regions[0], options),
        ctx_deleter);
    std::unique_ptr<ichol::precond::SubdomainSpSVContext, decltype(ctx_deleter)> ctx2(
        ichol::precond::create_subdomain_spsv_context(A, global_shape, regions[1], options),
        ctx_deleter);

    auto time_precond_start = std::chrono::high_resolution_clock::now();
    std::vector<ichol::precond::PrecondApply> preconds(2);
    preconds[0] = {&ichol::precond::apply_subdomain_exact_spsv, ctx1.get()};
    preconds[1] = {&ichol::precond::apply_subdomain_exact_spsv, ctx2.get()};
    auto time_precond_end = std::chrono::high_resolution_clock::now();

    std::cout << "Preconditioner contexts created in "
              << std::chrono::duration<double>(time_precond_end - time_precond_start).count()
              << " seconds\n";

    if (options.kind == ichol::precond::SubdomainPreconditionerKind::ExactCholesky)
    {
        const ichol::precond::SubdomainRegion check_region{0, 16, 0, 16, 0, 1};
        auto A_sub = extract_lower_subdomain_csr(A, global_shape, check_region);

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
        auto L = ichol::numeric::incomplete_cholesky_preconditioner<double>(A_sub, sym_plan, num_plan, ic_opts);
        const double err = exact_cholesky_residual_fro_norm(A_sub, L);
        const double a_norm = std::sqrt(std::inner_product(A_sub.values.begin(), A_sub.values.end(), A_sub.values.begin(), 0.0));
        const double rel = err / (a_norm > 0.0 ? a_norm : 1.0);
        std::cout << "[Exact Cholesky check] ||A-LL^T||_F=" << err << " rel=" << rel << "\n";
        EXPECT_LT(rel, 1e-8);
    }

    ichol::solver::PCGParams params;
    params.maxits = 500;
    params.tol = 1e-10;
    params.restart = 0;

    // keep numerics clean
    params.prec_gemm = ichol::solver::ComputePrecision::FP64;
    params.prec_spmm = ichol::solver::ComputePrecision::FP64;
    params.prec_precond = ichol::solver::ComputePrecision::FP64;

    params.store_P_hist = ichol::solver::ComputePrecision::FP64;
    params.store_W_hist = ichol::solver::ComputePrecision::FP64;

    auto t_mpcg_start = std::chrono::high_resolution_clock::now();
    auto result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds, b, x, params);
    auto t_mpcg_end = std::chrono::high_resolution_clock::now();
    const double mpcg_secs = std::chrono::duration<double>(t_mpcg_end - t_mpcg_start).count();

    std::cout << "[MPCG 2D_Poisson_DomainDecomposition] n=" << n
              << " N=" << A.num_rows
              << " restart=" << params.restart
              << " iters=" << result.iterations
              << " finalRes=" << result.finalRes
              << " time=" << mpcg_secs << "s\n";
}

TEST(MPCG, 3D_Poisson_DD)
{
    const int n = 32;
    auto A = ichol::io::gen_3dpoi<double>(n);

    std::vector<double> b(A.num_rows);
    {
        std::mt19937 rng(20260303);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < A.num_rows; ++i)
            b[i] = dist(rng);
    }
    apply_unit_col_prescaling_system(A, b);
    std::vector<double> x(A.num_rows, 0.0);

    const ichol::precond::GridShape global_shape{n, n, n};
    const ichol::precond::SubdomainSize subdomain_size{16, 16, 16};
    // const ichol::precond::SubdomainSize subdomain_size{8, 8, 8};
    const auto regions = ichol::precond::partition_subdomains(global_shape, subdomain_size);
    ASSERT_FALSE(regions.empty());

    ichol::precond::SubdomainPreconditionerOptions options;
    options.kind = ichol::precond::SubdomainPreconditionerKind::ExactCholesky;
    options.precision = ichol::solver::ComputePrecision::FP32;

    auto ctx_deleter = [](ichol::precond::SubdomainSpSVContext *p)
    {
        ichol::precond::destroy_subdomain_spsv_context(p);
    };

    auto t_mpcg_precond_start = std::chrono::high_resolution_clock::now();
    std::vector<std::future<ichol::precond::SubdomainSpSVContext *>> futures;
    futures.reserve(regions.size());
    for (const auto &reg : regions)
    {
        futures.emplace_back(std::async(std::launch::async, [&A, &global_shape, reg, options]()
                                        { return ichol::precond::create_subdomain_spsv_context(A, global_shape, reg, options); }));
    }

    std::vector<std::unique_ptr<ichol::precond::SubdomainSpSVContext, decltype(ctx_deleter)>> contexts;
    contexts.reserve(regions.size());
    for (auto &f : futures)
    {
        contexts.emplace_back(f.get(), ctx_deleter);
    }

    std::vector<ichol::precond::PrecondApply> preconds;
    preconds.reserve(contexts.size());
    for (auto &ctx : contexts)
    {
        preconds.push_back({&ichol::precond::apply_subdomain_exact_spsv, ctx.get()});
    }
    auto t_mpcg_precond_end = std::chrono::high_resolution_clock::now();
    const double mpcg_precond_secs = std::chrono::duration<double>(t_mpcg_precond_end - t_mpcg_precond_start).count();

    ichol::solver::PCGParams params;
    params.maxits = 100;
    params.tol = 1e-10;
    params.restart = 20;
    params.prec_gemm = ichol::solver::ComputePrecision::FP64;
    params.prec_spmm = ichol::solver::ComputePrecision::FP64;
    params.prec_precond = ichol::solver::ComputePrecision::FP32;
    params.store_P_hist = ichol::solver::ComputePrecision::FP64;
    params.store_W_hist = ichol::solver::ComputePrecision::FP64;

    const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
    auto t_mpcg_solve_start = std::chrono::high_resolution_clock::now();
    auto result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds, b, x, params);
    auto t_mpcg_solve_end = std::chrono::high_resolution_clock::now();
    const double mpcg_solve_secs = std::chrono::duration<double>(t_mpcg_solve_end - t_mpcg_solve_start).count();
    const double mpcg_e2e_secs = mpcg_precond_secs + mpcg_solve_secs;

    std::cout << "[MPCG 3D_Poisson_DD] n=" << n
              << " N=" << A.num_rows
              << " subdomains=" << regions.size()
              << " subdomain_size=(" << subdomain_size.w << "," << subdomain_size.h << "," << subdomain_size.d << ")"
              << " iters=" << result.iterations
              << " finalRes=" << result.finalRes
              << " end_to_end=" << mpcg_e2e_secs << "s"
              << " solve_time=" << mpcg_solve_secs << "s"
              << " precond_gen_time=" << mpcg_precond_secs << "s\n";

    std::vector<double> x_pcg(A.num_rows, 0.0);
    std::vector<double> h_D(A.num_rows, 1.0);
    auto t_pcg_precond_start = std::chrono::high_resolution_clock::now();
    auto L_block = ichol::precond::build_block_diagonal_exact_preconditioner_3d(
        A, n, subdomain_size.w, subdomain_size.h, subdomain_size.d);
    auto t_pcg_precond_end = std::chrono::high_resolution_clock::now();
    const double pcg_precond_secs = std::chrono::duration<double>(t_pcg_precond_end - t_pcg_precond_start).count();

    ichol::solver::PCGParams pcg_params = params;
    pcg_params.custom_precond = nullptr;
    pcg_params.maxits = 100;

    auto t_pcg_solve_start = std::chrono::high_resolution_clock::now();
    auto pcg_result = ichol::solver::pcg_cusparse_spsv<double>(
        A.row_ptr, A.col_ind, A.values,
        L_block.row_ptr, L_block.col_ind, L_block.values,
        b, x_pcg, h_D, pcg_params);
    auto t_pcg_solve_end = std::chrono::high_resolution_clock::now();
    const double pcg_solve_secs = std::chrono::duration<double>(t_pcg_solve_end - t_pcg_solve_start).count();
    const double pcg_e2e_secs = pcg_precond_secs + pcg_solve_secs;

    std::cout << "[PCG 3D_Poisson_BlockDiagonal] n=" << n
              << " N=" << A.num_rows
              << " subdomain_size=(" << subdomain_size.w << "," << subdomain_size.h << "," << subdomain_size.d << ")"
              << " iters=" << pcg_result.iterations
              << " finalRes=" << pcg_result.finalRes
              << " end_to_end=" << pcg_e2e_secs << "s"
              << " solve_time=" << pcg_solve_secs << "s"
              << " precond_gen_time=" << pcg_precond_secs << "s\n";

    EXPECT_GT(result.iterations, 0);
    EXPECT_LT(result.finalRes, params.tol * (bnorm > 0.0 ? bnorm : 1.0));
    EXPECT_GT(pcg_result.iterations, 0);
    EXPECT_LT(pcg_result.finalRes, pcg_params.tol * (bnorm > 0.0 ? bnorm : 1.0));
}

TEST(MPCG, 2D_Poisson_DD)
{
    const int n = 200;
    auto A = ichol::io::gen_2dpoi<double>(n, 1.0);

    std::vector<double> b(A.num_rows);
    {
        std::mt19937 rng(20260303);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < A.num_rows; ++i)
            b[i] = dist(rng);
    }
    apply_unit_col_prescaling_system(A, b);
    std::vector<double> x(A.num_rows, 0.0);

    const ichol::precond::GridShape global_shape{n, n, 1};
    const ichol::precond::SubdomainSize subdomain_size{8, 8, 1};
    const auto regions = ichol::precond::partition_subdomains(global_shape, subdomain_size);
    ASSERT_FALSE(regions.empty());

    ichol::precond::SubdomainPreconditionerOptions options;
    options.kind = ichol::precond::SubdomainPreconditionerKind::ExactCholesky;
    options.precision = ichol::solver::ComputePrecision::FP64;

    auto ctx_deleter = [](ichol::precond::SubdomainSpSVContext *p)
    {
        ichol::precond::destroy_subdomain_spsv_context(p);
    };

    std::vector<std::unique_ptr<ichol::precond::SubdomainSpSVContext, decltype(ctx_deleter)>> contexts;
    contexts.reserve(regions.size());
    for (const auto &reg : regions)
        contexts.emplace_back(
            ichol::precond::create_subdomain_spsv_context(A, global_shape, reg, options),
            ctx_deleter);

    std::vector<ichol::precond::PrecondApply> preconds;
    preconds.reserve(contexts.size());
    for (auto &ctx : contexts)
        preconds.push_back({&ichol::precond::apply_subdomain_exact_spsv, ctx.get()});

    ichol::solver::PCGParams params;
    params.maxits = 30;
    params.tol = 1e-10;
    params.restart = 0;
    params.prec_gemm = ichol::solver::ComputePrecision::FP64;
    params.prec_spmm = ichol::solver::ComputePrecision::FP64;
    params.prec_precond = ichol::solver::ComputePrecision::FP64;
    params.store_P_hist = ichol::solver::ComputePrecision::FP64;
    params.store_W_hist = ichol::solver::ComputePrecision::FP64;

    const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds, b, x, params);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[MPCG 2D_Poisson_DD] n=" << n
              << " N=" << A.num_rows
              << " subdomains=" << regions.size()
              << " subdomain_size=(" << subdomain_size.w << "," << subdomain_size.h << ",1)"
              << " iters=" << result.iterations
              << " finalRes=" << result.finalRes
              << " time=" << secs << "s\n";

    std::vector<double> x_pcg(A.num_rows, 0.0);
    std::vector<double> h_D(A.num_rows, 1.0);
    auto L_block = ichol::precond::build_block_diagonal_exact_preconditioner_2d(
        A, n, subdomain_size.w, subdomain_size.h);

    ichol::solver::PCGParams pcg_params = params;
    pcg_params.custom_precond = nullptr;

    pcg_params.maxits = 400;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto pcg_result = ichol::solver::pcg_cusparse_spsv<double>(
        A.row_ptr, A.col_ind, A.values,
        L_block.row_ptr, L_block.col_ind, L_block.values,
        b, x_pcg, h_D, pcg_params);
    auto t3 = std::chrono::high_resolution_clock::now();
    const double pcg_secs = std::chrono::duration<double>(t3 - t2).count();

    std::cout << "[PCG 2D_Poisson_BlockDiagonal_8xStar] n=" << n
              << " N=" << A.num_rows
              << " subdomain_size=(" << subdomain_size.w << "," << subdomain_size.h << ",1)"
              << " iters=" << pcg_result.iterations
              << " finalRes=" << pcg_result.finalRes
              << " time=" << pcg_secs << "s\n";

    EXPECT_GT(result.iterations, 0);
    EXPECT_LT(result.finalRes, params.tol * (bnorm > 0.0 ? bnorm : 1.0));
    EXPECT_GT(pcg_result.iterations, 0);
    EXPECT_LT(pcg_result.finalRes, pcg_params.tol * (bnorm > 0.0 ? bnorm : 1.0));
}

TEST(MPCG, 3D_Poisson_DD_Scale)
{
    const int n = 80;
    auto A = ichol::io::gen_3dpoi<double>(n);

    std::vector<double> b(A.num_rows);
    {
        std::mt19937 rng(20260303);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < A.num_rows; ++i)
            b[i] = dist(rng);
    }
    apply_unit_col_prescaling_system(A, b);

    const ichol::precond::GridShape global_shape{n, n, n};
    const std::vector<int> subdomain_extents{16, 32, 64};
    const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));

    auto ctx_deleter = [](ichol::precond::SubdomainSpSVContext *p)
    {
        ichol::precond::destroy_subdomain_spsv_context(p);
    };

    for (const int extent : subdomain_extents)
    {
        SCOPED_TRACE("subdomain_extent=" + std::to_string(extent));

        std::vector<double> x(A.num_rows, 0.0);
        const ichol::precond::SubdomainSize subdomain_size{extent, extent, extent};
        const auto regions = ichol::precond::partition_subdomains(global_shape, subdomain_size);
        ASSERT_FALSE(regions.empty());

        ichol::precond::SubdomainPreconditionerOptions options;
        options.kind = ichol::precond::SubdomainPreconditionerKind::ExactCholesky;
        options.precision = ichol::solver::ComputePrecision::FP32;

        auto t_mpcg_precond_start = std::chrono::high_resolution_clock::now();
        auto raw_contexts = ichol::precond::create_subdomain_preconditioner_contexts_parallel(
            A, global_shape, regions, options);

        std::vector<std::unique_ptr<ichol::precond::SubdomainSpSVContext, decltype(ctx_deleter)>> contexts;
        contexts.reserve(raw_contexts.size());
        for (auto *ctx : raw_contexts)
            contexts.emplace_back(ctx, ctx_deleter);

        std::vector<ichol::precond::PrecondApply> preconds;
        preconds.reserve(contexts.size());
        for (auto &ctx : contexts)
            preconds.push_back({&ichol::precond::apply_subdomain_exact_spsv, ctx.get()});
        auto t_mpcg_precond_end = std::chrono::high_resolution_clock::now();
        const double mpcg_precond_secs = std::chrono::duration<double>(t_mpcg_precond_end - t_mpcg_precond_start).count();

        ichol::solver::PCGParams params;
        params.maxits = 60;
        params.tol = 1e-10;
        params.restart = 0;
        params.prec_gemm = ichol::solver::ComputePrecision::FP64;
        params.prec_spmm = ichol::solver::ComputePrecision::FP64;
        // params.prec_precond = ichol::solver::ComputePrecision::FP64;
        params.prec_precond = options.precision;
        params.store_P_hist = ichol::solver::ComputePrecision::FP64;
        params.store_W_hist = ichol::solver::ComputePrecision::FP64;
        params.verbose = true;

        auto t_mpcg_solve_start = std::chrono::high_resolution_clock::now();
        auto result = ichol::solver::mpcg<double>(
            A.row_ptr, A.col_ind, A.values,
            preconds, b, x, params);
        auto t_mpcg_solve_end = std::chrono::high_resolution_clock::now();
        const double mpcg_solve_secs = std::chrono::duration<double>(t_mpcg_solve_end - t_mpcg_solve_start).count();
        const double mpcg_e2e_secs = mpcg_precond_secs + mpcg_solve_secs;

        std::cout << "[MPCG 3D_Poisson_DD_Scale] n=" << n
                  << " N=" << A.num_rows
                  << " subdomains=" << regions.size()
                  << " subdomain_size=(" << subdomain_size.w << "," << subdomain_size.h << "," << subdomain_size.d << ")"
                  << " iters=" << result.iterations
                  << " finalRes=" << result.finalRes
                  << " end_to_end=" << mpcg_e2e_secs << "s"
                  << " solve_time=" << mpcg_solve_secs << "s"
                  << " precond_gen_time=" << mpcg_precond_secs << "s\n";

        EXPECT_GT(result.iterations, 0);
        EXPECT_LT(result.finalRes, params.tol * (bnorm > 0.0 ? bnorm : 1.0));
    }
}

TEST(MPCG, 3D_Poisson_DD_ConvergencePlot)
{
    const int n = 64;
    auto A = ichol::io::gen_3dpoi<double>(n);

    std::vector<double> b(A.num_rows);
    {
        std::mt19937 rng(20260322);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < A.num_rows; ++i)
            b[i] = dist(rng);
    }
    apply_unit_col_prescaling_system(A, b);

    std::vector<double> x_mpcg(A.num_rows, 0.0);
    std::vector<double> x_mpcg20(A.num_rows, 0.0);
    std::vector<double> x_pcg(A.num_rows, 0.0);

    const ichol::precond::GridShape global_shape{n, n, n};
    const ichol::precond::SubdomainSize subdomain_size{16, 16, 16};
    const auto regions = ichol::precond::partition_subdomains(global_shape, subdomain_size);
    ASSERT_FALSE(regions.empty());

    ichol::precond::SubdomainPreconditionerOptions options;
    options.kind = ichol::precond::SubdomainPreconditionerKind::ExactCholesky;
    options.precision = ichol::solver::ComputePrecision::FP64;

    auto ctx_deleter = [](ichol::precond::SubdomainSpSVContext *p)
    {
        ichol::precond::destroy_subdomain_spsv_context(p);
    };

    auto raw_contexts = ichol::precond::create_subdomain_preconditioner_contexts_parallel(
        A, global_shape, regions, options);

    std::vector<std::unique_ptr<ichol::precond::SubdomainSpSVContext, decltype(ctx_deleter)>> contexts;
    contexts.reserve(raw_contexts.size());
    for (auto *ctx : raw_contexts)
        contexts.emplace_back(ctx, ctx_deleter);

    std::vector<ichol::precond::PrecondApply> preconds;
    preconds.reserve(contexts.size());
    for (auto &ctx : contexts)
        preconds.push_back({&ichol::precond::apply_subdomain_exact_spsv, ctx.get()});

    ichol::solver::PCGParams params;
    params.maxits = 50;
    params.tol = 1e-10;
    params.restart = 0;
    params.prec_gemm = ichol::solver::ComputePrecision::FP64;
    params.prec_spmm = ichol::solver::ComputePrecision::FP64;
    params.prec_precond = options.precision;
    params.store_P_hist = ichol::solver::ComputePrecision::FP64;
    params.store_W_hist = ichol::solver::ComputePrecision::FP64;
    params.verbose = true;

    const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
    auto mpcg_result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds, b, x_mpcg, params);

    ichol::solver::PCGParams params20 = params;
    params20.restart = 20;
    auto mpcg20_result = ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds, b, x_mpcg20, params20);

    std::vector<double> h_D(A.num_rows, 1.0);
    auto L_block = ichol::precond::build_block_diagonal_exact_preconditioner_3d(
        A, n, subdomain_size.w, subdomain_size.h, subdomain_size.d);

    ichol::solver::PCGParams pcg_params = params;
    pcg_params.custom_precond = nullptr;
    pcg_params.maxits = 100;
    auto pcg_result = ichol::solver::pcg_cusparse_spsv<double>(
        A.row_ptr, A.col_ind, A.values,
        L_block.row_ptr, L_block.col_ind, L_block.values,
        b, x_pcg, h_D, pcg_params);

    const auto artifact_dir = ensure_test_artifact_dir();
    const auto txt_path = artifact_dir / "pcg_vs_mpcg_3d_poisson_dd_convergence.txt";
    const auto png_path = artifact_dir / "pcg_vs_mpcg_3d_poisson_dd_convergence.png";
    const auto script_path = std::filesystem::path("python/plot_pcg_vs_mpcg_convergence.py");

    write_residual_history_txt(
        txt_path,
        pcg_result.relResiduals,
        mpcg_result.relResiduals,
        mpcg20_result.relResiduals);

    std::cout << "[MPCG 3D_Poisson_DD_ConvergencePlot] txt=" << txt_path
              << " plot_script=" << script_path
              << " png=" << png_path
              << " mpcg_iters=" << mpcg_result.iterations
              << " mpcg20_iters=" << mpcg20_result.iterations
              << " pcg_iters=" << pcg_result.iterations << "\n";

    EXPECT_GT(mpcg_result.iterations, 0);
    EXPECT_GT(mpcg20_result.iterations, 0);
    EXPECT_GT(pcg_result.iterations, 0);
    EXPECT_GT(mpcg_result.relResiduals.size(), 1u);
    EXPECT_GT(mpcg20_result.relResiduals.size(), 1u);
    EXPECT_GT(pcg_result.relResiduals.size(), 1u);
    EXPECT_LT(mpcg_result.finalRes, params.tol * (bnorm > 0.0 ? bnorm : 1.0));
    EXPECT_LT(mpcg20_result.finalRes, params20.tol * (bnorm > 0.0 ? bnorm : 1.0));
    EXPECT_LT(pcg_result.finalRes, pcg_params.tol * (bnorm > 0.0 ? bnorm : 1.0));
}

TEST(MPCG, PETSc_3D_Poisson_KSP)
{
    ensure_petsc_initialized();
    constexpr double kTol = 1e-10;

    PetscInt nx = 80;
    PetscInt ny = nx;
    PetscInt nz = nx;
    petsc_check(PetscOptionsGetInt(nullptr, nullptr, "-nx", &nx, nullptr), "PetscOptionsGetInt(-nx)");
    petsc_check(PetscOptionsGetInt(nullptr, nullptr, "-ny", &ny, nullptr), "PetscOptionsGetInt(-ny)");
    petsc_check(PetscOptionsGetInt(nullptr, nullptr, "-nz", &nz, nullptr), "PetscOptionsGetInt(-nz)");

    ASSERT_GT(nx, 0);
    ASSERT_GT(ny, 0);
    ASSERT_GT(nz, 0);
    ASSERT_EQ(nx, ny) << "PETSc Poisson test reuses ichol::io::gen_3dpoi, so -nx must equal -ny";
    ASSERT_EQ(nx, nz) << "PETSc Poisson test reuses ichol::io::gen_3dpoi, so -nx must equal -nz";

    const int n = static_cast<int>(nx);
    const auto A_csr = ichol::io::gen_3dpoi<double>(n);
    const PetscInt n_global = static_cast<PetscInt>(A_csr.num_rows);

    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    ASSERT_EQ(MPI_Comm_rank(PETSC_COMM_WORLD, &rank), MPI_SUCCESS);
    ASSERT_EQ(MPI_Comm_size(PETSC_COMM_WORLD, &size), MPI_SUCCESS);

    const PetscInt base = n_global / static_cast<PetscInt>(size);
    const PetscInt rem = n_global % static_cast<PetscInt>(size);
    const PetscInt rstart = static_cast<PetscInt>(rank) * base +
                            std::min<PetscInt>(static_cast<PetscInt>(rank), rem);
    const PetscInt n_local = base + (static_cast<PetscInt>(rank) < rem ? 1 : 0);
    const PetscInt rend = rstart + n_local;

    std::vector<PetscInt> d_nnz(static_cast<size_t>(n_local), 0);
    std::vector<PetscInt> o_nnz(static_cast<size_t>(n_local), 0);
    for (PetscInt row = rstart; row < rend; ++row)
    {
        PetscInt diag_nnz = 0;
        PetscInt offdiag_nnz = 0;
        for (int p = A_csr.row_ptr[static_cast<size_t>(row)];
             p < A_csr.row_ptr[static_cast<size_t>(row) + 1]; ++p)
        {
            const PetscInt col = static_cast<PetscInt>(A_csr.col_ind[static_cast<size_t>(p)]);
            if (col >= rstart && col < rend)
                ++diag_nnz;
            else
                ++offdiag_nnz;
        }
        d_nnz[static_cast<size_t>(row - rstart)] = diag_nnz;
        o_nnz[static_cast<size_t>(row - rstart)] = offdiag_nnz;
    }

    Mat A_petsc = nullptr;
    Vec b = nullptr;
    Vec x = nullptr;
    Vec r = nullptr;
    KSP ksp = nullptr;

    auto petsc_cleanup = [&]()
    {
        if (ksp)
            (void)KSPDestroy(&ksp);
        if (r)
            (void)VecDestroy(&r);
        if (x)
            (void)VecDestroy(&x);
        if (b)
            (void)VecDestroy(&b);
        if (A_petsc)
            (void)MatDestroy(&A_petsc);
    };

    try
    {
        petsc_check(MatCreate(PETSC_COMM_WORLD, &A_petsc), "MatCreate");
        petsc_check(MatSetSizes(A_petsc, n_local, n_local, n_global, n_global), "MatSetSizes");
        petsc_check(MatSetFromOptions(A_petsc), "MatSetFromOptions");
        petsc_check(MatXAIJSetPreallocation(A_petsc, 1, d_nnz.data(), o_nnz.data(), nullptr, nullptr),
                    "MatXAIJSetPreallocation");

        for (PetscInt row = rstart; row < rend; ++row)
        {
            const int row_begin = A_csr.row_ptr[static_cast<size_t>(row)];
            const int row_end = A_csr.row_ptr[static_cast<size_t>(row) + 1];
            const PetscInt ncols = static_cast<PetscInt>(row_end - row_begin);
            std::vector<PetscInt> cols(static_cast<size_t>(ncols));
            std::vector<PetscScalar> vals(static_cast<size_t>(ncols));
            for (PetscInt k = 0; k < ncols; ++k)
            {
                cols[static_cast<size_t>(k)] =
                    static_cast<PetscInt>(A_csr.col_ind[static_cast<size_t>(row_begin + k)]);
                vals[static_cast<size_t>(k)] =
                    static_cast<PetscScalar>(A_csr.values[static_cast<size_t>(row_begin + k)]);
            }
            petsc_check(MatSetValues(A_petsc, 1, &row, ncols, cols.data(), vals.data(), INSERT_VALUES),
                        "MatSetValues");
        }

        petsc_check(MatAssemblyBegin(A_petsc, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
        petsc_check(MatAssemblyEnd(A_petsc, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");
        petsc_check(MatSetOption(A_petsc, MAT_SYMMETRIC, PETSC_TRUE), "MatSetOption(MAT_SYMMETRIC)");
        petsc_check(MatSetOption(A_petsc, MAT_SPD, PETSC_TRUE), "MatSetOption(MAT_SPD)");

        petsc_check(VecCreate(PETSC_COMM_WORLD, &b), "VecCreate");
        petsc_check(VecSetSizes(b, n_local, n_global), "VecSetSizes");
        petsc_check(VecSetFromOptions(b), "VecSetFromOptions");
        petsc_check(VecDuplicate(b, &x), "VecDuplicate");
        petsc_check(VecDuplicate(b, &r), "VecDuplicate");
        petsc_check(VecSet(b, 1.0), "VecSet(b)");
        petsc_check(VecSet(x, 0.0), "VecSet(x)");

        petsc_check(KSPCreate(PETSC_COMM_WORLD, &ksp), "KSPCreate");
        petsc_check(KSPSetOperators(ksp, A_petsc, A_petsc), "KSPSetOperators");
        petsc_check(KSPSetType(ksp, KSPCG), "KSPSetType(KSPCG)");
        petsc_check(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE), "KSPSetInitialGuessNonzero");
        petsc_check(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED), "KSPSetNormType");

        PC pc = nullptr;
        petsc_check(KSPGetPC(ksp, &pc), "KSPGetPC");
        petsc_check(PCSetType(pc, PCGAMG), "PCSetType(PCGAMG)");

        petsc_check(KSPSetTolerances(ksp, kTol, PETSC_DEFAULT, PETSC_DEFAULT, 1000),
                    "KSPSetTolerances");
        petsc_check(KSPSetFromOptions(ksp), "KSPSetFromOptions");

        auto t_precond_start = std::chrono::high_resolution_clock::now();
        petsc_check(KSPSetUp(ksp), "KSPSetUp");
        auto t_precond_end = std::chrono::high_resolution_clock::now();

        auto t_solve_start = std::chrono::high_resolution_clock::now();
        petsc_check(KSPSolve(ksp, b, x), "KSPSolve");
        auto t_solve_end = std::chrono::high_resolution_clock::now();

        KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
        PetscInt its = 0;
        PetscReal ksp_res_norm = 0.0;
        PetscReal bnorm = 0.0;
        PetscReal true_res_norm = 0.0;
        petsc_check(KSPGetConvergedReason(ksp, &reason), "KSPGetConvergedReason");
        petsc_check(KSPGetIterationNumber(ksp, &its), "KSPGetIterationNumber");
        petsc_check(KSPGetResidualNorm(ksp, &ksp_res_norm), "KSPGetResidualNorm");
        petsc_check(VecNorm(b, NORM_2, &bnorm), "VecNorm(b)");
        petsc_check(MatMult(A_petsc, x, r), "MatMult");
        petsc_check(VecAYPX(r, -1.0, b), "VecAYPX");
        petsc_check(VecNorm(r, NORM_2, &true_res_norm), "VecNorm(r)");

        const double precond_secs = std::chrono::duration<double>(t_precond_end - t_precond_start).count();
        const double solve_secs = std::chrono::duration<double>(t_solve_end - t_solve_start).count();
        const double end_to_end_secs = precond_secs + solve_secs;
        const double rel_res = static_cast<double>(true_res_norm) /
                               (static_cast<double>(bnorm) > 0.0 ? static_cast<double>(bnorm) : 1.0);

        petsc_check(
            PetscPrintf(PETSC_COMM_WORLD,
                        "[PETSc 3D Poisson] grid=%" PetscInt_FMT "x%" PetscInt_FMT "x%" PetscInt_FMT
                        " N=%" PetscInt_FMT " ranks=%d"
                        " iters=%" PetscInt_FMT
                        " finalRes=%.6e relRes=%.6e kspRes=%.6e"
                        " end_to_end=%.6fs solve_time=%.6fs precond_gen_time=%.6fs\n",
                        nx, ny, nz, n_global, static_cast<int>(size), its,
                        static_cast<double>(true_res_norm), rel_res, static_cast<double>(ksp_res_norm),
                        end_to_end_secs, solve_secs, precond_secs),
            "PetscPrintf");

        EXPECT_GT(static_cast<int>(its), 0);
        EXPECT_GT(static_cast<int>(reason), 0);
        EXPECT_LT(rel_res, kTol);

        petsc_cleanup();
    }
    catch (...)
    {
        petsc_cleanup();
        throw;
    }
}

int main(int argc, char **argv)
{
    std::vector<std::string> petsc_args;
    petsc_args.emplace_back(argv[0] ? argv[0] : "test_mpcg");
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.rfind("--gtest_", 0) == 0)
        {
            if (arg.find('=') == std::string::npos && i + 1 < argc)
                ++i;
            continue;
        }
        if (arg.rfind("--case=", 0) == 0 || arg.rfind("--n=", 0) == 0)
            continue;
        if ((arg == "--case" || arg == "--n") && i + 1 < argc)
        {
            ++i;
            continue;
        }
        petsc_args.push_back(arg);
    }
    set_petsc_command_line(petsc_args);

    ::testing::InitGoogleTest(&argc, argv);

    std::string selected_case;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.rfind("--case=", 0) == 0)
        {
            selected_case = arg.substr(7);
        }
        else if (arg == "--case" && i + 1 < argc)
        {
            selected_case = argv[++i];
        }
        else if (arg.rfind("--n=", 0) == 0)
        {
            MPCGTest::n = std::stoi(arg.substr(4));
        }
        else
        {
            continue;
        }
    }

    if (!selected_case.empty())
        ::testing::GTEST_FLAG(filter) = selected_case;

    return RUN_ALL_TESTS();
}
