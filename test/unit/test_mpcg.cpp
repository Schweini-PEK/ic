// test/unit/test_mpcg.cpp
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <cuda_runtime.h>

#include "ichol/mtx_read.hpp"
#include "ichol/preconditioner.hpp"
#include "ichol/pcg.hpp"
#include "unit/test_utils.hpp"

class MPCGTest : public ::testing::Test
{
public:
    static int n;
};

int MPCGTest::n = 24;

namespace
{
    struct SubdomainExactCtx
    {
        int n_global = 0;   // global grid side length (n), global N = n*n
        int r0 = 0, r1 = 0; // rows [r0, r1)
        int c0 = 0, c1 = 0; // cols [c0, c1)

        int w = 0;    // subdomain width  = c1-c0
        int h = 0;    // subdomain height = r1-r0
        int bw = 0;   // half-bandwidth in local ordering = w
        int nsub = 0; // w*h

        // Banded Cholesky factor L in column-major band storage:
        // L(i,j) stored at band[(i-j) + (bw+1)*j], for i>=j and i-j<=bw.
        std::vector<double> Lband; // size (bw+1)*nsub
        std::vector<double> ytmp;  // size nsub

        // pinned host buffers for D2H/H2D
        double *h_rhs = nullptr;
        double *h_sol = nullptr;
    };

    static inline int gidx(const SubdomainExactCtx &ctx, int r, int c)
    {
        return r * ctx.n_global + c;
    }
    static inline int lidx(const SubdomainExactCtx &ctx, int r, int c)
    {
        return (r - ctx.r0) * ctx.w + (c - ctx.c0);
    }

    static inline double &band_at(std::vector<double> &a, int bw, int i, int j)
    {
        // requires i>=j and i-j<=bw
        return a[(i - j) + (bw + 1) * j];
    }

    static inline const double &band_at(const std::vector<double> &a, int bw, int i, int j)
    {
        return a[(i - j) + (bw + 1) * j];
    }

    static void banded_cholesky_factor(SubdomainExactCtx &ctx)
    {
        const int n = ctx.nsub;
        const int bw = ctx.bw;

        for (int j = 0; j < n; ++j)
        {
            // diagonal
            double sum = band_at(ctx.Lband, bw, j, j);
            const int k0 = std::max(0, j - bw);
            for (int k = k0; k < j; ++k)
            {
                const double ljk = band_at(ctx.Lband, bw, j, k);
                sum -= ljk * ljk;
            }
            if (sum <= 0.0)
                throw std::runtime_error("banded_cholesky_factor: non-positive pivot (matrix not SPD or build wrong)");
            band_at(ctx.Lband, bw, j, j) = std::sqrt(sum);

            // below diagonal within band
            const int i_max = std::min(n - 1, j + bw);
            for (int i = j + 1; i <= i_max; ++i)
            {
                double aij = band_at(ctx.Lband, bw, i, j); // currently A(i,j)
                // dot over k where both L(i,k) and L(j,k) exist
                const int kk0 = std::max({0, j - bw, i - bw});
                for (int k = kk0; k < j; ++k)
                {
                    const double lik = band_at(ctx.Lband, bw, i, k);
                    const double ljk = band_at(ctx.Lband, bw, j, k);
                    aij -= lik * ljk;
                }
                band_at(ctx.Lband, bw, i, j) = aij / band_at(ctx.Lband, bw, j, j);
            }
        }
    }

    static void banded_cholesky_solve(const SubdomainExactCtx &ctx, const double *rhs, double *x)
    {
        const int n = ctx.nsub;
        const int bw = ctx.bw;

        // forward: L y = rhs
        // use ctx.ytmp as workspace (const_cast to write)
        auto &y = const_cast<std::vector<double> &>(ctx.ytmp);

        for (int i = 0; i < n; ++i)
        {
            double sum = rhs[i];
            const int k0 = std::max(0, i - bw);
            for (int k = k0; k < i; ++k)
                sum -= band_at(ctx.Lband, bw, i, k) * y[k];
            y[i] = sum / band_at(ctx.Lband, bw, i, i);
        }

        // backward: L^T x = y
        for (int i = n - 1; i >= 0; --i)
        {
            double sum = y[i];
            const int k1 = std::min(n - 1, i + bw);
            for (int k = i + 1; k <= k1; ++k)
                sum -= band_at(ctx.Lband, bw, k, i) * x[k];
            x[i] = sum / band_at(ctx.Lband, bw, i, i);
        }
    }

    static void build_restricted_banded_from_global_csr_rect(const ichol::matrix::CsrMatrix<double> &A,
                                                             SubdomainExactCtx &ctx)
    {
        ctx.w = ctx.c1 - ctx.c0;
        ctx.h = ctx.r1 - ctx.r0;
        if (ctx.w <= 0 || ctx.h <= 0)
            throw std::runtime_error("build_restricted_banded_rect: empty subdomain");
        ctx.nsub = ctx.w * ctx.h;
        ctx.bw = ctx.w; // local row-major: neighbors differ by 1 or w

        ctx.Lband.assign((size_t)(ctx.bw + 1) * (size_t)ctx.nsub, 0.0);
        ctx.ytmp.assign((size_t)ctx.nsub, 0.0);

        const int N = A.num_rows;
        std::vector<int> g2l((size_t)N, -1);

        // build g2l map in local row-major
        for (int r = ctx.r0; r < ctx.r1; ++r)
            for (int c = ctx.c0; c < ctx.c1; ++c)
                g2l[gidx(ctx, r, c)] = lidx(ctx, r, c);

        // Fill lower-triangular band of the principal submatrix A(I,I)
        for (int r = ctx.r0; r < ctx.r1; ++r)
        {
            for (int c = ctx.c0; c < ctx.c1; ++c)
            {
                const int gi = gidx(ctx, r, c);
                const int li = g2l[gi];
                for (int kk = A.row_ptr[gi]; kk < A.row_ptr[gi + 1]; ++kk)
                {
                    const int gj = A.col_ind[kk];
                    const int lj = (gj >= 0 && gj < N) ? g2l[gj] : -1;
                    if (lj < 0)
                        continue; // outside subdomain => dropped (Dirichlet interface)

                    if (li < lj)
                        continue; // store lower only
                    const int d = li - lj;
                    if (d > ctx.bw)
                        continue; // should not happen for this local ordering
                    band_at(ctx.Lband, ctx.bw, li, lj) = A.values[kk];
                }
            }
        }

        // basic diagonal sanity
        for (int i = 0; i < ctx.nsub; ++i)
            if (band_at(ctx.Lband, ctx.bw, i, i) == 0.0)
                throw std::runtime_error("build_restricted_banded_rect: missing diagonal entry");
    }

    static void apply_subdomain_exact(void *vctx, const double *d_r, double *d_z, int N, cudaStream_t stream)
    {
        (void)N;
        auto *ctx = reinterpret_cast<SubdomainExactCtx *>(vctx);
        const int w = ctx->w;
        const int r0 = ctx->r0, r1 = ctx->r1;
        const int c0 = ctx->c0;
        const size_t row_bytes = (size_t)w * sizeof(double);

        for (int r = r0; r < r1; ++r)
        {
            const int loff = (r - r0) * w;
            const int goff = gidx(*ctx, r, c0);
            cudaMemcpyAsync(ctx->h_rhs + loff, d_r + goff, row_bytes, cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        banded_cholesky_solve(*ctx, ctx->h_rhs, ctx->h_sol);

        for (int r = r0; r < r1; ++r)
        {
            const int loff = (r - r0) * w;
            const int goff = gidx(*ctx, r, c0);
            cudaMemcpyAsync(d_z + goff, ctx->h_sol + loff, row_bytes, cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);
    }

    static bool rects_overlap(const SubdomainExactCtx &a, const SubdomainExactCtx &b)
    {
        const bool row_overlap = (a.r0 < b.r1) && (b.r0 < a.r1);
        const bool col_overlap = (a.c0 < b.c1) && (b.c0 < a.c1);
        return row_overlap && col_overlap;
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
    const int n = 64;
    const double epsilon = 0.5;
    auto A = ichol::io::gen_2dpoi<double>(n, epsilon);
    auto b = ichol::io::rhs_2d_poisson_manufactured(A, n);
    std::vector<double> x(A.num_rows, 0.0);

    ichol::matrix::CsrMatrix<double> M1, M2;
    ichol::precond::generateADIPreconditioners(n, 1.0, M1, M2);

    ichol::solver::PCGParams params;
    params.maxits = 5000;
    params.tol = 1e-10;
    params.restart = 1;
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

TEST(MPCG, 2D_Poisson_DomainDecomposition)
{
    const int n = 100; // 100x100 grid
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
    std::vector<double> x(A.num_rows, 0.0);

    SubdomainExactCtx ctx1;
    ctx1.n_global = n;
    ctx1.r0 = 0;
    ctx1.r1 = n;
    ctx1.c0 = 0;
    ctx1.c1 = n / 2; // left 100x50

    SubdomainExactCtx ctx2;
    ctx2.n_global = n;
    ctx2.r0 = 0;
    ctx2.r1 = n;
    ctx2.c0 = n / 2;
    ctx2.c1 = n; // right 100x50

    EXPECT_FALSE(rects_overlap(ctx1, ctx2))
        << "Subdomain overlap detected; expected disjoint decomposition for this test.";

    // Build & factorize the exact restricted subdomain matrices
    build_restricted_banded_from_global_csr_rect(A, ctx1);
    build_restricted_banded_from_global_csr_rect(A, ctx2);

    banded_cholesky_factor(ctx1);
    banded_cholesky_factor(ctx2);

    // pinned buffers
    cudaMallocHost(&ctx1.h_rhs, (size_t)ctx1.nsub * sizeof(double));
    cudaMallocHost(&ctx1.h_sol, (size_t)ctx1.nsub * sizeof(double));
    cudaMallocHost(&ctx2.h_rhs, (size_t)ctx2.nsub * sizeof(double));
    cudaMallocHost(&ctx2.h_sol, (size_t)ctx2.nsub * sizeof(double));

    std::vector<ichol::precond::PrecondApply> preconds(2);
    preconds[0] = {&apply_subdomain_exact, &ctx1};
    preconds[1] = {&apply_subdomain_exact, &ctx2};

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

    EXPECT_LT(result.iterations, 60); // paper reports ~37 vs ~49 for this 2-subdomain case

    cudaFreeHost(ctx1.h_rhs);
    cudaFreeHost(ctx1.h_sol);
    cudaFreeHost(ctx2.h_rhs);
    cudaFreeHost(ctx2.h_sol);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    if (argc > 1)
    {
        MPCGTest::n = std::stoi(argv[1]);
    }
    return RUN_ALL_TESTS();
}
