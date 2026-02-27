// test/unit/test_mpcg.cpp
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

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

    params.store_P_hist = ichol::solver::ComputePrecision::FP32;
    params.store_W_hist = ichol::solver::ComputePrecision::FP32;

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

    std::cout << "[MPCG 2D] iters: " << result.iterations
              << " finalRes: " << result.finalRes
              << " time: " << mpcg_secs << "s\n";

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
    std::cout << "[PCG 2D] iters: " << pcg_result.iterations
              << " finalRes: " << pcg_result.finalRes
              << " time: " << pcg_secs << "s\n";
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
