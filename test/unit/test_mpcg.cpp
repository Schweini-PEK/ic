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
            // b[i] = dist(rng);
            b[i] = 1.0;
        }
    }

    std::vector<double> x(A.num_rows, 0.0);

    const int maxits = 500;
    const double tol = 1e-10;
    const int restart = 0;

    std::vector<ichol::precond::ADIContext> ctxs;
    ctxs.push_back({n, ichol::precond::ADIDirection3D::X});
    ctxs.push_back({n, ichol::precond::ADIDirection3D::Y});
    ctxs.push_back({n, ichol::precond::ADIDirection3D::Z});

    std::vector<ichol::precond::PrecondApply> preconds;
    preconds.reserve(3);
    for (int t = 0; t < 3; ++t)
    {
        ichol::precond::PrecondApply P;
        P.apply = &ichol::precond::apply_adi3d_dir;
        P.ctx = static_cast<void *>(&ctxs[t]);
        preconds.push_back(P);
    }

    int iters = 0;
    double finalRes = 0.0;

    const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));

    auto t0 = std::chrono::high_resolution_clock::now();

    ichol::solver::mpcg<double>(
        A.row_ptr, A.col_ind, A.values,
        preconds,
        b, x,
        maxits, tol, restart,
        iters, finalRes);

    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    EXPECT_GT(iters, 0);
    EXPECT_LT(finalRes, tol * (bnorm > 0.0 ? bnorm : 1.0));

    std::cout << "[MPCG 3D_Poisson] n=" << nloc
              << " N=" << A.num_rows
              << " iters=" << iters
              << " finalRes=" << finalRes
              << " time=" << secs << "s\n";
}

TEST(MPCG, 2D_Poisson_Asymmetric)
{
    const int n = 32;
    const double epsilon = 0.5;
    auto A = ichol::io::gen_2dpoi<double>(n, epsilon);

    std::vector<double> b(A.num_rows, 1.0); // Constant RHS
    std::vector<double> x(A.num_rows, 0.0);

    // Contexts for Mx and My
    ichol::precond::ADI2DContext ctxX{n, ichol::precond::ADIDirection2D::X, 1.0};
    ichol::precond::ADI2DContext ctxY{n, ichol::precond::ADIDirection2D::Y, epsilon};

    std::vector<ichol::precond::PrecondApply> preconds(2);
    preconds[0] = {&ichol::precond::apply_adi2d_dir, &ctxX};
    preconds[1] = {&ichol::precond::apply_adi2d_dir, &ctxY};

    int iters;
    double res;
    ichol::solver::mpcg<double>(A.row_ptr, A.col_ind, A.values, preconds, b, x, 500, 1e-10, 0, iters, res);

    std::cout << "[MPCG 2D] iters: " << iters << "\n";
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