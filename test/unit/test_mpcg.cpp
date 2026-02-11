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

TEST(MPCG, 3D_Poisson)
{
    const int n = 8;
    ichol::matrix::CsrMatrix<double> A = ichol::io::gen_3dpoi<double>(n);

    std::vector<double> b(A.num_rows);
    {
        std::mt19937 rng(12345);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < A.num_rows; ++i)
            b[i] = dist(rng);
    }

    std::vector<double> x(A.num_rows, 0.0);

    const int maxits = 500;
    const double tol = 1e-10;
    const int restart = 1;

    std::vector<ichol::precond::ADIContext> ctxs;
    ctxs.push_back({n, ichol::precond::ADIDirection3D::X});
    ctxs.push_back({n, ichol::precond::ADIDirection3D::Y});
    ctxs.push_back({n, ichol::precond::ADIDirection3D::Z});

    std::vector<ichol::precond::PrecondApply> preconds;
    preconds.reserve(3);
    for (int t = 0; t < 3; ++t)
    {
        ichol::precond::PrecondApply P;
        P.apply = &ichol::precond::apply_adi_dir;
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

    std::cout << "[MPCG 3D_Poisson] n=" << n
              << " N=" << A.num_rows
              << " iters=" << iters
              << " finalRes=" << finalRes
              << " time=" << secs << "s\n";
}
