// test_pcg_double.cu
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "ichol/pcg.hpp"

TEST(ICPCG, Solves2x2_Double)
{
    const int n = 2;

    // CSR for A (row-major, 0-based):
    // row0: (0,4), (1,1)
    // row1: (0,1), (1,3)
    std::vector<int> rowPtrA = {0, 2, 4};
    std::vector<int> colIndA = {0, 1, 0, 1};
    std::vector<double> valA = {4.0, 1.0, 1.0, 3.0};

    // CSR for L
    // L =
    // [2      0     ]
    // [0.5    sqrt(2.75)]
    //
    // Row 0: (0,2)
    // Row 1: (0,0.5), (1, sqrt(2.75))
    double l11 = 2.0;
    double l21 = 0.5;
    double l22 = std::sqrt(2.75);

    std::vector<int> rowPtrL = {0, 1, 3};
    std::vector<int> colIndL = {0, 0, 1};
    std::vector<double> valL = {l11, l21, l22};

    // RHS
    std::vector<double> b = {1.0, 2.0};

    std::vector<double> x;
    int iters = 0;
    double finalRes = 0.0;

    std::vector<double> D = {1.0, 1.0}; // No scaling

    ichol::solver::pcg<double>(
        rowPtrA,
        colIndA,
        valA,
        rowPtrL,
        colIndL,
        valL,
        b,
        x,
        D,
        iters,
        finalRes);

    ASSERT_EQ(x.size(), 2);

    double x_true_0 = 1.0 / 11.0; // 0.090909...
    double x_true_1 = 7.0 / 11.0; // 0.636363...

    EXPECT_NEAR(x[0], x_true_0, 1e-10);
    EXPECT_NEAR(x[1], x_true_1, 1e-10);

    EXPECT_LT(finalRes, 1e-10);
    EXPECT_LT(iters, 50);
}
