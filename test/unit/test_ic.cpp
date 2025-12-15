// test_ic.cpp
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>

#include "ichol/matrix_formats.hpp"
#include "ichol/ictp.hpp"
#include "ichol/pcg.hpp"
#include "ichol/fact.hpp"

#include "../../src/io/mtx_read.hpp"

TEST(IC_Factorize, ProducesUsablePreconditionerOnMTX)
{
    std::string path = "test/data/HB/bcsstk11.mtx";
    ichol::CSR<double> Ahost = ichol::readMTXtoCSR<double>(path, false);

    const int n = Ahost.num_rows;

    ICTP_Params ictp_params;
    ictp_params.lfil_per_row = 100;
    ictp_params.drop_tol = 0.0;
    IC_Factorize_Params fparams;
    fparams.initial_shift = 1e-8;
    fparams.shift_growth = 2.0;
    fparams.max_restarts = 8;
    IC_Factorize_Info out_info;

    ichol::core::IC_Symbolic Sym = ichol::core::build_ic_symbolic(Ahost, 4);

    // Factorize using the new driver
    ichol::CSR<double> L = ichol::IC_factorize(Ahost, ictp_params, fparams, Sym, &out_info);
    ASSERT_GT(L.values.size(), 0u);

    std::vector<double> D(n, 1.0);
    if (!out_info.D.empty())
        D = out_info.D;

    /*
    Construct B y = b_tilde, where
    B = D^{-1} A D^{-1},
    and b_tilde = D^{-1} b
    */
    ichol::CSR<double> B = apply_symm_prescaling(Ahost, D);
    std::vector<double> b(n, 1.0);
    std::vector<double> b_tilde(n);
    for (int i = 0; i < n; ++i)
        b_tilde[i] = b[i] / D[i];

    // Prepare L for PCG
    std::vector<int> rowPtrL = L.row_ptr;
    std::vector<int> colIndL = L.col_ind;
    std::vector<double> valL = L.values;

    std::vector<double> y;
    int iters = 0;
    double finalRes = 0.0;

    /*
    Solve B y = b_tilde with preconditioner from L,
    where LL^T \approx D^{-1} A D^{-1} + \alpha I
    */
    ichol::icPreconditionedCG_GPU<double>(
        B.row_ptr,
        B.col_ind,
        B.values,
        rowPtrL,
        colIndL,
        valL,
        b_tilde,
        y,
        D,
        iters,
        finalRes);

    ASSERT_EQ(y.size(), static_cast<size_t>(n));

    auto vec_norm = [](const std::vector<double> &v)
    {
        double s = 0.0;
        for (double a : v)
            s += a * a;
        return std::sqrt(s);
    };

    // Symmetric matvec for CSR storing lower-triangular + diagonal only.
    // Assumes diagonal entry exists in every row.
    auto symm_lower_csr_matvec = [&](const ichol::CSR<double> &M,
                                     const std::vector<double> &x,
                                     std::vector<double> &y)
    {
        y.assign(n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            for (int p = M.row_ptr[i]; p < M.row_ptr[i + 1]; ++p)
            {
                int j = M.col_ind[p];
                double aij = M.values[p];
                y[i] += aij * x[j];
                if (j != i)
                {
                    y[j] += aij * x[i]; // add symmetric counterpart
                }
            }
        }
    };

    // 1) Scaled system residual: rB = B*y - b_tilde
    std::vector<double> By(n), rB(n);
    symm_lower_csr_matvec(B, y, By);
    for (int i = 0; i < n; ++i)
        rB[i] = By[i] - b_tilde[i];

    double rBnorm = vec_norm(rB);
    double bTildenorm = vec_norm(b_tilde);
    double relresB = (bTildenorm == 0.0) ? rBnorm : rBnorm / bTildenorm;

    std::cout << "Scaled-system relative residual (B y = b_tilde): "
              << relresB << "\n";
    std::cout << "Final residual from CG (reported ||r||_2): "
              << finalRes << "\n";

    EXPECT_LT(relresB, 1e-6);

    // 2) Original system residual: rA = A*x - b, with x = D^{-1} y
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i)
        x[i] = y[i] / D[i];

    std::vector<double> Ax(n), rA(n);
    symm_lower_csr_matvec(Ahost, x, Ax);
    for (int i = 0; i < n; ++i)
        rA[i] = Ax[i] - b[i];

    double rAnorm = vec_norm(rA);
    double bnorm = vec_norm(b);
    double relresA = (bnorm == 0.0) ? rAnorm : rAnorm / bnorm;

    std::cout << "Original-system relative residual (A x = b): "
              << relresA << "\n";
}
