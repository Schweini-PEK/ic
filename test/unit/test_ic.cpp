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
    // Load SPD test matrix
    std::string path = "test/data/bcsstk11.mtx";
    ichol::CSR<double> Ahost = ichol::readMTXtoCSR<double>(path);

    const int n = Ahost.num_rows;

    ICTP_Params ictp_params;

    // Robust outer policy (field set kept minimal to match your evolving struct)
    IC_Factorize_Params fparams;
    fparams.initial_shift = 1e-12; // avoid the "alpha stays 0" trap if your logic hasn't added min-shift yet
    fparams.shift_growth = 2.0;
    fparams.max_restarts = 8;

    IC_Factorize_Info out_info;

    // Factorize using the new driver
    ichol::CSR<double> L = ichol::IC_factorize(Ahost, ictp_params, fparams, &out_info);
    ASSERT_GT(L.values.size(), 0u);

    // Determine scaling D (identity if not provided)
    std::vector<double> D(n, 1.0);
    if (!out_info.D.empty())
        D = out_info.D;

    // Build scaled matrix B values (pattern = Ahost):
    // B_ij = A_ij / (D_i D_j)  (if D is identity, this is just A)
    std::vector<double> valB(Ahost.values.size());
    for (int i = 0; i < n; ++i)
    {
        for (int p = Ahost.row_ptr[i]; p < Ahost.row_ptr[i + 1]; ++p)
        {
            int j = Ahost.col_ind[p];
            valB[p] = Ahost.values[p] / (D[i] * D[j]);
        }
    }

    // RHS: ones
    std::vector<double> b(n, 1.0);

    // Scale RHS for the B-system: c = D^{-1} b
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

    // Solve B y = b_tilde with preconditioner from L
    // (Your PCG implementation likely treats L as M^{1/2} where M ≈ B + alpha I.)
    ichol::icPreconditionedCG_GPU<double>(
        Ahost.row_ptr,
        Ahost.col_ind,
        valB,
        rowPtrL,
        colIndL,
        valL,
        b_tilde,
        y,
        iters,
        finalRes);

    ASSERT_EQ(y.size(), static_cast<size_t>(n));

    // Recover x = D^{-1} y
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i)
        x[i] = y[i] / D[i];

    // Compute residual r = A x - b
    std::vector<double> r(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        double s = 0.0;
        for (int p = Ahost.row_ptr[i]; p < Ahost.row_ptr[i + 1]; ++p)
        {
            int j = Ahost.col_ind[p];
            s += Ahost.values[p] * x[j];
        }
        r[i] = s - b[i];
    }

    auto vec_norm = [](const std::vector<double> &v)
    {
        double s = 0.0;
        for (double a : v)
            s += a * a;
        return std::sqrt(s);
    };

    double rnorm = vec_norm(r);
    double bnorm = vec_norm(b);
    double relres = (bnorm == 0.0) ? rnorm : rnorm / bnorm;

    std::cout << "IC_factorize shift used: " << out_info.shift_used << "\n";
    std::cout << "IC_factorize restarts : " << out_info.restarts << "\n";
    std::cout << "Relative residual     : " << relres << "\n";
    std::cout << "Final residual (PCG)  : " << finalRes << "\n";

    EXPECT_LT(relres, 1e-6);
    EXPECT_LT(finalRes, 1e-6);
}
