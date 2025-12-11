// test_fact.cpp
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "ichol/matrix_formats.hpp"
#include "ichol/ictp.hpp"
#include "ichol/fact.hpp"
#include "../../src/io/mtx_read.hpp"
#include "../../include/ichol/matrix_norm.hpp"

TEST(FactTest, ICTP_KernelRuns)
{
    std::string path = "test/data/bcsstk11.mtx";
    ichol::CSR<double> csr = ichol::readMTXtoCSR<double>(path);

    std::cout << "CSR loaded: rows=" << csr.num_rows
              << ", cols=" << csr.num_cols
              << ", nnz=" << csr.nnz << std::endl;

    ICTP_Params ictp_params;
    IC_Attempt_Params attempt_params;
    attempt_params.pivot_tol = 0.0; // permissive for this smoke test

    ICTP_Factor_Info finfo;

    // New ICTP entry point
    auto L = ichol::ictp<double>(csr, ictp_params, attempt_params, &finfo);

    ASSERT_EQ(L.num_rows, csr.num_rows);
    ASSERT_EQ(L.num_cols, csr.num_cols);
    ASSERT_GT(L.values.size(), 0u);
    ASSERT_EQ(finfo.code, IC_Breakdown::None);

    std::cout << "ictp kernel ran successfully." << std::endl;

    // Residual norm for the *unshifted* ICTP kernel output
    double residual_norm = ichol::residual_l2_norm(csr, L);
    std::cout << "Residual L2 norm ||A - LL^T|| / ||A||: "
              << residual_norm << std::endl;
}

TEST(FactTest, IC_Factorize_Runs)
{
    std::string path = "test/data/bcsstk11.mtx";
    ichol::CSR<double> csr = ichol::readMTXtoCSR<double>(path);

    ICTP_Params ictp_params;
    ictp_params.lfil_per_row = 64;
    ictp_params.drop_tol = 0.0;

    IC_Attempt_Params attempt_params;
    attempt_params.pivot_tol = 0.0;

    IC_Factorize_Params fparams;
    fparams.initial_shift = 1e-8;
    fparams.shift_growth = 2.0;
    fparams.max_restarts = 8;

    IC_Factorize_Info info;

    auto L = ichol::IC_factorize(csr, ictp_params, fparams, &info);

    ASSERT_EQ(L.num_rows, csr.num_rows);
    ASSERT_EQ(L.num_cols, csr.num_cols);
    ASSERT_GT(L.values.size(), 0u);

    std::cout << "IC_factorize ran successfully. "
              << "shift_used=" << info.shift_used
              << ", restarts=" << info.restarts << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
