// test_fact.cpp
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "ichol/matrix_formats.hpp"
#include "ichol/ictp.hpp"
#include "ichol/fact.hpp"
#include "ichol/symbolic.hpp"
#include "../../src/io/mtx_read.hpp"
#include "../../include/ichol/matrix_norm.hpp"

TEST(FactTest, IC_Factorize_Runs)
{
    std::string path = "test/data/HB/bcsstk11.mtx";
    ichol::CSR<double> csr = ichol::readMTXtoCSR<double>(path, false);

    ICTP_Params ictp_params;
    ictp_params.lfil_per_row = 64;
    ictp_params.drop_tol = 0.0;

    IC_Factorize_Params fparams;
    fparams.initial_shift = 1e-8;
    fparams.shift_growth = 2.0;
    fparams.max_restarts = 8;

    IC_Factorize_Info info;

    ichol::core::IC_Symbolic Sym = ichol::core::build_ic_symbolic(csr, 4);

    std::string algo = "parict";
    auto L = ichol::IC_factorize<double>(algo, csr, ictp_params, fparams, Sym, &info);

    ASSERT_EQ(L.num_rows, csr.num_rows);
    ASSERT_EQ(L.num_cols, csr.num_cols);
    ASSERT_GT(L.values.size(), 0u);

    std::cout << "IC_factorize ran successfully. "
              << "shift_used=" << info.shift_used
              << ", restarts=" << info.restarts << std::endl;

    auto A_scale = apply_symm_prescaling(csr, info.D);
    auto A_tilde = add_diagonal_shift(A_scale, info.shift_used);
    double residual_norm = ichol::residual_l2_norm(A_tilde, L);
    std::cout << "Residual L2 norm ||A - LL^T|| / ||A||: "
              << residual_norm << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
