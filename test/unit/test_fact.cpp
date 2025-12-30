// test/unit/test_fact.cpp
#include <gtest/gtest.h>
#include <iostream>

#include "ichol/matrix_formats.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/matrix_norm.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/factorize.hpp"
#include "unit/test_utils.hpp"

TEST(FactTest, IC_Factorize_Runs)
{
    std::string path = "test/data/Kuu.mtx";
    ichol::matrix::CsrMatrix<double> A = ichol::io::mtx_to_csr<double>(path, false);

    ichol::SymbolicOptions sym_options;
    // sym_options.ordering = ichol::Ordering::AMD;
    sym_options.level_k = 2; // IC(2)

    ichol::IncompleteCholeskyOptions ic_options;
    ic_options.scaling = ichol::Scaling::UnitSqrtDiag;
    ic_options.pivot_shift_strategy = ichol::PivotShiftStrategy::Static;
    ic_options.static_shift = 1e-8;
    ic_options.lfil = 40;
    ic_options.drop_tol = 0.0;

    auto sym_plan = ichol::symbolic::ic_analyze<double>(A, sym_options);

    ichol::numeric::NumericPlan num_plan;
    auto L = ichol::numeric::incomplete_cholesky_preconditioner<double>(A, sym_plan, num_plan, ic_options);
    auto A_scaled = num_plan.A_scaled;
    ichol::numeric::add_diagonal_shift<double>(A_scaled, num_plan.ic_info.shift_used);

    double residual_norm = ichol::residual_l2_norm(A_scaled, L);
    std::cout << "Residual L2 norm ||A - LL^T|| / ||A||: "
              << residual_norm << std::endl;

    EXPECT_LT(residual_norm, 1e-1);
}