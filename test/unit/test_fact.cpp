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
    /**
     * load check
     */
    std::string path = "test/data/Kuu.mtx";
    ichol::matrix::CsrMatrix<double> A = ichol::io::mtx_to_csr<double>(path, false);

    const int n = A.row_ptr.size(); // number of rows + 1
    const int m = A.col_ind.size(); // number of non-zeros

    ASSERT_EQ(A.num_rows, A.num_cols);
    ichol::testutil::assert_diag_last_csr(A.row_ptr, A.col_ind);
    ichol::testutil::assert_lower_only_csr(A.row_ptr, A.col_ind);
    ichol::testutil::assert_diag_positive_csr(A.row_ptr, A.col_ind, A.values);

    /**
     * symbolic analysis check
     */
    ichol::SymbolicOptions sym_options;
    // // sym_options.ordering = ichol::Ordering::AMD;
    sym_options.level_k = 3; // IC(3)

    auto sym_plan = ichol::symbolic::ic_analyze<double>(A, sym_options);

    ASSERT_EQ(sym_plan.perm.perm.size(), A.num_rows);
    ASSERT_EQ(sym_plan.factor_pattern.row_ptr_L.back(), sym_plan.factor_pattern.col_ind_L.size());
    ichol::testutil::assert_diag_last_csr(sym_plan.factor_pattern.row_ptr_L, sym_plan.factor_pattern.col_ind_L);
    ichol::testutil::assert_lower_only_csr(sym_plan.factor_pattern.row_ptr_L, sym_plan.factor_pattern.col_ind_L);

    /**
     * numerical factorization check
     */
    ichol::IncompleteCholeskyOptions ic_options;
    ic_options.scaling = ichol::Scaling::UnitColNorm;
    ic_options.pivot_shift_strategy = ichol::PivotShiftStrategy::Static;
    ic_options.max_restarts = 8;
    ic_options.static_shift = 1e-2;
    ic_options.lfil = 40;
    ic_options.drop_tol = 0.0;

    ichol::numeric::NumericPlan num_plan;
    auto L = ichol::numeric::incomplete_cholesky_preconditioner<double>(A, sym_plan, num_plan, ic_options);

    ASSERT_EQ(L.num_rows, A.num_rows);
    ASSERT_EQ(L.num_cols, A.num_cols);
    ichol::testutil::assert_diag_positive_csr(L.row_ptr, L.col_ind, L.values);
    ichol::testutil::assert_row_nnz_le(L.row_ptr, ic_options.lfil + 1); // +1 for diag
    ichol::testutil::assert_dependency_L(L.row_ptr, L.col_ind, sym_plan.level_sets);

    if (ic_options.scaling == ichol::Scaling::UnitSqrtDiag)
    {
        // Check that the pre-scaling produced unit diagonal in A
        for (int i = 0; i < A.num_rows; ++i)
        {
            const int row_end = A.row_ptr[i + 1];
            const int diag_pos = row_end - 1;
            const double diag_val = A.values[diag_pos];
            double scale = std::max({1.0, std::abs(diag_val)});
            double tol = 16 * std::numeric_limits<double>::epsilon() * scale;
            ASSERT_NEAR(diag_val, 1.0, tol) << "Pre-scaling failed to produce unit diagonal at row " << i;
        }
    }

    ichol::matrix::CsrMatrix<double> A_org = ichol::io::mtx_to_csr<double>(path, false);
    ichol::symbolic::apply_permutation_csr<double>(A_org, sym_plan.perm);
    ichol::numeric::apply_prescaling(A_org, num_plan.prescaling.D);
    ichol::testutil::assert_matrices_equal(A, A_org);
    ichol::testutil::assert_diag_last_csr(A.row_ptr, A.col_ind);
    ichol::testutil::assert_lower_only_csr(A.row_ptr, A.col_ind);
    ichol::testutil::assert_diag_positive_csr(A.row_ptr, A.col_ind, A.values);

    /**
     * A is shifted from now on
     */
    ichol::numeric::add_diagonal_shift<double>(A, num_plan.ic_info.shift_used);
    ASSERT_EQ(n, A.row_ptr.size());
    ASSERT_EQ(m, A.col_ind.size());
    ichol::testutil::assert_shift_delta_equal(A_org, A, num_plan.ic_info.shift_used);

    ichol::testutil::assert_cols_sorted_unique(L.row_ptr, L.col_ind);
    ichol::testutil::assert_cols_sorted_unique(A.row_ptr, A.col_ind);
    ichol::testutil::assert_diag_last_csr(A.row_ptr, A.col_ind);

    double residual_norm = ichol::residual_l2_norm(A, L);
    std::cout << "Residual L2 norm ||A - LL^T|| / ||A||: "
              << residual_norm << std::endl;

    EXPECT_LT(residual_norm, 0.5);
}