// test/unit/test_fact.cpp
#include <gtest/gtest.h>
#include <iostream>

#include "ichol/matrix_formats.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/matrix_norm.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/factorize.hpp"

TEST(FactTest, IC_Factorize_Runs)
{
    std::string path = "test/data/HB/bcsstk11.mtx";
    ichol::matrix::CsrMatrix<double> A = ichol::io::mtx_to_csr<double>(path, false);

    ichol::SymbolicOptions sym_options;
    sym_options.ordering = ichol::Ordering::AMD;
    sym_options.level_k = 2; // IC(2)

    ichol::IncompleteCholeskyOptions ic_options;
    ic_options.scaling = ichol::Scaling::UnitSqrtDiag;
    ic_options.pivot_shift_strategy = ichol::PivotShiftStrategy::Static;

    auto sym_plan = ichol::symbolic::ic_analyze<double>(A, sym_options);

    ichol::numeric::NumericPlan num_plan;
    auto L = ichol::numeric::incomplete_cholesky_preconditioner<float>(A, sym_plan, num_plan, ic_options);
    auto A_scaled = num_plan.A_scaled;

    // Prepare L for PCG
    std::vector<int> rowPtrL = L.row_ptr;
    std::vector<int> colIndL = L.col_ind;
    std::vector<double> valL = ichol::io::toDoubleVector(L.values);

    std::vector<double> y;
    int iters = 0;
    double finalRes = 0.0;

    auto D = num_plan.prescaling.D;
    const int n = A.num_rows;
    std::vector<double> b_tilde(n);
    for (int i = 0; i < n; ++i)
        b_tilde[i] = 1.0 / D[i];
    /*
    Solve B y = b_tilde with preconditioner from L,
    where LL^T \approx D^{-1} A D^{-1} + \alpha I
    */
    // ichol::icPreconditionedCG_GPU<double>(
    //     A_scaled.row_ptr,
    //     A_scaled.col_ind,
    //     A_scaled.values,
    //     rowPtrL,
    //     colIndL,
    //     valL,
    //     b_tilde,
    //     y,
    //     D,
    //     iters,
    //     finalRes);

    // ASSERT_EQ(y.size(), static_cast<size_t>(n));

    // auto vec_norm = [](const std::vector<double> &v)
    // {
    //     double s = 0.0;
    //     for (double a : v)
    //         s += a * a;
    //     return std::sqrt(s);
    // };

    // // Symmetric matvec for CSR storing lower-triangular + diagonal only.
    // // Assumes diagonal entry exists in every row.
    // auto symm_lower_csr_matvec = [&](const ichol::matrix::CsrMatrix<double> &M,
    //                                  const std::vector<double> &x,
    //                                  std::vector<double> &y)
    // {
    //     y.assign(n, 0.0);
    //     for (int i = 0; i < n; ++i)
    //     {
    //         for (int p = M.row_ptr[i]; p < M.row_ptr[i + 1]; ++p)
    //         {
    //             int j = M.col_ind[p];
    //             double aij = M.values[p];
    //             y[i] += aij * x[j];
    //             if (j != i)
    //             {
    //                 y[j] += aij * x[i]; // add symmetric counterpart
    //             }
    //         }
    //     }
    // };

    // // 1) Scaled system residual: rB = B*y - b_tilde
    // std::vector<double> By(n), rB(n);
    // symm_lower_csr_matvec(B, y, By);
    // for (int i = 0; i < n; ++i)
    //     rB[i] = By[i] - b_tilde[i];

    // double rBnorm = vec_norm(rB);
    // double bTildenorm = vec_norm(b_tilde);
    // double relresB = (bTildenorm == 0.0) ? rBnorm : rBnorm / bTildenorm;

    // std::cout << "Scaled-system relative residual (B y = b_tilde): "
    //           << relresB << "\n";
    // std::cout << "Iterations taken by PCG: "
    //           << iters << "\n";
    // std::cout << "Final residual from CG (reported ||r||_2): "
    //           << finalRes << "\n";

    // EXPECT_LT(relresB, 1e-6);

    // // 2) Original system residual: rA = A*x - b, with x = D^{-1} y
    // std::vector<double> x(n);
    // for (int i = 0; i < n; ++i)
    //     x[i] = y[i] / D[i];

    // std::vector<double> Ax(n), rA(n);
    // symm_lower_csr_matvec(Ahost, x, Ax);
    // for (int i = 0; i < n; ++i)
    //     rA[i] = Ax[i] - b[i];

    // double rAnorm = vec_norm(rA);
    // double bnorm = vec_norm(b);
    // double relresA = (bnorm == 0.0) ? rAnorm : rAnorm / bnorm;

    // std::cout << "Original-system relative residual (A x = b): "
    //           << relresA << "\n";

    // auto L = ichol::IC_factorize<double>(algo, csr, ictp_params, fparams, Sym, &info);

    // std::cout << "IC_factorize ran successfully. "
    //           << "shift_used=" << info.shift_used
    //           << ", restarts=" << info.restarts << std::endl;

    // auto A_scale = apply_symm_prescaling(csr, info.D);
    // auto A_tilde = add_diagonal_shift(A_scale, info.shift_used);
    // double residual_norm = ichol::residual_l2_norm(A_tilde, L);
    // std::cout << "Residual L2 norm ||A - LL^T|| / ||A||: "
    //           << residual_norm << std::endl;
}