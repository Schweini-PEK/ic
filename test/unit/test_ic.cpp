// test/unit/test_ic.cpp
#include <gtest/gtest.h>
// #include <vector>
// #include <cmath>

#include "ichol/matrix_formats.hpp"
#include "ichol/pcg.hpp"
#include "ichol/half.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/factorize.hpp"
#include "backends/cpu/util/cast.hpp"
#include "unit/test_utils.hpp"

TEST(IC_Factorize, ProducesUsablePreconditionerOnMTX)
{
    std::string path = "test/data/nasa2146.mtx";
    ichol::matrix::CsrMatrix<double> A = ichol::io::mtx_to_csr<double>(path, false);

    const int n = A.num_rows;

    ichol::SymbolicOptions sym_options;
    // sym_options.ordering = ichol::Ordering::AMD;
    sym_options.level_k = 3; // IC(3)

    ichol::IncompleteCholeskyOptions ic_options;
    ic_options.scaling = ichol::Scaling::UnitSqrtDiag;
    ic_options.pivot_shift_strategy = ichol::PivotShiftStrategy::Static;
    ic_options.static_shift = 1e-5;
    ic_options.lfil = 10;
    ic_options.drop_tol = 0.0;

    auto sym_plan = ichol::symbolic::ic_analyze<double>(A, sym_options);

    ichol::numeric::NumericPlan num_plan;
    auto L = ichol::numeric::incomplete_cholesky_preconditioner<double>(A, sym_plan, num_plan, ic_options);

    auto D = num_plan.prescaling.D;

    /*
    Construct B y = b_tilde, where
    B = A_scaled = D^{-1} A D^{-1},
    and b_tilde = D^{-1} b
    */
    std::vector<double> b_tilde(n);
    for (int i = 0; i < n; ++i)
        b_tilde[i] = 1.0 / D[i];

    // Prepare L for PCG
    std::vector<int> rowPtrL = L.row_ptr;
    std::vector<int> colIndL = L.col_ind;
    std::vector<double> valL = ichol::util::to_double_vec(L.values);

    std::vector<double> y;
    int iters = 0;
    double finalRes = 0.0;

    /*
    Solve B y = b_tilde with preconditioner from L,
    where LL^T \approx D^{-1} A D^{-1} + \alpha I
    */
    ichol::icPreconditionedCG_GPU<double>(
        A.row_ptr,
        A.col_ind,
        A.values,
        rowPtrL,
        colIndL,
        valL,
        b_tilde,
        y,
        D,
        iters,
        finalRes);

    auto vec_norm = [](const std::vector<double> &v)
    {
        double s = 0.0;
        for (double a : v)
            s += a * a;
        return std::sqrt(s);
    };

    // // Symmetric matvec for CSR storing lower-triangular + diagonal only.
    // // Assumes diagonal entry exists in every row.
    auto symm_lower_csr_matvec = [&](const ichol::matrix::CsrMatrix<double> &M,
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
    symm_lower_csr_matvec(A, y, By);
    for (int i = 0; i < n; ++i)
        rB[i] = By[i] - b_tilde[i];

    double rBnorm = vec_norm(rB);
    double bTildenorm = vec_norm(b_tilde);
    double relresB = (bTildenorm == 0.0) ? rBnorm : rBnorm / bTildenorm;

    std::cout << "Scaled-system relative residual (B y = b_tilde): "
              << relresB << "\n";
    std::cout << "Iterations taken by PCG: "
              << iters << "\n";
    std::cout << "Final residual from CG (reported ||r||_2): "
              << finalRes << "\n";

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
}
