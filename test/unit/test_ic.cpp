// test/unit/test_ic.cpp
#include <gtest/gtest.h>
#include <petscsys.h>
#include <chrono>

#include "ichol/matrix_formats.hpp"
#include "ichol/pcg.hpp"
#include "ichol/half.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/factorize.hpp"
#include "backends/cpu/util/cast.hpp"
#include "unit/test_utils.hpp"
#include "linalg/norm.hpp"

TEST(IC_Factorize, ProducesUsablePreconditionerOnMTX)
{
    std::string path = "test/data/europe_osm.mtx";
    // ichol::matrix::CsrMatrix<double> A = ichol::io::mtx_to_csr<double>(path, false, 1.0);
    ichol::matrix::CsrMatrix<double> A = ichol::io::gen_3dlap_csr<double>(300);

    const int n = A.num_rows;

    ichol::SymbolicOptions sym_options;
    sym_options.ordering = ichol::Ordering::RCM;
    sym_options.level_k = 4; // IC(4)

    ichol::IncompleteCholeskyOptions ic_options;
    ic_options.scaling = ichol::Scaling::UnitSqrtDiag;
    ic_options.pivot_shift_strategy = ichol::PivotShiftStrategy::Static;
    ic_options.static_shift = 1e-5;
    ic_options.lfil = 100;
    ic_options.drop_tol = 0.0;

    auto sym_plan = ichol::symbolic::ic_analyze<double>(A, sym_options);

    ichol::numeric::NumericPlan num_plan;

    auto fact_start = std::chrono::high_resolution_clock::now();
    auto L = ichol::numeric::incomplete_cholesky_preconditioner<half_float::half>(A, sym_plan, num_plan, ic_options);
    auto fact_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fact_duration = fact_end - fact_start;
    std::cout << "IC factorization time: " << fact_duration.count() << " seconds.\n";

    auto D = num_plan.prescaling.D;

    /*
    Construct B y = b_tilde, where
    B = A_scaled = D^{-1} A D^{-1},
    and b_tilde = D^{-1} b
    */
    std::vector<double> b(n, 1.0);
    std::vector<double> b_perm = (sym_options.ordering == ichol::Ordering::Identity)
                                     ? b
                                     : ichol::symbolic::apply_permutation_vec(b, sym_plan.perm);
    std::vector<double> b_tilde(n);
    for (int i = 0; i < n; ++i)
        b_tilde[i] = b_perm[i] / D[i];

    // Prepare L for PCG
    std::vector<int> rowPtrL = L.row_ptr;
    std::vector<int> colIndL = L.col_ind;
    std::cout << "nnzs of L" << size_t(L.values.size()) << "\n";
    std::cout << "nnzs of predicted L: "
              << sym_plan.factor_pattern.col_ind_L.size() << "\n";

    std::vector<double> y;
    int iters = 0;
    double finalRes = 0.0;

    /*
    Solve B y = b_tilde with preconditioner from L,
    where LL^T \approx D^{-1} A D^{-1} + \alpha I
    */
    ichol::solver::pcg<half_float::half>(
        A.row_ptr,
        A.col_ind,
        A.values,
        rowPtrL,
        colIndL,
        L.values,
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

    // Symmetric matvec for CSR storing lower-triangular + diagonal only.
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

    // Scaled system residual: rB = B*y - b_tilde
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
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr)
        return ierr;

    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();

    ierr = PetscFinalize();
    if (ierr)
        return ierr;
    return rc;
}
