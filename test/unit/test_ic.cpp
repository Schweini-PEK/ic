// test_ic.cpp
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <cassert>

#include "ichol/matrix_formats.hpp"
#include "ichol/ictp.hpp"
#include "ichol/pcg.hpp"
#include "ichol/half.hpp"
#include "ichol/fact.hpp"

#include "ichol/mtx_read.hpp"

#include <limits>
#include <random>

namespace test_checks
{

    inline void assert_pos_finite_vec(const std::vector<double> &v, const char *name)
    {
        ASSERT_FALSE(v.empty()) << name;
        for (size_t i = 0; i < v.size(); ++i)
        {
            ASSERT_TRUE(std::isfinite(v[i])) << name << "[" << i << "] not finite: " << v[i];
            ASSERT_GT(v[i], 0.0) << name << "[" << i << "] not positive: " << v[i];
        }
    }

    template <typename T>
    inline void assert_csr_lower_diag_only_sorted(const ichol::CsrMatrix<T> &M,
                                                  const char *name,
                                                  bool require_diag_last = true)
    {
        const int n = M.num_rows;
        ASSERT_EQ(M.num_cols, n) << name << " not square";
        ASSERT_EQ((int)M.row_ptr.size(), n + 1) << name << " row_ptr size";
        ASSERT_EQ(M.col_ind.size(), M.values.size()) << name << " col/val size mismatch";

        for (int i = 0; i < n; ++i)
        {
            const int rs = M.row_ptr[i];
            const int re = M.row_ptr[i + 1];
            ASSERT_LT(rs, re) << name << " row " << i << " empty / missing diagonal";
            int prev = -1;
            bool seen_diag = false;

            for (int p = rs; p < re; ++p)
            {
                const int j = M.col_ind[p];
                const auto a = (double)M.values[p];

                ASSERT_GE(j, 0) << name << " row " << i << " col < 0";
                ASSERT_LT(j, n) << name << " row " << i << " col out of range: " << j;
                ASSERT_LE(j, i) << name << " row " << i << " has upper entry col=" << j;

                ASSERT_GT(j, prev) << name << " row " << i
                                   << " cols not strictly increasing (unsorted or duplicate)";
                prev = j;

                ASSERT_TRUE(std::isfinite(a)) << name << " has non-finite value at ("
                                              << i << "," << j << "): " << a;

                if (j == i)
                    seen_diag = true;
            }

            ASSERT_TRUE(seen_diag) << name << " row " << i << " missing diagonal";
            if (require_diag_last)
            {
                ASSERT_EQ(M.col_ind[re - 1], i) << name << " row " << i << " diag not last";
            }
        }
    }

    // Stronger L-specific check (needed for SpSV with NON_UNIT diagonal)
    template <typename T>
    inline void assert_L_lower_diag_pos_finite(const ichol::CsrMatrix<T> &L, const char *name)
    {
        assert_csr_lower_diag_only_sorted(L, name, /*require_diag_last=*/true);
        const int n = L.num_rows;
        for (int i = 0; i < n; ++i)
        {
            const int pdiag = L.row_ptr[i + 1] - 1; // diag must be last
            const double di = (double)L.values[pdiag];
            ASSERT_TRUE(std::isfinite(di)) << name << " diag not finite at row " << i;
            ASSERT_GT(di, 0.0) << name << " diag not positive at row " << i << ": " << di;
        }
    }

} // namespace test_checks

template <typename T>
void assert_diag_last(const ichol::CsrMatrix<T> &M)
{
    int n = M.num_rows;
    for (int i = 0; i < n; ++i)
    {
        int row_start = M.row_ptr[i];
        int row_end = M.row_ptr[i + 1];
        assert(row_end > row_start && "Row has no entries / diagonal missing");
        int last_col = M.col_ind[row_end - 1];
        if (last_col != i)
        {
            std::cerr << "CSR diag-check failed: row " << i
                      << " last column index " << last_col
                      << " != expected diag index " << i << "\n";
            assert(false);
        }
    }

    std::cout << "Diag Check passed!" << std::endl;
}

TEST(IC_Factorize, ProducesUsablePreconditionerOnMTX)
{
    std::string path = "test/data/nasa2146.mtx";
    ichol::CsrMatrix<double> Ahost = ichol::io::mtx_to_csr<double>(path, false);

    const int n = Ahost.num_rows;

    ICTP_Params ictp_params;
    ictp_params.lfil_per_row = 40;
    ictp_params.drop_tol = 0.0;
    IC_Factorize_Params fparams;
    fparams.initial_shift = 1e-10;
    fparams.shift_growth = 2.0;
    fparams.max_restarts = 8;
    IC_Factorize_Info out_info;

    ichol::core::IC_Symbolic Sym = ichol::core::build_ic_symbolic(Ahost, 4);

    std::string algo = "parict";
    ichol::CsrMatrix<float> L = ichol::IC_factorize<float>(algo, Ahost, ictp_params, fparams, Sym, &out_info);
    ASSERT_GT(L.values.size(), 0u);

    std::vector<double> D(n, 1.0);
    if (!out_info.D.empty())
        D = out_info.D;

    /*
    Construct B y = b_tilde, where
    B = D^{-1} A D^{-1},
    and b_tilde = D^{-1} b
    */
    ichol::CsrMatrix<double> B = apply_symm_prescaling(Ahost, D);
    std::vector<double> b(n, 1.0);
    std::vector<double> b_tilde(n);
    for (int i = 0; i < n; ++i)
        b_tilde[i] = b[i] / D[i];

    assert_diag_last(B);

    // Prepare L for PCG
    std::vector<int> rowPtrL = L.row_ptr;
    std::vector<int> colIndL = L.col_ind;
    std::vector<double> valL = ichol::io::toDoubleVector(L.values);

    std::vector<double> y;
    int iters = 0;
    double finalRes = 0.0;

    test_checks::assert_pos_finite_vec(D, "D");
    test_checks::assert_csr_lower_diag_only_sorted(Ahost, "Ahost");
    test_checks::assert_csr_lower_diag_only_sorted(B, "B");
    test_checks::assert_L_lower_diag_pos_finite(L, "L");

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
    auto symm_lower_csr_matvec = [&](const ichol::CsrMatrix<double> &M,
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
    std::cout << "Iterations taken by PCG: "
              << iters << "\n";
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
