// test_supernodal.cpp
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "factor/numerical/supernodal_numeric_ll.hpp"

namespace {

static double l2norm_sq(const std::vector<double>& v)
{
    double s = 0.0;
    for (double a : v) s += a * a;
    return s;
}

static double l2norm(const std::vector<double>& v)
{
    return std::sqrt(l2norm_sq(v));
}

static double l2norm_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) return std::numeric_limits<double>::quiet_NaN();
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}


// Treat A as symmetric, stored in LOWER triangle CSC (i >= j).
static void spmv_sym_lower_csc(const ichol::matrix::CscMatrix<double>& A,
                               const std::vector<double>& x,
                               std::vector<double>& y)
{
    const int n = A.num_cols;
    y.assign((size_t)n, 0.0);

    for (int j = 0; j < n; ++j) {
        for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
            const int i = A.row_ind[p];
            const double v = A.values[p];
            if (i < j) continue; // lower only

            y[(size_t)i] += v * x[(size_t)j];
            if (i != j) y[(size_t)j] += v * x[(size_t)i];
        }
    }
}

static double fro_norm_sym_lower_csc(const ichol::matrix::CscMatrix<double>& A)
{
    const int n = A.num_cols;
    double s = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
            const int i = A.row_ind[p];
            const double v = A.values[p];
            if (i < j) continue;
            if (i == j) s += v * v;
            else        s += 2.0 * v * v; // symmetric duplicate
        }
    }
    return std::sqrt(s);
}

// y = L * x, where L is stored in CHOLMOD-style packed supernodal blocks.
static void apply_L(const ichol::numeric::SuperNumeric& num,
                    const std::vector<double>& x,
                    std::vector<double>& y)
{
    const auto& sym = num.sym;
    const int n = (int)x.size();
    y.assign((size_t)n, 0.0);

    const int nsuper = (int)sym.super.size() - 1;
    for (int k = 0; k < nsuper; ++k) {
        const int scol  = sym.super[(size_t)k];
        const int ecol  = sym.super[(size_t)k + 1];
        const int nscol = ecol - scol;

        const int pi0   = sym.pi[(size_t)k];
        const int pi1   = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;

        const int px0   = sym.px[(size_t)k];

        for (int lc = 0; lc < nscol; ++lc) {
            const int col = scol + lc;
            const double alpha = x[(size_t)col];
            if (alpha == 0.0) continue;

            const size_t base = (size_t)px0 + (size_t)lc * (size_t)nsrow;
            for (int t = 0; t < nsrow; ++t) {
                const int row = sym.s[(size_t)(pi0 + t)];
                y[(size_t)row] += num.x[base + (size_t)t] * alpha;
            }
        }
    }
}

// y = L^T * x
static void apply_LT(const ichol::numeric::SuperNumeric& num,
                     const std::vector<double>& x,
                     std::vector<double>& y)
{
    const auto& sym = num.sym;
    const int n = (int)x.size();
    y.assign((size_t)n, 0.0);

    const int nsuper = (int)sym.super.size() - 1;
    for (int k = 0; k < nsuper; ++k) {
        const int scol  = sym.super[(size_t)k];
        const int ecol  = sym.super[(size_t)k + 1];
        const int nscol = ecol - scol;

        const int pi0   = sym.pi[(size_t)k];
        const int pi1   = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;

        const int px0   = sym.px[(size_t)k];

        for (int lc = 0; lc < nscol; ++lc) {
            const int col = scol + lc;
            const size_t base = (size_t)px0 + (size_t)lc * (size_t)nsrow;

            double sum = 0.0;
            for (int t = 0; t < nsrow; ++t) {
                const int row = sym.s[(size_t)(pi0 + t)];
                sum += num.x[base + (size_t)t] * x[(size_t)row];
            }
            y[(size_t)col] += sum;
        }
    }
}

static double estimate_fro_norm_LLt(const ichol::numeric::SuperNumeric& num, int trials)
{
    const int n = (int)(num.sym.super.empty() ? 0 : num.sym.super.back());
    std::mt19937 rng(12345);
    std::bernoulli_distribution coin(0.5);

    std::vector<double> g((size_t)n), t((size_t)n), v((size_t)n);
    double acc = 0.0;

    for (int k = 0; k < trials; ++k) {
        for (int i = 0; i < n; ++i) g[(size_t)i] = coin(rng) ? 1.0 : -1.0;
        apply_LT(num, g, t);
        apply_L(num, t, v);
        acc += l2norm_sq(v);
    }

    // Hutchinson: E ||M g||^2 = ||M||_F^2 for Rademacher / Gaussian g.
    return std::sqrt(acc / std::max(1, trials));
}

static double worst_random_rel_residual(const ichol::matrix::CscMatrix<double>& A,
                                       const ichol::numeric::SuperNumeric& num,
                                       int trials)
{
    const int n = A.num_cols;
    std::mt19937 rng(20240120);
    std::bernoulli_distribution coin(0.5);

    std::vector<double> x((size_t)n), yA, t, yL;

    double worst = 0.0;
    for (int k = 0; k < trials; ++k) {
        for (int i = 0; i < n; ++i) x[(size_t)i] = coin(rng) ? 1.0 : -1.0;

        spmv_sym_lower_csc(A, x, yA);
        apply_LT(num, x, t);
        apply_L(num, t, yL);

        const double nume = l2norm_diff(yA, yL);
        const double deno = std::max(l2norm(yA), 1e-30);
        worst = std::max(worst, nume / deno);
    }
    return worst;
}

} // namespace

TEST(SupernodalCPU, SymbolicThenNumeric_LL_Nasa2146)
{
    // Keep the same convention as the original project tests.
    const std::string path = "/tmp/ic/test/data/nasa2146.mtx";

    auto A = ichol::io::mtx_to_csc<double>(path, /*make_symmetric=*/false);
    ASSERT_GT(A.num_cols, 0);
    ASSERT_EQ(A.num_rows, A.num_cols) << "This test expects a square matrix.";

    // 1) Symbolic phase: build ALL symbolic info needed by numeric.
    ichol::SuperNodeOptions snopt;
    snopt.approximate = false;
    auto plan = ichol::symbolic::supernodal_ll_analyze(A, snopt);

    // Basic sanity on the symbolic product.
    ASSERT_GE((int)plan.sym.super.size(), 2);
    ASSERT_EQ((int)plan.sym.super.size(), (int)plan.sym.pi.size());
    ASSERT_EQ((int)plan.sym.super.size(), (int)plan.sym.px.size());
    ASSERT_EQ((int)plan.sym.s.size(), plan.sym.pi.back());

    // 2) Numeric phase (CPU): consume the symbolic plan and factorize.
    auto num = ichol::numeric::factorize_supernodal_ll(A, plan);

    ASSERT_TRUE(num.ok) << "Numeric factorization failed at snode="
                        << num.fail_snode << ", col_in_snode=" << num.fail_col_in_snode;

    // Numeric output is CHOLMOD-style packed supernodal storage of L.
    ASSERT_EQ((int)num.x.size(), plan.sym.px.back());

    // 3) Verification.
    // The ratio ||L L^T|| / ||A|| alone is a weak check (it only checks scale),
    // so we report it but also assert a meaningful relative residual.
    const double A_fro = fro_norm_sym_lower_csc(A);
    const double LLt_fro = estimate_fro_norm_LLt(num, /*trials=*/20);
    const double ratio = LLt_fro / std::max(A_fro, 1e-30);

    const double worst_rel = worst_random_rel_residual(A, num, /*trials=*/10);

    std::cout << "[SupernodalCPU] n=" << A.num_cols
              << " nsuper=" << (int)plan.sym.super.size() - 1
              << " ||A||_F=" << A_fro
              << " ||LL^T||_F~=" << LLt_fro
              << " ratio=" << ratio
              << " worst_rel_residual~=" << worst_rel
              << "\n";

    ASSERT_TRUE(std::isfinite(ratio));
    ASSERT_TRUE(std::isfinite(worst_rel));

    // Loose but meaningful default bound for double-precision Cholesky.
    // If this fails, it usually means: wrong symmetry interpretation, wrong
    // packed-block decoding, or numeric kernel bug.
    ASSERT_LT(worst_rel, 1e-8);
}
