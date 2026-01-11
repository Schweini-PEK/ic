// test_supernodal.cpp
#include <gtest/gtest.h>
#include <limits>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <utility>
#include <filesystem>
#include "ichol/mtx_read.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/symbolic/detail/symbolic_plan.hpp"
#include <cmath>
#include "factor/numerical/supernodal_numeric_ll.hpp"
#include "factor/numerical/super_sym.hpp"
// ---- SuiteSparse / CHOLMOD ----
extern "C" {
#include <cholmod.h>
}

using namespace ichol;
using namespace ichol::symbolic;

static void print_sn_range_list(const std::vector<std::pair<int,int>>& snodes, int limit = 20)
{
    int m = static_cast<int>(snodes.size());
    std::cout << "  total supernodes = " << m << "\n";
    int shown = std::min(m, limit);
    std::cout << "  first " << shown << " supernodes [start,end):\n";
    for (int i = 0; i < shown; ++i) {
        std::cout << "    [" << snodes[i].first << "," << snodes[i].second << ")";
        std::cout << " size=" << (snodes[i].second - snodes[i].first) << "\n";
    }
    if (m > shown) std::cout << "    ... (+" << (m - shown) << " more)\n";
}

static std::vector<int> snode_size_histogram(const std::vector<std::pair<int,int>>& snodes, int max_bucket = 20)
{
    std::vector<int> hist(max_bucket + 1, 0); // last bucket is ">= max_bucket"
    for (auto &p : snodes) {
        int sz = p.second - p.first;
        if (sz >= max_bucket) hist[max_bucket]++;
        else hist[sz]++;
    }
    return hist;
}

static void print_histogram(const std::vector<int>& hist)
{
    for (size_t i = 0; i + 1 < hist.size(); ++i) {
        std::cout << "    size=" << std::setw(2) << i << " -> " << hist[i] << "\n";
    }
    std::cout << "    size>=" << (hist.size()-1) << " -> " << hist.back() << "\n";
}

// =============== CHOLMOD integration helpers ===============

#ifdef ICHOL_CHOLMOD_LONG
    using cholmod_idx_t = SuiteSparse_long;
    #define CHM(name) cholmod_l_##name
#else
    using cholmod_idx_t = int;
    #define CHM(name) cholmod_##name
#endif

static cholmod_sparse* csc_to_cholmod_pattern(
    const ichol::matrix::CscMatrix<double>& A,
    int stype,                 // 0=unsym, -1=lower(sym), +1=upper(sym)
    cholmod_common* c)
{
    cholmod_sparse* S = CHM(allocate_sparse)(
        (size_t)A.num_rows, (size_t)A.num_cols, (size_t)A.nnz,
        /*sorted=*/1, /*packed=*/1,
        stype, CHOLMOD_PATTERN, c);

    auto* p = static_cast<cholmod_idx_t*>(S->p);
    auto* i = static_cast<cholmod_idx_t*>(S->i);

    for (int k = 0; k < (int)A.col_ptr.size(); ++k) p[k] = (cholmod_idx_t)A.col_ptr[k];
    for (int k = 0; k < (int)A.row_ind.size(); ++k) i[k] = (cholmod_idx_t)A.row_ind[k];

    return S;
}


static cholmod_sparse* csc_to_cholmod_real(
    const ichol::matrix::CscMatrix<double>& A,
    int stype,                 // 0=unsym, -1=lower(sym), +1=upper(sym)
    cholmod_common* c)
{
    cholmod_sparse* S = CHM(allocate_sparse)(
        (size_t)A.num_rows, (size_t)A.num_cols, (size_t)A.nnz,
        /*sorted=*/1, /*packed=*/1,
        stype, CHOLMOD_REAL, c);

    auto* p = static_cast<cholmod_idx_t*>(S->p);
    auto* i = static_cast<cholmod_idx_t*>(S->i);
    auto* x = static_cast<double*>(S->x);

    for (int k = 0; k < (int)A.col_ptr.size(); ++k) p[k] = (cholmod_idx_t)A.col_ptr[k];
    for (int k = 0; k < (int)A.row_ind.size(); ++k) i[k] = (cholmod_idx_t)A.row_ind[k];
    for (int k = 0; k < (int)A.values.size(); ++k) x[k] = (double)A.values[k];

    return S;
}

static double cholmod_solve_residual_norm2(
    cholmod_sparse* A,
    cholmod_factor* L,
    cholmod_common* cc)
{
    const int n = (int)A->ncol;

    // deterministic RHS b
    cholmod_dense* b = CHM(allocate_dense)((size_t)n, 1, (size_t)n, CHOLMOD_REAL, cc);
    auto* bx = static_cast<double*>(b->x);
    for (int i = 0; i < n; ++i) {
        // small-ish values, avoid overflow
        bx[i] = 0.001 * (double)(i % 997) + 1.0;
    }

    cholmod_dense* x = CHM(solve)(CHOLMOD_A, L, b, cc);
    if (!x) {
        CHM(free_dense)(&b, cc);
        return std::numeric_limits<double>::infinity();
    }

    cholmod_dense* y = CHM(allocate_dense)((size_t)n, 1, (size_t)n, CHOLMOD_REAL, cc);
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};
    CHM(sdmult)(A, /*transpose=*/0, alpha, beta, x, y, cc);

    // r = y - b
    auto* yx = static_cast<double*>(y->x);
    double n2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double ri = yx[i] - bx[i];
        n2 += ri * ri;
    }

    CHM(free_dense)(&y, cc);
    CHM(free_dense)(&x, cc);
    CHM(free_dense)(&b, cc);
    return std::sqrt(n2);
}


static std::vector<std::pair<int,int>> cholmod_symbolic_supernodes_identity(
    const ichol::matrix::CscMatrix<double>& A,
    int stype,
    bool disable_postorder,
    bool disable_relax,
    cholmod_common* cc_out
)
{
    cholmod_common cc_local;
    cholmod_common* cc = cc_out ? cc_out : &cc_local;

    CHM(start)(cc);

    cc->supernodal = CHOLMOD_SUPERNODAL;

    if (disable_postorder) cc->postorder = 0;
    if (disable_relax) {
        cc->nrelax[0] = cc->nrelax[1] = cc->nrelax[2] = 0;
        cc->zrelax[0] = cc->zrelax[1] = cc->zrelax[2] = 0.0;
    }

    cc->nmethods = 1;
    cc->method[0].ordering = CHOLMOD_GIVEN;

    const int n = A.num_cols;
    std::vector<cholmod_idx_t> perm((size_t)n);
    for (int i = 0; i < n; ++i) perm[i] = (cholmod_idx_t)i;

    cholmod_sparse* S = csc_to_cholmod_pattern(A, stype, cc);
    cholmod_factor* L = CHM(analyze_p)(S, perm.data(), nullptr, 0, cc);

    std::vector<std::pair<int,int>> sn_chol;

    if (!L) {
        std::cerr << "CHOLMOD analyze_p returned nullptr factor.\n";
    } else if (!L->is_super) {
        std::cerr << "CHOLMOD did not produce a supernodal factor (is_super = false).\n";
    } else {
        int nsuper = (int)L->nsuper;
        auto* super = static_cast<cholmod_idx_t*>(L->super);

        sn_chol.reserve((size_t)nsuper);
        for (int s = 0; s < nsuper; ++s) {
            int a = (int)super[s];
            int b = (int)super[s + 1];
            sn_chol.emplace_back(a, b);
        }
    }

    CHM(free_factor)(&L, cc);
    CHM(free_sparse)(&S, cc);

    if (!cc_out) CHM(finish)(cc);
    return sn_chol;
}

static void compare_partitions(
    const std::vector<std::pair<int,int>>& ours,
    const std::vector<std::pair<int,int>>& chol,
    const char* tag_ours,
    const char* tag_chol,
    int max_print = 20
)
{
    std::cout << "\n==== Partition compare: " << tag_ours << " vs " << tag_chol << " ====\n";
    std::cout << "  " << tag_ours << " supernodes: " << ours.size() << "\n";
    std::cout << "  " << tag_chol << " supernodes: " << chol.size() << "\n";

    size_t m = std::min(ours.size(), chol.size());
    size_t first_bad = m;
    for (size_t k = 0; k < m; ++k) {
        if (ours[k] != chol[k]) { first_bad = k; break; }
    }

    if (first_bad == m && ours.size() == chol.size()) {
        std::cout << "  ✅ partitions identical\n";
        return;
    }

    if (first_bad == m) {
        std::cout << "  ⚠️ first " << m << " blocks match, but counts differ.\n";
    } else {
        std::cout << "  ❌ first mismatch at block k=" << first_bad << "\n";
        std::cout << "     " << tag_ours << ": [" << ours[first_bad].first << "," << ours[first_bad].second << ")\n";
        std::cout << "     " << tag_chol << ": [" << chol[first_bad].first << "," << chol[first_bad].second << ")\n";
    }

    int shown = (int)std::min(m, (size_t)max_print);
    std::cout << "  showing first " << shown << " blocks side-by-side:\n";
    for (int k = 0; k < shown; ++k) {
        std::cout << "    k=" << std::setw(3) << k
                  << "  " << tag_ours << "=[" << ours[k].first << "," << ours[k].second << ")"
                  << "  " << tag_chol << "=[" << chol[k].first << "," << chol[k].second << ")\n";
    }
}

// =============== Test ===============
TEST(SupernodalIO, CompareConservativeAndApproxOnNasa_WithCHOLMOD)
{
    std::string path = "/tmp/ic/test/data/nasa2146.mtx";
    auto A = ichol::io::mtx_to_csc<double>(path, false);

    ASSERT_GT(A.num_cols, 0);
    std::cout << "Matrix: " << path << "  ncols=" << A.num_cols << " nnz=" << A.nnz << "\n";

    // ---- your pipeline ----
    auto etree = ichol::symbolic::build_etree<double>(A);
    auto fp = ichol::symbolic::compute_complete_cholesky_pattern<double>(A, etree);

    ASSERT_EQ(fp.row_ptr_L.back(), static_cast<int>(fp.col_ind_L.size()));

    SymbolicOptions symopts; // default
    auto col_ls = ichol::symbolic::build_level_sets(fp, symopts);

    // (A) ours: relaxed (matches CHOLMOD default super_symbolic behavior)
    auto sn_ours_relaxed = ichol::symbolic::detect_supernodes(fp, etree);
    std::cout << "Ours detect (CHOLMOD-style relaxed amalgamation):\n";
    print_sn_range_list(sn_ours_relaxed);
    auto hist_relaxed = snode_size_histogram(sn_ours_relaxed, 16);
    std::cout << "Ours(relaxed) size histogram:\n";
    print_histogram(hist_relaxed);

    // (B) ours: fundamental (relax=off), for strict comparison
    auto sn_ours_fund = ichol::symbolic::detect_supernodes_fundamental(etree);
    std::cout << "\nOurs detect (fundamental, relax=off):\n";
    print_sn_range_list(sn_ours_fund);
    auto hist_fund = snode_size_histogram(sn_ours_fund, 16);
    std::cout << "Ours(fundamental) size histogram:\n";
    print_histogram(hist_fund);

    // Approx versions unchanged
    auto sn_appx1 = ichol::symbolic::detect_supernodes_approx(fp, etree, 1.0);
    std::cout << "\nApproximate detect (threshold=1.0):\n";
    print_sn_range_list(sn_appx1);

    double thr = 0.8;
    auto sn_appx08 = ichol::symbolic::detect_supernodes_approx(fp, etree, thr);
    std::cout << "\nApproximate detect (threshold=" << thr << "):\n";
    print_sn_range_list(sn_appx08);

    EXPECT_LE(sn_appx08.size(), sn_ours_relaxed.size());

    // ---- CHOLMOD symbolic supernodes ----
    // MUST match our etree/colcount semantics: we use strict UPPER (prefer_upper).
    // So use stype = +1.
    // const int stype = +1;
    const int stype = -1;

    // (1) CHOLMOD default-ish: identity ordering, postorder ON, relax ON
    auto sn_chol_default = cholmod_symbolic_supernodes_identity(
        A, stype,
        /*disable_postorder=*/false,
        /*disable_relax=*/false,
        /*cc_out=*/nullptr
    );

    std::cout << "\nCHOLMOD supernodes (identity perm, postorder=ON, relax=ON):\n";
    print_sn_range_list(sn_chol_default);

    // (2) CHOLMOD fundamental: identity ordering, postorder ON, relax OFF
    auto sn_chol_fund = cholmod_symbolic_supernodes_identity(
        A, stype,
        /*disable_postorder=*/false,
        /*disable_relax=*/true,
        /*cc_out=*/nullptr
    );

    std::cout << "\nCHOLMOD supernodes (identity perm, postorder=ON, relax=OFF):\n";
    print_sn_range_list(sn_chol_fund);

    // ---- comparisons ----
    compare_partitions(sn_ours_relaxed, sn_chol_default, "ours(relaxed)", "cholmod(relaxed)", 30);
    compare_partitions(sn_ours_fund,    sn_chol_fund,    "ours(fund)",    "cholmod(fund)",    30);

    // Optional strict asserts (enable only when you're confident everything aligned):
    // EXPECT_EQ(sn_ours_relaxed, sn_chol_default);
    // EXPECT_EQ(sn_ours_fund,    sn_chol_fund);
}

// ============================================================
// Test
// ============================================================


struct CholmodRAII
{
    cholmod_common cc{};
    cholmod_sparse* S = nullptr;
    cholmod_factor* L = nullptr;

    ~CholmodRAII() {
        if (L) CHM(free_factor)(&L, &cc);
        if (S) CHM(free_sparse)(&S, &cc);
        CHM(finish)(&cc);
    }
};

static CholmodRAII cholmod_super_ll_identity_factorize(const ichol::matrix::CscMatrix<double>& A, int stype)
{
    CholmodRAII h;
    CHM(start)(&h.cc);

    h.cc.supernodal  = CHOLMOD_SUPERNODAL;
    h.cc.final_super = 1;
    h.cc.final_ll    = 1;

    h.cc.nmethods = 1;
    h.cc.method[0].ordering = CHOLMOD_GIVEN;

    const int n = A.num_cols;
    std::vector<cholmod_idx_t> perm((size_t)n);
    for (int i = 0; i < n; ++i) perm[(size_t)i] = (cholmod_idx_t)i;

    h.S = csc_to_cholmod_real(A, stype, &h.cc);
    h.L = CHM(analyze_p)(h.S, perm.data(), nullptr, 0, &h.cc);
    if (!h.L) return h;

    CHM(factorize)(h.S, h.L, &h.cc);
    return h;
}


static ichol::symbolic::SuperSym extract_cholmod_super_sym(const cholmod_factor* L)
{
    ichol::symbolic::SuperSym out;

    const int nsuper = (int)L->nsuper;

    auto* Lsuper = static_cast<cholmod_idx_t*>(L->super);
    auto* Lpi    = static_cast<cholmod_idx_t*>(L->pi);
    auto* Lpx    = static_cast<cholmod_idx_t*>(L->px);
    auto* Ls     = static_cast<cholmod_idx_t*>(L->s);

    out.super.resize((size_t)nsuper + 1);
    out.pi.resize((size_t)nsuper + 1);
    out.px.resize((size_t)nsuper + 1);

    for (int k = 0; k < nsuper + 1; ++k) {
        out.super[(size_t)k] = (int)Lsuper[k];
        out.pi[(size_t)k]    = (int)Lpi[k];
        out.px[(size_t)k]    = (int)Lpx[k];
    }

    const int s_len = out.pi.back();
    out.s.resize((size_t)s_len);
    for (int t = 0; t < s_len; ++t) out.s[(size_t)t] = (int)Ls[t];

    return out;
}

static bool compare_super_sym(const ichol::symbolic::SuperSym& ours,
                             const ichol::symbolic::SuperSym& chol,
                             int ncols)
{
    const int no = (int)ours.super.size() - 1;
    const int nc = (int)chol.super.size() - 1;
    if (no != nc) {
        std::cout << "[DIFF] nsuper mismatch ours=" << no << " chol=" << nc << "\n";
        return false;
    }

    for (int k = 0; k < no; ++k) {
        if (ours.super[(size_t)k] != chol.super[(size_t)k] ||
            ours.super[(size_t)k + 1] != chol.super[(size_t)k + 1]) {
            std::cout << "[DIFF] super boundary mismatch k=" << k << "\n";
            return false;
        }

        const int o_pi0 = ours.pi[(size_t)k], o_pi1 = ours.pi[(size_t)k + 1];
        const int c_pi0 = chol.pi[(size_t)k], c_pi1 = chol.pi[(size_t)k + 1];
        if ((o_pi1 - o_pi0) != (c_pi1 - c_pi0)) {
            std::cout << "[DIFF] nsrow mismatch k=" << k << "\n";
            return false;
        }

        // row list: must match exactly (already sorted & unique in your Step1 alignment)
        const int len = o_pi1 - o_pi0;
        for (int t = 0; t < len; ++t) {
            if (ours.s[(size_t)(o_pi0 + t)] != chol.s[(size_t)(c_pi0 + t)]) {
                std::cout << "[DIFF] rowlist mismatch k=" << k << " at t=" << t
                          << " ours=" << ours.s[(size_t)(o_pi0 + t)]
                          << " chol=" << chol.s[(size_t)(c_pi0 + t)] << "\n";
                return false;
            }
        }

        if (ours.px[(size_t)k] != chol.px[(size_t)k] ||
            ours.px[(size_t)k + 1] != chol.px[(size_t)k + 1]) {
            std::cout << "[DIFF] px mismatch k=" << k << "\n";
            return false;
        }
    }

    if (ours.super.back() != ncols || chol.super.back() != ncols) {
        std::cout << "[DIFF] super.back != ncols\n";
        return false;
    }
    return true;
}

// Compare only meaningful region of supernodal x:
// - ignore L11 upper triangle (i<j within pivot block), because CHOLMOD doesn't guarantee it
static void report_block_maxdiff_meaningful(
    const ichol::symbolic::SuperSym& sym,
    const std::vector<double>& ours_x,
    const std::vector<double>& chol_x,
    int k,
    double& max_abs,
    double& max_rel)
{
    const int scol  = sym.super[(size_t)k];
    const int ecol  = sym.super[(size_t)k + 1];
    const int nscol = ecol - scol;

    const int pi0   = sym.pi[(size_t)k];
    const int pi1   = sym.pi[(size_t)k + 1];
    const int nsrow = pi1 - pi0;

    const int px0   = sym.px[(size_t)k];

    max_abs = 0.0;
    max_rel = 0.0;

    for (int j = 0; j < nscol; ++j) {
        for (int i = 0; i < nsrow; ++i) {
            if (i < nscol && i < j) continue; // skip L11 upper triangle

            const int t = i + j * nsrow;
            const int idx = px0 + t;

            const double a = ours_x[(size_t)idx];
            const double b = chol_x[(size_t)idx];
            const double d = std::abs(a - b);

            max_abs = std::max(max_abs, d);
            const double denom = std::max(1.0, std::abs(b));
            max_rel = std::max(max_rel, d / denom);
        }
    }
}

// ============================================================
// Step0: CHOLMOD baseline sanity (optional)
// ============================================================

TEST(Step0_CHOLMOD, BaselineFactorization_Nasa2146)
{
    const std::string path = "/tmp/ic/test/data/nasa2146.mtx";
    auto A = ichol::io::mtx_to_csc<double>(path, false);
    ASSERT_GT(A.num_cols, 0);

    auto h = cholmod_super_ll_identity_factorize(A, /*stype=*/-1);
    ASSERT_EQ(h.cc.status, CHOLMOD_OK);
    ASSERT_NE(h.L, nullptr);
    ASSERT_TRUE(h.L->is_super);
    ASSERT_TRUE(h.L->is_ll);

    std::cout << "\n==== CHOLMOD baseline factorization ====\n";
    std::cout << "Matrix: " << path << "  n=" << A.num_cols << "  nnz=" << A.nnz << "\n";
    std::cout << "Common->status=" << h.cc.status << " (0=OK)\n";
    std::cout << "CHOLMOD factor: n=" << (int)h.L->n
              << "  is_ll=" << (int)h.L->is_ll
              << "  is_super=" << (int)h.L->is_super
              << "  nsuper=" << (int)h.L->nsuper
              << "  minor=" << (int)h.L->minor << "\n";
}

// ============================================================
// Step1: ours vs CHOLMOD super-symbolic  (NO implementation in test)
// ============================================================

TEST(Step1_Symbolic, OursVsCHOLMOD_SuperSymbolic_Nasa2146)
{
    const std::string path = "/tmp/ic/test/data/nasa2146.mtx";
    auto A = ichol::io::mtx_to_csc<double>(path, false);
    ASSERT_GT(A.num_cols, 0);
    const int n = A.num_cols;

    // --- call your library symbolic pipeline (do NOT implement in test) ---
    auto etree  = ichol::symbolic::build_etree<double>(A);
    auto pat    = ichol::symbolic::compute_complete_cholesky_pattern<double>(A, etree);
    auto snodes = ichol::symbolic::detect_supernodes(pat, etree);
    auto rows   = ichol::symbolic::compute_snode_rows(pat, snodes);

    SuperSym ours_sym = build_super_sym(snodes, rows);

    // CHOLMOD
    auto h = cholmod_super_ll_identity_factorize(A, /*stype=*/-1);
    ASSERT_EQ(h.cc.status, CHOLMOD_OK);
    ASSERT_NE(h.L, nullptr);
    ASSERT_TRUE(h.L->is_super);
    ASSERT_TRUE(h.L->is_ll);

    ichol::symbolic::SuperSym chol_sym = extract_cholmod_super_sym(h.L);

    std::cout << "\n==== Step1: ours vs CHOLMOD super-symbolic ====\n";
    std::cout << "Matrix: " << path << "  n=" << n << "  nnz=" << A.nnz << "\n";
    std::cout << "ours.nsuper=" << (int)ours_sym.super.size() - 1 << "\n";
    std::cout << "chol.nsuper=" << (int)chol_sym.super.size() - 1 << "\n";

    EXPECT_TRUE(compare_super_sym(ours_sym, chol_sym, n));
}

// ============================================================
// Step3: our numeric vs CHOLMOD x  (NO implementation in test)
// ============================================================

TEST(Step3_Numeric, OursVsCHOLMOD_SupernodalX_Nasa2146)
{
    const std::string path = "/tmp/ic/test/data/nasa2146.mtx";
    auto A = ichol::io::mtx_to_csc<double>(path, false);
    ASSERT_GT(A.num_cols, 0);
    const int n = A.num_cols;

    // our symbolic (library)
    auto etree  = ichol::symbolic::build_etree<double>(A);
    auto pat    = ichol::symbolic::compute_complete_cholesky_pattern<double>(A, etree);
    auto snodes = ichol::symbolic::detect_supernodes(pat, etree);
    auto rows   = ichol::symbolic::compute_snode_rows(pat, snodes);

    // your SuperSym (library)
    ichol::symbolic::SuperSym ours_sym = ichol::symbolic::build_super_sym(snodes, rows);

    // CHOLMOD baseline
    auto h = cholmod_super_ll_identity_factorize(A, /*stype=*/-1);
    ASSERT_EQ(h.cc.status, CHOLMOD_OK);
    ASSERT_NE(h.L, nullptr);
    ASSERT_TRUE(h.L->is_super);
    ASSERT_TRUE(h.L->is_ll);

    ichol::symbolic::SuperSym chol_sym = extract_cholmod_super_sym(h.L);
    ASSERT_TRUE(compare_super_sym(ours_sym, chol_sym, n));

    // extract CHOLMOD x
    const int xlen = chol_sym.px.back();
    ASSERT_NE(h.L->x, nullptr);
    std::vector<double> x_chol((size_t)xlen);
    std::copy((const double*)h.L->x, (const double*)h.L->x + (size_t)xlen, x_chol.begin());

    // our numeric (library) - 这里用你已经实现对齐 CHOLMOD 的 multifrontal 版本
    // 如果你的接口名不同，改这一行即可：
    //
    // 期望返回：struct { bool ok; int fail_snode; int fail_col_in_snode; std::vector<double> x; }
    //
    auto num = ichol::symbolic::factorize_supernodal_ll(A, ours_sym);

    ASSERT_TRUE(num.ok) << "Our LL failed at snode=" << num.fail_snode << " col=" << num.fail_col_in_snode;
    ASSERT_EQ((int)num.x.size(), xlen);

    // compare meaningful region
    double max_abs = 0.0, max_rel = 0.0;
    int worst_k = -1;
    double worst_abs = 0.0;

    const int nsuper = (int)chol_sym.super.size() - 1;
    for (int k = 0; k < nsuper; ++k) {
        double a = 0.0, r = 0.0;
        report_block_maxdiff_meaningful(chol_sym, num.x, x_chol, k, a, r);
        if (a > worst_abs) { worst_abs = a; worst_k = k; }
        max_abs = std::max(max_abs, a);
        max_rel = std::max(max_rel, r);
    }

    std::cout << "\n==== Step3: our numeric vs CHOLMOD x (meaningful region) ====\n";
    std::cout << "max_abs=" << std::setprecision(15) << max_abs
              << " max_rel=" << std::setprecision(15) << max_rel
              << " worst_k=" << worst_k
              << " worst_abs=" << worst_abs
              << "\n";

    // 你已经跑到 ~1e-12，直接用紧阈值
    EXPECT_LT(max_abs, 1e-9);
    EXPECT_LT(max_rel, 1e-9);
}




