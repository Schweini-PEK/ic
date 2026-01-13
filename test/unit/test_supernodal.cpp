// test_supernodal.cpp
#include <gtest/gtest.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <utility>
#include <filesystem>

// your project headers
#include "ichol/mtx_read.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/symbolic/detail/symbolic_plan.hpp"

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
