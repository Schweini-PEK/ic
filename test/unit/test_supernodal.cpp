// // test_supernodal.cpp
//
#include <gtest/gtest.h>
//
// // #include "factor/symbolic/symbolic.hpp"
// // #include "ichol/matrix_formats.hpp"
// // #include "factor/symbolic/symbolic.hpp"
// // namespace test_checks
// // {
// //     // (No additional checks needed here for supernodal tests yet)
// // } // namespace test_checks
// //
// // TEST(Supernodal, PlaceholderTest)
// // {
// //     std::string path = "test/data/nasa2146.mtx";
// //
// //     ASSERT_TRUE(true);
// // }
//
// #include <algorithm>
//
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "ichol/mtx_read.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/symbolic/detail/symbolic_plan.hpp"

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

// produce histogram of supernode sizes (columns per supernode)
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

TEST(SupernodalIO, CompareConservativeAndApproxOnNasa)
{
    std::string path = "F:/new/ic/test/data/nasa2146.mtx";
    // load CSC (we run supernode pipeline on CSC)
    auto A = ichol::io::mtx_to_csc<double>(path, false);

    ASSERT_GT(A.num_cols, 0);
    ASSERT_EQ(A.num_rows, A.num_cols);
    std::cout << "Matrix: " << path << "  ncols=" << A.num_cols << " nnz=" << A.nnz << "\n";

    // 1) build etree and factor pattern
    auto etree = ichol::symbolic::build_etree<double>(A);
    auto fp = ichol::symbolic::compute_complete_cholesky_pattern<double>(A, etree);

    // sanity checks
    ASSERT_EQ(static_cast<int>(fp.row_ptr_L.size()), A.num_cols + 1);
    ASSERT_EQ(fp.row_ptr_L.back(), static_cast<int>(fp.col_ind_L.size()));

    // 2) column-level level sets (used to derive snode levels)
    SymbolicOptions symopts; // default
    auto col_ls = ichol::symbolic::build_level_sets(fp, symopts);

    // 3) conservative detection
    auto sn_cons = ichol::symbolic::detect_supernodes(fp, etree);
    std::cout << "Conservative detect:\n";
    print_sn_range_list(sn_cons);
    auto hist_cons = snode_size_histogram(sn_cons, 16);
    std::cout << "Conservative size histogram:\n";
    print_histogram(hist_cons);

    // 4) approximate detection with threshold = 1.0 (should match conservative)
    auto sn_appx1 = ichol::symbolic::detect_supernodes_approx(fp, etree, 1.0);
    std::cout << "\nApproximate detect (threshold=1.0):\n";
    print_sn_range_list(sn_appx1);
    auto hist_appx1 = snode_size_histogram(sn_appx1, 16);
    std::cout << "Approx(1.0) size histogram:\n";
    print_histogram(hist_appx1);

    // 5) approximate detection with threshold = 0.8 (coarser, likely fewer snodes)
    double thr = 0.8;
    auto sn_appx08 = ichol::symbolic::detect_supernodes_approx(fp, etree, thr);
    std::cout << "\nApproximate detect (threshold=" << thr << "):\n";
    print_sn_range_list(sn_appx08);
    auto hist_appx08 = snode_size_histogram(sn_appx08, 16);
    std::cout << "Approx(0.8) size histogram:\n";
    print_histogram(hist_appx08);

    // 6) basic assertions about relationships
    ASSERT_EQ(sn_cons.size(), sn_appx1.size());
    EXPECT_LE(sn_appx08.size(), sn_cons.size());

    // 7) build col->snode map and check coverage
    auto col2s_cons = ichol::symbolic::build_col2snode(sn_cons, A.num_cols);
    auto col2s_appx08 = ichol::symbolic::build_col2snode(sn_appx08, A.num_cols);

    ASSERT_EQ(static_cast<int>(col2s_cons.size()), A.num_cols);
    ASSERT_EQ(static_cast<int>(col2s_appx08.size()), A.num_cols);

    // each column must map to exactly one snode id (>=0)
    for (int c = 0; c < A.num_cols; ++c) {
        EXPECT_GE(col2s_cons[c], 0);
        EXPECT_LT(col2s_cons[c], static_cast<int>(sn_cons.size()));
        EXPECT_GE(col2s_appx08[c], 0);
        EXPECT_LT(col2s_appx08[c], static_cast<int>(sn_appx08.size()));
    }

    // 8) compute snode rows for both and compare numbers
    auto snrows_cons = ichol::symbolic::compute_snode_rows(fp, sn_cons);
    auto snrows_appx08 = ichol::symbolic::compute_snode_rows(fp, sn_appx08);

    std::cout << "\nSnode rows: conservative has " << snrows_cons.size() << " blocks, approx(0.8) has " << snrows_appx08.size() << "\n";

    // 9) build snode-level-sets for approx(0.8) and conservative
    auto snode_level_cons = ichol::symbolic::build_snode_level_sets(col_ls, sn_cons);
    auto snode_level_appx08 = ichol::symbolic::build_snode_level_sets(col_ls, sn_appx08);

    std::cout << "\nSnode-level sets: conservative levels = " << (snode_level_cons.snode_level.size()) << "; level buckets = " << (snode_level_cons.level_sets.level_ptr.size()-1) << "\n";
    std::cout << "                   approx(0.8) levels = " << (snode_level_appx08.snode_level.size()) << "; level buckets = " << (snode_level_appx08.level_sets.level_ptr.size()-1) << "\n";

    // 10) print a short comparison summary
    std::cout << "\nSummary:\n";
    std::cout << "  conservative supernodes: " << sn_cons.size() << "\n";
    std::cout << "  approx(th=1.0) supernodes: " << sn_appx1.size() << "\n";
    std::cout << "  approx(th=0.8) supernodes: " << sn_appx08.size() << "\n";
    std::cout << "  conservative total snode rows nnz (sum sizes): ";
    int sum_cons = 0;
    for (auto &r : snrows_cons) sum_cons += (int)r.size();
    std::cout << sum_cons << "\n";
    int sum_appx08 = 0;
    for (auto &r : snrows_appx08) sum_appx08 += (int)r.size();
    std::cout << "  approx(0.8) total snode rows nnz (sum sizes): " << sum_appx08 << "\n";

    // small sanity checks
    EXPECT_GT(sn_cons.size(), 0u);
    EXPECT_GT(sn_appx08.size(), 0u);
}