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
#include "factor/symbolic/symbolic.hpp"
#include "ichol/symbolic.hpp"  // ETree, FactorPattern, declaration of compute_complete_cholesky_pattern
#include "ichol/mtx_read.hpp"  // CscMatrix type (or the header where CscMatrix is defined)
#include "ichol/matrix_formats.hpp"


// Helper to create a CSC matrix (n x n)
static ichol::matrix::CscMatrix<double> make_csc(int n, const std::vector<int>& col_ptr, const std::vector<int>& row_ind) {
    ichol::matrix::CscMatrix<double> A;
    A.num_rows = n;
    A.num_cols = n;
    A.nnz = static_cast<int>(row_ind.size());
    A.col_ptr = col_ptr;
    A.row_ind = row_ind;
    A.values.clear();
    return A;
}

TEST(PatternCSC, DiagonalOnly) {
    int n = 4;
    const std::vector<int> col_ptr = {0,1,2,3,4};
    const std::vector<int> row_ind = {0,1,2,3};
    ichol::matrix::CscMatrix<double> A = make_csc(n, col_ptr, row_ind);

    auto et = ichol::symbolic::build_etree<double>(A);
    auto pattern = ichol::symbolic::compute_complete_cholesky_pattern<double>(A, et);

    ASSERT_EQ(static_cast<int>(pattern.row_ptr_L.size()), n+1);
    for (int k = 0; k < n; ++k) {
        int s = pattern.row_ptr_L[k], e = pattern.row_ptr_L[k+1];
        ASSERT_EQ(e - s, 1);
        EXPECT_EQ(pattern.col_ind_L[s], k);
    }
}

TEST(PatternCSC, ColumnWithLowerEntries) {
    int n = 4;
    std::vector<int> col_ptr = {0,3,4,5,6};
    std::vector<int> row_ind = {0,1,2, 1, 2, 3}; // col0 rows 0,1,2 ; others diag
    auto A = make_csc(n, col_ptr, row_ind);

    auto et = ichol::symbolic::build_etree<double>(A);
    auto pattern = ichol::symbolic::compute_complete_cholesky_pattern<double>(A, et);

    const int s0 = pattern.row_ptr_L[0];
    // column 0 should contain 0,1,2 (order unspecified)
    int e0 = pattern.row_ptr_L[1];
    std::vector<int> col0(pattern.col_ind_L.begin() + s0, pattern.col_ind_L.begin() + e0);
    std::sort(col0.begin(), col0.end());
    const std::vector<int> expected0 = {0,1,2};
    EXPECT_EQ(col0, expected0);

    // other columns should contain at least their diagonal entry
    for (int k = 1; k < n; ++k) {
        int s = pattern.row_ptr_L[k], e = pattern.row_ptr_L[k+1];
        bool found_diag = false;
        for (int p = s; p < e; ++p) if (pattern.col_ind_L[p] == k) { found_diag = true; break; }
        EXPECT_TRUE(found_diag);
    }
}