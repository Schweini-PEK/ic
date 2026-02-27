#pragma once

#include <gtest/gtest.h>
#include "ichol/matrix_formats.hpp"
#include "factor/symbolic/symbolic.hpp"

namespace ichol::testutil
{
    /**
     * CSR SPD matrix checks
     * */
    inline void assert_diag_last_csr(const std::vector<int> &row_ptr,
                                     const std::vector<int> &col_ind)
    {
        const int n = static_cast<int>(row_ptr.size()) - 1;
        for (int i = 0; i < n; ++i)
        {
            const int row_end = row_ptr[i + 1];
            const int last_col = col_ind[row_end - 1];
            ASSERT_EQ(last_col, i) << "CSR diag-check failed: row " << i
                                   << " last column index " << last_col
                                   << " != expected diag index " << i;
        }
    }

    inline void assert_diag_positive_csr(const std::vector<int> &row_ptr,
                                         const std::vector<int> &col_ind,
                                         const std::vector<double> &values)
    {
        const int n = static_cast<int>(row_ptr.size()) - 1;
        for (int i = 0; i < n; ++i)
        {
            const int row_end = row_ptr[i + 1];
            const int diag_pos = row_end - 1;
            const double diag_val = values[diag_pos];
            ASSERT_GT(diag_val, 0.0) << "CSR diag-positivity check failed: row " << i
                                     << " diag value " << diag_val << " not positive.";
        }
    }

    inline void assert_lower_only_csr(const std::vector<int> &row_ptr,
                                      const std::vector<int> &col_ind)
    {
        const int n = static_cast<int>(row_ptr.size()) - 1;
        for (int i = 0; i < n; ++i)
        {
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];
            for (int p = row_start; p < row_end; ++p)
            {
                const int j = col_ind[p];
                ASSERT_LE(j, i) << "CSR lower-triangular check failed: row " << i
                                << " has upper entry at column " << j;
            }
        }
    }

    inline void assert_matrices_equal(const ichol::matrix::CsrMatrix<double> &A,
                                      const ichol::matrix::CsrMatrix<double> &B)
    {
        for (int i = 0; i < A.num_rows + 1; ++i)
        {
            ASSERT_EQ(A.row_ptr[i], B.row_ptr[i]) << "row_ptr mismatch at index " << i;
        }
        for (int i = 0; i < A.nnz; ++i)
        {
            ASSERT_EQ(A.col_ind[i], B.col_ind[i]) << "col_ind mismatch at index " << i;
            ASSERT_DOUBLE_EQ(A.values[i], B.values[i]) << "values mismatch at index " << i;
        }
    }

    inline void assert_shift_delta_equal(const ichol::matrix::CsrMatrix<double> &A,
                                         const ichol::matrix::CsrMatrix<double> &B,
                                         const double shift)
    {
        for (int i = 0; i < A.num_rows; ++i)
        {
            const int row_end = A.row_ptr[i + 1];
            const int diag_pos = row_end - 1;
            const double diag_A = A.values[diag_pos];
            const double diag_B = B.values[diag_pos];
            double expected = diag_A + shift;
            double scale = std::max({1.0, std::abs(diag_A), std::abs(diag_B), std::abs(expected)});
            double tol = 16 * std::numeric_limits<double>::epsilon() * scale;
            ASSERT_NEAR(diag_B, expected, tol) << "diagonal shift mismatch at row " << i;
        }
    }

    inline void assert_cols_sorted_unique(const std::vector<int> &row_ptr,
                                          const std::vector<int> &col_ind)
    {
        const int n = static_cast<int>(row_ptr.size()) - 1;
        for (int i = 0; i < n; ++i)
        {
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];
            int prev_col = -1;
            for (int p = row_start; p < row_end; ++p)
            {
                const int j = col_ind[p];
                ASSERT_GT(j, prev_col) << "CSR column indices not sorted/unique at row " << i;
                prev_col = j;
            }
        }
    }

    /**
     * Symbolic analysis checks
     */

    inline void assert_row_nnz_le(const std::vector<int> &row_ptr,
                                  int max_nnz)
    {
        const int n = static_cast<int>(row_ptr.size()) - 1;
        for (int i = 0; i < n; ++i)
        {
            const int row_nnz = row_ptr[i + 1] - row_ptr[i];
            ASSERT_LE(row_nnz, max_nnz) << "Row " << i << " has nnz " << row_nnz
                                        << " exceeding limit " << max_nnz;
        }
    }

    inline void assert_dependency_L(const std::vector<int> &row_ptr_L,
                                    const std::vector<int> &col_ind_L,
                                    const ichol::symbolic::LevelSets &level_sets)
    {
        const int n = static_cast<int>(row_ptr_L.size()) - 1;
        ASSERT_GE(n, 0);

        // LevelSets basic invariants
        ASSERT_EQ(static_cast<int>(level_sets.levels.size()), n);
        ASSERT_GE(level_sets.level_ptr.size(), 2u);
        ASSERT_EQ(level_sets.level_ptr.front(), 0);
        ASSERT_EQ(level_sets.level_ptr.back(), n);
        for (size_t i = 0; i + 1 < level_sets.level_ptr.size(); ++i)
            ASSERT_LE(level_sets.level_ptr[i], level_sets.level_ptr[i + 1]);

        // Build row -> level map (0-based)
        const int num_levels = static_cast<int>(level_sets.level_ptr.size()) - 1;
        std::vector<int> level_of(n, -1);

        for (int lev = 0; lev < num_levels; ++lev)
        {
            const int b = level_sets.level_ptr[lev];
            const int e = level_sets.level_ptr[lev + 1];
            for (int t = b; t < e; ++t)
            {
                const int row = level_sets.levels[t];
                ASSERT_GE(row, 0);
                ASSERT_LT(row, n);
                ASSERT_EQ(level_of[row], -1) << "Row appears in multiple levels: row=" << row;
                level_of[row] = lev;
            }
        }
        for (int i = 0; i < n; ++i)
            ASSERT_NE(level_of[i], -1) << "Row missing from level schedule: row=" << i;

        // Dependency check: for each offdiag (i,k) in L pattern, level(k) < level(i)
        for (int i = 0; i < n; ++i)
        {
            const int li = level_of[i];
            const int r0 = row_ptr_L[i];
            const int r1 = row_ptr_L[i + 1];
            ASSERT_LE(r0, r1);
            ASSERT_GE(r0, 0);
            ASSERT_LE(r1, static_cast<int>(col_ind_L.size()));

            // Skip the last entry (assumed diagonal) by convention
            for (int p = r0; p < r1 - 1; ++p)
            {
                const int k = col_ind_L[p];
                ASSERT_GE(k, 0);
                ASSERT_LT(k, n);
                ASSERT_LT(level_of[k], li)
                    << "Level dependency violated at row i=" << i
                    << " depends on k=" << k
                    << " with level(i)=" << li
                    << " level(k)=" << level_of[k];
            }
        }
    }
} // namespace ichol::testutil