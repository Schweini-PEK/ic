#pragma once

#include <gtest/gtest.h>
#include "ichol/matrix_formats.hpp"

namespace ichol::testutil
{
    template <typename T>
    inline void assert_diag_last_csr(const ichol::matrix::CsrMatrix<T> &M)
    {
        const int n = M.num_rows;
        for (int i = 0; i < n; ++i)
        {
            const int row_end = M.row_ptr[i + 1];
            const int last_col = M.col_ind[row_end - 1];
            ASSERT_EQ(last_col, i) << "CSR diag-check failed: row " << i
                                   << " last column index " << last_col
                                   << " != expected diag index " << i;
        }
    }
} // namespace ichol::testutil