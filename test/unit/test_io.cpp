// test/unit/test_io.cpp
#include <gtest/gtest.h>

#include "ichol/mtx_read.hpp"

namespace
{
    template <typename T>
    bool csr_diag_positive(const ichol::matrix::CsrMatrix<T> &csr)
    {
        int n = csr.num_rows;
        for (int i = 0; i < n; i++)
        {
            int row_end = csr.row_ptr[i + 1];
            if (csr.col_ind[row_end - 1] != i)
            {
                ADD_FAILURE() << "CSR diag missing at row " << i;
                return false;
            }
            T diag_val = csr.values[row_end - 1];
            if (diag_val <= T(0))
            {
                ADD_FAILURE() << "CSR diag not positive at row " << i << ": " << diag_val;
                return false;
            }
        }
        return true;
    }

    template <typename T>
    bool csc_diag_positive(const ichol::matrix::CscMatrix<T> &csc)
    {
        int n = csc.num_cols;
        for (int j = 0; j < n; j++)
        {
            int col_start = csc.col_ptr[j];
            if (csc.row_ind[col_start] != j)
            {
                ADD_FAILURE() << "CSC diag missing at col " << j;
                return false;
            }
            T diag_val = csc.values[col_start];
            if (diag_val <= T(0))
            {
                ADD_FAILURE() << "CSC diag not positive at col " << j << ": " << diag_val;
                return false;
            }
        }
        return true;
    }

    template <typename T>
    bool csr_sorted_check(const ichol::matrix::CsrMatrix<T> &csr)
    {
        int n = csr.num_rows;
        for (int i = 0; i < n; i++)
        {
            int row_start = csr.row_ptr[i];
            int row_end = csr.row_ptr[i + 1];
            for (int p = row_start + 1; p < row_end; p++)
            {
                if (csr.col_ind[p] <= csr.col_ind[p - 1])
                {
                    ADD_FAILURE() << "CSR row " << i << " not sorted at position " << p;
                    return false;
                }
            }
        }
        return true;
    }
}

template <typename T>
bool csc_sorted_check(const ichol::matrix::CscMatrix<T> &csc)
{
    int n = csc.num_cols;
    for (int j = 0; j < n; j++)
    {
        int col_start = csc.col_ptr[j];
        int col_end = csc.col_ptr[j + 1];
        for (int p = col_start + 1; p < col_end; p++)
        {
            if (csc.row_ind[p] <= csc.row_ind[p - 1])
            {
                ADD_FAILURE() << "CSC col " << j << " not sorted at position " << p;
                return false;
            }
        }
    }
    return true;
}

TEST(io, ProducesUsablePreconditionerOnMTX)
{
    std::string path = "test/data/nasa2146.mtx";
    ichol::matrix::CsrMatrix<double> Acsr = ichol::io::mtx_to_csr<double>(path, false);
    ichol::matrix::CscMatrix<double> Acsc = ichol::io::mtx_to_csc<double>(path, false);

    ASSERT_EQ(Acsr.num_rows, Acsc.num_rows);
    ASSERT_EQ(Acsr.num_cols, Acsc.num_cols);
    ASSERT_EQ(Acsr.num_rows, Acsc.num_rows);
    ASSERT_EQ(Acsr.row_ptr.size(), static_cast<size_t>(Acsr.num_rows + 1));
    ASSERT_EQ(Acsc.col_ptr.size(), static_cast<size_t>(Acsc.num_cols + 1));

    ASSERT_EQ(Acsr.nnz, Acsc.nnz);
    ASSERT_EQ(Acsr.row_ptr[Acsr.num_rows], Acsr.nnz);
    ASSERT_EQ(Acsr.row_ptr[0], 0);
    ASSERT_EQ(Acsc.col_ptr[Acsc.num_cols], Acsc.nnz);
    ASSERT_EQ(Acsc.col_ptr[0], 0);

    // Diag check
    ASSERT_TRUE(csr_diag_positive(Acsr));
    ASSERT_TRUE(csc_diag_positive(Acsc));

    // Sorted check
    ASSERT_TRUE(csr_sorted_check(Acsr));
    ASSERT_TRUE(csc_sorted_check(Acsc));
}