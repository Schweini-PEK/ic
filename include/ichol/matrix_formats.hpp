/**
 * @file matrix_formats.hpp
 * @brief Definitions of common sparse matrix formats (CSR, CSC)
 */
#ifndef INCHOL_MATRIX_FORMATS_HPP
#define INCHOL_MATRIX_FORMATS_HPP

#include <vector>
#include "ichol/half.hpp"

namespace ichol::matrix
{
    template <typename T>
    struct CooMatrix
    {
        std::vector<int> row_ind; // Row indices (size: nnz)
        std::vector<int> col_ind; // Column indices (size: nnz)
        std::vector<T> values;    // Non-zero values (size: nnz)
        int num_rows;             // Number of rows
        int num_cols;             // Number of columns
        int nnz;                  // Number of non-zero elements
    };

    template <typename T>
    struct CsrMatrix
    {
        std::vector<int> row_ptr; // Row pointers (size: num_rows + 1)
        std::vector<int> col_ind; // Column indices (size: nnz)
        std::vector<T> values;    // Non-zero values (size: nnz)
        int num_rows;             // Number of rows
        int num_cols;             // Number of columns
        int nnz;                  // Number of non-zero elements
    };

    template <typename T>
    struct CscMatrix
    {
        std::vector<int> col_ptr; // Column pointers (size: num_cols + 1)
        std::vector<int> row_ind; // Row indices (size: nnz)
        std::vector<T> values;    // Non-zero values (size: nnz)
        int num_rows;             // Number of rows
        int num_cols;             // Number of columns
        int nnz;                  // Number of non-zero elements
    };

    inline void csr_to_csc_pattern_only(
        int n,
        const std::vector<int> &csr_row_ptr,
        const std::vector<int> &csr_col_ind,
        std::vector<int> &csc_col_ptr,
        std::vector<int> &csc_row_ind,
        std::vector<int> &csc_to_csr_map)
    {
        const int nnz = csr_col_ind.size();
        csc_col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);

        for (int p = 0; p < nnz; ++p)
        {
            int c = csr_col_ind[p];
            ++csc_col_ptr[c + 1];
        }

        for (int c = 0; c < n; ++c)
        {
            csc_col_ptr[c + 1] += csc_col_ptr[c];
        }

        csc_row_ind.resize(static_cast<std::size_t>(nnz));
        csc_to_csr_map.resize(static_cast<std::size_t>(nnz));

        std::vector<int> next = csc_col_ptr;
        for (int i = 0; i < n; ++i)
        {
            for (int p = csr_row_ptr[i]; p < csr_row_ptr[i + 1]; ++p)
            {
                int c = csr_col_ind[p];
                int dest = next[c]++;
                csc_row_ind[dest] = i;
                csc_to_csr_map[dest] = p;
            }
        }
    }

    template <typename Tin, typename Tout>
    inline matrix::CsrMatrix<Tout> convert_csr_precision(const matrix::CsrMatrix<Tin> &src)
    {
        matrix::CsrMatrix<Tout> dst;
        dst.num_rows = src.num_rows;
        dst.num_cols = src.num_cols;
        const int nnz = static_cast<int>(src.values.size());
        dst.nnz = nnz;
        dst.row_ptr = src.row_ptr; // copy structure
        dst.col_ind = src.col_ind; // copy structure
        dst.values.resize(static_cast<std::size_t>(nnz));
        for (int i = 0; i < nnz; ++i)
            dst.values[static_cast<std::size_t>(i)] = static_cast<Tout>(src.values[static_cast<std::size_t>(i)]);
        return dst;
    }
} // namespace ichol::matrix

#endif // INCHOL_MATRIX_FORMATS_HPP