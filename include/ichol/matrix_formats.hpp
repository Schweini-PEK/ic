/**
 * @file matrix_formats.hpp
 * @brief Definitions of common sparse matrix formats (CSR, CSC)
 */
#ifndef INCHOL_MATRIX_FORMATS_HPP
#define INCHOL_MATRIX_FORMATS_HPP

#include <vector>

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

    } // namespace ichol::matrix
}

#endif // INCHOL_MATRIX_FORMATS_HPP