#ifndef INCHOL_MATRIX_FORMATS_HPP
#define INCHOL_MATRIX_FORMATS_HPP

#include <vector>

namespace ichol
{

    template <typename T>
    struct CSR
    {
        std::vector<int> row_ptr; // Row pointers (size: num_rows + 1)
        std::vector<int> col_ind; // Column indices (size: nnz)
        std::vector<T> values;    // Non-zero values (size: nnz)
        int num_rows;             // Number of rows
        int num_cols;             // Number of columns
        int nnz;                  // Number of non-zero elements
    };

    template <typename T>
    struct CSC
    {
        std::vector<int> col_ptr; // Column pointers (size: num_cols + 1)
        std::vector<int> row_ind; // Row indices (size: nnz)
        std::vector<T> values;    // Non-zero values (size: nnz)
        int num_rows;             // Number of rows
        int num_cols;             // Number of columns
        int nnz;                  // Number of non-zero elements
    };
} // namespace ichol

#endif // INCHOL_MATRIX_FORMATS_HPP