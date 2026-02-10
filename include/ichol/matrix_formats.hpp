/**
 * @file matrix_formats.hpp
 * @brief Definitions of common sparse matrix formats (CSR, CSC)
 */
#ifndef INCHOL_MATRIX_FORMATS_HPP
#define INCHOL_MATRIX_FORMATS_HPP

#include <vector>
#include <cassert>
#include <numeric>
#include <cstring>
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
        const int n,
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

    inline void csc_to_csr_pattern_only(
        const int n,
        const std::vector<int> &csc_col_ptr,
        const std::vector<int> &csc_row_ind,
        std::vector<int> &csr_row_ptr,
        std::vector<int> &csr_col_ind,
        std::vector<int> &csr_to_csc_map)
    {
        const int nnz = csc_row_ind.size();
        csr_row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);

        for (int p = 0; p < nnz; ++p)
        {
            int r = csc_row_ind[p];
            ++csr_row_ptr[r + 1];
        }

        for (int r = 0; r < n; ++r)
        {
            csr_row_ptr[r + 1] += csr_row_ptr[r];
        }

        csr_col_ind.resize(static_cast<std::size_t>(nnz));
        csr_to_csc_map.resize(static_cast<std::size_t>(nnz));

        std::vector<int> next = csr_row_ptr;
        for (int c = 0; c < n; ++c)
        {
            for (int p = csc_col_ptr[c]; p < csc_col_ptr[c + 1]; ++p)
            {
                int r = csc_row_ind[p];
                int dest = next[r]++;
                csr_col_ind[dest] = c;
                csr_to_csc_map[dest] = p;
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

    template <typename T>
    inline matrix::CooMatrix<T> csr_to_coo(const matrix::CsrMatrix<T> &csr)
    {
        matrix::CooMatrix<T> coo;
        coo.num_rows = csr.num_rows;
        coo.num_cols = csr.num_cols;
        coo.nnz = csr.nnz;
        coo.row_ind.resize(static_cast<std::size_t>(coo.nnz));
        coo.col_ind.resize(static_cast<std::size_t>(coo.nnz));
        coo.values.resize(static_cast<std::size_t>(coo.nnz));

        for (int i = 0; i < csr.num_rows; ++i)
        {
            for (int p = csr.row_ptr[static_cast<std::size_t>(i)]; p < csr.row_ptr[static_cast<std::size_t>(i) + 1]; ++p)
            {
                const int j = csr.col_ind[static_cast<std::size_t>(p)];
                const T v = csr.values[static_cast<std::size_t>(p)];
                const std::size_t k = static_cast<std::size_t>(p);
                coo.row_ind[k] = i;
                coo.col_ind[k] = j;
                coo.values[k] = v;
            }
        }

        return coo;
    }

    template <typename T>
    inline matrix::CooMatrix<T> csc_to_coo(const matrix::CscMatrix<T> &csc)
    {
        matrix::CooMatrix<T> coo;
        coo.num_rows = csc.num_rows;
        coo.num_cols = csc.num_cols;
        coo.nnz = csc.nnz;
        coo.row_ind.resize(static_cast<std::size_t>(coo.nnz));
        coo.col_ind.resize(static_cast<std::size_t>(coo.nnz));
        coo.values.resize(static_cast<std::size_t>(coo.nnz));

        for (int j = 0; j < csc.num_cols; ++j)
        {
            for (int p = csc.col_ptr[static_cast<std::size_t>(j)]; p < csc.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            {
                const int i = csc.row_ind[static_cast<std::size_t>(p)];
                const T v = csc.values[static_cast<std::size_t>(p)];
                const std::size_t k = static_cast<std::size_t>(p);
                coo.row_ind[k] = i;
                coo.col_ind[k] = j;
                coo.values[k] = v;
            }
        }

        return coo;
    }
} // namespace ichol::matrix

#endif // INCHOL_MATRIX_FORMATS_HPP