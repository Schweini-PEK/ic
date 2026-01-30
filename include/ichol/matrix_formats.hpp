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

    // Input: CSR(L), square, lower-tri + diag, per-row sorted by col, and diagonal is last entry in each row.
    // Output: CSR(L^T), still square, per-row sorted by col (off-diags), and diagonal moved to be last entry per row.
    template <typename T>
    CsrMatrix<T> csr_lower_to_csr_transpose_diag_last(const CsrMatrix<T> &L)
    {
        const int n = L.num_rows;
        assert(L.num_rows == L.num_cols);
        assert((int)L.row_ptr.size() == n + 1);
        assert((int)L.col_ind.size() == L.nnz);
        assert((int)L.values.size() == L.nnz);

        CsrMatrix<T> Lt;
        Lt.num_rows = n;
        Lt.num_cols = n;
        Lt.nnz = L.nnz;
        Lt.row_ptr.assign(n + 1, 0);
        Lt.col_ind.resize(L.nnz);
        Lt.values.resize(L.nnz);

        // Count nnz per row of Lt (i.e., per column of L).
        for (int i = 0; i < n; ++i)
        {
            for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p)
            {
                const int j = L.col_ind[p];
                // If L is truly lower+diag, j <= i. Not required for building, but useful as an invariant.
                // assert(j <= i);
                Lt.row_ptr[j + 1]++;
            }
        }

        // Prefix sum to row_ptr.
        std::partial_sum(Lt.row_ptr.begin(), Lt.row_ptr.end(), Lt.row_ptr.begin());

        // Fill (unsorted but will be in increasing col order for each row because i increases globally).
        std::vector<int> next = Lt.row_ptr;
        for (int i = 0; i < n; ++i)
        {
            for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p)
            {
                const int j = L.col_ind[p];
                const int dst = next[j]++;
                Lt.col_ind[dst] = i; // transpose: (i,j) -> (j,i)
                Lt.values[dst] = L.values[p];
            }
        }

        // Enforce "diag is last" per row of Lt while keeping off-diagonals sorted.
        // After the fill above, each Lt row r has col_ind in strictly increasing order (by construction).
        // So we only need to move the diagonal (col==r) to the end by shifting.
        for (int r = 0; r < n; ++r)
        {
            const int s = Lt.row_ptr[r];
            const int e = Lt.row_ptr[r + 1];
            assert(e > s); // diagonal exists

            // Find diagonal position in this row.
            int d = -1;
            for (int k = s; k < e; ++k)
            {
                if (Lt.col_ind[k] == r)
                {
                    d = k;
                    break;
                }
            }
            assert(d != -1);

            if (d == e - 1)
                continue; // already last

            const int diag_col = Lt.col_ind[d];
            const T diag_val = Lt.values[d];

            // Shift entries left to fill the gap at d.
            const int move_count = (e - 1) - d;
            std::memmove(&Lt.col_ind[d], &Lt.col_ind[d + 1], sizeof(int) * move_count);
            std::memmove(&Lt.values[d], &Lt.values[d + 1], sizeof(T) * move_count);

            // Put diagonal at the end.
            Lt.col_ind[e - 1] = diag_col; // == r
            Lt.values[e - 1] = diag_val;

            assert(Lt.col_ind[e - 1] == r);
        }

        return Lt;
    }
} // namespace ichol::matrix

#endif // INCHOL_MATRIX_FORMATS_HPP