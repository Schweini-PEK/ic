// include/ichol/matrix_norm.hpp
#pragma once
#ifndef INCHOL_MATRIX_NORM_HPP
#define INCHOL_MATRIX_NORM_HPP

#include <ichol/matrix_formats.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace ichol
{
    using DenseMatrix = std::vector<std::vector<double>>;

    inline double l2_norm(const matrix::CsrMatrix<double> &mat)
    {
        long double sum_sq = 0.0L;
        for (double v : mat.values)
        {
            const long double x = static_cast<long double>(v);
            sum_sq += x * x;
        }
        return std::sqrt(static_cast<double>(sum_sq));
    }

    namespace detail
    {
        // CSR transpose: returns a CSR matrix that represents A^T.
        inline matrix::CsrMatrix<double> transpose_csr(const matrix::CsrMatrix<double> &A)
        {
            matrix::CsrMatrix<double> AT;
            AT.num_rows = A.num_cols;
            AT.num_cols = A.num_rows;

            const int nnz = static_cast<int>(A.values.size());
            AT.row_ptr.assign(AT.num_rows + 1, 0);
            AT.col_ind.assign(nnz, 0);
            AT.values.assign(nnz, 0.0);

            // Count nnz per column (row in AT).
            for (int p = 0; p < nnz; ++p)
            {
                const int c = A.col_ind[p];
                if (0 <= c && c < AT.num_rows)
                    ++AT.row_ptr[c + 1];
            }

            // Prefix sum to row_ptr.
            for (int i = 0; i < AT.num_rows; ++i)
                AT.row_ptr[i + 1] += AT.row_ptr[i];

            std::vector<int> next = AT.row_ptr;

            // Fill.
            for (int i = 0; i < A.num_rows; ++i)
            {
                for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
                {
                    const int c = A.col_ind[p];
                    if (!(0 <= c && c < AT.num_rows))
                        continue;

                    const int dst = next[c]++;
                    AT.col_ind[dst] = i;
                    AT.values[dst] = A.values[p];
                }
            }
            return AT;
        }

        // Compute ||A - L*L^T||_F^2 without forming dense matrices or the full product.
        inline long double residual_frob_norm_sq(
            const matrix::CsrMatrix<double> &A,
            const matrix::CsrMatrix<double> &L,
            double drop_tol = 0.0)
        {
            const int n = A.num_rows;

            const auto LT = transpose_csr(L); // LT is CSR of L^T.

            // Accumulator for one row of L*L^T.
            std::vector<int> marker(static_cast<std::size_t>(LT.num_cols), -1);
            std::vector<int> acc_cols;
            std::vector<double> acc_vals;
            std::vector<int> order;
            std::vector<std::pair<int, double>> brow;

            long double sum_sq = 0.0L;

            for (int i = 0; i < n; ++i)
            {
                acc_cols.clear();
                acc_vals.clear();

                // Row i of product: (L * LT)[i, j] = sum_k L[i,k] * LT[k,j].
                for (int p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p)
                {
                    const int k = L.col_ind[p];
                    const double lik = L.values[p];

                    if (!(0 <= k && k < LT.num_rows))
                        continue;
                    if (lik == 0.0)
                        continue;

                    for (int q = LT.row_ptr[k]; q < LT.row_ptr[k + 1]; ++q)
                    {
                        const int j = LT.col_ind[q];
                        const double lkj = LT.values[q];
                        const double prod = lik * lkj;

                        if (!(0 <= j && j < LT.num_cols))
                            continue;

                        const int pos = marker[static_cast<std::size_t>(j)];
                        if (pos < 0)
                        {
                            marker[static_cast<std::size_t>(j)] = static_cast<int>(acc_cols.size());
                            acc_cols.push_back(j);
                            acc_vals.push_back(prod);
                        }
                        else
                        {
                            acc_vals[static_cast<std::size_t>(pos)] += prod;
                        }
                    }
                }

                // Sort accumulated columns and build a compact row representation.
                order.resize(acc_cols.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                          [&](int a, int b)
                          { return acc_cols[static_cast<std::size_t>(a)] < acc_cols[static_cast<std::size_t>(b)]; });

                brow.clear();
                brow.reserve(order.size());
                for (int idx : order)
                {
                    const int col = acc_cols[static_cast<std::size_t>(idx)];
                    const double val = acc_vals[static_cast<std::size_t>(idx)];
                    if (drop_tol == 0.0 || std::fabs(val) > drop_tol)
                        brow.emplace_back(col, val);
                }

                // Merge row i of A with computed row i of L*L^T to accumulate ||A - B||_F^2.
                int ap = A.row_ptr[i];
                const int aend = A.row_ptr[i + 1];
                std::size_t bp = 0;
                const std::size_t bend = brow.size();

                while (ap < aend || bp < bend)
                {
                    if (bp >= bend)
                    {
                        const long double d = static_cast<long double>(A.values[ap]);
                        sum_sq += d * d;
                        ++ap;
                        continue;
                    }
                    if (ap >= aend)
                    {
                        const long double d = -static_cast<long double>(brow[bp].second);
                        sum_sq += d * d;
                        ++bp;
                        continue;
                    }

                    const int ac = A.col_ind[ap];
                    const int bc = brow[bp].first;

                    if (ac < bc)
                    {
                        const long double d = static_cast<long double>(A.values[ap]);
                        sum_sq += d * d;
                        ++ap;
                    }
                    else if (bc < ac)
                    {
                        const long double d = -static_cast<long double>(brow[bp].second);
                        sum_sq += d * d;
                        ++bp;
                    }
                    else
                    {
                        const long double d = static_cast<long double>(A.values[ap]) - static_cast<long double>(brow[bp].second);
                        sum_sq += d * d;
                        ++ap;
                        ++bp;
                    }
                }

                // Reset markers for the next row.
                for (int col : acc_cols)
                    marker[static_cast<std::size_t>(col)] = -1;
            }

            return sum_sq;
        }
    } // namespace detail

    inline double residual_l2_norm(const matrix::CsrMatrix<double> &A, const matrix::CsrMatrix<double> &L)
    {
        const double denom = l2_norm(A);
        if (denom == 0.0)
            return std::numeric_limits<double>::infinity();

        const long double num_sq = detail::residual_frob_norm_sq(A, L, /*drop_tol=*/0.0);
        const double num = std::sqrt(static_cast<double>(num_sq));
        return num / denom;
    }

} // namespace ichol

#endif // INCHOL_MATRIX_NORM_HPP
