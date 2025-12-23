#ifndef INCHOL_MATRIX_NORM_HPP
#define INCHOL_MATRIX_NORM_HPP

#include <ichol/matrix_formats.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include <limits> // added

namespace ichol
{

    using DenseMatrix = std::vector<std::vector<double>>;

    DenseMatrix csr_to_dense(const CsrMatrix<double> &csr)
    {
        for (int i = 0; i < csr.num_rows; i++)
        {
            for (int p = csr.row_ptr[i]; p < csr.row_ptr[i + 1]; ++p)
            {
                int c = csr.col_ind[p];
                if (c < 0 || c >= csr.num_cols)
                {
                    std::cout << "bad col " << c << " at row " << i << "\n";
                }
            }
        }

        int r = csr.num_rows, c = csr.num_cols;
        DenseMatrix dense(r, std::vector<double>(c, 0.0));
        for (int i = 0; i < r; i++)
        {
            for (int p = csr.row_ptr[i]; p < csr.row_ptr[i + 1]; ++p)
            {
                int col = csr.col_ind[p];
                dense[i][col] = csr.values[p];
            }
        }
        return dense;
    }

    DenseMatrix transpose(const DenseMatrix &mat)
    {
        int n = mat.size();
        DenseMatrix t(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                t[j][i] = mat[i][j];
            }
        }
        return t;
    }

    DenseMatrix multiply(const DenseMatrix &A, const DenseMatrix &B)
    {
        int n = A.size();
        DenseMatrix C(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                for (int k = 0; k < n; ++k)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    DenseMatrix subtract(const DenseMatrix &A, const DenseMatrix &B)
    {
        int n = A.size();
        DenseMatrix C(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    double l2_norm_dense(const DenseMatrix &mat)
    {
        double sum_sq = 0.0;
        for (const auto &row : mat)
        {
            for (double val : row)
            {
                sum_sq += val * val;
            }
        }
        return std::sqrt(sum_sq);
    }

    double residual_l2_norm(const CsrMatrix<double> &A, const CsrMatrix<double> &L)
    {
        auto has_bad = [](const ichol::CsrMatrix<double> &M)
        {
            for (double v : M.values)
                if (!std::isfinite(v))
                    return true;
            return false;
        };
        std::cout << "A finite? " << !has_bad(A) << "\n";
        std::cout << "L finite? " << !has_bad(L) << "\n";

        auto A_dense = csr_to_dense(A);
        auto L_dense = csr_to_dense(L);
        auto LT_dense = transpose(L_dense);
        auto LLt = multiply(L_dense, LT_dense);
        auto residual = subtract(A_dense, LLt);

        double num = l2_norm_dense(residual);
        double denom = l2_norm_dense(A_dense);
        if (denom == 0.0)
        {
            return std::numeric_limits<double>::infinity();
        }
        return num / denom;
    }

    double l2_norm(const CsrMatrix<double> &mat)
    {
        double sum_sq = 0.0;
        for (const auto &val : mat.values)
        {
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq);
    }

} // namespace ichol

#endif // INCHOL_MATRIX_NORM_HPP