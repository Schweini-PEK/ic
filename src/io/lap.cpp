#include "ichol/mtx_read.hpp"

namespace ichol::io
{
    template <typename T>
    ichol::matrix::CsrMatrix<T> gen_laplacian_csr(int n)
    {
        ichol::matrix::CsrMatrix<T> laplacian;
        laplacian.num_rows = n * n;
        laplacian.num_cols = n * n;
        // Store only lower triangle + diagonal to match solver expectations.
        laplacian.nnz = 3 * n * n - 2 * n;
        laplacian.row_ptr.resize(laplacian.num_rows + 1);
        laplacian.col_ind.resize(laplacian.nnz);
        laplacian.values.resize(laplacian.nnz);

        int nnz_index = 0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int row = i * n + j;
                laplacian.row_ptr[row] = nnz_index;

                // Left neighbor (lower triangle)
                if (j > 0)
                {
                    laplacian.col_ind[nnz_index] = row - 1;
                    laplacian.values[nnz_index] = -1;
                    nnz_index++;
                }

                // Top neighbor (lower triangle)
                if (i > 0)
                {
                    laplacian.col_ind[nnz_index] = row - n;
                    laplacian.values[nnz_index] = -1;
                    nnz_index++;
                }

                // Diagonal element (must be last in row)
                laplacian.col_ind[nnz_index] = row;
                laplacian.values[nnz_index] = 4;
                nnz_index++;
            }
        }
        laplacian.row_ptr[laplacian.num_rows] = nnz_index;

        return laplacian;
    }

    template ichol::matrix::CsrMatrix<double> gen_laplacian_csr(int n);
    template ichol::matrix::CsrMatrix<float> gen_laplacian_csr(int n);
    template ichol::matrix::CsrMatrix<half_float::half> gen_laplacian_csr(int n);
} // namespace ichol::io
