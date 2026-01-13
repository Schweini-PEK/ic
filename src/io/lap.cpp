#include "ichol/mtx_read.hpp"

template <typename T>
ichol::matrix::CsrMatrix<T> gen_laplacian_csr(int n)
{
    ichol::matrix::CsrMatrix<T> laplacian;
    laplacian.num_rows = n * n;
    laplacian.num_cols = n * n;
    laplacian.nnz = 5 * n * n - 4 * n;
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

            // Diagonal element
            laplacian.col_ind[nnz_index] = row;
            laplacian.values[nnz_index] = 4;
            nnz_index++;

            // Left neighbor
            if (j > 0)
            {
                laplacian.col_ind[nnz_index] = row - 1;
                laplacian.values[nnz_index] = -1;
                nnz_index++;
            }

            // Right neighbor
            if (j < n - 1)
            {
                laplacian.col_ind[nnz_index] = row + 1;
                laplacian.values[nnz_index] = -1;
                nnz_index++;
            }

            // Top neighbor
            if (i > 0)
            {
                laplacian.col_ind[nnz_index] = row - n;
                laplacian.values[nnz_index] = -1;
                nnz_index++;
            }

            // Bottom neighbor
            if (i < n - 1)
            {
                laplacian.col_ind[nnz_index] = row + n;
                laplacian.values[nnz_index] = -1;
                nnz_index++;
            }
        }
    }
    laplacian.row_ptr[laplacian.num_rows] = nnz_index;

    return laplacian;
}