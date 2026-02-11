#include "ichol/mtx_read.hpp"
#include <limits>
#include <stdexcept>
#include <vector>

namespace ichol::io
{
    template <typename T>
    ichol::matrix::CsrMatrix<T> gen_3dpoi(int n)
    {
        if (n <= 0)
            throw std::runtime_error("gen_3dpoi: n must be positive");

        const int64_t N64 = 1LL * n * n * n;
        if (N64 > std::numeric_limits<int>::max())
            throw std::runtime_error("gen_3dpoi: n^3 exceeds int index range");

        const int N = static_cast<int>(N64);

        // For a full 7-point stencil, per row nnz is:
        // 1 (diag) + neighbors in +/- x, +/- y, +/- z directions.
        // Interior points have 7 nnz; boundary points have fewer.
        // Total nnz = N + 2 * (count of forward edges in 3 directions)
        const int64_t edges_per_dir = 1LL * (n - 1) * n * n;
        const int64_t nnz64 = 1LL * N + 6LL * edges_per_dir;

        ichol::matrix::CsrMatrix<T> A;
        A.num_rows = N;
        A.num_cols = N;
        A.nnz = static_cast<int>(nnz64);

        A.row_ptr.resize(static_cast<size_t>(N) + 1);
        A.col_ind.reserve(static_cast<size_t>(A.nnz));
        A.values.reserve(static_cast<size_t>(A.nnz));

        auto id = [n](int x, int y, int z) -> int
        {
            return x + y * n + z * n * n;
        };

        int nnz_so_far = 0;
        A.row_ptr[0] = 0;

        for (int z = 0; z < n; ++z)
        {
            for (int y = 0; y < n; ++y)
            {
                for (int x = 0; x < n; ++x)
                {
                    const int i = id(x, y, z);
                    A.row_ptr[i] = nnz_so_far;

                    // Neighbors emitted in strictly increasing column order for CSR compliance:
                    // 1. Z-minus (i - n^2)
                    if (z > 0)
                    {
                        A.col_ind.push_back(i - n * n);
                        A.values.push_back(static_cast<T>(-1));
                        ++nnz_so_far;
                    }
                    // 2. Y-minus (i - n)
                    if (y > 0)
                    {
                        A.col_ind.push_back(i - n);
                        A.values.push_back(static_cast<T>(-1));
                        ++nnz_so_far;
                    }
                    // 3. X-minus (i - 1)
                    if (x > 0)
                    {
                        A.col_ind.push_back(i - 1);
                        A.values.push_back(static_cast<T>(-1));
                        ++nnz_so_far;
                    }
                    // 4. Diagonal (i)
                    A.col_ind.push_back(i);
                    A.values.push_back(static_cast<T>(6));
                    ++nnz_so_far;

                    // 5. X-plus (i + 1)
                    if (x < n - 1)
                    {
                        A.col_ind.push_back(i + 1);
                        A.values.push_back(static_cast<T>(-1));
                        ++nnz_so_far;
                    }
                    // 6. Y-plus (i + n)
                    if (y < n - 1)
                    {
                        A.col_ind.push_back(i + n);
                        A.values.push_back(static_cast<T>(-1));
                        ++nnz_so_far;
                    }
                    // 7. Z-plus (i + n^2)
                    if (z < n - 1)
                    {
                        A.col_ind.push_back(i + n * n);
                        A.values.push_back(static_cast<T>(-1));
                        ++nnz_so_far;
                    }
                }
            }
        }
        A.row_ptr[N] = nnz_so_far;
        A.nnz = nnz_so_far;

        return A;
    }

    template ichol::matrix::CsrMatrix<double> gen_3dpoi(int n);
}