#include "symbolic.hpp"

namespace ichol::symbolic
{
    ichol::symbolic::Permutation identity_permutation(int n)
    {
        ichol::symbolic::Permutation P;
        P.perm.resize(n);
        P.inv_perm.resize(n);
        for (int i = 0; i < n; ++i)
        {
            P.perm[i] = i;
            P.inv_perm[i] = i;
        }
        return P;
    }

    template <typename T>
    void apply_permutation_csr(ichol::matrix::CsrMatrix<T> &A,
                               const ichol::symbolic::Permutation &P)
    {
        const int n = A.num_rows;
        const auto &inv = P.inv_perm;

        std::vector<int> row_counts(n, 0);

        for (int i = 0; i < n; ++i)
        {
            const int ni = inv[i];
            const int p0 = A.row_ptr[i];
            const int p1 = A.row_ptr[i + 1];
            for (int p = p0; p < p1; ++p)
            {
                const int nj = inv[A.col_ind[p]];
                const int r = (ni >= nj) ? ni : nj;
                ++row_counts[r];
            }
        }

        ichol::matrix::CsrMatrix<T> B;
        B.num_rows = n;
        B.num_cols = n;
        B.row_ptr.resize(n + 1);
        B.row_ptr[0] = 0;
        for (int r = 0; r < n; ++r)
            B.row_ptr[r + 1] = B.row_ptr[r] + row_counts[r];

        const int nnz = B.row_ptr[n];
        B.nnz = nnz;
        B.col_ind.resize(nnz);
        B.values.resize(nnz);

        // Pass 2: fill.
        std::vector<int> cursor = B.row_ptr;
        for (int i = 0; i < n; ++i)
        {
            const int ni = inv[i];
            const int p0 = A.row_ptr[i];
            const int p1 = A.row_ptr[i + 1];
            for (int p = p0; p < p1; ++p)
            {
                const int nj = inv[A.col_ind[p]];
                const int r = (ni >= nj) ? ni : nj;
                const int c = (ni >= nj) ? nj : ni;
                const int dst = cursor[r]++;
                B.col_ind[dst] = c;
                B.values[dst] = A.values[p];
            }
        }

        // Sort columns within each row.
        struct Entry
        {
            int c;
            T v;
        };
        int max_row = 0;
        for (int r = 0; r < n; ++r)
            if (row_counts[r] > max_row)
                max_row = row_counts[r];
        std::vector<Entry> tmp;
        tmp.resize(max_row);

        for (int r = 0; r < n; ++r)
        {
            const int begin = B.row_ptr[r];
            const int end = B.row_ptr[r + 1];
            const int len = end - begin;
            if (len <= 1)
                continue;

            for (int k = 0; k < len; ++k)
            {
                tmp[k].c = B.col_ind[begin + k];
                tmp[k].v = B.values[begin + k];
            }

            std::sort(tmp.begin(), tmp.begin() + len,
                      [](const Entry &a, const Entry &b)
                      { return a.c < b.c; });

            for (int k = 0; k < len; ++k)
            {
                B.col_ind[begin + k] = tmp[k].c;
                B.values[begin + k] = tmp[k].v;
            }
        }

        return B;
    }

    template void apply_permutation_csr(ichol::matrix::CsrMatrix<double> &A,
                                        const ichol::symbolic::Permutation &P);
    template void apply_permutation_csr(ichol::matrix::CsrMatrix<float> &A,
                                        const ichol::symbolic::Permutation &P);
    template void apply_permutation_csr(ichol::matrix::CsrMatrix<half_float::half> &A,
                                        const ichol::symbolic::Permutation &P);
} // namespace ichol::symbolic