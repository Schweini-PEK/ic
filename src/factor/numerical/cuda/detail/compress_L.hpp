#pragma once

#include "ichol/matrix_formats.hpp"
#include "backends/CUDA/util/host_cast.hpp"

namespace ichol::numeric::util
{
    template <class T, class G>
    static ichol::matrix::CsrMatrix<T> compress_fixed_pattern_L(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<G> &val_fixed)
    {
        ichol::matrix::CsrMatrix<T> L;
        L.num_rows = n;
        L.num_cols = n;
        L.row_ptr.assign(n + 1, 0);

        for (int i = 0; i < n; ++i)
        {
            int r0 = row_ptr[i], r1 = row_ptr[i + 1];
            int diagp = r1 - 1;

            int cnt = 0;
            for (int p = r0; p < diagp; ++p)
            {
                if (ichol::cuda::util::host_to_double(val_fixed[p]) != 0.0)
                    cnt++;
            }
            cnt += 1;
            L.row_ptr[i + 1] = L.row_ptr[i] + cnt;
        }

        L.nnz = L.row_ptr[n];
        L.col_ind.resize(L.nnz);
        L.values.resize(L.nnz);

        for (int i = 0; i < n; ++i)
        {
            int r0 = row_ptr[i], r1 = row_ptr[i + 1];
            int diagp = r1 - 1;
            int outp = L.row_ptr[i];

            for (int p = r0; p < diagp; ++p)
            {
                G v = val_fixed[p];
                if (ichol::cuda::util::host_to_double(v) != 0.0)
                {
                    L.col_ind[outp] = col_ind[p];
                    L.values[outp] = (T)ichol::cuda::util::host_to_double(v);
                    outp++;
                }
            }
            L.col_ind[outp] = i;
            L.values[outp] = (T)ichol::cuda::util::host_to_double(val_fixed[diagp]);
        }

        return L;
    }
} // namespace ichol::numeric::util