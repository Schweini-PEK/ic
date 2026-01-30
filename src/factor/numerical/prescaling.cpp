//src/factor/numerical/prescaling.cpp
#include "factorize.hpp"

namespace ichol::numeric
{
    std::vector<double> scale_diag_sqrt(const ichol::matrix::CsrMatrix<double> &A)
    {
        const int n = A.num_rows;
        std::vector<double> D(n, 1.0);
        double temp = 0.0;
        for (int i = 0; i < n; ++i)
        {
            int last = A.row_ptr[i + 1] - 1;
            temp = double(std::sqrt(A.values[last]));
            if (temp > 0.0 && std::isfinite(temp))
                D[i] = temp;
            else
                D[i] = 1e-12;
        }
        return D;
    }

    std::vector<double> scale_col_norm(const ichol::matrix::CsrMatrix<double> &A)
    {
        const int n = A.num_rows;
        std::vector<double> col_sq(n, 0.0);

        for (int i = 0; i < n; ++i)
        {
            for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
            {
                const int j = A.col_ind[p];
                const double v = static_cast<double>(A.values[p]);
                const double vv = v * v;

                col_sq[j] += vv;
                if (j != i)
                {
                    col_sq[i] += vv;
                }
            }
        }

        std::vector<double> D(n, 1.0);
        for (int j = 0; j < n; ++j)
        {
            double nrm = std::sqrt(col_sq[j]);
            if (nrm > 0.0 && std::isfinite(nrm))
                D[j] = nrm;
        }
        return D;
    }

    void apply_prescaling(ichol::matrix::CsrMatrix<double> &A,
                          const std::vector<double> &D)
    {
        const int n = A.num_rows;
        for (int i = 0; i < n; ++i)
        {
            const double invDi = 1.0 / D[i];
            for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
            {
                const int j = A.col_ind[p];
                const double s = invDi / D[j];
                A.values[p] = A.values[p] * s;
            }
        }
    }
} // namespace ichol::numeric