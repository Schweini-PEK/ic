#include "factorize.hpp"

namespace ichol::numeric
{
    template <class T>
    void add_diagonal_shift(ichol::matrix::CsrMatrix<T> &A, T alpha)
    {
        if (alpha == T(0))
            return;

        const int n = A.num_rows;

        for (int i = 0; i < n; ++i)
        {
            int row_end = A.row_ptr[i + 1];
            if (A.col_ind[row_end - 1] != i)
                throw std::runtime_error("add_diagonal_shift: Matrix missing diagonal entry.");
            A.values[A.row_ptr[i + 1] - 1] += alpha;
        }
    }

    template void add_diagonal_shift<double>(ichol::matrix::CsrMatrix<double> &A, double alpha);
    template void add_diagonal_shift<float>(ichol::matrix::CsrMatrix<float> &A, float alpha);
    template void add_diagonal_shift<half_float::half>(ichol::matrix::CsrMatrix<half_float::half> &A, half_float::half alpha);
} // namespace ichol::numeric