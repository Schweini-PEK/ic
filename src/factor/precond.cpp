#include "ichol/matrix_formats.hpp"
#include "ichol/preconditioner.hpp"

namespace ichol::precond
{
    /**
     * Generates Mx and My as defined in Section 4.1.1 of the paper.
     * A = Mx + My, where Mx is the x-derivatives and My is the y-derivatives.
     */
    template <typename T>
    void generateADIPreconditioners(int n, T epsilon, ichol::matrix::CsrMatrix<T> &Mx, ichol::matrix::CsrMatrix<T> &My)
    {
        int N = n * n;
        T h2 = (T)1.0 / ((n + 1) * (n + 1));

        auto initMat = [&](ichol::matrix::CsrMatrix<T> &M)
        {
            M.num_rows = N;
            M.num_cols = N;
            M.row_ptr.assign(N + 1, 0);
            M.col_ind.clear();
            M.values.clear();
        };

        initMat(Mx);
        initMat(My);

        for (int i = 0; i < n; ++i)
        { // y-index
            for (int j = 0; j < n; ++j)
            { // x-index
                int row = i * n + j;

                // --- Construct Mx (x-direction tridiagonal blocks) ---
                if (j > 0)
                {
                    Mx.col_ind.push_back(row - 1);
                    Mx.values.push_back(-1.0 / h2);
                }
                Mx.col_ind.push_back(row);
                Mx.values.push_back(2.0 / h2);
                if (j < n - 1)
                {
                    Mx.col_ind.push_back(row + 1);
                    Mx.values.push_back(-1.0 / h2);
                }
                Mx.row_ptr[row + 1] = Mx.col_ind.size();

                // --- Construct My (y-direction tridiagonal blocks) ---
                if (i > 0)
                {
                    My.col_ind.push_back(row - n);
                    My.values.push_back(-epsilon / h2);
                }
                My.col_ind.push_back(row);
                My.values.push_back(2.0 * epsilon / h2);
                if (i < n - 1)
                {
                    My.col_ind.push_back(row + n);
                    My.values.push_back(-epsilon / h2);
                }
                My.row_ptr[row + 1] = My.col_ind.size();
            }
        }
        Mx.nnz = Mx.values.size();
        My.nnz = My.values.size();
    }

    template void generateADIPreconditioners(int n, double epsilon, ichol::matrix::CsrMatrix<double> &Mx, ichol::matrix::CsrMatrix<double> &My);
} // namespace ichol::precond