#include <vector>

#include "factor/symbolic/symbolic.hpp"

namespace
{
    void update_etree(int i, int j,
                      std::vector<int> &Parent,
                      std::vector<int> &Ancestor)
    {
        // Loop while the child i is valid and strictly below j (i < j).
        while (i != -1)
        {
            int a = Ancestor[i];

            // Early exit when i is already compressed up to j.
            if (a == j)
                break;

            Ancestor[i] = j;

            if (a == -1)
            {
                Parent[i] = j;
                break; // parent found.
            }

            // Otherwise, keep climbing with path compression.
            i = a;
        }
    }
} // namespace

namespace ichol::symbolic
{
    template <typename T>
    ichol::symbolic::ETree build_etree(const ichol::matrix::CsrMatrix<T> &A)
    {
        const int n = A.num_rows;

        ichol::symbolic::ETree tree = ichol::symbolic::ETree();
        tree.parent.resize(n, -1);
        std::vector<int> Ancestor(n, -1);

        for (int j = 0; j < n; ++j)
        {
            const int row_start = A.row_ptr[j];
            const int row_end = A.row_ptr[j + 1];

            for (int t = row_start; t < row_end; ++t)
            {
                const int i = A.col_ind[t];
                if (i < j)
                {
                    update_etree(i, j, tree.parent, Ancestor);
                }
            }
        }

        return tree;
    }

    template ichol::symbolic::ETree build_etree<double>(const ichol::matrix::CsrMatrix<double> &A);
    template ichol::symbolic::ETree build_etree<float>(const ichol::matrix::CsrMatrix<float> &A);
    template ichol::symbolic::ETree build_etree<half_float::half>(const ichol::matrix::CsrMatrix<half_float::half> &A);
} // namespace ichol::symbolic