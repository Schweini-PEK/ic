#include <vector>

#include "symbolic.hpp"

namespace ichol::symbolic
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

    template <typename T>
    ichol::symbolic::ETree build_etree(const ichol::matrix::CscMatrix<T>& A) {
        // Assumes 0-based indices. A.col_ptr size == A.num_cols + 1, A.row_ind size == A.nnz.
        int n = A.num_cols; // number of columns / nodes
        ichol::symbolic::ETree etree;
        etree.parent.assign(n, -1);

        // ancestor array for path compression (initialized to -1)
        std::vector<int> ancestor(n, -1);

        // For each column j, walk its row indices i > j and update parent relationships.
        for (int j = 0; j < n; ++j) {
            for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
                int i = A.row_ind[p];
                if (i <= j) continue; // only consider strictly lower rows (i > j)
                int k = i;
                // climb ancestors of k until we find a node already visited in this column
                while (k != -1 && k != j) {
                    int next = ancestor[k];
                    ancestor[k] = j;            // point ancestor to current column j (mark)
                    if (next == -1) {
                        // if k had no previous ancestor, its parent becomes j
                        // But careful: parent should store parent of node k in etree (the first
                        // time it meets someone above it). For CSC algorithm, set parent[k] = j
                        // when next == -1 and k != j.
                        etree.parent[k] = j;
                    }
                    k = next;
                }
            }
        }
        return etree;
    }

    template ichol::symbolic::ETree build_etree<double>(const ichol::matrix::CsrMatrix<double> &A);
    template ichol::symbolic::ETree build_etree<double>(const ichol::matrix::CscMatrix<double> &A);
    template ichol::symbolic::ETree build_etree<float>(const ichol::matrix::CsrMatrix<float> &A);
    template ichol::symbolic::ETree build_etree<float>(const ichol::matrix::CscMatrix<float> &A);
    template ichol::symbolic::ETree build_etree<half_float::half>(const ichol::matrix::CsrMatrix<half_float::half> &A);
    template ichol::symbolic::ETree build_etree<half_float::half>(const ichol::matrix::CscMatrix<half_float::half> &A);
}