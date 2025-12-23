#include <algorithm>
#include <cassert>
#include <limits>

#include "ichol/symbolic.hpp"
#include "ichol/matrix_formats.hpp"

inline void update_etree(int i, int j,
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

inline std::vector<int> build_etree(const ichol::CsrMatrix<double> &A)
{
    const int n = A.num_rows;

    std::vector<int> Parent(n, -1);
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
                update_etree(i, j, Parent, Ancestor);
            }
        }
    }

    return Parent;
}

static inline std::vector<std::vector<int>>
build_col_neighbors_from_lower(const ichol::CsrMatrix<double> &A)
{
    const int n = A.num_rows;
    std::vector<std::vector<int>> col_neighbors(n);

    for (int i = 0; i < n; ++i)
    {
        const int rs = A.row_ptr[i];
        const int re = A.row_ptr[i + 1];

        for (int p = rs; p < re; ++p)
        {
            const int j = A.col_ind[p];
            if (j < i)                         // strict lower entry A(i,j)
                col_neighbors[j].push_back(i); // contributes to column j below diag
        }
    }
    return col_neighbors;
}

ichol::core::IC_Symbolic build_ic_symbolic_full(const ichol::CsrMatrix<double> &A,
                                                const std::vector<int> &parent)
{
    const int n = A.num_rows;
    assert(n == A.num_cols);

    ichol::core::IC_Symbolic sym;
    sym.n = n;
    sym.row_ptr_L.resize(n + 1);

    auto col_neighbors = build_col_neighbors_from_lower(A);

    std::vector<std::vector<int>> rows(n); // rows[i] stores cols j < i
    std::vector<int> mark(n, -1);
    std::vector<int> stack(n);

    for (int j = 0; j < n; ++j)
    {
        int top = 0;

        // iterate i > j from column j (below diagonal), derived from CSR(lower)
        for (int i : col_neighbors[j])
        {
            int v = i;
            while (v != -1 && v > j && mark[v] != j)
            {
                stack[top++] = v;
                mark[v] = j;
                v = (v < (int)parent.size()) ? parent[v] : -1;
            }
        }

        for (int s = 0; s < top; ++s)
        {
            const int i = stack[s];
            rows[i].push_back(j); // j < i
        }
    }

    std::vector<int> col_ind_L;
    col_ind_L.reserve(A.nnz + n); // +n for diagonal

    for (int i = 0; i < n; ++i)
    {
        auto &r = rows[i];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());

        sym.row_ptr_L[i] = (int)col_ind_L.size();

        // strict-lower part
        for (int j : r)
            if (j < i)
                col_ind_L.push_back(j);

        // include diagonal explicitly (lower+diag storage)
        col_ind_L.push_back(i);
    }

    sym.row_ptr_L[n] = (int)col_ind_L.size();
    sym.col_ind_L = std::move(col_ind_L);
    return sym;
}

ichol::core::IC_Symbolic build_ic_symbolic_levelk(const ichol::CsrMatrix<double> &A, int k)
{
    const int n = A.num_rows;
    assert(n == A.num_cols);
    assert(k >= 0);

    // row-wise pattern + level for L (includes diagonal internally)
    std::vector<std::vector<int>> L_row(n);
    std::vector<std::vector<int>> L_level(n);

    std::vector<int> marker(n, -1);
    std::vector<int> work_col(n);
    std::vector<int> work_level(n);

    for (int i = 0; i < n; ++i)
    {
        int used = 0;

        const int row_start = A.row_ptr[i];
        const int row_end = A.row_ptr[i + 1];

        // structural entries of A with j <= i
        for (int p = row_start; p < row_end; ++p)
        {
            int j = A.col_ind[p];
            if (j > i)
                continue;

            if (marker[j] != i)
            {
                marker[j] = i;
                work_col[used++] = j;
                work_level[j] = 0;
            }
            else if (0 < work_level[j])
            {
                work_level[j] = 0;
            }
        }

        // ensure diagonal exists internally
        if (marker[i] != i)
        {
            marker[i] = i;
            work_col[used++] = i;
            work_level[i] = 0;
        }

        // propagate fill from rows j < i
        for (int pos = 0; pos < used; ++pos)
        {
            int j = work_col[pos];
            if (j >= i)
                continue;

            int lev_ij = work_level[j];
            if (lev_ij > k)
                continue;

            const auto &Lj_row = L_row[j];
            const auto &Lj_level = L_level[j];

            const int len_j = (int)Lj_row.size();
            for (int idx = 0; idx < len_j; ++idx)
            {
                int m = Lj_row[idx]; // m <= j
                int lev_jm = Lj_level[idx];
                int lev_new = lev_ij + lev_jm + 1;

                if (lev_new > k)
                    continue;

                if (marker[m] != i)
                {
                    marker[m] = i;
                    work_col[used++] = m;
                    work_level[m] = lev_new;
                }
                else if (lev_new < work_level[m])
                {
                    work_level[m] = lev_new;
                }
            }
        }

        std::sort(work_col.begin(), work_col.begin() + used);

        auto &Li_row = L_row[i];
        auto &Li_level = L_level[i];

        Li_row.reserve(used);
        Li_level.reserve(used);

        for (int t = 0; t < used; ++t)
        {
            int col = work_col[t]; // col <= i
            Li_row.push_back(col);
            Li_level.push_back(work_level[col]);
        }
    }

    // compress into IC_Symbolic
    ichol::core::IC_Symbolic sym;
    sym.n = n;
    sym.row_ptr_L.resize(n + 1);

    std::vector<int> col_ind_L;
    col_ind_L.reserve(A.nnz * (k + 1));

    for (int i = 0; i < n; ++i)
    {
        sym.row_ptr_L[i] = static_cast<int>(col_ind_L.size());
        const auto &r = L_row[i];

        for (int col : r)
        {
            if (col > i)
                continue; // drop anything above diagonal
            col_ind_L.push_back(col);
        }
    }

    sym.row_ptr_L[n] = static_cast<int>(col_ind_L.size());
    sym.col_ind_L = std::move(col_ind_L);
    return sym;
}

namespace ichol
{
    namespace core
    {
        IC_Symbolic build_ic_symbolic(const ichol::CsrMatrix<double> &A,
                                      int k)
        {
            if (k < 0)
            {
                // full pattern: complete sparse Cholesky
                auto parent = build_etree(A);
                return build_ic_symbolic_full(A, parent);
            }
            else
            {
                // IC(k) pattern via level-of-fill
                return build_ic_symbolic_levelk(A, k);
            }
        }
    } // namespace core
} // namespace ichol
