#include <cassert>
#include <deque>
#include <limits>

#include "symbolic.hpp"

namespace ichol::symbolic
{
    template <typename T>
    ichol::symbolic::FactorPattern compute_complete_cholesky_pattern(const ichol::matrix::CsrMatrix<T> &A,
                                                                     const ichol::symbolic::ETree &etree)
    {
        const int n = A.num_rows;
        ichol::symbolic::FactorPattern factor_pattern;

        return factor_pattern;
    }

    static inline void sort_unique(std::vector<int> &v)
    {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // NOTE: We used to merge child patterns into a parent column via repeated
    // sorted unions (allocate+merge per child). That is correct but very slow.
    // The implementation below uses a stamp-based set union (dedup on the fly)
    // and sorts only once per column.

    // compute_complete_cholesky_pattern - CSC implementation (fast ereach -> CSC)
//
// The previous implementation built L(:,k) by repeatedly merging child column patterns
// into the parent (correct but can be superlinear: elements get re-scanned many times up
// the etree). On modest matrices this already becomes 10s of milliseconds.
//
// Here we switch to a standard near-optimal symbolic algorithm:
//   1) Build the symmetric *upper* pattern U = triu(A) as CSC-like adjacency:
//        Ucol[k] = { i | i < k and A(i,k) != 0 }   (deduped, sorted)
//   2) For each k, compute the nonzero pattern of row k of L (left of diagonal) via etree reach:
//        pattern(L(k,1:k-1)) = ereach(U(:,k), parent).
//      (Classic cs_ereach algorithm from CSparse; uses only the upper part of A.)
//   3) Convert row-patterns to column-patterns:
//        if L(k,j) is nonzero (j<k) then row k belongs to column j of L.
//
// Result: exact simplicial CSC pattern of the *lower* Cholesky factor L (rows >= col),
// with columns already sorted (diagonal first, then increasing row index).
template <typename T>
FactorPattern compute_complete_cholesky_pattern(
    const ichol::matrix::CscMatrix<T> &A,
    const ichol::symbolic::ETree &etree)
{
    const int n = A.num_cols;
    assert((int)etree.parent.size() == n);

    // ------------------------------------------------------------
    // Build symmetric upper-triangular adjacency by column:
    // Ucol[k] contains i < k where A(i,k) is structurally nonzero.
    // We map any (i,j) to (min(i,j), max(i,j)) so pattern becomes symmetric.
    // ------------------------------------------------------------
    std::vector<std::vector<int>> Ucol(n);
    for (int j = 0; j < n; ++j)
    {
        for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p)
        {
            int i = A.row_ind[p];
            if (i == j) continue;
            int u = (i < j) ? i : j;
            int v = (i < j) ? j : i;
            if (u == v) continue;
            Ucol[v].push_back(u); // u < v, stored in column v
        }
    }
    for (int k = 0; k < n; ++k) sort_unique(Ucol[k]);

    // ------------------------------------------------------------
    // Workspace for ereach + store row-patterns (strictly lower part):
    // row_ptr[k]..row_ptr[k+1]-1 are the columns j<k where L(k,j) is nonzero.
    // ------------------------------------------------------------
    std::vector<int> w(n, -1);    // mark array: w[i] == k means visited in current k
    std::vector<int> s(n, -1);    // temporary stack
    std::vector<int> row_ptr(n + 1, 0);
    std::vector<int> row_ind;

    // reserve using colcount sum if available (robust even if mismatch)
    if ((int)etree.colcount.size() == n)
    {
        long long nnzL_est = 0;
        for (int k = 0; k < n; ++k) nnzL_est += std::max(1, etree.colcount[k]);
        long long strict_lower = std::max(0LL, nnzL_est - (long long)n);
        if (strict_lower > 0) row_ind.reserve((size_t)strict_lower);
    }

    for (int k = 0; k < n; ++k)
    {
        const int stamp = k;
        int top = n;

        // mark k as visited (so paths stop at k)
        w[k] = stamp;

        // traverse each i in triu(A(:,k)) (i <= k) => here Ucol[k] has i < k
        for (int idx = 0; idx < (int)Ucol[k].size(); ++idx)
        {
            int i = Ucol[k][idx];

            int len = 0;
            while (i >= 0 && w[i] != stamp)
            {
                s[len++] = i;
                w[i] = stamp;
                i = etree.parent[i];
            }
            while (len > 0) s[--top] = s[--len];
        }

        row_ptr[k] = (int)row_ind.size();
        for (int p = top; p < n; ++p)
        {
            const int j = s[p];
            if (j >= 0 && j < k) row_ind.push_back(j);
        }
        row_ptr[k + 1] = (int)row_ind.size();
    }

    // ------------------------------------------------------------
    // Column counts from row-patterns (diag included).
    // ------------------------------------------------------------
    std::vector<int> col_count(n, 1); // diagonal
    for (int k = 0; k < n; ++k)
    {
        for (int p = row_ptr[k]; p < row_ptr[k + 1]; ++p)
        {
            const int j = row_ind[p];
            if (j >= 0 && j < n) col_count[j]++;
        }
    }

    // ------------------------------------------------------------
    // Pack into FactorPattern (CSC of L pattern; row_ptr_L is actually col_ptr_L here).
    // ------------------------------------------------------------
    FactorPattern pattern;
    pattern.row_ptr_L.resize(n + 1);
    pattern.row_ptr_L[0] = 0;
    for (int j = 0; j < n; ++j)
    {
        pattern.row_ptr_L[j + 1] = pattern.row_ptr_L[j] + col_count[j];
    }
    pattern.col_ind_L.resize(pattern.row_ptr_L[n]);

    // next write position per column
    std::vector<int> next = pattern.row_ptr_L;

    // diagonal first
    for (int j = 0; j < n; ++j)
    {
        pattern.col_ind_L[next[j]++] = j;
    }

    // scatter strictly-lower part: increasing k => columns become sorted by row
    for (int k = 0; k < n; ++k)
    {
        for (int p = row_ptr[k]; p < row_ptr[k + 1]; ++p)
        {
            const int j = row_ind[p];
            pattern.col_ind_L[next[j]++] = k;
        }
    }

#ifndef NDEBUG
    for (int j = 0; j < n; ++j)
    {
        assert(next[j] == pattern.row_ptr_L[j + 1] && "col_count mismatch in CSC pattern build");
    }
#endif

    return pattern;
}
template <typename T>
    ichol::symbolic::FactorPattern
    compute_ic_factor_pattern(const ichol::matrix::CsrMatrix<T> &A, int k)
    {
        const int n = A.num_rows;
        const int INF = std::numeric_limits<int>::max() / 8;

        std::vector<int> adj_head(n, -1); // Points to start of list for column 'j'

        std::vector<int> adj_next;
        std::vector<int> adj_row;
        std::vector<int> adj_lvl;

        // Reserve memory to prevent frequent reallocs. Estimate fill-in factor ~3x-5x
        size_t est_nnz = A.nnz * 3;
        adj_next.reserve(est_nnz);
        adj_row.reserve(est_nnz);
        adj_lvl.reserve(est_nnz);

        auto add_dependency = [&](int col, int row, int level)
        {
            int idx = static_cast<int>(adj_next.size());
            adj_next.push_back(adj_head[col]);
            adj_row.push_back(row);
            adj_lvl.push_back(level);
            adj_head[col] = idx;
        };

        std::vector<int> mark(n, -1);
        std::vector<int> lvl(n, INF);
        std::vector<int> touched;
        touched.reserve(256); // Thread-local scratch if parallelized

        std::vector<int> inQ(n, -1);
        std::vector<int> Q;
        Q.reserve(64);

        auto enqueue = [&](int i, int p)
        {
            if (p < i && inQ[p] != i)
            {
                inQ[p] = i;
                Q.push_back(p);
            }
        };

        auto activate_or_decrease = [&](int i, int j, int newlvl)
        {
            if (j > i)
                return;
            if (j != i && newlvl > k)
                return;

            if (mark[j] != i)
            {
                mark[j] = i;
                lvl[j] = newlvl;
                touched.push_back(j);
                enqueue(i, j);
            }
            else if (newlvl < lvl[j])
            {
                lvl[j] = newlvl;
                enqueue(i, j);
            }
        };

        ichol::symbolic::FactorPattern fp;
        fp.row_ptr_L.assign(n + 1, 0);
        fp.col_ind_L.reserve(est_nnz);

        // Main Loop: Serial (Inherently sequential for IC(k) wavefront)
        for (int i = 0; i < n; ++i)
        {
            touched.clear();
            Q.clear();

            // Seed from A
            for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
            {
                int j = A.col_ind[p];
                if (j <= i)
                    activate_or_decrease(i, j, 0);
            }

            // Ensure diagonal
            if (mark[i] != i)
            {
                mark[i] = i;
                lvl[i] = 0;
                touched.push_back(i);
            }
            else
            {
                lvl[i] = 0;
            }

            // Propagate
            while (!Q.empty())
            {
                int p = Q.back();
                Q.pop_back();
                inQ[p] = -1;

                if (mark[p] != i)
                    continue;
                int lvl_ip = lvl[p];
                if (lvl_ip >= k)
                    continue;

                int curr = adj_head[p];
                while (curr != -1)
                {
                    int r = adj_row[curr];
                    int newlvl = lvl_ip + adj_lvl[curr] + 1;
                    activate_or_decrease(i, r, newlvl);
                    curr = adj_next[curr];
                }
            }

            std::sort(touched.begin(), touched.end());

            fp.row_ptr_L[i] = (int)fp.col_ind_L.size();
            fp.col_ind_L.insert(fp.col_ind_L.end(), touched.begin(), touched.end());
            fp.row_ptr_L[i + 1] = (int)fp.col_ind_L.size();

            // Update column adjacency
            for (int j : touched)
            {
                if (j < i)
                {
                    // Optimized push
                    add_dependency(j, i, lvl[j]);
                }
            }
        }
        return fp;
    }

    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<double>(const ichol::matrix::CsrMatrix<double> &A,
                                                                                      const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<double>(const ichol::matrix::CsrMatrix<double> &A,
                                                                              int level_k);
    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<float>(const ichol::matrix::CsrMatrix<float> &A,
                                                                                     const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<float>(const ichol::matrix::CsrMatrix<float> &A,
                                                                             int level_k);
    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<half_float::half>(const ichol::matrix::CsrMatrix<half_float::half> &A,
                                                                                                const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<half_float::half>(const ichol::matrix::CsrMatrix<half_float::half> &A,
                                                                                        int level_k);

    template FactorPattern compute_complete_cholesky_pattern<double>(
        const ichol::matrix::CscMatrix<double> &A,
        const ichol::symbolic::ETree &etree);

    template FactorPattern compute_complete_cholesky_pattern<float>(
        const ichol::matrix::CscMatrix<float> &A,
        const ichol::symbolic::ETree &etree);

    template FactorPattern compute_complete_cholesky_pattern<half_float::half>(
        const ichol::matrix::CscMatrix<half_float::half> &A,
        const ichol::symbolic::ETree &etree);
} // namespace ichol::symbolic