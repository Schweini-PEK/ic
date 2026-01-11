#include <cassert>
#include <deque>

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

    // merge: a := union(a, b\{skip}), both a and b sorted unique, result sorted unique
    static inline void union_sorted_skip(std::vector<int> &a,
                                         const std::vector<int> &b,
                                         int skip)
    {
        std::vector<int> out;
        out.reserve(a.size() + b.size());

        size_t i = 0, j = 0;
        while (i < a.size() || j < b.size())
        {
            int va;
            if (j >= b.size() || (i < a.size() && a[i] < b[j]))
            {
                va = a[i++];
            }
            else if (i >= a.size() || (j < b.size() && b[j] < a[i]))
            {
                va = b[j++];
            }
            else
            {
                va = a[i];
                ++i;
                ++j;
            }

            if (va == skip) continue;
            if (out.empty() || out.back() != va) out.push_back(va);
        }

        a.swap(out);
    }

    // compute_complete_cholesky_pattern - CSC implementation (CHOLMOD-style symbolic)
    // Build exact simplicial column pattern of L for A (symmetric pattern),
    // using elimination tree and child-to-parent merge:
    //   L_k := A_k ∪ (⋃_{j: parent[j]=k} (L_j \ {j})) , then keep only rows >= k.
    template <typename T>
    FactorPattern compute_complete_cholesky_pattern(
        const ichol::matrix::CscMatrix<T> &A,
        const ichol::symbolic::ETree &etree)
    {
        const int n = A.num_cols;
        assert((int)etree.parent.size() == n);

        // ------------------------------------------------------------
        // Build lower-triangular pattern of A as adjacency-by-column:
        // Acol[c] contains rows r >= c where A(r,c) is nonzero (excluding diag handled later).
        // We map any (i,j) to (min(i,j), max(i,j)) so pattern becomes symmetric.
        // ------------------------------------------------------------
        std::vector<std::vector<int>> Acol(n);
        for (int j = 0; j < n; ++j)
        {
            for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p)
            {
                int i = A.row_ind[p];
                if (i == j) continue;
                int c = (i < j) ? i : j;
                int r = (i < j) ? j : i;
                if (r == c) continue;
                Acol[c].push_back(r);
            }
        }
        for (int c = 0; c < n; ++c) sort_unique(Acol[c]);

        // ------------------------------------------------------------
        // children lists from etree.parent
        // ------------------------------------------------------------
        std::vector<std::vector<int>> children(n);
        for (int j = 0; j < n; ++j)
        {
            int p = etree.parent[j];
            if (p >= 0) children[p].push_back(j);
        }

        // ------------------------------------------------------------
        // Build L column patterns by increasing k (children always < parent in etree)
        // ------------------------------------------------------------
        std::vector<std::vector<int>> Lcol(n);
        for (int k = 0; k < n; ++k)
        {
            std::vector<int> rows = Acol[k];
            rows.push_back(k);
            sort_unique(rows);

            // merge child patterns into parent, skipping child pivot index itself
            for (int child : children[k])
            {
                // child columns should be ready since child < k in elim tree
                union_sorted_skip(rows, Lcol[child], child);
            }

            // keep only lower triangle (rows >= k)
            auto it = std::lower_bound(rows.begin(), rows.end(), k);
            if (it != rows.begin()) rows.erase(rows.begin(), it);

            // ensure diagonal exists
            if (rows.empty() || rows.front() != k)
            {
                rows.insert(rows.begin(), k);
            }

            Lcol[k].swap(rows);
        }

        // ------------------------------------------------------------
        // Pack into FactorPattern (CSC of L pattern)
        // ------------------------------------------------------------
        FactorPattern pattern;
        pattern.row_ptr_L.resize(n + 1);
        pattern.row_ptr_L[0] = 0;
        for (int k = 0; k < n; ++k)
        {
            pattern.row_ptr_L[k + 1] =
                pattern.row_ptr_L[k] + (int)Lcol[k].size();
        }

        const int nnzL = pattern.row_ptr_L[n];
        pattern.col_ind_L.resize(nnzL);
        for (int k = 0; k < n; ++k)
        {
            int base = pattern.row_ptr_L[k];
            for (size_t t = 0; t < Lcol[k].size(); ++t)
            {
                pattern.col_ind_L[base + (int)t] = Lcol[k][t];
            }
        }

        return pattern;
    }

    template <typename T>
    ichol::symbolic::FactorPattern
    compute_ic_factor_pattern(const ichol::matrix::CsrMatrix<T> &A, int k)
    {
        const int n = A.num_rows;
        const int INF = std::numeric_limits<int>::max() / 8;

        struct AdjEntry
        {
            int row;
            int lvl;
        };

        // For each column p: list of (row r, level(r,p)) for L(r,p), with r increasing.
        std::vector<std::vector<AdjEntry>> col_adj(n);

        // Per-row workspace (stamp-based)
        std::vector<int> mark(n, -1);
        std::vector<int> lvl(n, INF);
        std::vector<int> touched;
        touched.reserve(256);

        // Queue membership per row
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
                return; // lower only
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
        fp.col_ind_L.clear();

        for (int i = 0; i < n; ++i)
        {
            touched.clear();
            Q.clear();

            // Seed from A(i, j) for j <= i, level 0
            for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
            {
                int j = A.col_ind[p];
                if (j <= i)
                    activate_or_decrease(i, j, 0);
            }

            // Ensure diagonal at level 0
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
                inQ[p] = -1; // allow re-enqueue in same row

                if (mark[p] != i)
                    continue;
                int lvl_ip = lvl[p];
                if (lvl_ip >= k)
                    continue; // pruning: cannot generate <=k fills

                const auto &adj = col_adj[p];
                for (const auto &e : adj)
                {
                    int r = e.row;
                    if (r >= i)
                        break; // rows appended in increasing order
                    int newlvl = lvl_ip + e.lvl + 1;
                    activate_or_decrease(i, r, newlvl);
                }
            }

            std::sort(touched.begin(), touched.end()); // already unique

            // Emit CSR row i
            fp.row_ptr_L[i] = (int)fp.col_ind_L.size();
            fp.col_ind_L.insert(fp.col_ind_L.end(), touched.begin(), touched.end());
            fp.row_ptr_L[i + 1] = (int)fp.col_ind_L.size();

            // Update column adjacency for future rows
            for (int j : touched)
                if (j < i)
                    col_adj[j].push_back({i, lvl[j]});
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