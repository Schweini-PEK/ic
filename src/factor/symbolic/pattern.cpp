#include <deque>

#include "symbolic.hpp"

namespace ichol::symbolic
{
    template <typename T>
    ichol::symbolic::FactorPattern compute_complete_cholesky_pattern(const ichol::CsrMatrix<T> &A,
                                                                     const ichol::symbolic::ETree &etree)
    {
        const int n = A.num_rows;
        ichol::symbolic::FactorPattern factor_pattern;

        return factor_pattern;
    }

    template <typename T>
    ichol::symbolic::FactorPattern compute_ic_factor_pattern(const ichol::CsrMatrix<T> &A,
                                                             int level_k)
    {
        const int n = A.num_rows;
        ichol::symbolic::FactorPattern factor_pattern;

        if (level_k == 0) // IC(0)
        {
            factor_pattern.row_ptr_L = A.row_ptr;
            factor_pattern.col_ind_L = A.col_ind;

            return factor_pattern;
        }

        // Store final per-row pattern+levels while constructing later rows.
        std::vector<std::vector<int>> L_cols(n);
        std::vector<std::vector<int>> L_lvls(n);

        // Workspace reused across rows:
        // mark[col] == current_row means col is in the active set for this row.
        std::vector<int> mark(n, -1);
        std::vector<int> lvl(n, std::numeric_limits<int>::max() / 4);

        // touched columns for fast reset (no O(n) clears).
        std::vector<int> touched;
        touched.reserve(256);

        // Queue membership flag (only needed up to current row index; keep full size for simplicity).
        std::vector<unsigned char> in_queue(n, 0);

        const int INF = std::numeric_limits<int>::max() / 4;

        for (int i = 0; i < n; ++i)
        {
            // Active set starts empty.
            touched.clear();
            std::deque<int> Q;

            auto activate_or_decrease = [&](int j, int newlvl)
            {
                // Keep only if within level_k; diagonal handled separately by caller.
                if (j != i && newlvl > level_k)
                    return;

                if (mark[j] != i)
                {
                    // First time we see column j in row i
                    mark[j] = i;
                    lvl[j] = newlvl;
                    touched.push_back(j);

                    // Any off-diagonal entry (i,j) with j<i can act as a pivot p=j for propagation.
                    if (j < i && !in_queue[j])
                    {
                        in_queue[j] = 1;
                        Q.push_back(j);
                    }
                }
                else
                {
                    // Already present: keep the minimum level (core of Saad/PETSc rule).
                    if (newlvl < lvl[j])
                    {
                        lvl[j] = newlvl;
                        // If this column can act as a pivot and is not queued, queue it.
                        if (j < i && !in_queue[j])
                        {
                            in_queue[j] = 1;
                            Q.push_back(j);
                        }
                    }
                }
            };

            // 1) Initialize from A's stored lower row i: all those edges are level 0.
            for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx)
            {
                const int j = A.col_ind[idx];
                if (j <= i)
                {
                    activate_or_decrease(j, 0);
                }
                // If input accidentally contains upper entries (j>i), ignore for lower pattern.
            }

            // Ensure diagonal exists (always keep).
            if (mark[i] != i)
            {
                mark[i] = i;
                lvl[i] = 0;
                touched.push_back(i);
            }
            else
            {
                lvl[i] = 0; // diag is level 0
            }

            // 2) Propagate fill via pivots p in the current row i.
            //
            // Intuition:
            //   If (i,p) is in the pattern, then eliminating through p allows coupling i with
            //   everything that p is coupled with (entries (p,j) in L row p), producing candidates (i,j).
            //
            // For each pivot p popped from the queue:
            //   for each (p,j) with j<p in row p:
            //       propose (i,j) with level = lvl(i,p) + lvl(p,j) + 1
            while (!Q.empty())
            {
                const int p = Q.front();
                Q.pop_front();
                in_queue[p] = 0;

                const int lvl_ip = (mark[p] == i) ? lvl[p] : INF;
                if (lvl_ip == INF)
                    continue;

                // Row p is already finalized (since p < i). Merge its strictly-lower entries j<p.
                const auto &rowp_cols = L_cols[p];
                const auto &rowp_lvls = L_lvls[p];

                for (size_t t = 0; t < rowp_cols.size(); ++t)
                {
                    const int j = rowp_cols[t];
                    if (j >= p)
                        continue; // skip diagonal (j==p) and anything above (shouldn't exist)
                    const int lvl_pj = rowp_lvls[t];

                    const int newlvl = lvl_ip + lvl_pj + 1; // PETSc/Saad ICC level recurrence (mirrored)
                    activate_or_decrease(j, newlvl);
                }
            }

            // 3) Finalize row i: collect active columns, sort, and store levels aligned.
            //
            // touched contains each active column exactly once (because mark prevents duplicates),
            // but in arbitrary order.
            std::vector<int> cols_i = touched;
            std::sort(cols_i.begin(), cols_i.end());
            cols_i.erase(std::unique(cols_i.begin(), cols_i.end()), cols_i.end());

            // Enforce lower-triangular constraint and diagonal presence.
            // (Triangular constraint should already hold; this is just structural hygiene.)
            cols_i.erase(std::remove_if(cols_i.begin(), cols_i.end(),
                                        [&](int c)
                                        { return c > i; }),
                         cols_i.end());
            if (cols_i.empty() || cols_i.back() != i)
            {
                cols_i.insert(std::upper_bound(cols_i.begin(), cols_i.end(), i), i);
                mark[i] = i;
                lvl[i] = 0;
            }

            std::vector<int> lvls_i;
            lvls_i.reserve(cols_i.size());
            for (int c : cols_i)
            {
                // All retained entries satisfy lvl<=level_k (except diag which is 0).
                lvls_i.push_back((mark[c] == i) ? lvl[c] : INF);
            }

            L_cols[i] = std::move(cols_i);
            L_lvls[i] = std::move(lvls_i);
        }

        // 4) Pack into CSR FactorPattern
        factor_pattern.row_ptr_L.resize(n + 1);
        factor_pattern.row_ptr_L[0] = 0;
        for (int i = 0; i < n; ++i)
        {
            factor_pattern.row_ptr_L[i + 1] = factor_pattern.row_ptr_L[i] + (int)L_cols[i].size();
        }

        factor_pattern.col_ind_L.resize(factor_pattern.row_ptr_L[n]);
        for (int i = 0; i < n; ++i)
        {
            int out = factor_pattern.row_ptr_L[i];
            for (int c : L_cols[i])
                factor_pattern.col_ind_L[out++] = c;
        }

        return factor_pattern;
    }

    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<double>(const ichol::CsrMatrix<double> &A,
                                                                                      const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<double>(const ichol::CsrMatrix<double> &A,
                                                                              int level_k);
    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<float>(const ichol::CsrMatrix<float> &A,
                                                                                     const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<float>(const ichol::CsrMatrix<float> &A,
                                                                             int level_k);
    template ichol::symbolic::FactorPattern compute_complete_cholesky_pattern<half_float::half>(const ichol::CsrMatrix<half_float::half> &A,
                                                                                                const ichol::symbolic::ETree &etree);
    template ichol::symbolic::FactorPattern compute_ic_factor_pattern<half_float::half>(const ichol::CsrMatrix<half_float::half> &A,
                                                                                        int level_k);
} // namespace ichol::symbolic
