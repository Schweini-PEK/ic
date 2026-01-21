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

    // compute_complete_cholesky_pattern - CSC native implementation
    // Returns FactorPattern with col_ptr (n+1) and row_idx (total nnz of factor)
    template <typename T>
    FactorPattern compute_complete_cholesky_pattern(
        const ichol::matrix::CscMatrix<T> &A,
        const ichol::symbolic::ETree &etree)
    {
        const int n = A.num_cols;

        // per-column discovered nodes
        std::vector<std::vector<int>> cols(n);
        std::vector<int> marker(n, -1); // marker[v] == k 表示 v 已被列 k 访问
        std::vector<int> stack;
        stack.reserve(n);

        for (int k = 0; k < n; ++k)
        {
            stack.clear();

            if (marker[k] != k)
            {
                marker[k] = k;
                stack.push_back(k);
            }

            for (int p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p)
            {
                int i = A.row_ind[p];
                if (i == k)
                    continue;
                int r = std::max(i, k); // 确保落到 L 的下三角（行>=列）
                if (marker[r] != k)
                {
                    marker[r] = k;
                    stack.push_back(r);
                }
            }

            // propagate up the elimination tree: 对 stack 中的每个节点加入其未访问过的祖先
            for (size_t idx = 0; idx < stack.size(); ++idx)
            {
                int v = stack[idx];
                int par = etree.parent[v];
                while (par != -1 && marker[par] != k)
                {
                    marker[par] = k;
                    stack.push_back(par);
                    par = etree.parent[par];
                }
            }

            // CHOLMOD 里列模式通常是递增且唯一的（至少在符号阶段会保证可比性）
            // 这里统一：只保留 >=k（下三角 L 的行），排序+去重
            std::sort(stack.begin(), stack.end());
            stack.erase(std::unique(stack.begin(), stack.end()), stack.end());
            auto it = std::lower_bound(stack.begin(), stack.end(), k);
            cols[k].assign(it, stack.end());
        }

        FactorPattern pattern;
        pattern.row_ptr_L.resize(n + 1);
        pattern.row_ptr_L[0] = 0;
        for (int k = 0; k < n; ++k)
        {
            pattern.row_ptr_L[k + 1] = pattern.row_ptr_L[k] + static_cast<int>(cols[k].size());
        }

        const int total_nnz = pattern.row_ptr_L[n];
        pattern.col_ind_L.resize(total_nnz);
        for (int k = 0; k < n; ++k)
        {
            const int base = pattern.row_ptr_L[k];
            for (size_t t = 0; t < cols[k].size(); ++t)
            {
                pattern.col_ind_L[base + static_cast<int>(t)] = cols[k][t];
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