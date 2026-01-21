// supernode.cpp
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>

#include "symbolic.hpp"

namespace ichol::symbolic
{

    // ============================================================================
    // Supernode 检测（symbolic, supernodal）
    //
    // 这里实现了与 CHOLMOD（cholmod_super_symbolic.c）一致的两阶段逻辑：
    //   1) fundamental supernodes：只基于 etree/colcount 的“基本超节点”划分
    //   2) relaxed amalgamation：按 (nrelax, zrelax) 阈值在消去树上做合并
    //
    // 备注：
    //  - detect_supernodes_fundamental：仅执行步骤 1（方便对齐/调试）
    //  - detect_supernodes（默认）：执行 1 + 2（与 CHOLMOD 默认行为一致）
    //  - detect_supernodes_approx：之前的启发式方法，保留作对比
    // ============================================================================


    //------------------------------------------------------------------------------
    // Fundamental supernodes (CHOLMOD/Supernodal/cholmod_super_symbolic.c)
    //------------------------------------------------------------------------------
    std::vector<std::pair<int, int>> detect_supernodes_fundamental(const ETree &etree)
    {
        const int n = (int)etree.parent.size();
        std::vector<std::pair<int, int>> sn;
        if (n <= 0) return sn;

        const auto &Parent = etree.parent;
        const auto &ColCount = etree.colcount;
        assert((int)ColCount.size() == n);

        // Wi: child counts of each node in etree
        std::vector<int> Wi(n, 0);
        for (int j = 0; j < n; ++j)
        {
            int p = Parent[j];
            if (p >= 0 && p < n) Wi[p]++;
        }

        // Super list (start indices)
        std::vector<int> Super;
        Super.reserve(n + 1);
        Super.push_back(0);

        for (int j = 1; j < n; ++j)
        {
            const bool new_super =
                (Parent[j - 1] != j) ||
                (ColCount[j - 1] != ColCount[j] + 1) ||
                (Wi[j] > 1);

            if (new_super) Super.push_back(j);
        }
        Super.push_back(n);

        sn.reserve(Super.size() - 1);
        for (size_t s = 0; s + 1 < Super.size(); ++s)
        {
            sn.emplace_back(Super[s], Super[s + 1]);
        }
        return sn;
    }

    //------------------------------------------------------------------------------
    // Relaxed amalgamation (CHOLMOD default nrelax/zrelax)
    //------------------------------------------------------------------------------
    static std::vector<std::pair<int, int>> detect_supernodes_relaxed_default(const ETree &etree)
    {
        const int n = (int)etree.parent.size();
        std::vector<std::pair<int, int>> out;
        if (n <= 0) return out;

        const auto &Parent = etree.parent;
        const auto &ColCount = etree.colcount;
        assert((int)ColCount.size() == n);

        // CHOLMOD defaults (Common->nrelax / Common->zrelax)
        constexpr int nrelax0 = 4;
        constexpr int nrelax1 = 16;
        constexpr int nrelax2 = 48;
        constexpr double zrelax0 = 0.8;
        constexpr double zrelax1 = 0.1;
        constexpr double zrelax2 = 0.05;

        // -------------------------------------------------------------------------
        // Step 1: fundamental supernodes (Super[0..nfsuper])
        // -------------------------------------------------------------------------
        std::vector<int> Wi(n, 0);
        for (int j = 0; j < n; ++j)
        {
            int p = Parent[j];
            if (p >= 0 && p < n) Wi[p]++;
        }

        std::vector<int> Super;
        Super.reserve(n + 1);
        int nfsuper = (n == 0) ? 0 : 1;
        Super.push_back(0);

        for (int j = 1; j < n; ++j)
        {
            if (Parent[j - 1] != j ||
                (ColCount[j - 1] != ColCount[j] + 1) ||
                Wi[j] > 1)
            {
                Super.push_back(j);
                nfsuper++;
            }
        }
        Super.push_back(n);
        assert((int)Super.size() == nfsuper + 1);

        // -------------------------------------------------------------------------
        // Step 2: SuperMap and fundamental supernodal etree (Sparent)
        // -------------------------------------------------------------------------
        std::vector<int> SuperMap(n, -1);
        for (int s = 0; s < nfsuper; ++s)
        {
            for (int k = Super[s]; k < Super[s + 1]; ++k)
                SuperMap[k] = s;
        }

        std::vector<int> Sparent(nfsuper, -1);
        for (int s = 0; s < nfsuper; ++s)
        {
            const int jlast = Super[s + 1] - 1;
            const int p = Parent[jlast];
            Sparent[s] = (p == -1) ? -1 : SuperMap[p];
        }

        // -------------------------------------------------------------------------
        // Step 3: relaxed amalgamation (Merged/Nscol/Zeros/Snz)
        // -------------------------------------------------------------------------
        std::vector<int> Merged(nfsuper, -1);
        std::vector<int> Nscol(nfsuper, 0);
        std::vector<int> Zeros(nfsuper, 0);
        std::vector<int> Snz(nfsuper, 0);

        for (int s = 0; s < nfsuper; ++s)
        {
            Nscol[s] = Super[s + 1] - Super[s];
            Zeros[s] = 0;
            Snz[s] = ColCount[Super[s]]; // leading column count
        }

        for (int s = nfsuper - 2; s >= 0; --s)
        {
            int ss = Sparent[s];
            if (ss == -1) continue; // root

            // find current parent of s (path compression through Merged)
            int sparent = ss;
            while (Merged[sparent] != -1) sparent = Merged[sparent];
            // compress along the way
            for (ss = Sparent[s]; Merged[ss] != -1; )
            {
                const int snext = Merged[ss];
                Merged[ss] = sparent;
                ss = snext;
            }

            // only merge if s+1 is the (current) parent of s
            if (sparent != s + 1) continue;

            const int nscol0 = Nscol[s];
            const int nscol1 = Nscol[s + 1];
            const int ns = nscol0 + nscol1;

            int totzeros = Zeros[s + 1];
            const double lnz1 = (double)Snz[s + 1];

            bool merge = false;
            if (ns <= nrelax0)
            {
                merge = true;
            }
            else
            {
                const double lnz0 = (double)Snz[s];
                const double xnewzeros = (double)nscol0 * (lnz1 + (double)nscol0 - lnz0);

                if (xnewzeros == 0.0)
                {
                    merge = true;
                }
                else
                {
                    const double xtotzeros = (double)totzeros + xnewzeros;
                    const double xns = (double)ns;
                    const double xtotsize = (xns * (xns + 1.0) / 2.0) + xns * (lnz1 - (double)nscol1);
                    const double z = xtotzeros / xtotsize;
                    totzeros += nscol0 * (Snz[s + 1] + nscol0 - Snz[s]);

                    merge = ((ns <= nrelax1 && z < zrelax0) ||
                             (ns <= nrelax2 && z < zrelax1) ||
                             (z < zrelax2));
                }
            }

            if (merge)
            {
                Zeros[s] = totzeros;
                Merged[s + 1] = s;
                Snz[s] = nscol0 + Snz[s + 1];
                Nscol[s] += Nscol[s + 1];
            }
        }

        // -------------------------------------------------------------------------
        // Step 4: build relaxed supernode list: keep only "live" supernodes
        // -------------------------------------------------------------------------
        std::vector<int> SuperR;
        SuperR.reserve(nfsuper + 1);
        for (int s = 0; s < nfsuper; ++s)
        {
            if (Merged[s] == -1)
                SuperR.push_back(Super[s]);
        }
        SuperR.push_back(n);

        out.reserve(SuperR.size() - 1);
        for (size_t s = 0; s + 1 < SuperR.size(); ++s)
            out.emplace_back(SuperR[s], SuperR[s + 1]);
        return out;
    }

    //------------------------------------------------------------------------------
    // Public API: match CHOLMOD(cholmod_super_symbolic) default behavior
    //   - When supernodal mode is used, CHOLMOD performs relaxed amalgamation.
    //   - The FactorPattern is not needed for the relax decision itself.
    //------------------------------------------------------------------------------
    std::vector<std::pair<int, int>> detect_supernodes(const FactorPattern & /*pattern*/,
                                                       const ETree &etree)
    {
        return detect_supernodes_relaxed_default(etree);
    }

    //------------------------------------------------------------------------------
    // Approx supernodes , unchanged
    //------------------------------------------------------------------------------
    std::vector<std::pair<int, int>> detect_supernodes_approx(const FactorPattern &pattern,
                                                              const ETree & /*etree*/,
                                                              double overlap_threshold)
    {
        assert(overlap_threshold > 0.0 && overlap_threshold <= 1.0);
        int n = static_cast<int>(pattern.row_ptr_L.size()) - 1;
        std::vector<std::pair<int, int>> sn;
        if (n <= 0) return sn;

        int cur_start = 0;
        for (int c = 1; c < n; ++c)
        {
            int s1 = pattern.row_ptr_L[c - 1], e1 = pattern.row_ptr_L[c - 1 + 1];
            int s2 = pattern.row_ptr_L[c],     e2 = pattern.row_ptr_L[c + 1];
            int len1 = e1 - s1;
            int len2 = e2 - s2;
            if (len1 == 0 || len2 == 0)
            {
                if (!(len1 == 0 && len2 == 0))
                {
                    sn.emplace_back(cur_start, c);
                    cur_start = c;
                }
                continue;
            }

            int i = s1, j = s2, inter = 0;
            while (i < e1 && j < e2)
            {
                int v1 = pattern.col_ind_L[i];
                int v2 = pattern.col_ind_L[j];
                if (v1 == v2) { ++inter; ++i; ++j; }
                else if (v1 < v2) ++i;
                else ++j;
            }

            double avg_len = 0.5 * (len1 + len2);
            double ratio = static_cast<double>(inter) / avg_len;
            if (ratio < overlap_threshold)
            {
                sn.emplace_back(cur_start, c);
                cur_start = c;
            }
        }
        sn.emplace_back(cur_start, n);
        return sn;
    }

    std::vector<int> build_col2snode(const std::vector<std::pair<int, int>> &snodes, int ncols)
    {
        std::vector<int> col2s(ncols, -1);
        for (size_t id = 0; id < snodes.size(); ++id)
        {
            int s = snodes[id].first;
            int e = snodes[id].second;
            for (int c = s; c < e; ++c) col2s[c] = static_cast<int>(id);
        }
        return col2s;
    }

    std::vector<std::vector<int>> compute_snode_rows(const FactorPattern &pat,
                                                     const std::vector<std::pair<int, int>> &snodes)
    {
        std::vector<std::vector<int>> out;
        out.reserve(snodes.size());
        for (const auto &pr : snodes)
        {
            int s = pr.first;
            int e = pr.second;
            std::vector<int> rows;
            for (int c = s; c < e; ++c)
            {
                int a = pat.row_ptr_L[c];
                int b = pat.row_ptr_L[c + 1];
                rows.insert(rows.end(), pat.col_ind_L.begin() + a, pat.col_ind_L.begin() + b);
            }
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            out.push_back(std::move(rows));
        }
        return out;
    }

} // namespace ichol::symbolic
