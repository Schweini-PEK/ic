#include "detail/symbolic_plan.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/half.hpp"

#include <numeric>
#include <vector>
#include <algorithm>

namespace ichol::symbolic
{
    ichol::symbolic::LevelSets build_level_sets(const ichol::symbolic::FactorPattern &factor_pattern)
    {
        int n = (int)factor_pattern.row_ptr_L.size() - 1;
        int max_level = -1;
        std::vector<int> level_of(n, -1);

        for (int i = 0; i < n; ++i)
        {
            int best = -1;
            // Skip the last one per row, i.e., the diagonal entry
            for (int p = factor_pattern.row_ptr_L[i]; p < factor_pattern.row_ptr_L[i + 1] - 1; ++p)
            {
                best = std::max(best, level_of[factor_pattern.col_ind_L[p]]);
            }
            level_of[i] = best + 1;
            max_level = std::max(max_level, level_of[i]);
        }

        const int num_levels = max_level + 1;
        std::vector<int> counts(num_levels, 0);
        for (int i = 0; i < n; ++i)
        {
            counts[level_of[i]]++;
        }

        ichol::symbolic::LevelSets level_sets;
        level_sets.level_ptr.assign(max_level + 1, 0);
        for (int i = 0; i < n; ++i)
        {
            level_sets.level_ptr[level_of[i]]++;
        }

        level_sets.level_ptr.resize(num_levels + 1);
        level_sets.level_ptr[0] = 0;
        std::partial_sum(counts.begin(), counts.end(), level_sets.level_ptr.begin() + 1);

        level_sets.levels.resize(n);
        std::vector<int> next = level_sets.level_ptr;
        for (int i = 0; i < n; ++i)
        {
            int L = level_of[i];
            level_sets.levels[next[L]++] = i;
        }

        return level_sets;
    }

    SnodeLevelSets build_snode_level_sets(const LevelSets &col_level_sets,
                                          const std::vector<std::pair<int,int>> &snodes)
    {
        int num_col_levels = static_cast<int>(col_level_sets.level_ptr.size()) - 1;
        int ncols = 0;
        if (num_col_levels >= 0) ncols = col_level_sets.level_ptr.back();

        // build level_of[col]
        std::vector<int> level_of(ncols, 0);
        for (int L = 0; L < num_col_levels; ++L) {
            int s = col_level_sets.level_ptr[L];
            int e = col_level_sets.level_ptr[L + 1];
            for (int idx = s; idx < e; ++idx) {
                int col = col_level_sets.levels[idx];
                level_of[col] = L;
            }
        }

        int m = static_cast<int>(snodes.size());
        std::vector<int> snode_level(m, 0);
        int max_level = 0;
        for (int id = 0; id < m; ++id) {
            int s = snodes[id].first;
            int e = snodes[id].second;
            int lv = 0;
            if (s < e) {
                lv = level_of[s];
                for (int c = s + 1; c < e; ++c) lv = std::max(lv, level_of[c]);
            }
            snode_level[id] = lv;
            if (lv > max_level) max_level = lv;
        }

        LevelSets snode_sets;
        snode_sets.level_ptr.assign(max_level + 2, 0); // size = max_level+1 + 1
        for (int id = 0; id < m; ++id) snode_sets.level_ptr[snode_level[id] + 1]++;

        std::partial_sum(snode_sets.level_ptr.begin(), snode_sets.level_ptr.end(), snode_sets.level_ptr.begin());
        snode_sets.levels.resize(m);
        std::vector<int> next(snode_sets.level_ptr.begin(), snode_sets.level_ptr.end() - 1);
        for (int id = 0; id < m; ++id) {
            int L = snode_level[id];
            snode_sets.levels[next[L]++] = id;
        }

        SnodeLevelSets out;
        out.snode_level = std::move(snode_level);
        out.level_sets = std::move(snode_sets);
        return out;
    }


} // namespace