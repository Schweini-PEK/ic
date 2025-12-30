#include "detail/symbolic_plan.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/options.hpp"
#include "ichol/half.hpp"

#include <numeric>
#include <vector>
#include <algorithm>

namespace ichol::symbolic
{
    ichol::symbolic::LevelSets build_level_sets(const ichol::symbolic::FactorPattern &factor_pattern,
                                                const ichol::SymbolicOptions &options)
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
} // namespace ichol::symbolic