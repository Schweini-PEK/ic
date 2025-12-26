// src/factor/symbolic/detail/symbolic_plan.hpp
#pragma once

#include <vector>

namespace ichol::symbolic
{
    struct Permutation
    {
        std::vector<int> perm;     // size n
        std::vector<int> inv_perm; // size n
    };

    struct ETree
    {
        std::vector<int> parent; // size n
    };

    struct LevelSets
    {
        std::vector<int> level_ptr; // size num_levels + 1
        std::vector<int> levels;    // size n
    };

    struct FactorPattern
    {
        std::vector<int> row_ptr_L; // size n + 1
        std::vector<int> col_ind_L; // size nnz_L
    };

    struct SymbolicPlan
    {
        Permutation perm;
        ETree etree;
        LevelSets level_sets;
        FactorPattern factor_pattern;
    };
}