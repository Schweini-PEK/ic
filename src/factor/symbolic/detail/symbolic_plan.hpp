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
        std::vector<int> colcount;  // size n, CHOLMOD-style simplicial column counts

    };

    struct LevelSets
    {
        std::vector<int> level_ptr; // size num_levels + 1
        std::vector<int> levels;    // size n
    };

    struct SnodeLevelSets {
        std::vector<int> snode_level;
        LevelSets level_sets; // level_sets.levels are snode ids
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
    struct SupernodalSymbolicPlan
    {
        ETree etree;                         // from CSC
        FactorPattern factor_pattern;        // column-level L pattern

        std::vector<std::pair<int,int>> snodes; // [start_col, end_col)
        std::vector<int> col2snode;             // size = ncols

        SnodeLevelSets snode_level_sets;         // supernode DAG scheduling
    };
}