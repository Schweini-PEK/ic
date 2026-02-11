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
        std::vector<int> parent;   // size n
        std::vector<int> colcount; // size n, CHOLMOD-style simplicial column counts

        // Cached strict-upper adjacency (optional). When present, callers can reuse
        // it to avoid re-scanning A to build an adjacency for ereach-like traversals.
        // Upper triangle in compressed form:
        //   upper_ind[ upper_ptr[j] .. upper_ptr[j+1] ) are i < j with A(i,j) != 0
        std::vector<int> upper_ptr; // size n+1
        std::vector<int> upper_ind; // size nnz_strict_upper
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
} // namespace ichol::symbolic