// src/factor/symbolic/detail/symbolic_plan.hpp
#pragma once

#include <vector>

#include "factor/symbolic/super_sym.hpp"

namespace ichol::symbolic
{
    struct Permutation
    {
        std::vector<int> perm;     // size n
        std::vector<int> inv_perm; // size n
    };

    struct ETree
    {
        std::vector<int> parent;    // size n
        std::vector<int> colcount;  // size n, CHOLMOD-style simplicial column counts

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
    // Scheduling information for supernodes.
    //
    // NOTE: For the LL supernodal numeric path, the canonical schedule is
    // derived from the SuperSym rowlist (CHOLMOD-style). For other algorithms,
    // a simpler level-set scheduling based on column dependencies can also be used.
    struct SnodeLevelSets {
        std::vector<int> levels;                    // per-snode level
        std::vector<int> parent;                    // per-snode parent (optional)
        std::vector<std::vector<int>> children;     // adjacency list (optional)
        std::vector<std::vector<int>> buckets;      // buckets[level] = snode ids
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

        // CHOLMOD-like packed supernodal symbolic (consumed by numeric)
        std::vector<std::vector<int>> snode_rows; // union rowlist per snode
        SuperSym sym;

        SnodeLevelSets snode_level_sets;         // supernode DAG scheduling
    };
}
