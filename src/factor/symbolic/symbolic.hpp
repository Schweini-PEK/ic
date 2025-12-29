// src/factor/symbolic/symbolic.hpp
#pragma once

#include "detail/symbolic_plan.hpp"
#include "ichol/options.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/half.hpp"

namespace ichol::symbolic
{
    /**
     * Construct the elimination tree from the pattern of A.
     */
    template <typename T>
    ichol::symbolic::ETree build_etree(const ichol::matrix::CsrMatrix<T> &A);

    template <typename T>
    ichol::symbolic::ETree build_etree(const matrix::CscMatrix<T>& A);

    template <typename T>
    ichol::symbolic::FactorPattern compute_complete_cholesky_pattern(const ichol::matrix::CsrMatrix<T> &A,
                                                                     const ichol::symbolic::ETree &etree);
    template <typename T>
    ichol::symbolic::FactorPattern compute_complete_cholesky_pattern(const ichol::matrix::CscMatrix<T> &A,
                                                                     const ichol::symbolic::ETree &etree);
    template <typename T>
    ichol::symbolic::FactorPattern compute_ic_factor_pattern(const ichol::matrix::CsrMatrix<T> &A,
                                                             int level_k);

    // declaration to add
    std::vector<std::pair<int,int>> detect_supernodes(const FactorPattern &pattern, const ETree &etree);
    std::vector<std::pair<int,int>> detect_supernodes_approx(const FactorPattern &pattern, const ETree &etree, double overlap_threshold);
    std::vector<int> build_col2snode(const std::vector<std::pair<int,int>>& snodes, int ncols);
    std::vector<std::vector<int>> compute_snode_rows(const FactorPattern& pattern, const std::vector<std::pair<int,int>>& snodes);


    /**
     * Build level scheduling given the sparsity pattern of L.
     */
    ichol::symbolic::LevelSets build_level_sets(const ichol::symbolic::FactorPattern &factor_pattern,
                                                const ichol::SymbolicOptions &options);

    SnodeLevelSets build_snode_level_sets(const LevelSets &col_level_sets,
                                          const std::vector<std::pair<int,int>> &snodes);
    /**
     * Perform symbolic analysis for IC or IC(k) factorization.
     */
    template <typename T>
    SymbolicPlan ic_analyze(const ichol::matrix::CsrMatrix<T> &A,
                            const SymbolicOptions &options);
} // namespace ichol::symbolic