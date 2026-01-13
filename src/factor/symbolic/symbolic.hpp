// src/factor/symbolic/symbolic.hpp
#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <utility>

#include "detail/symbolic_plan.hpp"
#include "ichol/options.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/half.hpp"

extern "C"
{
#include <amd.h>
}

namespace ichol::symbolic
{
    Permutation identity_permutation(int n);

    Permutation amd_from_csr(int n,
                             const std::vector<int> &row_ptr,
                             const std::vector<int> &col_ind);

    /**
     * In-place permutation of CSR matrix: A := P * A * P^T
     */
                             template <typename T>
    void apply_permutation_csr(ichol::matrix::CsrMatrix<T> &A,
                               const Permutation &P);

    /**
     * Construct the elimination tree from the pattern of A.
     */
    template <typename T>
    ichol::symbolic::ETree build_etree(const ichol::matrix::CsrMatrix<T> &A);

    template <typename T>
    ichol::symbolic::FactorPattern compute_complete_cholesky_pattern(const ichol::matrix::CsrMatrix<T> &A,
                                                                     const ichol::symbolic::ETree &etree);

    template <typename T>
    ichol::symbolic::FactorPattern compute_ic_factor_pattern(const ichol::matrix::CsrMatrix<T> &A,
                                                             int level_k);

    /**
     * Build 0-based level scheduling given the sparsity pattern of L.
     * 
     * If a row has no dependencies, it is assigned to level 0.
     * @param LevelSets.level_ptr gives the start indices of each level
     * @param LevelSets.levels holds the row indices grouped contiguously by levels
     */
    ichol::symbolic::LevelSets build_level_sets(const ichol::symbolic::FactorPattern &factor_pattern);

    /**
     * Perform symbolic analysis for IC or IC(k) factorization.
     *
     * After this function, A is permuted according to @param options.ordering.
     * The permutation vector is available in returned SymbolicPlan.perm.perm
     */
    template <typename T>
    SymbolicPlan ic_analyze(ichol::matrix::CsrMatrix<T> &A,
                            const SymbolicOptions &options);
} // namespace ichol::symbolic