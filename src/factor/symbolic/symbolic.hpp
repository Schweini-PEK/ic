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
    ichol::symbolic::FactorPattern compute_complete_cholesky_pattern(const ichol::matrix::CsrMatrix<T> &A,
                                                                     const ichol::symbolic::ETree &etree);

    template <typename T>
    ichol::symbolic::FactorPattern compute_ic_factor_pattern(const ichol::matrix::CsrMatrix<T> &A,
                                                             int level_k);

    /**
     * Build level scheduling given the sparsity pattern of L.
     */
    ichol::symbolic::LevelSets build_level_sets(const ichol::symbolic::FactorPattern &factor_pattern,
                                                const ichol::SymbolicOptions &options);

    /**
     * Perform symbolic analysis for IC or IC(k) factorization.
     */
    template <typename T>
    SymbolicPlan ic_analyze(const ichol::matrix::CsrMatrix<T> &A,
                            const SymbolicOptions &options);
} // namespace ichol::symbolic