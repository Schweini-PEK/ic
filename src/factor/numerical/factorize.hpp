#pragma once
#ifndef ICHOL_FACTOR_NUMERICAL_FACTORIZE_HPP
#define ICHOL_FACTOR_NUMERICAL_FACTORIZE_HPP

#include "ichol/matrix_formats.hpp"
#include "ichol/half.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/detail/numeric_plan.hpp"

/**
 * For now the original matrix A is always expected to be in double precision.
 */
namespace ichol::numeric
{
    /**
     * Generate a vector whose ith element is the sqrt of A(ii).
     */
    std::vector<double> scale_diag_sqrt(const ichol::matrix::CsrMatrix<double> &A);

    /**
     * Generate a vector whose ith element is the norm of the ith column of A.
     */
    std::vector<double> scale_col_norm(const ichol::matrix::CsrMatrix<double> &A);

    /**
     * Apply symmetric diagonal scaling to A in place: A := D^{-1} A D^{-1}
     */
    void apply_prescaling(ichol::matrix::CsrMatrix<double> &A, const std::vector<double> &D);

    /**
     * Apply consistent RHS scaling for A := D^{-1} A D^{-1}: b := D^{-1} b
     */
    void apply_rhs_prescaling(std::vector<double> &b, const std::vector<double> &D);

    /**
     * Add a shift alpha to the diagonal of A in place: A(ii) := A(ii) + alpha
     */
    template <class T>
    void add_diagonal_shift(ichol::matrix::CsrMatrix<T> &A, T alpha);

    template <typename T>
    ichol::matrix::CsrMatrix<T> incomplete_cholesky_preconditioner(ichol::matrix::CsrMatrix<double> &A,
                                                                   const ichol::symbolic::SymbolicPlan &sym_plan,
                                                                   ichol::numeric::NumericPlan &num_plan,
                                                                   ichol::IncompleteCholeskyOptions &options);
} // namespace ichol::numeric

#endif // ICHOL_FACTOR_NUMERICAL_FACTORIZE_HPP
