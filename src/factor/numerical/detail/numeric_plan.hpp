// src/factor/numerical/detail/numeric_plan.hpp
#pragma once
#ifndef ICHOL_FACTOR_NUMERICAL_FACTOIRZE_HPP
#define ICHOL_FACTOR_NUMERICAL_FACTOIRZE_HPP

#include <vector>
#include "ichol/matrix_formats.hpp"

namespace ichol::numeric
{
    enum class ICBreakdown
    {
        None = 0,
        B1_SmallOrNegativePivot, // diag pivot too small/negative
        B2_ScaleOverflow,        // low-precision risk (optional)
        B3_UpdateOverflow,       // low-precision risk (optional)
        OtherNumericalIssue
    };

    struct PrescalingVectors
    {
        std::vector<double> D; // Diagonal scaling entries
    };

    /**
     * Information about the IC factorization process.
     */
    struct ICInfo
    {
        int restarts = 0;
        double shift_used = 0.0;
        ICBreakdown code = ICBreakdown::None;

        // ParICT specific
        int step = -1;
    };

    struct NumericPlan
    {
        PrescalingVectors prescaling;
        ICInfo ic_info;
        ichol::matrix::CsrMatrix<double> A_scaled; // Prescaled matrix
        // ichol::matrix::CsrMatrix<double> L;        // Final IC factor in double precision
    };
} // namespace ichol::numeric

#endif // ICHOL_FACTOR_NUMERICAL_FACTOIRZE_HPP