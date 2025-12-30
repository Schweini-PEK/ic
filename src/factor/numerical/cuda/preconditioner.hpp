#pragma once

#include "ichol/matrix_formats.hpp"
#include "ichol/options.hpp"
#include "ichol/half.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "factor/numerical/detail/numeric_plan.hpp"

namespace ichol::numeric::cuda
{
    template <class T>
    ichol::matrix::CsrMatrix<T> ickdt(const ichol::matrix::CsrMatrix<T> &Ahost,
                                      const ichol::symbolic::SymbolicPlan &sym_plan,
                                      ichol::numeric::NumericPlan &out_plan,
                                      const ichol::IncompleteCholeskyOptions &options);
} // namespace ichol::numeric::cuda