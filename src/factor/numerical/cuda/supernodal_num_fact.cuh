#pragma once

#include "factor/numerical/supernodal_numeric_ll.hpp"
#include "factor/symbolic/super_sym.hpp"
#include "factor/symbolic/snode_schedule.hpp"

namespace ichol::numeric
{
    numeric::SuperNumeric factorize_supernodal_ll_gpu(
        const ichol::matrix::CscMatrix<double> &A,
        const symbolic::SupernodalLLPlan &plan);

} // namespace ichol::numeric