#pragma once
#include <vector>
#include <utility>

#include "super_sym.hpp"
#include "ichol/matrix_formats.hpp"

namespace ichol::symbolic {



    struct SuperNumeric
    {
        SuperSym sym;
        std::vector<double> x;  // size = sym.px.back(), CHOLMOD block layout
        bool ok = true;
        int fail_snode = -1;
        int fail_col_in_snode = -1;
    };

    SuperNumeric factorize_supernodal_ll(
        const matrix::CscMatrix<double>& A,
        const SuperSym& sym);


} // namespace ichol::symbolic
