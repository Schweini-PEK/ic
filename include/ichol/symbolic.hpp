// ichol/symbolic.hpp

#pragma once
#ifndef ICHOL_SYMBOLIC_HPP
#define ICHOL_SYMBOLIC_HPP

#include <vector>
#include "matrix_formats.hpp"

namespace ichol
{
    namespace core
    {
        /**
         * Symbolic phase result.
         */
        struct IC_Symbolic
        {
            int n;                      // number of rows/cols
            std::vector<int> row_ptr_L; // length n+1
            std::vector<int> col_ind_L; // length nnz_L
        };

        /**
         * Build IC symbolic factorization with either full pattern or IC(k)
         */
        IC_Symbolic build_ic_symbolic(const ichol::matrix::CsrMatrix<double> &A,
                                      int k);

    }

}

#endif // ICHOL_SYMBOLIC_HPP