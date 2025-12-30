#ifndef INCHOL_PARICT_HPP
#define INCHOL_PARICT_HPP

#include <ichol/matrix_formats.hpp>
#include "ichol/symbolic.hpp"

struct ICTP_Params;
struct IC_Attempt_Params;
struct ICTP_Factor_Info;

namespace ichol
{

    struct ParICT_Params
    {
        int max_steps = 5; // number of "steps" in paper terminology
    };

    template <class T>
    matrix::CsrMatrix<T> parict(const matrix::CsrMatrix<T> &Ahost,
                                const ICTP_Params &row_params,
                                const IC_Attempt_Params &fparams,
                                const core::IC_Symbolic &Sym,
                                ICTP_Factor_Info *info);

} // namespace ichol

#endif // INCHOL_PARICT_HPP