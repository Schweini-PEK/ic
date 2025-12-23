#ifndef ICTP_PAR_HPP
#define ICTP_PAR_HPP

#include <ichol/matrix_formats.hpp>
#include <ichol/symbolic.hpp>

struct ICTP_Params;
struct IC_Attempt_Params;
struct ICTP_Factor_Info;

namespace ichol
{
    template <class T>
    CsrMatrix<T> ictp_par(const CsrMatrix<T> &Ahost,
                    const ICTP_Params &row_params,
                    const IC_Attempt_Params &fparams,
                    const core::IC_Symbolic &Sym,
                    ICTP_Factor_Info *info);
}

#endif // ICTP_PAR_HPP