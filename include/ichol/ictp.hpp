#ifndef ICHOL_ICTP_HPP
#define ICHOL_ICTP_HPP

#include <ichol/matrix_formats.hpp>
#include <ichol/symbolic.hpp>

struct ICTP_Params;
struct IC_Attempt_Params;
struct ICTP_Factor_Info;

namespace ichol
{
    /**
     * Parameters used only in ICTP.
     *
     * It controls the maximum number of nonzeros per row and the drop threshold.
     */

    template <class T>
    CsrMatrix<T> ictp(const CsrMatrix<T> &Ahost,
                const ICTP_Params &row_params,
                const IC_Attempt_Params &fparams,
                const core::IC_Symbolic &Sym,
                ICTP_Factor_Info *info);

} // namespace ichol

#endif // INCHOL_ICTP_HPP