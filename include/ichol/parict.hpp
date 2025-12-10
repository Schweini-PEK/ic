#ifndef INCHOL_PARICT_HPP
#define INCHOL_PARICT_HPP

#include <ichol/matrix_formats.hpp>

namespace ichol
{

    struct ParICT_Params
    {
        int max_steps = 5; // number of "steps" in paper terminology
    };

    CSR<double> parict_factorize(const CSR<double> &Ahost, const ParICT_Params &params);

} // namespace ichol

#endif // INCHOL_PARICT_HPP