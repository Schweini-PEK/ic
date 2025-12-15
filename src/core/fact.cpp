#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

#include "ichol/matrix_formats.hpp"
#include "ichol/ictp.hpp"
#include "ichol/fact.hpp"
#include "ichol/symbolic.hpp"

namespace ichol
{
    template <class T>
    CSR<T> IC_factorize(const CSR<T> &Ahost,
                        const ICTP_Params &ictp_params,
                        const IC_Factorize_Params &params,
                        const core::IC_Symbolic &Sym,
                        IC_Factorize_Info *out_info)
    {
        IC_Factorize_Info info;

        // (1) Prescaling once
        info.D = col_norm_scale(Ahost);
        CSR<T> B = apply_symm_prescaling(Ahost, info.D);

        // (2) Prepare attempt-level params (no scaling, no shift here)
        IC_Attempt_Params attempt_params;
        attempt_params.pivot_tol = params.pivot_tol;
        attempt_params.enable_safe_fp16_checks = params.enable_safe_fp16_checks;

        // (3) Shift-restart loop
        T alpha = params.initial_shift;

        CSR<T> Atry, L;
        ICTP_Factor_Info finfo;
        for (int attempt = 0; attempt < params.max_restarts; ++attempt)
        {
            // Apply shifting at the beginning of each attempt
            Atry = add_diagonal_shift(B, alpha);
            L = ictp<T>(Atry, ictp_params, attempt_params, Sym, &finfo);

            if (finfo.code == IC_Breakdown::None)
            {
                info.shift_used = alpha;
                info.restarts = attempt;
                info.last_code = IC_Breakdown::None;
                if (out_info)
                    *out_info = std::move(info);
                return L;
            }

            info.last_code = finfo.code;

            alpha *= params.shift_growth;
        }

        info.shift_used = alpha / params.shift_growth;

        if (out_info)
            *out_info = std::move(info);

        throw std::runtime_error("IC_factorize: failed to prevent breakdown within max_restarts.");
    }

    template CSR<double> IC_factorize<double>(const CSR<double> &Ahost,
                                              const ICTP_Params &ictp_params,
                                              const IC_Factorize_Params &params,
                                              const core::IC_Symbolic &Sym,
                                              IC_Factorize_Info *out_info);

    template CSR<float> IC_factorize<float>(const CSR<float> &Ahost,
                                            const ICTP_Params &ictp_params,
                                            const IC_Factorize_Params &params,
                                            const core::IC_Symbolic &Sym,
                                            IC_Factorize_Info *out_info);

    // template CSR<half> IC_factorize<float>(const CSR<float> &Ahost,
    // const ICTP_Params &ictp_params,
    // const IC_Factorize_Params &params,
    // IC_Factorize_Info *out_info);

} // namespace ichol