#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>
#include "ichol/matrix_formats.hpp"
#include "ichol/ictp.hpp"
#include "ichol/ictp_par.hpp"
#include "ichol/parict.hpp"
#include "ichol/fact.hpp"
#include "ichol/symbolic.hpp"
#include "ichol/half.hpp"

namespace ichol
{
    template <class T>
    matrix::CsrMatrix<T> convert_CSR_precision(const matrix::CsrMatrix<double> &src)
    {
        matrix::CsrMatrix<T> dst;
        dst.num_rows = src.num_rows;
        dst.num_cols = src.num_cols;
        const int nnz = (int)src.values.size();
        dst.nnz = nnz;
        dst.row_ptr = src.row_ptr; // copy structure
        dst.col_ind = src.col_ind; // copy structure
        dst.values.resize(nnz);
        for (int i = 0; i < nnz; ++i)
            dst.values[i] = static_cast<T>(src.values[i]);
        return dst;
    }

    template <class T>
    matrix::CsrMatrix<T> IC_factorize(const std::string &algo,
                                      const matrix::CsrMatrix<double> &Ahost,
                                      const ICTP_Params &ictp_params,
                                      const IC_Factorize_Params &params,
                                      const core::IC_Symbolic &Sym,
                                      IC_Factorize_Info *out_info)
    {
        IC_Factorize_Info info;

        // (1) Prescaling once in double precision and convert the precision to u_l
        std::cout << "Using scaling " << params.scaling << std::endl;
        if (params.scaling == "none")
        {
            info.D = std::vector<double>(Ahost.num_rows, 1.0);
        }
        else if (params.scaling == "col_norm")
        {
            info.D = col_norm_scale(Ahost);
        }
        else if (params.scaling == "pivot")
        {
            info.D = pivot_scale(Ahost);
        }
        else
        {
            throw std::runtime_error("IC_factorize: unknown scaling option: " + params.scaling);
        }
        ichol::matrix::CsrMatrix<T> B = convert_CSR_precision<T>(apply_symm_prescaling(Ahost, info.D));

        // (2) Prepare attempt-level params (no scaling, no shift here)
        IC_Attempt_Params attempt_params;
        attempt_params.pivot_tol = params.pivot_tol;
        attempt_params.enable_safe_fp16_checks = params.enable_safe_fp16_checks;

        // (3) Shift-restart loop in u_l precision
        T alpha = T(params.initial_shift);

        ichol::matrix::CsrMatrix<T> Atry, L;
        ICTP_Factor_Info finfo;
        for (int attempt = 0; attempt < params.max_restarts; ++attempt)
        {
            // Apply shifting at the beginning of each attempt
            Atry = add_diagonal_shift(B, alpha);
            if (algo == "ictp")
                L = ictp<T>(Atry, ictp_params, attempt_params, Sym, &finfo);
            else if (algo == "ictp_par")
                L = ictp_par<T>(Atry, ictp_params, attempt_params, Sym, &finfo);
            else if (algo == "parict")
                L = parict<T>(Atry, ictp_params, attempt_params, Sym, &finfo);
            else
                throw std::runtime_error("Unknown algorithm: " + algo);

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

    template matrix::CsrMatrix<double> IC_factorize<double>(const std::string &algo,
                                                            const matrix::CsrMatrix<double> &Ahost,
                                                            const ICTP_Params &ictp_params,
                                                            const IC_Factorize_Params &params,
                                                            const core::IC_Symbolic &Sym,
                                                            IC_Factorize_Info *out_info);

    template matrix::CsrMatrix<float> IC_factorize<float>(const std::string &algo,
                                                          const matrix::CsrMatrix<double> &Ahost,
                                                          const ICTP_Params &ictp_params,
                                                          const IC_Factorize_Params &params,
                                                          const core::IC_Symbolic &Sym,
                                                          IC_Factorize_Info *out_info);

    template matrix::CsrMatrix<half_float::half> IC_factorize<half_float::half>(const std::string &algo,
                                                                                const matrix::CsrMatrix<double> &Ahost,
                                                                                const ICTP_Params &ictp_params,
                                                                                const IC_Factorize_Params &params,
                                                                                const core::IC_Symbolic &Sym,
                                                                                IC_Factorize_Info *out_info);

} // namespace ichol