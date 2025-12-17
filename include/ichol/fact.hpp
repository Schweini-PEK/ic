#ifndef ICHOL_FACT_HPP
#define ICHOL_FACT_HPP

#include <cmath>

#include "matrix_formats.hpp"
#include "ictp.hpp"
#include "symbolic.hpp"

/**
 * Generate a vector whose ith element is the norm of the ith column of A.
 *
 * It is later used as a diagonal scaling matrix D.
 * The vector is computed and stored in double precision.
 * The input matrix @param A is supposed to be stored as lower tri + diag.
 */
template <class T>
inline std::vector<double> col_norm_scale(const ichol::CSR<T> &A)
{
    const int n = A.num_rows;
    std::vector<double> col_sq(n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
        {
            const int j = A.col_ind[p];
            const double v = static_cast<double>(A.values[p]);
            const double vv = v * v;

            col_sq[j] += vv;
            if (j != i) // Ad twice for off-diag
            {
                col_sq[j] += vv;
            }
        }
    }

    std::vector<double> D(n, 1.0);
    for (int j = 0; j < n; ++j)
    {
        double nrm = std::sqrt(col_sq[j]);
        if (nrm > 0.0 && std::isfinite(nrm))
            D[j] = nrm;
    }
    return D;
}

/**
 * Apply symmetric diagonal scaling to A: B = D^{-1} A D^{-1}
 *
 * D is given as a vector of its diagonal entries.
 */
template <class T>
inline ichol::CSR<T> apply_symm_prescaling(const ichol::CSR<T> &A,
                                           const std::vector<double> &D)
{
    ichol::CSR<T> B;
    B.num_rows = A.num_rows;
    B.num_cols = A.num_cols;
    B.row_ptr = A.row_ptr;
    B.col_ind = A.col_ind;
    B.values.resize(A.values.size());

    for (int i = 0; i < A.num_rows; ++i)
    {
        T di = D[i];
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
        {
            int j = A.col_ind[p];
            T dj = D[j];
            T aij = static_cast<T>(A.values[p]);
            B.values[p] = aij / (di * dj);
        }
    }
    return B;
}

template <class T>
inline ichol::CSR<T> add_diagonal_shift(const ichol::CSR<T> &A, T alpha)
{
    if (alpha == T(0))
        return A;

    const int n = A.num_rows;
    ichol::CSR<T> S = A;

    for (int i = 0; i < n; ++i)
    {
        const int rs = S.row_ptr[i];
        const int re = S.row_ptr[i + 1];

        if (re > rs && S.col_ind[re - 1] == i)
        {
            S.values[re - 1] += alpha;
            continue;
        }
        else
        {
            throw std::runtime_error("add_diagonal_shift: missing diagonal entry in row " + std::to_string(i));
        }
    }

    return S;
}

struct ICTP_Params
{
    int lfil_per_row = 20;  // Level of fill per row
    double drop_tol = 1e-4; // Drop tolerance
};

/**
 * Breakdown event code for incomplete Cholesky.
 *
 * Defined by "Avoiding Breakdown in Incomplete Factorizations in Low Precision Arithmetic".
 */
enum class IC_Breakdown
{
    None = 0,
    B1_SmallOrNegativePivot, // diag pivot too small/negative
    B2_ScaleOverflow,        // low-precision risk (optional)
    B3_UpdateOverflow,       // low-precision risk (optional)
    OtherNumericalIssue
};

struct ICTP_Factor_Info
{
    IC_Breakdown code = IC_Breakdown::None;
    int step = -1;
    double pivot_value = 0.0;
};

/**
 * Parameters used to check only.
 *
 * To be passed into factorization routines like ICTP.
 */
struct IC_Attempt_Params
{
    double pivot_tol = 0.0;               // threshold for B1 checks
    bool enable_safe_fp16_checks = false; // For lower precision cases
};

/**
 * Parameters used to prevent breakdown.
 *
 */
struct IC_Factorize_Params
{
    double pivot_tol = 0.0;
    bool enable_safe_fp16_checks = false;

    double initial_shift = 1e-10;
    double shift_growth = 2.0;
    int max_restarts = 5;
};

struct IC_Factorize_Info
{
    double shift_used = 0.0;
    int restarts = 0;
    IC_Breakdown last_code = IC_Breakdown::None;
    std::vector<double> D; // scaling used (empty if none)
};

namespace ichol
{

    /**
     * The routine that implements the idea from "Avoiding Breakdown in Incomplete Factorizations in Low Precision Arithmetic"
     *
     * The scaling matrix and pivot shifting are all in precision T
     */
    template <class T>
    CSR<T> IC_factorize(const CSR<double> &Ahost,
                        const ICTP_Params &ictp_params,
                        const IC_Factorize_Params &params,
                        const core::IC_Symbolic &Sym,
                        IC_Factorize_Info *out_info);
} // namespace ichol

#endif // ICHOL_FACT_HPP