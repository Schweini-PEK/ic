#include "factor/numerical/factorize.hpp"
#include "factor/numerical/cuda/preconditioner.hpp"
#include "backends/cpu/util/cast.hpp"

namespace
{
    std::vector<double> compute_prescaling_vector(ichol::matrix::CsrMatrix<double> &A,
                                                  ichol::IncompleteCholeskyOptions &options)
    {
        std::vector<double> D;
        switch (options.scaling)
        {
        case ichol::Scaling::None:
            // No scaling applied
            D = std::vector<double>(A.num_rows, 1.0);
            break;

        case ichol::Scaling::UnitSqrtDiag:
            D = ichol::numeric::scale_diag_sqrt(A);
            break;

        case ichol::Scaling::UnitRowNorm:
            D = ichol::numeric::scale_col_norm(A);
            break;
        }
        return D;
    }

    template <typename T>
    T get_shift(ichol::IncompleteCholeskyOptions &options)
    {
        T shift = T(0.0);
        switch (options.pivot_shift_strategy)
        {
        case ichol::PivotShiftStrategy::None:
            shift = ichol::util::cast_fp_type<T, double>(0.0);
            break;

        case ichol::PivotShiftStrategy::MachineEpsilon:
            shift = T(std::numeric_limits<T>::epsilon() * 100.0);
            break;

        case ichol::PivotShiftStrategy::Static:
            shift = ichol::util::cast_fp_type<T, double>(options.static_shift);
            break;

        case ichol::PivotShiftStrategy::Dynamic:
            // STUB
            break;
        }
        return shift;
    }

    template <typename T>
    ichol::matrix::CsrMatrix<T> compute_ic_factor(ichol::matrix::CsrMatrix<T> &A_work,
                                                  ichol::symbolic::SymbolicPlan &sym_plan,
                                                  ichol::numeric::NumericPlan &num_plan,
                                                  ichol::IncompleteCholeskyOptions &options)
    {
        ichol::matrix::CsrMatrix<T> L;

        switch (options.algorithm)
        {
        case ichol::FactorizationAlgorithm::ICKDT:
            L = ichol::numeric::cuda::ickdt<T>(A_work, sym_plan, num_plan, options);
            break;

        default:
            break;
        }

        return L;
    }
}

namespace ichol::numeric
{
    template <typename T>
    ichol::matrix::CsrMatrix<T> incomplete_cholesky_preconditioner(ichol::matrix::CsrMatrix<double> &A,
                                                                   ichol::symbolic::SymbolicPlan &sym_plan,
                                                                   ichol::numeric::NumericPlan &num_plan,
                                                                   ichol::IncompleteCholeskyOptions &options)
    {
        num_plan.prescaling.D = compute_prescaling_vector(A, options);

        ichol::numeric::apply_prescaling(A, num_plan.prescaling.D);
        num_plan.A_scaled = A; // For PCG use

        ichol::matrix::CsrMatrix<T> A_work = ichol::matrix::convert_csr_precision<double, T>(A);
        ichol::matrix::CsrMatrix<T> L;
        T shift = get_shift<T>(options);
        for (int attempt = 0; attempt < options.max_restarts; ++attempt)
        {
            ichol::numeric::add_diagonal_shift<T>(A_work, shift);
            L = compute_ic_factor<T>(A_work, sym_plan, num_plan, options);

            if (num_plan.ic_info.code == ICBreakdown::None)
            {
                // Successful factorization, wrap up
                num_plan.ic_info.shift_used = static_cast<double>(shift);
                num_plan.ic_info.restarts = attempt;

                return L;
            }
            else // Factorization failed; increase shift and retry
            {
                ++num_plan.ic_info.restarts;
                shift *= ichol::util::cast_fp_type<T, double>(options.shift_growth);
                ichol::numeric::add_diagonal_shift<T>(A_work, shift);
            }
        }

        // Max restarts reached; abort
        throw std::runtime_error("Incomplete Cholesky factorization failed: maximum restarts reached.");
    }

    template ichol::matrix::CsrMatrix<double> incomplete_cholesky_preconditioner<double>(ichol::matrix::CsrMatrix<double> &A,
                                                                                         ichol::symbolic::SymbolicPlan &sym_plan,
                                                                                         ichol::numeric::NumericPlan &num_plan,
                                                                                         ichol::IncompleteCholeskyOptions &options);

    template ichol::matrix::CsrMatrix<float> incomplete_cholesky_preconditioner<float>(ichol::matrix::CsrMatrix<double> &A,
                                                                                       ichol::symbolic::SymbolicPlan &sym_plan,
                                                                                       ichol::numeric::NumericPlan &num_plan,
                                                                                       ichol::IncompleteCholeskyOptions &options);
    template ichol::matrix::CsrMatrix<half_float::half> incomplete_cholesky_preconditioner<half_float::half>(ichol::matrix::CsrMatrix<double> &A,
                                                                                                             ichol::symbolic::SymbolicPlan &sym_plan,
                                                                                                             ichol::numeric::NumericPlan &num_plan,
                                                                                                             ichol::IncompleteCholeskyOptions &options);

} // namespace ichol::numeric