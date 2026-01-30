#include "factor/numerical/cuda/preconditioner.hpp"
#include "factor/numerical/cuda/detail/ickdt_factorize_impl.hpp"
#include "factor/numerical/cuda/detail/compress_L.hpp"

namespace ichol::numeric::cuda
{
    template <class T>
    ichol::matrix::CsrMatrix<T> ickdt(const ichol::matrix::CsrMatrix<T> &Ahost,
                                      const ichol::symbolic::SymbolicPlan &sym_plan,
                                      ichol::numeric::NumericPlan &out_plan,
                                      const ichol::IncompleteCholeskyOptions &options)
    {
        using G =
            std::conditional_t<std::is_same<T, double>::value, double,
                               std::conditional_t<std::is_same<T, float>::value, float,
                                                  __half>>;

        std::vector<int> L_row_ptr_fixed, L_col_ind_fixed;
        std::vector<G> L_val_fixed;
        int fail_row = -1;

        try
        {
            icdtk_gpu<T, G>(
                Ahost,
                sym_plan.factor_pattern,
                sym_plan.level_sets,
                L_row_ptr_fixed,
                L_col_ind_fixed,
                L_val_fixed,
                fail_row,
                options);

            if (fail_row >= 0)
            {
                out_plan.ic_info.code = ichol::numeric::ICBreakdown::B1_SmallOrNegativePivot;
                out_plan.ic_info.step = fail_row;
                return ichol::matrix::CsrMatrix<T>{};
            }

            ichol::matrix::CsrMatrix<T> Lhost = ichol::numeric::util::compress_fixed_pattern_L<T, G>(
                Ahost.num_rows, L_row_ptr_fixed, L_col_ind_fixed, L_val_fixed);

            out_plan.ic_info.code = ichol::numeric::ICBreakdown::None;
            out_plan.ic_info.step = 0;
            return Lhost;
        }
        catch (...)
        {
            out_plan.ic_info.code = ichol::numeric::ICBreakdown::OtherNumericalIssue;
            out_plan.ic_info.step = 0;
            throw;
        }
    }

    template ichol::matrix::CsrMatrix<double> ickdt(const ichol::matrix::CsrMatrix<double> &Ahost,
                                                    const ichol::symbolic::SymbolicPlan &sym_plan,
                                                    ichol::numeric::NumericPlan &out_plan,
                                                    const ichol::IncompleteCholeskyOptions &options);

    template ichol::matrix::CsrMatrix<float> ickdt(const ichol::matrix::CsrMatrix<float> &Ahost,
                                                   const ichol::symbolic::SymbolicPlan &sym_plan,
                                                   ichol::numeric::NumericPlan &out_plan,
                                                   const ichol::IncompleteCholeskyOptions &options);

    template ichol::matrix::CsrMatrix<half_float::half> ickdt(const ichol::matrix::CsrMatrix<half_float::half> &Ahost,
                                                              const ichol::symbolic::SymbolicPlan &sym_plan,
                                                              ichol::numeric::NumericPlan &out_plan,
                                                              const ichol::IncompleteCholeskyOptions &options);
} // namespace ichol::numeric::cuda