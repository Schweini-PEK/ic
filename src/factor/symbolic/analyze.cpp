#include "symbolic.hpp"
#include "ichol/half.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"
namespace ichol::symbolic
{
    // NOTE: Supernodal LL pipeline entrypoint is `supernodal_ll_analyze`
    // in `src/factor/symbolic/supernodal_ll_plan.hpp` (returns SupernodalLLPlan).

    template <typename T>
    SymbolicPlan ic_analyze(const ichol::matrix::CsrMatrix<T> &A,
                            const SymbolicOptions &options)
    {
        SymbolicPlan plan;

        if (options.level_k == -1) // Complete Cholesky
        {
            // CHOLMOD-aligned etree + colcount
            plan.etree = build_etree<T>(A);

            // Pattern of complete Cholesky using that etree
            plan.factor_pattern = compute_complete_cholesky_pattern<T>(A, plan.etree);
        }
        else // IC(k)
        {
            // IC(k) pattern doesn't need etree
            plan.factor_pattern = compute_ic_factor_pattern<T>(A, options.level_k);
        }

        plan.level_sets = build_level_sets(plan.factor_pattern, options);
        return plan;
    }

    template SymbolicPlan ic_analyze<double>(const ichol::matrix::CsrMatrix<double> &A,
                                             const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<float>(const ichol::matrix::CsrMatrix<float> &A,
                                            const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<half_float::half>(const ichol::matrix::CsrMatrix<half_float::half> &A,
                                                       const SymbolicOptions &options);
} // namespace ichol::symbolic
