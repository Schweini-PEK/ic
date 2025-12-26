#include "symbolic.hpp"
#include "ichol/half.hpp"

namespace ichol::symbolic
{
    template <typename T>
    SymbolicPlan ic_analyze(const ichol::CsrMatrix<T> &A,
                            const SymbolicOptions &options)
    {
        SymbolicPlan plan;

        if (options.level_k == -1) // Complete Cholesky
        {
            plan.etree = build_etree<T>(A);
            plan.factor_pattern = compute_complete_cholesky_pattern<T>(A, plan.etree);
        }
        else // IC(k)
        {
            plan.factor_pattern = compute_ic_factor_pattern<T>(A, options.level_k);
        }

        plan.level_sets = build_level_sets(plan.factor_pattern, options);

        return plan;
    }

    template <typename T>
    SymbolicPlan supernodal_analyze(const ichol::CscMatrix<T> &A,
                                    const SuperNodeOptions &sn_options)
    {
        SymbolicPlan plan;

        // To be implemented: supernodal analysis
        // This is a placeholder for future implementation.

        return plan;
    }

    template SymbolicPlan ic_analyze<double>(const ichol::CsrMatrix<double> &A,
                                             const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<float>(const ichol::CsrMatrix<float> &A,
                                            const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<half_float::half>(const ichol::CsrMatrix<half_float::half> &A,
                                                       const SymbolicOptions &options);
}