#include "factor/symbolic/symbolic.hpp"

namespace ichol::symbolic
{
    template <typename T>
    SymbolicPlan ic_analyze(ichol::matrix::CsrMatrix<T> &A,
                            const SymbolicOptions &options)
    {
        SymbolicPlan plan;

        switch (options.ordering)
        {
        case ichol::Ordering::Identity:
            plan.perm = identity_permutation(A.num_rows);
            break;

        case ichol::Ordering::AMD:
            plan.perm = amd_from_csr(A.num_rows, A.row_ptr, A.col_ind);
            ichol::symbolic::apply_permutation_csr<T>(A, plan.perm);
            break;
        }

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
    SymbolicPlan supernodal_analyze(const ichol::matrix::CscMatrix<T> &A,
                                    const SuperNodeOptions &sn_options)
    {
        SymbolicPlan plan;

        // To be implemented: supernodal analysis
        // This is a placeholder for future implementation.

        return plan;
    }

    template SymbolicPlan ic_analyze<double>(ichol::matrix::CsrMatrix<double> &A,
                                             const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<float>(ichol::matrix::CsrMatrix<float> &A,
                                            const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<half_float::half>(ichol::matrix::CsrMatrix<half_float::half> &A,
                                                       const SymbolicOptions &options);
}