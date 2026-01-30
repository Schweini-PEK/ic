#include "factor/symbolic/symbolic.hpp"

namespace ichol::symbolic
{
    // NOTE: Supernodal LL pipeline entrypoint is `supernodal_ll_analyze`
    // in `src/factor/symbolic/supernodal_ll_plan.hpp` (returns SupernodalLLPlan).

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
            break;

        case ichol::Ordering::NestedDissection:
            plan.perm = nd_from_csr(A.num_rows, A.row_ptr, A.col_ind);
            break;

        case ichol::Ordering::RCM:
            plan.perm = rcm_from_csr(A.num_rows, A.row_ptr, A.col_ind);
            break;
        }

        if (options.profile)
        {
            ichol::util::ScopedTimer timer("Apply permutation to CSR", options.timer_sink);
            ichol::symbolic::apply_permutation_csr<T>(A, plan.perm);
        }
        else
        {
            ichol::symbolic::apply_permutation_csr<T>(A, plan.perm);
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

        plan.level_sets = build_level_sets(plan.factor_pattern);

        return plan;
    }


    template SymbolicPlan ic_analyze<double>(ichol::matrix::CsrMatrix<double> &A,
                                             const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<float>(ichol::matrix::CsrMatrix<float> &A,
                                            const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<half_float::half>(ichol::matrix::CsrMatrix<half_float::half> &A,
                                                       const SymbolicOptions &options);
} // namespace ichol::symbolic
