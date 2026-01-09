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
    SupernodalSymbolicPlan
    supernodal_analyze(const ichol::matrix::CscMatrix<T> &A,
                       const SuperNodeOptions &sn_options)
    {
        SupernodalSymbolicPlan plan;

        // 1) Elimination tree (CHOLMOD-aligned, CSC-native)
        plan.etree = build_etree<T>(A);

        // 2) Complete Cholesky pattern (CSC-native)
        plan.factor_pattern = compute_complete_cholesky_pattern<T>(A, plan.etree);

        // 3) Detect supernodes
        if (sn_options.approximate)
        {
            plan.snodes = detect_supernodes_approx(
                plan.factor_pattern,
                plan.etree,
                sn_options.overlap_threshold
            );
        }
        else
        {
            // By default: CHOLMOD super_symbolic behavior (relaxed amalgamation)
            // If you want to compare fundamental supernodes (relax=off),
            // compile with -DICHOL_SUPERNODES_FUNDAMENTAL
        #ifdef ICHOL_SUPERNODES_FUNDAMENTAL
                    plan.snodes = detect_supernodes_fundamental(plan.etree);
        #else
                    plan.snodes = detect_supernodes(plan.factor_pattern, plan.etree);
        #endif
        }

        // 4) Column -> supernode mapping
        const int ncols = A.num_cols;
        plan.col2snode = build_col2snode(plan.snodes, ncols);

        // 5) Column-level scheduling (reuse existing implementation)
        SymbolicOptions dummy;
        auto col_level_sets = build_level_sets(plan.factor_pattern, dummy);

        // 6) Supernode-level scheduling
        plan.snode_level_sets = build_snode_level_sets(col_level_sets, plan.snodes);

        return plan;
    }

    template SymbolicPlan ic_analyze<double>(ichol::matrix::CsrMatrix<double> &A,
                                             const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<float>(ichol::matrix::CsrMatrix<float> &A,
                                            const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<half_float::half>(ichol::matrix::CsrMatrix<half_float::half> &A,
                                                       const SymbolicOptions &options);
} // namespace ichol::symbolic
