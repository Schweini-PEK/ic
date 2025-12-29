#include "symbolic.hpp"
#include "ichol/half.hpp"

namespace ichol::symbolic
{
    template <typename T>
    SymbolicPlan ic_analyze(const ichol::matrix::CsrMatrix<T> &A,
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
    SupernodalSymbolicPlan
    supernodal_analyze(const ichol::matrix::CscMatrix<T> &A,
                       const SuperNodeOptions &sn_options)
    {
        SupernodalSymbolicPlan plan;

        // 1. Elimination tree (CSC-native)
        plan.etree = build_etree<T>(A);

        // 2. Complete Cholesky pattern (CSC-native)
        plan.factor_pattern =
            compute_complete_cholesky_pattern<T>(A, plan.etree);

        // 3. Detect supernodes
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
            plan.snodes = detect_supernodes(
                plan.factor_pattern,
                plan.etree
            );
        }

        // 4. Column -> supernode mapping
        int ncols = A.num_cols;
        plan.col2snode = build_col2snode(plan.snodes, ncols);

        // 5. Column-level scheduling
        // reuse existing implementation
        SymbolicOptions dummy;
        auto col_level_sets =
            build_level_sets(plan.factor_pattern, dummy);

        // 6. Supernode-level scheduling
        plan.snode_level_sets =
            build_snode_level_sets(col_level_sets, plan.snodes);

        return plan;
    }


    template SymbolicPlan ic_analyze<double>(const ichol::matrix::CsrMatrix<double> &A,
                                             const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<float>(const ichol::matrix::CsrMatrix<float> &A,
                                            const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<half_float::half>(const ichol::matrix::CsrMatrix<half_float::half> &A,
                                                       const SymbolicOptions &options);
}