#include "factor/symbolic/symbolic.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"

#include <stdexcept>
#include <vector>
#include <type_traits>
#include <iostream>

extern "C"
{
#include <cholmod.h>
}

namespace
{
    template <typename T>
    ichol::symbolic::Permutation get_permutation_csr(ichol::matrix::CsrMatrix<T> &A, const ichol::Ordering ordering)
    {
        switch (ordering)
        {
        case ichol::Ordering::Identity:
            return ichol::symbolic::identity_permutation(A.num_rows);
        case ichol::Ordering::AMD:
            return ichol::symbolic::amd_from_csr(A.num_rows, A.row_ptr, A.col_ind);
        case ichol::Ordering::NestedDissection:
            return ichol::symbolic::nd_from_csr(A.num_rows, A.row_ptr, A.col_ind);
        case ichol::Ordering::RCM:
            return ichol::symbolic::rcm_from_csr(A.num_rows, A.row_ptr, A.col_ind);
        default:
            return ichol::symbolic::identity_permutation(A.num_rows);
        }
    }

    template <typename T>
    ichol::symbolic::Permutation get_permutation_csc(ichol::matrix::CscMatrix<T> &A, const ichol::Ordering ordering)
    {
        switch (ordering)
        {
        case ichol::Ordering::Identity:
            return ichol::symbolic::identity_permutation(A.num_cols);
        case ichol::Ordering::AMD:
            return ichol::symbolic::amd_from_csc(A.num_cols, A.col_ptr, A.row_ind);
        case ichol::Ordering::NestedDissection:
            return ichol::symbolic::nd_from_csc(A.num_cols, A.col_ptr, A.row_ind);
        case ichol::Ordering::RCM:
            return ichol::symbolic::rcm_from_csc(A.num_cols, A.col_ptr, A.row_ind);
        default:
            return ichol::symbolic::identity_permutation(A.num_cols);
        }
    }
}

namespace ichol::symbolic
{
    /**
     * @brief Perform ordering, in-place permutation and compute symbolic plan for IC/IC(k)
     *
     * @param A Input matrix in CSR, lower triangular, diagonal present. Each row should be sorted, no duplicates. No validity checks.
     * @param options Symbolic options (ordering, level_k, etc).
     * @return SymbolicPlan containing permutation, etree, factor pattern, and level sets.
     */
    template <typename T>
    SymbolicPlan ic_analyze(ichol::matrix::CsrMatrix<T> &A,
                            const SymbolicOptions &options)
    {
        SymbolicPlan plan;  //封装符号分析阶段用到的矩阵结构信息（重排序需要的置换向量、消去树、L的非零结构、层级关系）

        plan.perm = get_permutation_csr(A, options.ordering);   //构造重排序的置换向量
        apply_permutation_csr<T>(A, plan.perm); //对原矩阵A进行重排序，并更新存储

        if (options.level_k == -1) // Complete Cholesky
        {
            plan.etree = build_etree<T>(A);
            plan.factor_pattern = compute_complete_cholesky_pattern<T>(A, plan.etree);
        }
        else // IC(k)
        {
            plan.factor_pattern = compute_ic_factor_pattern<T>(A, options.level_k); //计算IC分解会用到的结构信息（填充量、非零位置、etc）
        }

        plan.level_sets = build_level_sets(plan.factor_pattern, options);   //构建层级关系
        return plan;
    }

    template SymbolicPlan ic_analyze<double>(ichol::matrix::CsrMatrix<double> &A,
                                             const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<float>(ichol::matrix::CsrMatrix<float> &A,
                                            const SymbolicOptions &options);
    template SymbolicPlan ic_analyze<half_float::half>(ichol::matrix::CsrMatrix<half_float::half> &A,
                                                       const SymbolicOptions &options);

    /**
     * @brief Perform symbolic analysis for supernodal symbolic left-looking Cholesky via CHOLMOD.
     */
    template <typename T>
    SupernodalLLPlan supernodal_analyze(matrix::CscMatrix<T> &A,
                                        const SymbolicOptions &options)
    {
        SupernodalLLPlan supernodal_plan;

        supernodal_plan.perm = get_permutation_csc(A, options.ordering);
        apply_permutation_csc<T>(A, supernodal_plan.perm);

        const int n = (int)A.col_ptr.size() - 1;

        cholmod_common cc;
        cholmod_start(&cc);
        cc.itype = CHOLMOD_INT;
        cc.nmethods = 1;
        cc.method[0].ordering = CHOLMOD_NATURAL;
        cc.postorder = 1;
        cc.supernodal = CHOLMOD_SUPERNODAL;
        cc.supernodal_switch = 0;

        const int nnz = (int)A.row_ind.size();
        cholmod_sparse *S = cholmod_allocate_sparse(
            (size_t)n, (size_t)n, (size_t)nnz,
            /*sorted=*/1, // each column is sorted by row indices, no duplicates
            /*packed=*/1,
            /*stype=*/-1, // lower triangular for a symmetric pattern
            CHOLMOD_PATTERN,
            &cc);

        auto *Sp = reinterpret_cast<int *>(S->p);
        auto *Si = reinterpret_cast<int *>(S->i);
        for (int j = 0; j < n + 1; ++j)
            Sp[(size_t)j] = (int)A.col_ptr[(size_t)j];
        for (int p = 0; p < nnz; ++p)
            Si[(size_t)p] = (int)A.row_ind[(size_t)p];

        cholmod_factor *L = nullptr;
        L = cholmod_analyze(S, &cc);

        // Copy CHOLMOD supernodal symbolic (rowlists, block offsets)
        const int nsuper = (int)L->nsuper;
        const auto *super = reinterpret_cast<const int *>(L->super);
        const auto *pi = reinterpret_cast<const int *>(L->pi);
        const auto *px = reinterpret_cast<const int *>(L->px);
        const auto *s = reinterpret_cast<const int *>(L->s);

        supernodal_plan.sym.super.assign((size_t)nsuper + 1, 0);
        supernodal_plan.sym.pi.assign((size_t)nsuper + 1, 0);
        supernodal_plan.sym.px.assign((size_t)nsuper + 1, 0);
        for (int k = 0; k < nsuper + 1; ++k)
        {
            supernodal_plan.sym.super[(size_t)k] = (int)super[(size_t)k];
            supernodal_plan.sym.pi[(size_t)k] = (int)pi[(size_t)k];
            supernodal_plan.sym.px[(size_t)k] = (int)px[(size_t)k];
        }

        const int ns = supernodal_plan.sym.pi.back();
        supernodal_plan.sym.s.assign((size_t)ns, 0);
        for (int t = 0; t < ns; ++t)
            supernodal_plan.sym.s[(size_t)t] = (int)s[(size_t)t];

        // 5) Build execution schedule for CPU/GPU numeric
        fill_schedule_from_sym(supernodal_plan, n);

        cholmod_free_factor(&L, &cc);
        cholmod_free_sparse(&S, &cc);
        cholmod_finish(&cc);

        return supernodal_plan;
    }

    template SupernodalLLPlan supernodal_analyze<double>(matrix::CscMatrix<double> &,
                                                         const SymbolicOptions &);
    template SupernodalLLPlan supernodal_analyze<float>(matrix::CscMatrix<float> &,
                                                        const SymbolicOptions &);

} // namespace ichol::symbolic
