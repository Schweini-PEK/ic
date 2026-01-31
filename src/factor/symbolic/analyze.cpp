#include "factor/symbolic/symbolic.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"

extern "C" {
#include <cholmod.h>
}

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

        {
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

// ----------------------------------------------------------------------------
// Supernodal (LL) symbolic using CHOLMOD.
//
// Goal: avoid re-implementing CHOLMOD internals (supernodes/rowlists/etc.).
// We let CHOLMOD:
//   1) choose/apply an ordering (or accept a user-provided one),
//   2) build the supernodal symbolic (L->super, L->pi, L->px, L->s, ...).
//
// For our numeric (CPU/GPU) code we additionally keep a permuted copy of the
// input matrix A in CSC form:  A := P*A*P^T (lower triangle).  This is done via
// CHOLMOD's permutation routines, not by our own CSC reindexing code.
// ----------------------------------------------------------------------------

namespace ichol::symbolic
{
    template <typename T>
    SupernodalLLPlan supernodal_ll_analyze_fast(matrix::CscMatrix<T>& A,
                                                const SuperNodeOptions& /*sn_options*/,
                                                const SymbolicOptions& options)
    {
        if (A.num_cols != A.num_rows)
            throw std::runtime_error("supernodal_ll_analyze_fast: A must be square");

        const int n = A.num_cols;
        SupernodalLLPlan plan;

        // ---- 1) CHOLMOD analyze (ordering + supernodal symbolic) ----
        cholmod_common cc;
        cholmod_start(&cc);
        cc.itype = CHOLMOD_LONG;
        cc.dtype = CHOLMOD_DOUBLE; // symbolic uses pattern only; keep a stable default
        cc.nmethods = 1;

        // Keep the style consistent with ic_analyze: choose ordering, then apply it.
        // NOTE: CHOLMOD has no native RCM ordering; for RCM we compute perm externally
        // and provide it via cholmod_analyze_p.
        switch (options.ordering)
        {
        case Ordering::Identity:
            cc.method[0].ordering = CHOLMOD_NATURAL;
            break;
        case Ordering::AMD:
            cc.method[0].ordering = CHOLMOD_AMD;
            break;
        case Ordering::NestedDissection:
            cc.method[0].ordering = CHOLMOD_NESDIS;
            break;
        case Ordering::RCM:
            cc.method[0].ordering = CHOLMOD_GIVEN;
            break;
        }

        cc.postorder = 0;
        cc.supernodal = CHOLMOD_SUPERNODAL;
        cc.supernodal_switch = 0;
        cc.final_ll = 1;
        cc.final_super = 1;
        cc.final_asis = 0;

        // Build a CHOLMOD sparse view (pattern-only) of our lower-triangular CSC.
        const int nnz = (int)A.row_ind.size();
        cholmod_sparse* S = cholmod_allocate_sparse(
            (size_t)n, (size_t)n, (size_t)nnz,
            /*sorted=*/1,
            /*packed=*/1,
            /*stype=*/-1,
            CHOLMOD_PATTERN,
            &cc);
        if (!S)
        {
            cholmod_finish(&cc);
            throw std::runtime_error("supernodal_ll_analyze_fast: cholmod_allocate_sparse failed");
        }

        auto* Sp = reinterpret_cast<SuiteSparse_long*>(S->p);
        auto* Si = reinterpret_cast<SuiteSparse_long*>(S->i);
        for (int j = 0; j < n + 1; ++j) Sp[(std::size_t)j] = (SuiteSparse_long)A.col_ptr[(std::size_t)j];
        for (int p = 0; p < nnz; ++p)   Si[(std::size_t)p] = (SuiteSparse_long)A.row_ind[(std::size_t)p];

        cholmod_factor* L = nullptr;
        std::vector<SuiteSparse_long> user_perm_long;

        if (options.ordering == Ordering::RCM)
        {
            const auto P = rcm_from_csc(n, A.col_ptr, A.row_ind);
            user_perm_long.resize((std::size_t)n);
            for (int k = 0; k < n; ++k) user_perm_long[(std::size_t)k] = (SuiteSparse_long)P.perm[(std::size_t)k];
            L = cholmod_l_analyze_p(S, user_perm_long.data(), /*fset=*/nullptr, /*fsize=*/0, &cc);
        }
        else
        {
            L = cholmod_l_analyze(S, &cc);
        }

        if (!L || !L->Perm)
        {
            cholmod_free_factor(&L, &cc);
            cholmod_free_sparse(&S, &cc);
            cholmod_finish(&cc);
            throw std::runtime_error("supernodal_ll_analyze_fast: cholmod_analyze failed");
        }

        // ---- 2) Extract permutation (CHOLMOD convention: perm[new] = old) ----
        const auto* Perm = reinterpret_cast<const SuiteSparse_long*>(L->Perm);
        plan.perm.perm.assign((std::size_t)n, 0);
        plan.perm.inv_perm.assign((std::size_t)n, 0);
        bool is_identity = true;
        for (int k = 0; k < n; ++k)
        {
            const int orig = (int)Perm[(std::size_t)k];
            plan.perm.perm[(std::size_t)k] = orig;
            plan.perm.inv_perm[(std::size_t)orig] = k;
            is_identity = is_identity && (orig == k);
        }

        // ---- 3) Apply ordering to A (in-place) ----
        if (!is_identity)
        {
            {
                ichol::symbolic::apply_symmetric_permutation_csc_lower_inplace(A, plan.perm);
            }
            else
            {
                ichol::symbolic::apply_symmetric_permutation_csc_lower_inplace(A, plan.perm);
            }
        }

        // ---- 4) Copy CHOLMOD supernodal symbolic (rowlists, block offsets) ----
        if (!L->is_super || !L->super || !L->pi || !L->px || !L->s)
        {
            cholmod_free_factor(&L, &cc);
            cholmod_free_sparse(&S, &cc);
            cholmod_finish(&cc);
            throw std::runtime_error("supernodal_ll_analyze_fast: CHOLMOD did not produce a supernodal factor");
        }

        const int nsuper = (int)L->nsuper;
        const auto* super = reinterpret_cast<const SuiteSparse_long*>(L->super);
        const auto* pi    = reinterpret_cast<const SuiteSparse_long*>(L->pi);
        const auto* px    = reinterpret_cast<const SuiteSparse_long*>(L->px);
        const auto* s     = reinterpret_cast<const SuiteSparse_long*>(L->s);

        plan.sym.super.assign((std::size_t)nsuper + 1, 0);
        plan.sym.pi.assign((std::size_t)nsuper + 1, 0);
        plan.sym.px.assign((std::size_t)nsuper + 1, 0);
        for (int k = 0; k < nsuper + 1; ++k)
        {
            plan.sym.super[(std::size_t)k] = (int)super[(std::size_t)k];
            plan.sym.pi[(std::size_t)k]    = (int)pi[(std::size_t)k];
            plan.sym.px[(std::size_t)k]    = (int)px[(std::size_t)k];
        }

        const int ns = plan.sym.pi.back();
        plan.sym.s.assign((std::size_t)ns, 0);
        for (int t = 0; t < ns; ++t) plan.sym.s[(std::size_t)t] = (int)s[(std::size_t)t];

        // ---- 5) Build execution schedule for CPU/GPU numeric ----
        fill_schedule_from_sym(plan, n);

        cholmod_free_factor(&L, &cc);
        cholmod_free_sparse(&S, &cc);
        cholmod_finish(&cc);

        return plan;
    }

    template SupernodalLLPlan supernodal_ll_analyze_fast<double>(matrix::CscMatrix<double>&,
                                                                 const SuperNodeOptions&,
                                                                 const SymbolicOptions&);
    template SupernodalLLPlan supernodal_ll_analyze_fast<float>(matrix::CscMatrix<float>&,
                                                                const SuperNodeOptions&,
                                                                const SymbolicOptions&);
}