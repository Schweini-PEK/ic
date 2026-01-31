#include "factor/symbolic/symbolic.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"

#include <stdexcept>
#include <vector>

extern "C" {
#include <cholmod.h>
}

namespace ichol::symbolic
{

// ----------------------------------------------------------------------------
// IC / IC(k) symbolic: ordering + permutation + pattern
// ----------------------------------------------------------------------------

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
    default:
        plan.perm = identity_permutation(A.num_rows);
        break;
    }

    ichol::symbolic::apply_permutation_csr<T>(A, plan.perm);

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

template SymbolicPlan ic_analyze<double>(ichol::matrix::CsrMatrix<double> &A,
                                         const SymbolicOptions &options);
template SymbolicPlan ic_analyze<float>(ichol::matrix::CsrMatrix<float> &A,
                                        const SymbolicOptions &options);
template SymbolicPlan ic_analyze<half_float::half>(ichol::matrix::CsrMatrix<half_float::half> &A,
                                                   const SymbolicOptions &options);

// ----------------------------------------------------------------------------
// Supernodal (LL) symbolic using CHOLMOD.
// ----------------------------------------------------------------------------

template <typename T>
SupernodalLLPlan supernodal_ll_analyze_fast(matrix::CscMatrix<T> &A,
                                            const SuperNodeOptions & /*sn_options*/,
                                            const SymbolicOptions &options)
{
    if (A.num_cols != A.num_rows)
        throw std::runtime_error("supernodal_ll_analyze_fast: A must be square");

    const int n = A.num_cols;
    SupernodalLLPlan plan;

    // 1) CHOLMOD analyze: choose ordering and build supernodal symbolic
    cholmod_common cc;
    cholmod_start(&cc);
    // Keep CHOLMOD in its default 32-bit index mode (CHOLMOD_INT) to match our
    // internal index storage (std::vector<int>) and to remain compatible with
    // SuiteSparse builds that are not configured for 64-bit indices.
    cc.itype = CHOLMOD_INT;
    cc.nmethods = 1;

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
        cc.method[0].ordering = CHOLMOD_GIVEN; // we provide permutation
        break;
    default:
        cc.method[0].ordering = CHOLMOD_NATURAL;
        break;
    }

    cc.postorder = 0;
    cc.supernodal = CHOLMOD_SUPERNODAL;
    cc.supernodal_switch = 0;
    cc.final_ll = 1;
    cc.final_super = 1;
    cc.final_asis = 0;

    const int nnz = (int)A.row_ind.size();
    cholmod_sparse *S = cholmod_allocate_sparse(
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

    auto *Sp = reinterpret_cast<int *>(S->p);
    auto *Si = reinterpret_cast<int *>(S->i);
    for (int j = 0; j < n + 1; ++j)
        Sp[(size_t)j] = (int)A.col_ptr[(size_t)j];
    for (int p = 0; p < nnz; ++p)
        Si[(size_t)p] = (int)A.row_ind[(size_t)p];

    cholmod_factor *L = nullptr;
    std::vector<int> user_perm;

    if (options.ordering == Ordering::RCM)
    {
        const auto P = rcm_from_csc(n, A.col_ptr, A.row_ind);
        user_perm.resize((size_t)n);
        for (int k = 0; k < n; ++k)
            user_perm[(size_t)k] = (int)P.perm[(size_t)k];
        L = cholmod_analyze_p(S, user_perm.data(), /*fset=*/nullptr, /*fsize=*/0, &cc);
    }
    else
    {
        L = cholmod_analyze(S, &cc);
    }

    if (!L || !L->Perm)
    {
        cholmod_free_factor(&L, &cc);
        cholmod_free_sparse(&S, &cc);
        cholmod_finish(&cc);
        throw std::runtime_error("supernodal_ll_analyze_fast: cholmod_analyze failed");
    }

    // 2) Extract permutation (CHOLMOD convention: perm[new] = old)
    const auto *Perm = reinterpret_cast<const int *>(L->Perm);
    plan.perm.perm.assign((size_t)n, 0);
    plan.perm.inv_perm.assign((size_t)n, 0);

    bool is_identity = true;
    for (int k = 0; k < n; ++k)
    {
        const int orig = (int)Perm[(size_t)k];
        plan.perm.perm[(size_t)k] = orig;
        plan.perm.inv_perm[(size_t)orig] = k;
        is_identity = is_identity && (orig == k);
    }

    // 3) Apply ordering to A (in-place) so numeric consumes permuted matrix
    if (!is_identity)
        apply_symmetric_permutation_csc_lower_inplace(A, plan.perm);

    // 4) Copy CHOLMOD supernodal symbolic (rowlists, block offsets)
    if (!L->is_super || !L->super || !L->pi || !L->px || !L->s)
    {
        cholmod_free_factor(&L, &cc);
        cholmod_free_sparse(&S, &cc);
        cholmod_finish(&cc);
        throw std::runtime_error("supernodal_ll_analyze_fast: CHOLMOD did not produce a supernodal factor");
    }

    const int nsuper = (int)L->nsuper;
    const auto *super = reinterpret_cast<const int *>(L->super);
    const auto *pi = reinterpret_cast<const int *>(L->pi);
    const auto *px = reinterpret_cast<const int *>(L->px);
    const auto *s = reinterpret_cast<const int *>(L->s);

    plan.sym.super.assign((size_t)nsuper + 1, 0);
    plan.sym.pi.assign((size_t)nsuper + 1, 0);
    plan.sym.px.assign((size_t)nsuper + 1, 0);
    for (int k = 0; k < nsuper + 1; ++k)
    {
        plan.sym.super[(size_t)k] = (int)super[(size_t)k];
        plan.sym.pi[(size_t)k] = (int)pi[(size_t)k];
        plan.sym.px[(size_t)k] = (int)px[(size_t)k];
    }

    const int ns = plan.sym.pi.back();
    plan.sym.s.assign((size_t)ns, 0);
    for (int t = 0; t < ns; ++t)
        plan.sym.s[(size_t)t] = (int)s[(size_t)t];

    // 5) Build execution schedule for CPU/GPU numeric
    fill_schedule_from_sym(plan, n);

    cholmod_free_factor(&L, &cc);
    cholmod_free_sparse(&S, &cc);
    cholmod_finish(&cc);

    return plan;
}

template SupernodalLLPlan supernodal_ll_analyze_fast<double>(matrix::CscMatrix<double> &,
                                                             const SuperNodeOptions &,
                                                             const SymbolicOptions &);
template SupernodalLLPlan supernodal_ll_analyze_fast<float>(matrix::CscMatrix<float> &,
                                                            const SuperNodeOptions &,
                                                            const SymbolicOptions &);

} // namespace ichol::symbolic
