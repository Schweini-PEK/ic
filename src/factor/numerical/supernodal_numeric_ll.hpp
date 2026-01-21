#pragma once
#include <vector>
#include <utility>

#include "factor/symbolic/super_sym.hpp" // shim -> canonical symbolic SuperSym
#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "ichol/matrix_formats.hpp"

namespace ichol::numeric {



    struct SuperNumeric
    {
        symbolic::SuperSym sym;
        std::vector<double> x;  // size = sym.px.back(), CHOLMOD block layout
        bool ok = true;
        int fail_snode = -1;
        int fail_col_in_snode = -1;
        // ---- Step4 debug/trace ----
        int threads_used = 1;              // 本次 factorize 实际使用的线程数（OpenMP）
        std::vector<int> thread_work;      // thread_work[tid] = 该线程处理的 supernode 数

    };

    // Numeric phase ONLY: consumes a pre-built symbolic plan.
    SuperNumeric factorize_supernodal_ll(
        const matrix::CscMatrix<double>& A,
        const symbolic::SupernodalLLPlan& plan);

    // Numeric phase ONLY: consumes a pre-built symbolic plan.
    SuperNumeric factorize_supernodal_ll_cuda(
        const ichol::matrix::CscMatrix<double>& A,
        const symbolic::SupernodalLLPlan& plan);

    // -------------------------------------------------------------------------
    // Backward-compatible overloads (discouraged).
    // They wrap the provided SuperSym into a SupernodalLLPlan so that all
    // symbolic logic remains centralized and identical.
    // -------------------------------------------------------------------------
    inline SuperNumeric factorize_supernodal_ll(
        const matrix::CscMatrix<double>& A,
        const symbolic::SuperSym& sym)
    {
        return factorize_supernodal_ll(A, symbolic::ll_plan_from_sym(sym, A.num_cols));
    }

    inline SuperNumeric factorize_supernodal_ll_cuda(
        const ichol::matrix::CscMatrix<double>& A,
        const symbolic::SuperSym& sym)
    {
        return factorize_supernodal_ll_cuda(A, symbolic::ll_plan_from_sym(sym, A.num_cols));
    }

} // namespace ichol::symbolic
