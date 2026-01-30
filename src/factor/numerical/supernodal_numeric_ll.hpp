#pragma once
#include <vector>
#include <utility>

#include "factor/symbolic/super_sym.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "ichol/matrix_formats.hpp"

namespace ichol::numeric {

    // Options for CUDA supernodal numeric factorization (single GPU, multi-stream).
    // Defaults follow the IA3'2014/CHOLMOD-style setting commonly used in practice.
    struct CudaSupernodalOptions {
        int device = 0;
        int streams = 16;              // default stream count (paper-style)
        bool verbose = false;          // print streams/work distribution
        bool print_schedule = false;   // print level buckets + mapping preview
        int schedule_print_limit = 8;  // how many snodes to preview per level
    };


    struct SuperNumeric
    {
        symbolic::SuperSym sym;
        std::vector<double> x;  // size = sym.px
        bool ok = true;
        int fail_snode = -1;
        int fail_col_in_snode = -1;
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

    // CUDA numeric phase with explicit options.
    SuperNumeric factorize_supernodal_ll_cuda(
        const ichol::matrix::CscMatrix<double>& A,
        const symbolic::SupernodalLLPlan& plan,
        const CudaSupernodalOptions& opt);

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

} // namespace ichol::numeric
