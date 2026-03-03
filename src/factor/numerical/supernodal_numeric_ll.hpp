#pragma once
#include <vector>
#include <utility>

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

        // Fast path: CHOLMOD typically assumes SPD for LL; if your input may be
        // indefinite, set this to false to keep per-supernode info checks.
        // When true, we avoid a per-supernode stream sync on POTRF, which is
        // a major overhead on workloads with many small fronts.
        bool assume_spd = true;

        // Heuristic: for very small supernodes, GPU POTRF/TRSM launch/setup
        // overhead can dominate. If nscol <= cpu_fallback_max_n, we perform
        // POTRF/TRSM/SYRK on CPU (simple unblocked kernels) and only keep the
        // GPU path for larger fronts. Set to 0 to disable.
        int cpu_fallback_max_n = 48;
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
