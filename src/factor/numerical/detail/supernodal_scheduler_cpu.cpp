#include "factor/numerical/detail/supernodal_numeric_ll_internal.hpp"

#if defined(ICHOL_USE_OPENMP)
  #include <omp.h>
#endif

namespace ichol::numeric::detail {

void schedule_levels_cpu(
    const matrix::CscMatrix<double>& A,
    const symbolic::SupernodalLLPlan& plan,
    std::vector<UpdatePack>& up,
    std::vector<double>& x,
    std::atomic<bool>& ok,
    std::atomic<int>& fail_snode,
    std::atomic<int>& fail_col,
    int max_front,
    int& threads_used,
    std::vector<int>& thread_work)
{
    const auto& buckets = plan.buckets;
    const int maxL = (int)buckets.size() - 1;

#if defined(ICHOL_USE_OPENMP)
    // OpenMP schedule: parallel within each level bucket
    const int max_threads = omp_get_max_threads();
    std::vector<std::atomic<int>> work_atomic((size_t)std::max(1, max_threads));
    for (auto& a : work_atomic) a.store(0);

    std::atomic<int> threads_used_atomic{1};

#pragma omp parallel
    {
#pragma omp single
        { threads_used_atomic.store(omp_get_num_threads()); }

        SupernodalWorkspace ws;
        ws.init(A.num_cols, max_front);

        for (int L = 0; L <= maxL; ++L) {
            const auto& nodes = buckets[(size_t)L];

#pragma omp for schedule(dynamic,1)
            for (int ii = 0; ii < (int)nodes.size(); ++ii) {
                const int k = nodes[(size_t)ii];
                compute_one_supernode_cpu(
                    k, A, plan.sym, plan.children, up, x,
                    ok, fail_snode, fail_col, ws);

                const int tid = omp_get_thread_num();
                work_atomic[(size_t)tid].fetch_add(1);
            }

#pragma omp barrier
            if (!ok.load(std::memory_order_relaxed)) break;
        }
    }

    threads_used = threads_used_atomic.load();
    thread_work.assign((size_t)threads_used, 0);
    for (int t = 0; t < threads_used && t < (int)work_atomic.size(); ++t) {
        thread_work[(size_t)t] = work_atomic[(size_t)t].load();
    }
#else
    // Sequential schedule (bucket order preserved)
    SupernodalWorkspace ws;
    ws.init(A.num_cols, max_front);

    threads_used = 1;
    thread_work.assign(1, 0);

    for (int L = 0; L <= maxL; ++L) {
        const auto& nodes = buckets[(size_t)L];
        for (int ii = 0; ii < (int)nodes.size(); ++ii) {
            const int k = nodes[(size_t)ii];
            compute_one_supernode_cpu(
                k, A, plan.sym, plan.children, up, x,
                ok, fail_snode, fail_col, ws);
            thread_work[0] += 1;
            if (!ok.load(std::memory_order_relaxed)) break;
        }
        if (!ok.load(std::memory_order_relaxed)) break;
    }
#endif
}

} // namespace ichol::numeric::detail
