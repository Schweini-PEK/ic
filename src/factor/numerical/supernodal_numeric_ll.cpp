#include "factor/numerical/supernodal_numeric_ll.hpp"
#include "factor/numerical/detail/supernodal_numeric_ll_internal.hpp"

#include <atomic>
#include <algorithm>
#include <vector>

namespace ichol::numeric {

    SuperNumeric factorize_supernodal_ll(
        const ichol::matrix::CscMatrix<double>& A,
        const symbolic::SupernodalLLPlan& plan)
    {
        SuperNumeric out;
        out.sym = plan.sym;

        const int nsuper = (int)plan.sym.super.size() - 1;

        // Allocate numeric storage in CHOLMOD-like block layout.
        const int xsz = plan.sym.px.empty() ? 0 : plan.sym.px.back();
        out.x.assign((size_t)xsz, 0.0);

        // Packed updates (one per supernode, consumed by parent).
        std::vector<detail::UpdatePack> up((size_t)nsuper);

        // Workspace sizing (prefer plan's precomputed bound).
        int max_front = plan.max_front_dim;
        if (max_front <= 0) {
            for (int k = 0; k < nsuper; ++k) {
                const int nsrow = plan.sym.pi[(size_t)k + 1] - plan.sym.pi[(size_t)k];
                if (nsrow > max_front) max_front = nsrow;
            }
        }

        // Failure flags
        std::atomic<bool> ok{true};
        std::atomic<int> fail_snode{-1};
        std::atomic<int> fail_col{-1};

        // Run schedule (OpenMP if enabled, otherwise sequential).
        detail::schedule_levels_cpu(
            A, plan, up, out.x,
            ok, fail_snode, fail_col,
            max_front,
            out.threads_used,
            out.thread_work);

        out.ok = ok.load(std::memory_order_relaxed);
        out.fail_snode = fail_snode.load(std::memory_order_relaxed);
        out.fail_col_in_snode = fail_col.load(std::memory_order_relaxed);
        return out;
    }

} // namespace ichol::numeric
