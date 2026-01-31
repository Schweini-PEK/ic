#pragma once

#include <vector>
#include <atomic>
#include <cstddef>
#include <algorithm>

#include "ichol/matrix_formats.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "factor/numerical/supernodal_workspace.hpp"

namespace ichol::numeric::detail {

// -----------------------------------------------------------------------------
// Small helpers (column-major access)
// -----------------------------------------------------------------------------
inline double& CM(double* a, int ld, int i, int j) {
    return a[(size_t)i + (size_t)ld * (size_t)j];
}
inline double CMc(const double* a, int ld, int i, int j) {
    return a[(size_t)i + (size_t)ld * (size_t)j];
}

// -----------------------------------------------------------------------------
// Sparse view helper (for fast column access)
// -----------------------------------------------------------------------------
template <typename T>
struct CscView {
    const matrix::CscMatrix<T>& A;
    explicit CscView(const matrix::CscMatrix<T>& Ain) : A(Ain) {}
    int cb(int j) const { return A.col_ptr[j]; }
    int ce(int j) const { return A.col_ptr[j + 1]; }
    int row(int p) const { return A.row_ind[p]; }
    T   val(int p) const { return A.values[p]; }
};

// -----------------------------------------------------------------------------
// Per-supernode packed update (lives until parent consumes it)
// -----------------------------------------------------------------------------
struct UpdatePack {
    int nupd = 0;                 // number of update rows
    std::vector<int> idx;         // global row ids of update rows (length nupd)
    std::vector<double> S;        // dense symmetric update matrix (nupd x nupd, col-major)
};

// -----------------------------------------------------------------------------
// Kernels / scheduler entry points (internal)
// -----------------------------------------------------------------------------
void compute_one_supernode_cpu(
    int k,
    const matrix::CscMatrix<double>& A,
    const symbolic::SuperSym& sym,
    const std::vector<std::vector<int>>& children,
    std::vector<UpdatePack>& up,
    std::vector<double>& x,                       // CHOLMOD block layout
    std::atomic<bool>& ok,
    std::atomic<int>& fail_snode,
    std::atomic<int>& fail_col,
    SupernodalWorkspace& ws);

// Schedule supernodes by level buckets (OpenMP if enabled, otherwise sequential).
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
    std::vector<int>& thread_work);

} // namespace ichol::numeric::detail
