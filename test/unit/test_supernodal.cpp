// test_supernodal.cpp
// Timing-only symbolic comparison: our supernodal symbolic vs SuiteSparse CHOLMOD.
//
// Usage:
//   ICHOL_MTX=/tmp/ic/test/data/nasa2146.mtx /tmp/ic/test/test_supernodal --gtest_color=no
// If ICHOL_MTX is not set, defaults to /tmp/ic/test/data/nasa2146.mtx

#include <gtest/gtest.h>

extern "C" {
#include <cholmod.h>
}

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"

// Your repo has a fast symbolic entrypoint (added during optimization). We forward-declare it here so
// this timing test does not depend on a specific header layout.
namespace ichol::symbolic {
    SupernodalLLPlan supernodal_ll_analyze_fast(const ichol::matrix::CscMatrix<double>& A,
                                                const ichol::SuperNodeOptions& opt);
}

namespace {

using Clock = std::chrono::steady_clock;

static std::string get_mtx_path()
{
    const char* p = std::getenv("ICHOL_MTX");
    if (p && *p) return std::string(p);
    return std::string("/tmp/ic/test/data/apache2.mtx");
}

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static cholmod_sparse* to_cholmod_sparse_lower_csc(const ichol::matrix::CscMatrix<double>& A,
                                                  cholmod_common* cc)
{
    const int n = A.num_cols;
    const size_t nz = static_cast<size_t>(A.nnz);

    // stype = -1 means symmetric, stored in LOWER triangle.
    cholmod_sparse* S = cholmod_allocate_sparse(
        (size_t)n, (size_t)n, nz,
        /*sorted=*/1, /*packed=*/1,
        /*stype=*/-1,
        /*xtype=*/CHOLMOD_REAL,
        cc);
    if (!S) return nullptr;

    auto* Sp = static_cast<int32_t*>(S->p);
    auto* Si = static_cast<int32_t*>(S->i);
    auto* Sx = static_cast<double*>(S->x);

    // Copy pointers/indices/values.
    for (int j = 0; j <= n; ++j) Sp[j] = static_cast<int32_t>(A.col_ptr[(size_t)j]);
    for (int k = 0; k < A.nnz; ++k) {
        Si[(size_t)k] = static_cast<int32_t>(A.row_ind[(size_t)k]);
        Sx[(size_t)k] = (A.values.empty() ? 1.0 : A.values[(size_t)k]);
    }

    return S;
}

static void run_ours_once(const ichol::matrix::CscMatrix<double>& A, const ichol::SuperNodeOptions& opt)
{
    (void)ichol::symbolic::supernodal_ll_analyze_fast(A, opt);
}

} // namespace

TEST(SupernodalSymbolic, Once_OursVsCHOLMOD)
{
    const std::string path = get_mtx_path();
    std::cout << "[SymbolicOnce] matrix=" << path << "\n";

    // Our reader stores lower-triangle + diag (symmetric convention) for symbolic.
    auto A = ichol::io::mtx_to_csc<double>(path, /*verify=*/false);
    ASSERT_GT(A.num_cols, 0);
    ASSERT_EQ(A.num_rows, A.num_cols);

    const int n = A.num_cols;
    std::cout << "[SymbolicOnce] n=" << n << " nnz=" << A.nnz << "\n";

    ichol::SuperNodeOptions snopt;
    snopt.approximate = false;

    // --- CHOLMOD setup ---
    cholmod_common cc;
    cholmod_start(&cc);
    cc.postorder = 0;
    cc.nmethods = 1;
    cc.method[0].ordering = CHOLMOD_GIVEN;
    cc.supernodal = CHOLMOD_SUPERNODAL;   // 强制 supernodal analyze
    cc.supernodal_switch = 0;            // 可选：避免启发式切回 simplicial
    cc.final_ll = 1;                     // 可选：最终 LL'
    cc.final_super = 1;                  // 可选：保持 supernodal 形式
    cc.final_asis = 0;                   // 可选：不要忽略 final_* 参数
    cholmod_sparse* S = to_cholmod_sparse_lower_csc(A, &cc);
    ASSERT_NE(S, nullptr);

    std::vector<int32_t> perm((size_t)n);
    for (int i = 0; i < n; ++i) perm[(size_t)i] = (int32_t)i;

    // --- time ours (once) ---
    const auto t0 = Clock::now();
    run_ours_once(A, snopt);
    const auto t1 = Clock::now();

    // --- time CHOLMOD (once) ---
    const auto t2 = Clock::now();
    cholmod_factor* L = cholmod_analyze_p(S, perm.data(), nullptr, 0, &cc);
    const auto t3 = Clock::now();
    ASSERT_NE(L, nullptr);

    const double ours_ms = elapsed_ms(t0, t1);
    const double chol_ms = elapsed_ms(t2, t3);

    std::cout << "[SymbolicOnce] ours=" << ours_ms << " ms  cholmod=" << chol_ms
              << " ms  ratio=" << (chol_ms > 0 ? (ours_ms / chol_ms) : 0.0) << "\n";

    cholmod_free_factor(&L, &cc);
    cholmod_free_sparse(&S, &cc);
    cholmod_finish(&cc);
}
