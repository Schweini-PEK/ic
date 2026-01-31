// test_supernodal.cpp
//
// Supernodal LL: symbolic + numeric validation against SuiteSparse CHOLMOD.
// - Symbolic: timing comparison (ours vs CHOLMOD analyze)
// - Numeric : CPU and CUDA(single-GPU, multi-stream) vs CHOLMOD factorization
//
// Extra CLI flags (in addition to gtest flags):
//   --ichol_mtx=PATH
//   --ichol_cuda_device=N
//   --ichol_cuda_streams=N            (default 16, paper/CHOLMOD-style)
//   --ichol_cuda_verbose
//   --ichol_cuda_print_schedule
//   --ichol_cuda_schedule_limit=N
//
// Example:
//   /tmp/ic/test/test_supernodal --gtest_color=no \
//       --ichol_mtx=/tmp/ic/test/data/nasa2146.mtx \
//       --ichol_cuda_streams=16 --ichol_cuda_verbose --ichol_cuda_print_schedule

#include <gtest/gtest.h>

extern "C" {
#include <cholmod.h>
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "factor/numerical/supernodal_numeric_ll.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct CliOpt {
    std::string mtx_path;
    ichol::numeric::CudaSupernodalOptions cuda_opt;
};
static CliOpt g_cli;

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static std::string get_mtx_path()
{
    if (!g_cli.mtx_path.empty()) return g_cli.mtx_path;
    const char* p = std::getenv("ICHOL_MTX");
    if (p && *p) return std::string(p);
    return std::string("/tmp/ic/test/data/apache2.mtx");
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

    for (int j = 0; j <= n; ++j) Sp[j] = static_cast<int32_t>(A.col_ptr[(size_t)j]);
    for (int k = 0; k < A.nnz; ++k) {
        Si[(size_t)k] = static_cast<int32_t>(A.row_ind[(size_t)k]);
        Sx[(size_t)k] = (A.values.empty() ? 1.0 : A.values[(size_t)k]);
    }
    return S;
}

struct CompareResult {
    double max_abs = 0.0;
    double max_rel = 0.0;
    uint64_t worst_key = 0;
    int common = 0;
    int onlyA = 0;
    int onlyB = 0;
};

static inline uint64_t pack_key(int32_t row, int32_t col)
{
    return (uint64_t)(uint32_t)col << 32 | (uint32_t)row;
}

static CompareResult compare_L_maps(const std::unordered_map<uint64_t,double>& A,
                                   const std::unordered_map<uint64_t,double>& B)
{
    CompareResult r;
    // common / onlyA
    for (const auto& kv : A) {
        auto it = B.find(kv.first);
        if (it == B.end()) {
            r.onlyA++;
            continue;
        }
        r.common++;
        const double va = kv.second;
        const double vb = it->second;
        const double diff = std::abs(va - vb);
        r.max_abs = std::max(r.max_abs, diff);
        const double denom = std::max(1e-300, std::abs(vb));
        const double rel = diff / denom;
        if (rel > r.max_rel) {
            r.max_rel = rel;
            r.worst_key = kv.first;
        }
    }
    // onlyB
    for (const auto& kv : B) {
        if (A.find(kv.first) == A.end()) r.onlyB++;
    }
    return r;
}

static void build_our_L_map(const ichol::numeric::SuperNumeric& num,
                            std::unordered_map<uint64_t,double>& out,
                            std::vector<double>* diag_out)
{
    const auto& sym = num.sym;
    const int nsuper = (int)sym.super.size() - 1;
    const int n = sym.super.back();

    out.clear();
    out.reserve((size_t)n * 8);

    std::vector<double> diag((size_t)n, 0.0);

    for (int k = 0; k < nsuper; ++k) {
        const int scol  = sym.super[(size_t)k];
        const int ecol  = sym.super[(size_t)k + 1];
        const int nscol = ecol - scol;

        const int pi0   = sym.pi[(size_t)k];
        const int pi1   = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;

        const int px0   = sym.px[(size_t)k];

        for (int j = 0; j < nscol; ++j) {
            const int gcol = scol + j;
            for (int i = j; i < nsrow; ++i) {
                const int grow = sym.s[(size_t)(pi0 + i)];
                const double v = num.x[(size_t)px0 + (size_t)i + (size_t)j * (size_t)nsrow];
                out.emplace(pack_key((int32_t)grow, (int32_t)gcol), v);
                if (grow == gcol) diag[(size_t)gcol] = v;
            }
        }
    }

    if (diag_out) *diag_out = std::move(diag);
}

static void build_cholmod_L_map(cholmod_factor* L,
                               cholmod_common* cc,
                               std::unordered_map<uint64_t,double>& out)
{
    out.clear();

    cholmod_sparse* Ls = cholmod_factor_to_sparse(L, cc);
    if (!Ls) return;

    const int n = (int)Ls->ncol;
    auto* Lp = static_cast<int32_t*>(Ls->p);
    auto* Li = static_cast<int32_t*>(Ls->i);
    auto* Lx = static_cast<double*>(Ls->x);

    const int nnz = (int)Lp[n];
    out.reserve((size_t)nnz * 2);

    for (int col = 0; col < n; ++col) {
        for (int p = Lp[col]; p < Lp[col + 1]; ++p) {
            const int row = Li[p];
            const double v = Lx[p];
            if (row < col) continue; // keep lower
            out.emplace(pack_key((int32_t)row, (int32_t)col), v);
        }
    }

    cholmod_free_sparse(&Ls, cc);
}

} // namespace

TEST(SupernodalSymbolic, Once_OursVsCHOLMOD)
{
    const std::string path = get_mtx_path();
    std::cout << "[SymbolicOnce] matrix=" << path << "\n";

    auto A0 = ichol::io::mtx_to_csc<double>(path, /*verify=*/false);
    auto A = A0;
    ASSERT_GT(A.num_cols, 0);
    ASSERT_EQ(A.num_rows, A.num_cols);

    const int n = A.num_cols;
    std::cout << "[SymbolicOnce] n=" << n << " nnz=" << A.nnz << "\n";

    ichol::SuperNodeOptions snopt;
    snopt.approximate = false;

    ichol::SymbolicOptions symopt;
    symopt.ordering = ichol::Ordering::AMD;

    // --- time ours (includes CHOLMOD analyze + apply permutation, if any) ---
    const auto t0 = Clock::now();
    auto plan = ichol::symbolic::supernodal_ll_analyze_fast(A, snopt, symopt);
    const auto t1 = Clock::now();

    // --- CHOLMOD setup ---
    cholmod_common cc;
    cholmod_start(&cc);
    cc.postorder = 0;
    cc.nmethods = 1;
    cc.method[0].ordering = CHOLMOD_AMD;
    cc.supernodal = CHOLMOD_SUPERNODAL;
    cc.supernodal_switch = 0;
    cc.final_ll = 1;
    cc.final_super = 1;
    cc.final_asis = 0;

    cholmod_sparse* S = to_cholmod_sparse_lower_csc(A0, &cc);
    ASSERT_NE(S, nullptr);

    const auto t2 = Clock::now();
    cholmod_factor* L = cholmod_analyze(S, &cc);
    const auto t3 = Clock::now();

    ASSERT_NE(L, nullptr);

    std::cout << "[SymbolicOnce] ours=" << elapsed_ms(t0, t1) << " ms"
              << "  cholmod=" << elapsed_ms(t2, t3) << " ms"
              << "  ratio=" << (elapsed_ms(t0, t1) / std::max(1e-9, elapsed_ms(t2, t3))) << "\n";

    cholmod_free_factor(&L, &cc);
    cholmod_free_sparse(&S, &cc);
    cholmod_finish(&cc);
}

TEST(SupernodalNumeric, OursCPUAndGPU_Vs_CHOLMOD_SupernodalLL)
{
    const std::string path = get_mtx_path();
    std::cout << "[Numeric] matrix=" << path << "\n";
    auto A0 = ichol::io::mtx_to_csc<double>(path, /*verify=*/false);
    auto A = A0;
    ASSERT_GT(A.num_cols, 0);
    ASSERT_EQ(A.num_rows, A.num_cols);

    const int n = A.num_cols;

    ichol::SuperNodeOptions snopt;
    snopt.approximate = false;

    ichol::SymbolicOptions symopt;
    symopt.ordering = ichol::Ordering::AMD;

    // --- time symbolic (ours vs CHOLMOD analyze) ---
    auto A_sym = A; // our analyze permutes A in-place
    const auto ts0 = Clock::now();
    auto plan = ichol::symbolic::supernodal_ll_analyze_fast(A_sym, snopt, symopt);
    const auto ts1 = Clock::now();

    cholmod_common cc;
    cholmod_start(&cc);
    cc.postorder = 0;
    cc.nmethods = 1;
    cc.method[0].ordering = CHOLMOD_AMD;
    cc.supernodal = CHOLMOD_SUPERNODAL;
    cc.supernodal_switch = 0;
    cc.final_ll = 1;
    cc.final_super = 1;
    cc.final_asis = 0;

    cholmod_sparse* S = to_cholmod_sparse_lower_csc(A0, &cc);
    ASSERT_NE(S, nullptr);

    const auto ts2 = Clock::now();
    cholmod_factor* L = cholmod_analyze(S, &cc);
    const auto ts3 = Clock::now();
    ASSERT_NE(L, nullptr);

    const double ours_ms = elapsed_ms(ts0, ts1);
    const double cholmod_ms = elapsed_ms(ts2, ts3);
    std::cout << "[NumericSymbolic] ours=" << ours_ms << " ms"
              << "  cholmod=" << cholmod_ms << " ms"
              << "  ratio=" << (ours_ms / std::max(1e-9, cholmod_ms)) << "";

    cholmod_free_factor(&L, &cc);
    cholmod_free_sparse(&S, &cc);
    cholmod_finish(&cc);

    // Use permuted matrix for numeric factorization, consistent with the plan.
    A = std::move(A_sym);

    // --- our CPU ---
    auto num_cpu = ichol::numeric::factorize_supernodal_ll(A, plan);
    ASSERT_TRUE(num_cpu.ok);
    g_cli.cuda_opt.print_schedule = true;
    // --- our GPU ---
    auto num_gpu = ichol::numeric::factorize_supernodal_ll_cuda(A, plan, g_cli.cuda_opt);
    if (!num_gpu.ok) {
        GTEST_SKIP() << "CUDA unavailable or GPU factorization failed";
    }

    // --- CHOLMOD factorize ---
    cholmod_start(&cc);
    cc.postorder = 0;
    cc.nmethods = 1;
    cc.method[0].ordering = CHOLMOD_GIVEN;
    cc.supernodal = CHOLMOD_SUPERNODAL;
    cc.supernodal_switch = 0;
    cc.final_ll = 1;
    cc.final_super = 1;
    cc.final_asis = 0;

    ASSERT_NE(S, nullptr);

    std::vector<int32_t> perm((size_t)n);
    for (int i = 0; i < n; ++i) perm[(size_t)i] = (int32_t)i;

    ASSERT_NE(L, nullptr);
    ASSERT_TRUE(cholmod_factorize(S, L, &cc));

    std::unordered_map<uint64_t,double> map_chol;
    build_cholmod_L_map(L, &cc, map_chol);

    std::unordered_map<uint64_t,double> map_cpu, map_gpu;
    std::vector<double> diag_cpu, diag_gpu;
    build_our_L_map(num_cpu, map_cpu, &diag_cpu);
    build_our_L_map(num_gpu, map_gpu, &diag_gpu);

    auto rcpu = compare_L_maps(map_cpu, map_chol);
    auto rgpu = compare_L_maps(map_gpu, map_chol);

    std::cout << "[Compare][CPU vs CHOLMOD] max_abs=" << rcpu.max_abs
              << " max_rel=" << rcpu.max_rel
              << " worst_key=" << rcpu.worst_key
              << " common=" << rcpu.common
              << " onlyCPU=" << rcpu.onlyA
              << " onlyCHOL=" << rcpu.onlyB << "\n";

    std::cout << "[Compare][GPU vs CHOLMOD] max_abs=" << rgpu.max_abs
              << " max_rel=" << rgpu.max_rel
              << " worst_key=" << rgpu.worst_key
              << " common=" << rgpu.common
              << " onlyGPU=" << rgpu.onlyA
              << " onlyCHOL=" << rgpu.onlyB << "\n";

    auto print_first10 = [](const char* tag, const std::vector<double>& d){
        std::cout << tag << " first10:";
        for (int i = 0; i < 10 && i < (int)d.size(); ++i) std::cout << " " << d[(size_t)i];
        std::cout << "\n";
    };
    print_first10("[Diag][CPU]", diag_cpu);
    print_first10("[Diag][GPU]", diag_gpu);

    std::cout << "[GPU streams] streams_used=" << num_gpu.threads_used << " work_per_stream:";
    for (int v : num_gpu.thread_work) std::cout << " " << v;
    std::cout << "\n";

    cholmod_free_factor(&L, &cc);
    cholmod_free_sparse(&S, &cc);
    cholmod_finish(&cc);
}

// --- Custom main: parse our flags, then run gtest ---
static bool starts_with(const std::string& s, const std::string& p) {
    return s.rfind(p, 0) == 0;
}

int main(int argc, char** argv)
{
    g_cli.cuda_opt = ichol::numeric::CudaSupernodalOptions{}; // defaults (streams=16)

    std::vector<char*> gtest_argv;
    gtest_argv.reserve((size_t)argc);
    gtest_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);

        if (starts_with(a, "--ichol_mtx=")) {
            g_cli.mtx_path = a.substr(std::string("--ichol_mtx=").size());
            continue;
        }
        if (starts_with(a, "--ichol_cuda_device=")) {
            g_cli.cuda_opt.device = std::atoi(a.c_str() + std::string("--ichol_cuda_device=").size());
            continue;
        }
        if (starts_with(a, "--ichol_cuda_streams=")) {
            g_cli.cuda_opt.streams = std::atoi(a.c_str() + std::string("--ichol_cuda_streams=").size());
            continue;
        }
        if (a == "--ichol_cuda_verbose") {
            g_cli.cuda_opt.verbose = true;
            continue;
        }
        if (a == "--ichol_cuda_print_schedule") {
            g_cli.cuda_opt.print_schedule = true;
            continue;
        }
        if (starts_with(a, "--ichol_cuda_schedule_limit=")) {
            g_cli.cuda_opt.schedule_print_limit = std::atoi(a.c_str() + std::string("--ichol_cuda_schedule_limit=").size());
            continue;
        }

        // Keep all other args for gtest
        gtest_argv.push_back(argv[i]);
    }

    int gargc = (int)gtest_argv.size();
    ::testing::InitGoogleTest(&gargc, gtest_argv.data());
    return RUN_ALL_TESTS();
}
