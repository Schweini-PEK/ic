// test_supernodal.cpp
//
// Supernodal LL: symbolic + numeric validation against SuiteSparse CHOLMOD.
// - Symbolic: timing comparison (ours vs CHOLMOD analyze)
// - Numeric : CPU and CUDA(single-GPU, multi-stream) vs CHOLMOD factorization
//
// Extra CLI flags (in addition to gtest flags):
//   --ichol_mtx=PATH
//   --ichol_cuda_device=N
//   --ichol_cuda_streams=N            (default 16)
//   --ichol_cuda_verbose
//   --ichol_cuda_print_schedule
//   --ichol_cuda_schedule_limit=N
//   --ichol_compare=sample|full|none  (default sample)
//   --ichol_compare_max_n=N           (skip compare if n > N and compare != full; default 200000)
//   --ichol_compare_samples=N         (sample size when compare=sample; default 200000)
//   --ichol_compare_seed=U64          (default 12345)
//
// Notes:
// - For large matrices, converting CHOLMOD factor to sparse + building full hash maps can look like a hang.
//   Default comparison mode is bounded (sample) and auto-skips very large n.
// - For correctness, CHOLMOD is forced to use OUR permutation (CHOLMOD_GIVEN + cholmod_analyze_p with plan.perm).
//
// Example:
//   /tmp/ic/test/test_supernodal --gtest_color=no \
//       --ichol_mtx=/tmp/ic/test/data/nasa2146.mtx \
//       --ichol_cuda_streams=16 --ichol_cuda_verbose --ichol_cuda_print_schedule \
//       --ichol_compare=sample --ichol_compare_samples=200000

#include <gtest/gtest.h>

extern "C" {
#include <cholmod.h>
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ---- CUDA runtime (optional, for warm-up timing) ----
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define ICHOL_TEST_HAS_CUDA_RUNTIME 1
#else
#define ICHOL_TEST_HAS_CUDA_RUNTIME 0
#endif

#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "factor/numerical/supernodal_numeric_ll.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct CliOpt {
    std::string mtx_path;
    ichol::numeric::CudaSupernodalOptions cuda_opt;

    std::string compare = "sample"; // sample|full|none
    int compare_max_n = 200000;
    int compare_samples = 200000;
    uint64_t compare_seed = 12345;
};
static CliOpt g_cli;

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool starts_with(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }

static uint64_t parse_u64(const std::string& s)
{
    const char* str = s.c_str();
    char* end = nullptr;
    return std::strtoull(str, &end, 0);
}

static std::string get_mtx_path()
{
    if (!g_cli.mtx_path.empty()) return g_cli.mtx_path;
    const char* p = std::getenv("ICHOL_MTX");
    if (p && *p) return std::string(p);
    return std::string("test/data/nasa2146.mtx");
}

static cholmod_sparse* to_cholmod_sparse_lower_csc(const ichol::matrix::CscMatrix<double>& A,
                                                  cholmod_common* cc)
{
    const int n = A.num_cols;
    const size_t nz = static_cast<size_t>(A.nnz);

    // stype = -1 means symmetric, stored in LOWER triangle.
    cholmod_sparse* S = cholmod_allocate_sparse(
        (size_t)n, (size_t)n, nz,
        /*sorted=*/0, /*packed=*/1,
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
    uint64_t worst_key_abs = 0;
    uint64_t worst_key_rel = 0;
    int common = 0;
    int onlyA = 0;
    int onlyB = 0;
};

static inline uint64_t pack_key(int32_t row, int32_t col)
{
    // keep consistent with previous code: col in high bits, row in low bits
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
        if (diff > r.max_abs) { r.max_abs = diff; r.worst_key_abs = kv.first; }

        const double denom = std::max(1e-300, std::max(std::abs(va), std::abs(vb)));
        const double rel = diff / denom;
        if (rel > r.max_rel) {
            r.max_rel = rel;
            r.worst_key_rel = kv.first;
        }
    }
    // onlyB
    for (const auto& kv : B) {
        if (A.find(kv.first) == A.end()) r.onlyB++;
    }
    return r;
}

static void build_our_L_map_full(const ichol::numeric::SuperNumeric& num,
                                std::unordered_map<uint64_t,double>& out,
                                std::vector<double>* diag_out)
{
    const auto& sym = num.sym;
    const int nsuper = (int)sym.super.size() - 1;
    const int n = sym.super.back();

    out.clear();
    out.reserve((size_t)std::min(n, 100000) * 8);

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

static void build_our_L_map_filtered(const ichol::numeric::SuperNumeric& num,
                                    const std::unordered_set<uint64_t>& want,
                                    std::unordered_map<uint64_t,double>& out,
                                    std::vector<double>* diag_out)
{
    const auto& sym = num.sym;
    const int nsuper = (int)sym.super.size() - 1;
    const int n = sym.super.back();

    out.clear();
    out.reserve(want.size() * 2 + 16);

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
                const uint64_t key = pack_key((int32_t)grow, (int32_t)gcol);
                if (want.find(key) == want.end()) continue;

                const double v = num.x[(size_t)px0 + (size_t)i + (size_t)j * (size_t)nsrow];
                out.emplace(key, v);
                if (grow == gcol) diag[(size_t)gcol] = v;
            }
        }
    }

    if (diag_out) *diag_out = std::move(diag);
}

// Build CHOLMOD L map. If max_samples > 0, do reservoir sampling (bounded work/memory).
static void build_cholmod_L_map(cholmod_factor* L,
                               cholmod_common* cc,
                               std::unordered_map<uint64_t,double>& out,
                               int max_samples,
                               uint64_t seed)
{
    out.clear();

    cholmod_sparse* Ls = cholmod_factor_to_sparse(L, cc);
    if (!Ls) return;

    const int n = (int)Ls->ncol;
    auto* Lp = static_cast<int32_t*>(Ls->p);
    auto* Li = static_cast<int32_t*>(Ls->i);
    auto* Lx = static_cast<double*>(Ls->x);

    if (max_samples <= 0) {
        const int nnz = (int)Lp[n];
        out.reserve((size_t)nnz * 2);
        for (int col = 0; col < n; ++col) {
            for (int p = Lp[col]; p < Lp[col + 1]; ++p) {
                const int row = Li[p];
                if (row < col) continue;
                out.emplace(pack_key((int32_t)row, (int32_t)col), Lx[p]);
            }
        }
        cholmod_free_sparse(&Ls, cc);
        return;
    }

    struct KV { uint64_t key; double val; };
    std::vector<KV> reservoir;
    reservoir.reserve((size_t)max_samples);

    std::mt19937_64 rng(seed);
    uint64_t seen = 0;

    for (int col = 0; col < n; ++col) {
        for (int p = Lp[col]; p < Lp[col + 1]; ++p) {
            const int row = Li[p];
            if (row < col) continue;
            const uint64_t key = pack_key((int32_t)row, (int32_t)col);
            const double val = Lx[p];

            ++seen;
            if ((int)reservoir.size() < max_samples) {
                reservoir.push_back({key, val});
            } else {
                std::uniform_int_distribution<uint64_t> dist(0, seen - 1);
                const uint64_t j = dist(rng);
                if (j < (uint64_t)max_samples) reservoir[(size_t)j] = {key, val};
            }
        }
    }

    out.reserve(reservoir.size() * 2 + 16);
    for (const auto& kv : reservoir) out.emplace(kv.key, kv.val);

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

    ichol::SymbolicOptions symopt;
    symopt.ordering = ichol::Ordering::AMD;

    // --- time ours (includes CHOLMOD analyze + apply permutation, if any) ---
    const auto t0 = Clock::now();
    (void)ichol::symbolic::supernodal_analyze(A, symopt);
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

    ichol::SymbolicOptions symopt;
    symopt.ordering = ichol::Ordering::AMD;

    // --- our analyze (permutes A_sym in-place) ---
    auto A_sym = A;
    const auto ts0 = Clock::now();
    auto plan = ichol::symbolic::supernodal_analyze(A_sym, symopt);
    const auto ts1 = Clock::now();
    const double ours_sym_ms = elapsed_ms(ts0, ts1);

    // --- CHOLMOD analyze with GIVEN permutation (ours perm) ---
    cholmod_common cc;
    cholmod_start(&cc);
    cc.postorder = 0;
    cc.nmethods = 1;
    cc.method[0].ordering = CHOLMOD_GIVEN;
    cc.supernodal = CHOLMOD_SUPERNODAL;
    cc.supernodal_switch = 0;
    cc.final_ll = 1;
    cc.final_super = 1;
    cc.final_asis = 0;

    cholmod_sparse* S = to_cholmod_sparse_lower_csc(A0, &cc);
    ASSERT_NE(S, nullptr);

    std::vector<int32_t> perm((size_t)n);
    for (int i = 0; i < n; ++i) perm[(size_t)i] = (int32_t)plan.perm.perm[(size_t)i];

    const auto ts2 = Clock::now();
    cholmod_sort(S, &cc);
    cholmod_factor* L = cholmod_analyze_p(S, perm.data(), nullptr, 0, &cc);
    const auto ts3 = Clock::now();
    ASSERT_NE(L, nullptr);

    const double cholmod_sym_ms = elapsed_ms(ts2, ts3);
    std::cout << "[NumericSymbolic] ours=" << ours_sym_ms << " ms"
              << "  cholmod=" << cholmod_sym_ms << " ms"
              << "  ratio=" << (ours_sym_ms / std::max(1e-9, cholmod_sym_ms)) << "\n";

    // Use permuted matrix for numeric factorization, consistent with the plan.
    A = std::move(A_sym);

    // --- our CPU factorize (timed) ---
    const auto tcpu0 = Clock::now();
    auto num_cpu = ichol::numeric::factorize_supernodal_ll(A, plan);
    const auto tcpu1 = Clock::now();
    ASSERT_TRUE(num_cpu.ok);
    const double cpu_ms = elapsed_ms(tcpu0, tcpu1);

    // --- our GPU factorize ---
    // NOTE: The first CUDA call in a process can pay large one-time costs
    // (context creation, module loading/JIT, allocator init). For fair timing,
    // we warm up CUDA runtime/context and the factorization path, then time a second run.
    double gpu_ms = -1.0;
    double gpu_warmup_ms = -1.0;

#if ICHOL_TEST_HAS_CUDA_RUNTIME
    {
        // Force CUDA runtime/context initialization (not timed).
        cudaSetDevice(g_cli.cuda_opt.device);
        (void)cudaFree(nullptr); // creates context on first call
        (void)cudaDeviceSynchronize();
    }
#endif

    // Warm-up run (not timed)
    {
        const auto tw0 = Clock::now();
        auto warm = ichol::numeric::factorize_supernodal_ll_cuda(A, plan, g_cli.cuda_opt);
        const auto tw1 = Clock::now();
        gpu_warmup_ms = elapsed_ms(tw0, tw1);
        if (!warm.ok) {
            std::cout << "[NumericFactorize] ours_gpu_warmup=FAILED\n";
        }
    }

    // Timed run
    const auto tgpu0 = Clock::now();
    auto num_gpu = ichol::numeric::factorize_supernodal_ll_cuda(A, plan, g_cli.cuda_opt);
    const auto tgpu1 = Clock::now();
    if (!num_gpu.ok) {
        std::cout << "[NumericFactorize] ours_gpu=FAILED\n";
    } else {
        gpu_ms = elapsed_ms(tgpu0, tgpu1);
    }

    // --- CHOLMOD factorize (timed) ---
    const auto tch0 = Clock::now();
    const int ok = cholmod_factorize(S, L, &cc);
    std::cout << "[CHOLMOD] is_super=" << L->is_super
          << " supernodal=" << cc.supernodal
          << " final_ll=" << cc.final_ll
          << "\n";
    const auto tch1 = Clock::now();
    ASSERT_TRUE(ok != 0);
    const double chol_fac_ms = elapsed_ms(tch0, tch1);
    //
    std::cout << "[NumericFactorize] ours_gpu_warmup=" << gpu_warmup_ms << " ms\n";

    std::cout << "[NumericFactorize] ours_gpu=" << (num_gpu.ok ? gpu_ms : -1.0) << " ms"
              << "  (gpu_device=" << g_cli.cuda_opt.device << ", streams=" << g_cli.cuda_opt.streams << ")\n";
    std::cout << "[NumericFactorize] cholmod=" << chol_fac_ms << " ms"
              << "  speedup(chol/ours_gpu)=" << (num_gpu.ok ? (chol_fac_ms / std::max(1e-9, gpu_ms)) : 0.0)
              << "\n";

    // ---- bounded correctness check ----
    const bool full = (g_cli.compare == "full");
    if (g_cli.compare == "none") {
        std::cout << "[Compare] disabled (--ichol_compare=none)\n";
    } else if (!full && n > g_cli.compare_max_n) {
        std::cout << "[Compare] skipped: n=" << n << " > compare_max_n=" << g_cli.compare_max_n
                  << " (use --ichol_compare=full to force)\n";
    } else {
        const int max_samples = full ? 0 : std::max(1, g_cli.compare_samples);

        std::unordered_map<uint64_t,double> map_chol;
        build_cholmod_L_map(L, &cc, map_chol, max_samples, g_cli.compare_seed);

        std::unordered_set<uint64_t> want;
        want.reserve(map_chol.size() * 2 + 16);
        for (const auto& kv : map_chol) want.insert(kv.first);

        std::unordered_map<uint64_t,double> map_cpu, map_gpu;
        std::vector<double> diag_cpu, diag_gpu;

        if (full) {
            build_our_L_map_full(num_cpu, map_cpu, &diag_cpu);
            if (num_gpu.ok) build_our_L_map_full(num_gpu, map_gpu, &diag_gpu);
        } else {
            build_our_L_map_filtered(num_cpu, want, map_cpu, &diag_cpu);
            if (num_gpu.ok) build_our_L_map_filtered(num_gpu, want, map_gpu, &diag_gpu);
        }

        auto rcpu = compare_L_maps(map_cpu, map_chol);
        std::cout << "[Compare][CPU vs CHOLMOD] mode=" << (full ? "full" : "sample")
                  << " keys=" << map_chol.size()
                  << " max_abs=" << rcpu.max_abs
                  << " max_rel=" << rcpu.max_rel
                  << " worst_abs_key=" << rcpu.worst_key_abs << " worst_rel_key=" << rcpu.worst_key_rel
                  << " common=" << rcpu.common
                  << " onlyCPU=" << rcpu.onlyA
                  << " onlyCHOL=" << rcpu.onlyB << "\n";

        if (num_gpu.ok) {
            auto rgpu = compare_L_maps(map_gpu, map_chol);
            std::cout << "[Compare][GPU vs CHOLMOD] mode=" << (full ? "full" : "sample")
                      << " keys=" << map_chol.size()
                      << " max_abs=" << rgpu.max_abs
                      << " max_rel=" << rgpu.max_rel
                      << " worst_abs_key=" << rgpu.worst_key_abs << " worst_rel_key=" << rgpu.worst_key_rel
                      << " common=" << rgpu.common
                      << " onlyGPU=" << rgpu.onlyA
                      << " onlyCHOL=" << rgpu.onlyB << "\n";

            // --- Debug: decode worst_abs_key / worst_rel_key to (row,col) and print values ---
            auto decode_row = [](uint64_t key) -> int32_t { return (int32_t)(uint32_t)(key & 0xffffffffu); };
            auto decode_col = [](uint64_t key) -> int32_t { return (int32_t)(uint32_t)(key >> 32); };

            auto dump_key = [&](const char* tag, uint64_t key) {
                int32_t r = decode_row(key);
                int32_t c = decode_col(key);
                auto itg = map_gpu.find(key);
                auto itc = map_chol.find(key);
                double vg = (itg == map_gpu.end()) ? 0.0 : itg->second;
                double vc = (itc == map_chol.end()) ? 0.0 : itc->second;

                // locate supernode by column
                const auto& sup = num_cpu.sym.super;
                int sk = -1;
                {
                    int lo = 0, hi = (int)sup.size() - 2;
                    while (lo <= hi) {
                        int mid = (lo + hi) >> 1;
                        int a = sup[(size_t)mid];
                        int b = sup[(size_t)mid + 1];
                        if (c < a) hi = mid - 1;
                        else if (c >= b) lo = mid + 1;
                        else { sk = mid; break; }
                    }
                }

                std::cout << "  [" << tag << "] (row=" << r << ", col=" << c << ")"
                          << " snode=" << sk
                          << " gpu=" << vg << " chol=" << vc
                          << " abs=" << std::abs(vg - vc)
                          << " rel=" << (std::max(1e-300, std::max(std::abs(vg), std::abs(vc))) > 0
                                        ? std::abs(vg - vc) / std::max(1e-300, std::max(std::abs(vg), std::abs(vc)))
                                        : 0.0)
                          << "\n";
            };

            dump_key("WorstAbsKey[GPU vs CHOLMOD]", rgpu.worst_key_abs);
            dump_key("WorstRelKey[GPU vs CHOLMOD]", rgpu.worst_key_rel);
}

        auto print_first10 = [](const char* tag, const std::vector<double>& d){
            std::cout << tag << " first10:";
            for (int i = 0; i < 10 && i < (int)d.size(); ++i) std::cout << " " << d[(size_t)i];
            std::cout << "\n";
        };
        print_first10("[Diag][CPU]", diag_cpu);
        if (num_gpu.ok) print_first10("[Diag][GPU]", diag_gpu);

        if (num_gpu.ok) {
            std::cout << "[GPU streams] streams_used=" << num_gpu.threads_used << " work_per_stream:";
            for (int v : num_gpu.thread_work) std::cout << " " << v;
            std::cout << "\n";
        }
    }

    cholmod_free_factor(&L, &cc);
    cholmod_free_sparse(&S, &cc);
    cholmod_finish(&cc);
}


// --- Custom main: parse our flags, then run gtest ---
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

        if (starts_with(a, "--ichol_compare=")) {
            g_cli.compare = a.substr(std::string("--ichol_compare=").size());
            continue;
        }
        if (starts_with(a, "--ichol_compare_max_n=")) {
            g_cli.compare_max_n = std::atoi(a.c_str() + std::string("--ichol_compare_max_n=").size());
            continue;
        }
        if (starts_with(a, "--ichol_compare_samples=")) {
            g_cli.compare_samples = std::atoi(a.c_str() + std::string("--ichol_compare_samples=").size());
            continue;
        }
        if (starts_with(a, "--ichol_compare_seed=")) {
            g_cli.compare_seed = parse_u64(a.substr(std::string("--ichol_compare_seed=").size()));
            continue;
        }

        // Keep all other args for gtest
        gtest_argv.push_back(argv[i]);
    }

    if (g_cli.compare != "none" && g_cli.compare != "sample" && g_cli.compare != "full") {
        std::cerr << "[CLI] invalid --ichol_compare=" << g_cli.compare << " (use sample|full|none)\n";
        return 2;
    }

    int gargc = (int)gtest_argv.size();
    ::testing::InitGoogleTest(&gargc, gtest_argv.data());
    return RUN_ALL_TESTS();
}
