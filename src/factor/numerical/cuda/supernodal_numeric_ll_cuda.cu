// ============================================================
// FILE: src/factor/numerical/cuda/supernodal_numeric_ll_cuda.cu
// Single-GPU supernodal LL numeric factorization (CHOLMOD-style):
//   - Symbolic scheduling comes from SupernodalLLPlan (parent/children/level buckets).
//   - Numeric runs level-by-level; snodes in the same level are independent and can be
//     processed concurrently using multiple CUDA streams (and CPU threads to drive them).
//   - For correctness and robustness, we avoid any async D2H writes into unstable addresses.
// ============================================================
#include "factor/numerical/supernodal_numeric_ll.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <dlfcn.h>
#include <link.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "factor/symbolic/snode_schedule.hpp"

namespace ichol::numeric {

static std::string dl_path_of_loaded(const char* soname)
{
    void* h = dlopen(soname, RTLD_NOW);
    if (!h) return std::string("dlopen failed: ") + dlerror();

    struct link_map* lm = nullptr;
    if (dlinfo(h, RTLD_DI_LINKMAP, &lm) != 0 || !lm || !lm->l_name) {
        dlclose(h);
        return "dlinfo failed";
    }
    std::string path = lm->l_name;
    dlclose(h);
    return path;
}

static bool cuda_runtime_usable(std::string* why = nullptr)
{
    // Print which libcuda is actually loaded (WSL stub vs real driver is a common pitfall).
    std::string libcuda_path = dl_path_of_loaded("libcuda.so.1");
    std::cerr << "[CUDA] libcuda.so.1 loaded from: " << libcuda_path << "\n";
    if (libcuda_path.find("stubs") != std::string::npos) {
        if (why) {
            *why = "Loaded CUDA stub library: " + libcuda_path +
                   " (remove /usr/local/cuda/lib64/stubs from LD_LIBRARY_PATH / rpath)";
        }
        return false;
    }

    int devCount = 0;
    cudaError_t e3 = cudaGetDeviceCount(&devCount);
    if (e3 != cudaSuccess || devCount <= 0) {
        if (why) *why = "No CUDA device available";
        return false;
    }
    return true;
}

// ------------------ CUDA CHECK ------------------
#define CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
  } \
} while(0)

#define CUBLAS_CHECK(x) do { \
  cublasStatus_t _s = (x); \
  if (_s != CUBLAS_STATUS_SUCCESS) { \
    throw std::runtime_error("cuBLAS error"); \
  } \
} while(0)

#define CUSOLVER_CHECK(x) do { \
  cusolverStatus_t _s = (x); \
  if (_s != CUSOLVER_STATUS_SUCCESS) { \
    throw std::runtime_error("cuSOLVER error"); \
  } \
} while(0)

// column-major helper (host)
static inline double& HCM(std::vector<double>& a, int ld, int r, int c) {
    return a[(size_t)r + (size_t)c * (size_t)ld];
}
static inline double HCMc(const std::vector<double>& a, int ld, int r, int c) {
    return a[(size_t)r + (size_t)c * (size_t)ld];
}

struct UpdatePack {
    int nupd = 0;
    std::vector<int> idx;       // global indices of update rows (size nupd)
    std::vector<double> S;      // dense symmetric update, full stored col-major (ld=nupd)
};

template <class T>
struct CscView {
    const ichol::matrix::CscMatrix<T>& A;
    int cb(int j) const { return A.col_ptr[j]; }
    int ce(int j) const { return A.col_ptr[j + 1]; }
    int row(int p) const { return A.row_ind[p]; }
    T   val(int p) const { return A.values[p]; }
};

// -------------- CUDA context (per-stream, reuse buffers) --------------
struct CudaCtx {
    bool inited = false;
    int device = 0;

    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    cudaStream_t stream = nullptr;

    double* dC = nullptr;      // nsrow x nscol (column-major, ld=nsrow)
    double* dS = nullptr;      // nupd x nupd (full)
    int*    dinfo = nullptr;
    void*   dwork = nullptr;
    int     lwork = 0;

    int cap_nsrow = 0;
    int cap_nscol = 0;
    int cap_nupd  = 0;

    void init(int dev)
    {
        if (inited) return;
        device = dev;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
        CUDA_CHECK(cudaMalloc((void**)&dinfo, sizeof(int)));
        inited = true;
    }

    void ensure_C(int nsrow, int nscol)
    {
        if (!inited) throw std::runtime_error("CudaCtx not initialized");
        if (nsrow <= cap_nsrow && nscol <= cap_nscol && dC) return;
        if (dC) CUDA_CHECK(cudaFree(dC));
        cap_nsrow = std::max(cap_nsrow, nsrow);
        cap_nscol = std::max(cap_nscol, nscol);
        CUDA_CHECK(cudaMalloc((void**)&dC, sizeof(double) * (size_t)cap_nsrow * (size_t)cap_nscol));
    }

    void ensure_S(int nupd)
    {
        if (!inited) throw std::runtime_error("CudaCtx not initialized");
        if (nupd <= cap_nupd && dS) return;
        if (dS) CUDA_CHECK(cudaFree(dS));
        cap_nupd = std::max(cap_nupd, nupd);
        CUDA_CHECK(cudaMalloc((void**)&dS, sizeof(double) * (size_t)cap_nupd * (size_t)cap_nupd));
    }

    void ensure_potrf_workspace(int n, int lda)
    {
        if (!inited) throw std::runtime_error("CudaCtx not initialized");
        int lw = 0;
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, n, dC, lda, &lw));
        if (lw <= lwork && dwork) return;
        if (dwork) CUDA_CHECK(cudaFree(dwork));
        lwork = lw;
        CUDA_CHECK(cudaMalloc(&dwork, sizeof(double) * (size_t)lwork));
    }

    void shutdown()
    {
        if (!inited) return;
        CUDA_CHECK(cudaSetDevice(device));

        if (dwork) { CUDA_CHECK(cudaFree(dwork)); dwork = nullptr; }
        if (dS)    { CUDA_CHECK(cudaFree(dS)); dS = nullptr; }
        if (dC)    { CUDA_CHECK(cudaFree(dC)); dC = nullptr; }
        if (dinfo) { CUDA_CHECK(cudaFree(dinfo)); dinfo = nullptr; }

        if (cusolver) { CUSOLVER_CHECK(cusolverDnDestroy(cusolver)); cusolver = nullptr; }
        if (cublas)   { CUBLAS_CHECK(cublasDestroy(cublas)); cublas = nullptr; }
        if (stream)   { CUDA_CHECK(cudaStreamDestroy(stream)); stream = nullptr; }

        inited = false;
        cap_nsrow = cap_nscol = cap_nupd = 0;
        lwork = 0;
    }
};

// GPU: POTRF on L11, TRSM for L21, SYRK for update S
static bool gpu_factor_and_update(
    CudaCtx& ctx,
    int nsrow, int nscol,
    const std::vector<double>& C_host, // nsrow x nscol
    int nupd,
    std::vector<double>& C_out_host,   // nsrow x nscol (L block)
    std::vector<double>& S_out_host)   // nupd x nupd (full)
{
    ctx.ensure_C(nsrow, nscol);

    // Copy C to device
    CUDA_CHECK(cudaMemcpyAsync(ctx.dC, C_host.data(),
                               sizeof(double) * (size_t)nsrow * (size_t)nscol,
                               cudaMemcpyHostToDevice, ctx.stream));

    // POTRF on top-left nscol x nscol of dC
    ctx.ensure_potrf_workspace(nscol, nsrow);
    CUSOLVER_CHECK(cusolverDnDpotrf(ctx.cusolver, CUBLAS_FILL_MODE_LOWER,
                                    nscol, ctx.dC, nsrow,
                                    (double*)ctx.dwork, ctx.lwork,
                                    ctx.dinfo));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpyAsync(&info_h, ctx.dinfo, sizeof(int),
                               cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    if (info_h != 0) return false;

    // TRSM: L21 = A21 * inv(L11^T)
    if (nupd > 0) {
        const double alpha = 1.0;
        double* dA21 = ctx.dC + (size_t)nscol; // row offset
        CUBLAS_CHECK(cublasDtrsm(ctx.cublas,
                                CUBLAS_SIDE_RIGHT,
                                CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT,
                                nupd, nscol,
                                &alpha,
                                ctx.dC, nsrow,
                                dA21, nsrow));
    }

    // Copy back L block to host
    C_out_host.resize((size_t)nsrow * (size_t)nscol);
    CUDA_CHECK(cudaMemcpyAsync(C_out_host.data(), ctx.dC,
                               sizeof(double) * (size_t)nsrow * (size_t)nscol,
                               cudaMemcpyDeviceToHost, ctx.stream));

    // Update: S = F22 - L21*L21^T
    if (nupd > 0) {
        ctx.ensure_S(nupd);

        // initial S = F22 (host)
        CUDA_CHECK(cudaMemcpyAsync(ctx.dS, S_out_host.data(),
                                   sizeof(double) * (size_t)nupd * (size_t)nupd,
                                   cudaMemcpyHostToDevice, ctx.stream));

        // SYRK: S(lower) = (-1)*L21*L21^T + 1*S
        const double alpha = -1.0;
        const double beta  = 1.0;
        const double* dL21 = ctx.dC + (size_t)nscol; // row offset
        CUBLAS_CHECK(cublasDsyrk(ctx.cublas,
                                CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_N,
                                nupd, nscol,
                                &alpha,
                                dL21, nsrow,
                                &beta,
                                ctx.dS, nupd));

        CUDA_CHECK(cudaMemcpyAsync(S_out_host.data(), ctx.dS,
                                   sizeof(double) * (size_t)nupd * (size_t)nupd,
                                   cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        // Symmetrize full (host) from lower
        for (int j = 0; j < nupd; ++j)
            for (int i = 0; i < j; ++i)
                S_out_host[(size_t)i + (size_t)j * (size_t)nupd] =
                    S_out_host[(size_t)j + (size_t)i * (size_t)nupd];
    } else {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }

    // Deterministic: zero upper of L11 in C_out_host (nscol x nscol)
    for (int j = 0; j < nscol; ++j)
        for (int i = 0; i < j; ++i)
            C_out_host[(size_t)i + (size_t)j * (size_t)nsrow] = 0.0;

    return true;
}

struct ThreadLocal {
    std::vector<int> g2p;       // size n, maps global row -> position in current front
    std::vector<int> touched;   // list of global rows touched in g2p (for fast reset)
};

// Assemble C(nsrow x nscol) and initial S(nupd x nupd) without forming full F(nsrow x nsrow).
static void assemble_C_and_S(
    int k,
    const ichol::matrix::CscMatrix<double>& A,
    const symbolic::SuperSym& sym,
    const std::vector<std::vector<int>>& children,
    const std::vector<UpdatePack>& up,   // children updates (already computed)
    ThreadLocal& tl,
    std::vector<double>& C,
    std::vector<double>& S,
    int& nsrow_out,
    int& nscol_out,
    int& nupd_out,
    int& px0_out)
{
    CscView<double> Ac{A};

    const int scol  = sym.super[(size_t)k];
    const int ecol  = sym.super[(size_t)k + 1];
    const int nscol = ecol - scol;

    const int pi0   = sym.pi[(size_t)k];
    const int pi1   = sym.pi[(size_t)k + 1];
    const int nsrow = pi1 - pi0;

    const int px0   = sym.px[(size_t)k];

    nsrow_out = nsrow;
    nscol_out = nscol;
    nupd_out  = nsrow - nscol;
    px0_out   = px0;

    // Ensure mapping storage
    if ((int)tl.g2p.size() < A.num_cols) {
        tl.g2p.assign((size_t)A.num_cols, -1);
    }
    tl.touched.clear();

    // build g2p for rows in this front
    for (int t = 0; t < nsrow; ++t) {
        int r = sym.s[(size_t)(pi0 + t)];
        tl.g2p[(size_t)r] = t;
        tl.touched.push_back(r);
    }

    C.assign((size_t)nsrow * (size_t)nscol, 0.0);

    const int nupd = nsrow - nscol;
    if (nupd > 0) {
        S.assign((size_t)nupd * (size_t)nupd, 0.0);
    } else {
        S.clear();
    }

    // Scatter A lower (stype=-1 convention): only i>=j
    // Contributions land in C (columns are supernode columns), with symmetric fill inside F11.
    for (int j = scol; j < ecol; ++j) {
        int jpos = tl.g2p[(size_t)j];
        if (jpos < 0) continue;
        // j should be within first nscol positions
        for (int p = Ac.cb(j); p < Ac.ce(j); ++p) {
            int i = Ac.row(p);
            if (i < j) continue;
            int ipos = tl.g2p[(size_t)i];
            if (ipos < 0) continue;

            const double v = (double)Ac.val(p);

            // column jpos (supernode column)
            HCM(C, nsrow, ipos, jpos) += v;

            // symmetric counterpart only needed inside the leading nscol x nscol block
            if (ipos < nscol && ipos != jpos) {
                HCM(C, nsrow, jpos, ipos) += v;
            }
        }
    }

    // Add children updates (dense symmetric over child's update rows).
    for (int c : children[(size_t)k]) {
        const UpdatePack& uc = up[(size_t)c];
        const int m = uc.nupd;
        if (m <= 0) continue;

        for (int j = 0; j < m; ++j) {
            int gj = uc.idx[(size_t)j];
            int pj = tl.g2p[(size_t)gj];
            if (pj < 0) continue;

            for (int i = j; i < m; ++i) {
                int gi = uc.idx[(size_t)i];
                int pi = tl.g2p[(size_t)gi];
                if (pi < 0) continue;

                const double v = HCMc(uc.S, m, i, j);

                // Cases based on whether (pi,pj) are pivot( <nscol ) or update( >=nscol )
                if (pj < nscol) {
                    // column in supernode -> goes into C
                    HCM(C, nsrow, pi, pj) += v;
                    if (pi < nscol && pi != pj) {
                        // symmetric inside F11
                        HCM(C, nsrow, pj, pi) += v;
                    }
                } else if (pi < nscol) {
                    // symmetric counterpart lands in column pi (a supernode column)
                    HCM(C, nsrow, pj, pi) += v;
                } else {
                    // both in update region -> S
                    if (nupd > 0) {
                        int ui = pi - nscol;
                        int uj = pj - nscol;
                        S[(size_t)ui + (size_t)uj * (size_t)nupd] += v;
                        if (ui != uj) S[(size_t)uj + (size_t)ui * (size_t)nupd] += v;
                    }
                }
            }
        }
    }

    // reset g2p quickly
    for (int r : tl.touched) tl.g2p[(size_t)r] = -1;
}

SuperNumeric factorize_supernodal_ll_cuda(
    const ichol::matrix::CscMatrix<double>& A,
    const symbolic::SupernodalLLPlan& plan)
{
    CudaSupernodalOptions opt;
    return factorize_supernodal_ll_cuda(A, plan, opt);
}

SuperNumeric factorize_supernodal_ll_cuda(
    const ichol::matrix::CscMatrix<double>& A,
    const symbolic::SupernodalLLPlan& plan,
    const CudaSupernodalOptions& opt)
{
    SuperNumeric out;
    out.ok = true;
    out.fail_snode = -1;
    out.fail_col_in_snode = -1;
    out.sym = plan.sym;
    out.x.assign((size_t)plan.sym.px.back(), 0.0);

    std::string why;
    if (!cuda_runtime_usable(&why)) {
        out.ok = false;
        std::cerr << "[CUDA] unavailable: " << why << "\n";
        return out; // let tests decide to skip
    }

    const int n = A.num_cols;
    const int nsuper = (int)plan.sym.super.size() - 1;

    // Scheduling info should come from plan.
    // For legacy callers (if plan.children empty), compute a fallback schedule from SuperSym.
    std::vector<std::vector<int>> children_fallback;
    const std::vector<std::vector<int>>* children_ptr = &plan.children;
    std::vector<std::vector<int>> buckets_fallback;
    const std::vector<std::vector<int>>* buckets_ptr = &plan.buckets;

    if (plan.children.empty() || plan.buckets.empty()) {
        auto col2s  = symbolic::build_col2snode(plan.sym.super, n);
        auto parent = symbolic::build_snode_parent_from_rowlist(plan.sym, col2s);
        children_fallback = symbolic::build_children(parent);
        children_ptr = &children_fallback;

        auto level = symbolic::compute_level_from_leaves(children_fallback);
        buckets_fallback = symbolic::bucket_by_level(level);
        buckets_ptr = &buckets_fallback;
    }
    const auto& children = *children_ptr;
    const auto& buckets  = *buckets_ptr;

    // Decide stream/thread count
    int streams_used = std::max(1, opt.streams);
#ifdef _OPENMP
    // Respect the user's request, but don't exceed buckets parallelism too much.
    // (We still cap by a reasonable number to avoid oversubscription.)
    streams_used = std::min(streams_used, 64);
#else
    if (streams_used != 1) {
        std::cerr << "[GPU streams] OpenMP not enabled at compile time; forcing streams=1\n";
        streams_used = 1;
    }
#endif

    // Init per-stream contexts (no static global ctx -> avoids destructor-order issues).
    std::vector<CudaCtx> ctxs((size_t)streams_used);
    for (int s = 0; s < streams_used; ++s) {
        ctxs[(size_t)s].init(opt.device);
    }

    out.threads_used = streams_used;
    out.thread_work.assign((size_t)streams_used, 0);

    if (opt.print_schedule) {
        std::cerr << "[GPU schedule] levels=" << buckets.size()
                  << " streams_used=" << streams_used
                  << " device=" << opt.device << "\n";
        const int limit = std::max(0, opt.schedule_print_limit);
        for (size_t lv = 0; lv < buckets.size(); ++lv) {
            const auto& b = buckets[lv];
            std::cerr << "  level " << lv << ": snodes=" << b.size();
            if (limit > 0 && !b.empty()) {
                std::cerr << " preview:";
                for (int t = 0; t < (int)b.size() && t < limit; ++t) {
                    std::cerr << " " << b[(size_t)t] << "->s" << (t % streams_used);
                }
            }
            std::cerr << "\n";
        }
    }

    std::vector<UpdatePack> up((size_t)nsuper);
    std::atomic<bool> ok(true);

    // Thread-local mapping buffers (one per stream/thread)
    std::vector<ThreadLocal> tls((size_t)streams_used);
    for (int s = 0; s < streams_used; ++s) {
        tls[(size_t)s].g2p.assign((size_t)n, -1);
        tls[(size_t)s].touched.reserve(4096);
    }

    // Process level-by-level
    for (size_t lv = 0; lv < buckets.size(); ++lv) {
        const auto& bucket = buckets[lv];
        if (bucket.empty()) continue;

#ifdef _OPENMP
        #pragma omp parallel num_threads(streams_used)
#endif
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            CudaCtx& ctx = ctxs[(size_t)tid];
            ThreadLocal& tl = tls[(size_t)tid];

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int bi = 0; bi < (int)bucket.size(); ++bi) {
                if (!ok.load(std::memory_order_relaxed)) continue;

                const int k = bucket[(size_t)bi];

                std::vector<double> C, S;
                int nsrow=0, nscol=0, nupd=0, px0=0;

                assemble_C_and_S(k, A, plan.sym, children, up, tl, C, S, nsrow, nscol, nupd, px0);

                std::vector<double> Cfact;
                bool gok = gpu_factor_and_update(ctx, nsrow, nscol, C, nupd, Cfact, S);
                if (!gok) {
                    bool expected = true;
                    if (ok.compare_exchange_strong(expected, false)) {
                        out.ok = false;
                        out.fail_snode = k;
                        out.fail_col_in_snode = -1;
                    }
                    continue;
                }

                // write block to x
                std::copy(Cfact.begin(), Cfact.end(), out.x.begin() + (size_t)px0);

                // store update pack (for parent)
                UpdatePack& uk = up[(size_t)k];
                uk.nupd = nupd;
                uk.idx.clear();
                uk.S.clear();
                if (nupd > 0) {
                    uk.idx.resize((size_t)nupd);
                    const int pi0 = plan.sym.pi[(size_t)k];
                    for (int t = 0; t < nupd; ++t) {
                        uk.idx[(size_t)t] = plan.sym.s[(size_t)(pi0 + nscol + t)];
                    }
                    uk.S = std::move(S);
                }

#ifdef _OPENMP
                #pragma omp atomic
#endif
                out.thread_work[(size_t)tid] += 1;
            }
        } // end parallel

        if (!ok.load()) break;
    }

    if (opt.verbose) {
        std::cerr << "[GPU streams] streams_used=" << streams_used << " work_per_stream:";
        for (int s = 0; s < streams_used; ++s) std::cerr << " " << out.thread_work[(size_t)s];
        std::cerr << "\n";
    }

    // Cleanup
    for (auto& c : ctxs) c.shutdown();

    return out;
}

} // namespace ichol::numeric
