#include "factor/numerical/supernodal_numeric_ll.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <dlfcn.h>
#include <link.h>

#include "factor/symbolic/supernodal_ll_plan.hpp"

namespace ichol::numeric
{
    // ------------------ helpers: dlopen path check ------------------
    static std::string dl_path_of_loaded(const char* soname)
    {
        void* h = dlopen(soname, RTLD_NOW);
        if (!h) return std::string("dlopen failed: ") + dlerror();

        link_map* lm = nullptr;
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
        // WSL/容器里常见坑：误加载 stubs/libcuda
        std::string libcuda_path = dl_path_of_loaded("libcuda.so.1");
        std::cerr << "[CUDA] libcuda.so.1 loaded from: " << libcuda_path << "\n";

        if (libcuda_path.find("stubs") != std::string::npos) {
            if (why) {
                *why = "Loaded CUDA stub library: " + libcuda_path +
                       " (remove /usr/local/cuda/lib64/stubs from LD_LIBRARY_PATH / rpath)";
            }
            return false;
        }

        int driver = 0, runtime = 0;
        cudaError_t e1 = cudaDriverGetVersion(&driver);
        cudaError_t e2 = cudaRuntimeGetVersion(&runtime);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            if (why) *why = "cudaDriverGetVersion/cudaRuntimeGetVersion failed";
            return false;
        }
        if (driver < runtime) {
            if (why) *why = "CUDA driver too old: driver=" + std::to_string(driver) +
                            " runtime=" + std::to_string(runtime);
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

    template <class T>
    struct CscView {
        const ichol::matrix::CscMatrix<T>& A;
        int cb(int j) const { return A.col_ptr[(size_t)j]; }
        int ce(int j) const { return A.col_ptr[(size_t)j + 1]; }
        int row(int p) const { return A.row_ind[(size_t)p]; }
        T   val(int p) const { return A.values[(size_t)p]; }
    };

    // ------------------ update pack (host) ------------------
    struct UpdatePack {
        int nupd = 0;
        std::vector<int> idx;       // global row indices of update rows
        std::vector<double> S;      // dense symmetric (full stored), col-major ld=nupd
    };

    // ------------------ CPU fallback: factor (panel) + update ------------------
    static bool cpu_factor_and_update(int nsrow, int nscol,
                                     std::vector<double>& C,   // in/out, ld=nsrow
                                     int nupd,
                                     std::vector<double>& S)   // in/out full, ld=nupd
    {
        for (int j = 0; j < nscol; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                const double ljk = HCMc(C, nsrow, j, k);
                sum += ljk * ljk;
            }
            double ajj = HCMc(C, nsrow, j, j) - sum;
            if (ajj <= 0.0 || !std::isfinite(ajj)) return false;
            const double ljj = std::sqrt(ajj);
            HCM(C, nsrow, j, j) = ljj;

            for (int i = j + 1; i < nsrow; ++i) {
                double s = 0.0;
                for (int k = 0; k < j; ++k) {
                    s += HCMc(C, nsrow, i, k) * HCMc(C, nsrow, j, k);
                }
                HCM(C, nsrow, i, j) = (HCMc(C, nsrow, i, j) - s) / ljj;
            }

            for (int i = 0; i < j; ++i) HCM(C, nsrow, i, j) = 0.0;
        }

        if (nupd > 0) {
            const int row0 = nscol;
            for (int j = 0; j < nupd; ++j) {
                for (int i = j; i < nupd; ++i) {
                    double acc = 0.0;
                    for (int k = 0; k < nscol; ++k) {
                        const double lik = HCMc(C, nsrow, row0 + i, k);
                        const double ljk = HCMc(C, nsrow, row0 + j, k);
                        acc += lik * ljk;
                    }
                    const double v = S[(size_t)i + (size_t)j * (size_t)nupd] - acc;
                    S[(size_t)i + (size_t)j * (size_t)nupd] = v;
                    S[(size_t)j + (size_t)i * (size_t)nupd] = v;
                }
            }
        }
        return true;
    }

    // ------------------ CUDA multi-stream context ------------------
    struct CudaStreamState {
        cudaStream_t stream = nullptr;
        cublasHandle_t cublas = nullptr;
        cusolverDnHandle_t cusolver = nullptr;

        double* dC = nullptr;   // cap_nsrow * cap_nscol
        double* dS = nullptr;   // cap_nupd * cap_nupd
        int* dinfo = nullptr;
        void* dwork = nullptr;
        int lwork = 0;

        int cap_nsrow = 0;
        int cap_nscol = 0;
        int cap_nupd  = 0;
    };

    struct CudaCtx {
        bool inited = false;
        int device = -1;
        int nstreams = 0;
        std::vector<CudaStreamState> st;

        void init(int dev, int streams)
        {
            if (inited && dev == device && streams == nstreams) return;
            shutdown();

            device = dev;
            nstreams = std::max(1, streams);
            CUDA_CHECK(cudaSetDevice(device));

            (void)cudaFree(nullptr);
            (void)cudaDeviceSynchronize();

            st.resize((size_t)nstreams);

            for (int s = 0; s < nstreams; ++s) {
                auto& ss = st[(size_t)s];
                CUDA_CHECK(cudaStreamCreateWithFlags(&ss.stream, cudaStreamNonBlocking));
                CUBLAS_CHECK(cublasCreate(&ss.cublas));
                CUSOLVER_CHECK(cusolverDnCreate(&ss.cusolver));
                CUBLAS_CHECK(cublasSetStream(ss.cublas, ss.stream));
                CUSOLVER_CHECK(cusolverDnSetStream(ss.cusolver, ss.stream));
                CUDA_CHECK(cudaMalloc((void**)&ss.dinfo, sizeof(int)));
            }
            inited = true;
        }

        void ensure_C(int sid, int nsrow, int nscol)
        {
            auto& ss = st[(size_t)sid];
            if (nsrow <= ss.cap_nsrow && nscol <= ss.cap_nscol && ss.dC) return;
            // If there are queued ops on this stream using the old buffer,
            // freeing/reallocating it would corrupt results. Synchronize the
            // stream before reallocation.
            if (ss.dC) {
                CUDA_CHECK(cudaStreamSynchronize(ss.stream));
                CUDA_CHECK(cudaFree(ss.dC));
            }
            ss.cap_nsrow = std::max(ss.cap_nsrow, nsrow);
            ss.cap_nscol = std::max(ss.cap_nscol, nscol);
            CUDA_CHECK(cudaMalloc((void**)&ss.dC, sizeof(double) * (size_t)ss.cap_nsrow * (size_t)ss.cap_nscol));
            ss.lwork = 0;
            if (ss.dwork) {
                CUDA_CHECK(cudaStreamSynchronize(ss.stream));
                CUDA_CHECK(cudaFree(ss.dwork));
                ss.dwork = nullptr;
            }
        }

        void ensure_S(int sid, int nupd)
        {
            auto& ss = st[(size_t)sid];
            if (nupd <= ss.cap_nupd && ss.dS) return;
            if (ss.dS) {
                CUDA_CHECK(cudaStreamSynchronize(ss.stream));
                CUDA_CHECK(cudaFree(ss.dS));
            }
            ss.cap_nupd = std::max(ss.cap_nupd, nupd);
            CUDA_CHECK(cudaMalloc((void**)&ss.dS, sizeof(double) * (size_t)ss.cap_nupd * (size_t)ss.cap_nupd));
        }

        void ensure_potrf_workspace(int sid, int n, int lda)
        {
            auto& ss = st[(size_t)sid];
            int lw = 0;
            CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(ss.cusolver, CUBLAS_FILL_MODE_LOWER, n, ss.dC, lda, &lw));
            if (lw <= ss.lwork && ss.dwork) return;
            if (ss.dwork) {
                CUDA_CHECK(cudaStreamSynchronize(ss.stream));
                CUDA_CHECK(cudaFree(ss.dwork));
            }
            ss.lwork = lw;
            CUDA_CHECK(cudaMalloc(&ss.dwork, sizeof(double) * (size_t)ss.lwork));
        }

        void sync_all()
        {
            for (int s = 0; s < nstreams; ++s) {
                CUDA_CHECK(cudaStreamSynchronize(st[(size_t)s].stream));
            }
        }

        void shutdown()
        {
            if (!inited) return;

            for (auto& ss : st) {
                if (ss.dC) CUDA_CHECK(cudaFree(ss.dC));
                if (ss.dS) CUDA_CHECK(cudaFree(ss.dS));
                if (ss.dwork) CUDA_CHECK(cudaFree(ss.dwork));
                if (ss.dinfo) CUDA_CHECK(cudaFree(ss.dinfo));

                ss.dC = ss.dS = nullptr;
                ss.dwork = nullptr;
                ss.dinfo = nullptr;
                ss.lwork = 0;

                if (ss.cusolver) CUSOLVER_CHECK(cusolverDnDestroy(ss.cusolver));
                if (ss.cublas)   CUBLAS_CHECK(cublasDestroy(ss.cublas));
                if (ss.stream)   CUDA_CHECK(cudaStreamDestroy(ss.stream));

                ss.cusolver = nullptr;
                ss.cublas = nullptr;
                ss.stream = nullptr;
                ss.cap_nsrow = ss.cap_nscol = ss.cap_nupd = 0;
            }

            st.clear();
            inited = false;
            device = -1;
            nstreams = 0;
        }
    };

    static CudaCtx& cuda_ctx()
    {
        static CudaCtx ctx;
        return ctx;
    }

    // enqueue POTRF/TRSM/SYRK + D2H.
    // By default everything is enqueued asynchronously onto the stream.
    // For debugging correctness issues (race/lifetime), you can force
    // serialization by setting env ICHOL_CUDA_SERIALIZE=1.
    static bool gpu_factor_and_update_enqueue(CudaCtx& ctx,
                                             int sid,
                                             int nsrow, int nscol,
                                             const std::vector<double>& C_host,
                                             int nupd,
                                             std::vector<double>& C_out_host,
                                             std::vector<double>& S_out_host,
                                             int* host_info)
    {
        auto& ss = ctx.st[(size_t)sid];
        const bool serialize = (std::getenv("ICHOL_CUDA_SERIALIZE") != nullptr);
        ctx.ensure_C(sid, nsrow, nscol);

        if (serialize) {
            CUDA_CHECK(cudaMemcpy(ss.dC, C_host.data(),
                                  sizeof(double) * (size_t)nsrow * (size_t)nscol,
                                  cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(ss.dC, C_host.data(),
                                       sizeof(double) * (size_t)nsrow * (size_t)nscol,
                                       cudaMemcpyHostToDevice, ss.stream));
        }

        ctx.ensure_potrf_workspace(sid, nscol, nsrow);
        CUSOLVER_CHECK(cusolverDnDpotrf(ss.cusolver,
                                       CUBLAS_FILL_MODE_LOWER,
                                       nscol,
                                       ss.dC,
                                       nsrow,
                                       (double*)ss.dwork,
                                       ss.lwork,
                                       ss.dinfo));

        if (host_info) {
            if (serialize) {
                CUDA_CHECK(cudaMemcpy(host_info, ss.dinfo, sizeof(int), cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(host_info, ss.dinfo, sizeof(int),
                                           cudaMemcpyDeviceToHost, ss.stream));
            }
        }

        if (nsrow > nscol) {
            const double one = 1.0;
            double* dL11 = ss.dC;
            double* dA21 = ss.dC + (size_t)nscol;
            CUBLAS_CHECK(cublasDtrsm(ss.cublas,
                                    CUBLAS_SIDE_RIGHT,
                                    CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_T,
                                    CUBLAS_DIAG_NON_UNIT,
                                    nsrow - nscol,
                                    nscol,
                                    &one,
                                    dL11, nsrow,
                                    dA21, nsrow));
        }

        C_out_host.resize((size_t)nsrow * (size_t)nscol);
        if (serialize) {
            CUDA_CHECK(cudaMemcpy(C_out_host.data(), ss.dC,
                                  sizeof(double) * (size_t)nsrow * (size_t)nscol,
                                  cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(C_out_host.data(), ss.dC,
                                       sizeof(double) * (size_t)nsrow * (size_t)nscol,
                                       cudaMemcpyDeviceToHost, ss.stream));
        }

        if (nupd > 0) {
            ctx.ensure_S(sid, nupd);

            if (serialize) {
                CUDA_CHECK(cudaMemcpy(ss.dS, S_out_host.data(),
                                      sizeof(double) * (size_t)nupd * (size_t)nupd,
                                      cudaMemcpyHostToDevice));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(ss.dS, S_out_host.data(),
                                           sizeof(double) * (size_t)nupd * (size_t)nupd,
                                           cudaMemcpyHostToDevice, ss.stream));
            }

            const double alpha = -1.0;
            const double beta  = 1.0;
            double* dL21 = ss.dC + (size_t)nscol;

            CUBLAS_CHECK(cublasDsyrk(ss.cublas,
                                    CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_N,
                                    nupd, nscol,
                                    &alpha,
                                    dL21, nsrow,
                                    &beta,
                                    ss.dS, nupd));

            if (serialize) {
                CUDA_CHECK(cudaMemcpy(S_out_host.data(), ss.dS,
                                      sizeof(double) * (size_t)nupd * (size_t)nupd,
                                      cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(S_out_host.data(), ss.dS,
                                           sizeof(double) * (size_t)nupd * (size_t)nupd,
                                           cudaMemcpyDeviceToHost, ss.stream));
            }
        }

        if (serialize) {
            // Ensure all ops are complete before we return.
            CUDA_CHECK(cudaStreamSynchronize(ss.stream));
        }

        return true;
    }

    // Build depth buckets from parent pointers.
    // Depth is measured from the root (root depth = 0). For numeric factorization we
    // must process **leaves -> root** so that each parent front merges all child
    // update blocks that were produced earlier.
    static std::vector<std::vector<int>> build_levels_from_parent(const std::vector<int>& parent)
    {
        const int n = (int)parent.size();
        std::vector<int> depth((size_t)n, 0);
        int maxd = 0;

        for (int i = 0; i < n; ++i) {
            int d = 0;
            int p = parent[(size_t)i];
            while (p >= 0) {
                ++d;
                p = parent[(size_t)p];
            }
            depth[(size_t)i] = d;
            maxd = std::max(maxd, d);
        }

        std::vector<std::vector<int>> levels((size_t)maxd + 1);
        for (int i = 0; i < n; ++i) {
            levels[(size_t)depth[(size_t)i]].push_back(i);
        }
        return levels;
    }

    // GPU pending task
    struct PendingGpuTask {
        int k = -1;
        int sid = 0;

        int nsrow = 0;
        int nscol = 0;
        int nupd  = 0;
        int px0   = 0;

        int info = 0; // potrf info

        // Keep host-side input alive until the stream finishes the H2D copy.
        // Using a temporary vector for C_host and then enqueueing cudaMemcpyAsync
        // can lead to use-after-free if the copy is truly async.
        std::vector<double> C_in;

        // Keep a copy of the pre-update F22 (S) for optional debug checks.
        std::vector<double> S_in;

        std::vector<double> Cfact;  // D2H writes here
        std::vector<double> S;      // D2H writes here (full)
        std::vector<int> upd_idx;   // update row global indices
    };

    // Assemble A contribution into F, extract C and initial F22 (S), plus update idx.
    // (simple correctness-first assemble)
    static void assemble_front_C_and_F22(int k,
                                         const ichol::matrix::CscMatrix<double>& A,
                                         const symbolic::SupernodalLLPlan& plan,
                                         std::vector<int>& g2p,
                                         int& nsrow, int& nscol, int& nupd,
                                         int& scol, int& pi0, int& px0,
                                         std::vector<double>& C,
                                         std::vector<double>& S,
                                         std::vector<int>& upd_idx)
    {
        const auto& sym = plan.sym;
        CscView<double> Ac{A};

        scol  = sym.super[(size_t)k];
        const int ecol  = sym.super[(size_t)k + 1];
        nscol = ecol - scol;

        pi0   = sym.pi[(size_t)k];
        const int pi1   = sym.pi[(size_t)k + 1];
        nsrow = pi1 - pi0;

        px0   = sym.px[(size_t)k];
        nupd  = nsrow - nscol;

        std::vector<double> F((size_t)nsrow * (size_t)nsrow, 0.0);

        std::fill(g2p.begin(), g2p.end(), -1);
        for (int t = 0; t < nsrow; ++t) {
            int r = sym.s[(size_t)(pi0 + t)];
            g2p[(size_t)r] = t;
        }

        for (int jnew = scol; jnew < ecol; ++jnew) {
            const int jold = plan.perm.perm.empty() ? jnew : plan.perm.perm[(size_t)jnew];
            const int jpos = g2p[(size_t)jnew];
            if (jpos < 0) continue;

            for (int p = Ac.cb(jold); p < Ac.ce(jold); ++p) {
                const int iold = Ac.row(p);

                // Map to new ordering if a permutation is present.
                const int inew = plan.perm.inv_perm.empty() ? iold : plan.perm.inv_perm[(size_t)iold];

                // We only assemble the lower triangle in the *new* ordering.
                if (inew < jnew) continue;

                const int ipos = g2p[(size_t)inew];
                if (ipos < 0) continue;

                const double v = (double)Ac.val(p);
                HCM(F, nsrow, ipos, jpos) += v;
                if (ipos != jpos) HCM(F, nsrow, jpos, ipos) += v;
            }
        }


        // C = F(:, 0:nscol-1)
        C.assign((size_t)nsrow * (size_t)nscol, 0.0);
        for (int j = 0; j < nscol; ++j)
            for (int i = 0; i < nsrow; ++i)
                HCM(C, nsrow, i, j) = HCMc(F, nsrow, i, j);

        // S = F22
        S.clear();
        upd_idx.clear();
        if (nupd > 0) {
            S.assign((size_t)nupd * (size_t)nupd, 0.0);
            for (int j = 0; j < nupd; ++j) {
                for (int i = j; i < nupd; ++i) {
                    double v = HCMc(F, nsrow, nscol + i, nscol + j);
                    S[(size_t)i + (size_t)j * (size_t)nupd] = v;
                    S[(size_t)j + (size_t)i * (size_t)nupd] = v;
                }
            }
            upd_idx.resize((size_t)nupd);
            for (int t = 0; t < nupd; ++t)
                upd_idx[(size_t)t] = sym.s[(size_t)(pi0 + nscol + t)];
        }
    }

// Merge child update blocks (from up[]) into parent's S (F22)
static void merge_children_updates_into_S(int k,
                                         const symbolic::SupernodalLLPlan& plan,
                                         const std::vector<UpdatePack>& up,
                                         const std::vector<int>& g2p,
                                         int nscol, int nupd,
                                         std::vector<double>& S)
{
    if (nupd <= 0) return;
    const auto& children = plan.children;
    if (children.empty()) return;

    for (int c : children[(size_t)k]) {
        const UpdatePack& uc = up[(size_t)c];
        const int m = uc.nupd;
        if (m <= 0) continue;

        std::vector<int> pos((size_t)m, -1);
        for (int a = 0; a < m; ++a) {
            int p = g2p[(size_t)uc.idx[(size_t)a]];
            if (p >= nscol) pos[(size_t)a] = p - nscol;
        }

        for (int jj = 0; jj < m; ++jj) {
            int pj = pos[(size_t)jj];
            if (pj < 0 || pj >= nupd) continue;
            for (int ii = jj; ii < m; ++ii) {
                int pi = pos[(size_t)ii];
                if (pi < 0 || pi >= nupd) continue;

                double v = HCMc(uc.S, m, ii, jj);
                S[(size_t)pi + (size_t)pj * (size_t)nupd] += v;
                S[(size_t)pj + (size_t)pi * (size_t)nupd] += v;
            }
        }
    }
}

// finalize after barrier: mutate task (so we can edit Cfact/S), then writeback
static bool finalize_gpu_task(PendingGpuTask& t,
                             std::vector<double>& x,
                             std::vector<UpdatePack>& up)
{
    // Optional supernode-level correctness debug:
    //   export ICHOL_DEBUG_SNODE=<k>
    // This will run the CPU fallback kernel on the *same assembled front* and
    // print the max difference against the GPU result for that one supernode.
    // (Useful to pinpoint whether the divergence starts at POTRF/TRSM/SYRK
    //  or later writeback/merge logic.)
    if (t.info != 0) {
        std::cerr << "[CUDA] potrf info != 0 for snode " << t.k << " info=" << t.info << "\n";
        return false;
    }

    // Optional: compare GPU (POTRF/TRSM/SYRK) against the CPU fallback kernels
    // on the exact same assembled front to pinpoint where divergence starts.
    if (const char* dbg = std::getenv("ICHOL_DEBUG_SNODE")) {
        const int dbg_k = std::atoi(dbg);
        if (dbg_k == t.k) {
            std::vector<double> Ccpu = t.C_in;   // nsrow*nscol
            std::vector<double> Scpu = t.S_in;   // nupd*nupd (full)
            bool ok = cpu_factor_and_update(t.nsrow, t.nscol, Ccpu, t.nupd, Scpu);
            if (!ok) {
                std::cerr << "[DBG] CPU fallback failed for snode " << t.k << "\n";
                return false;
            }

            double max_abs_C = 0.0;
            for (size_t i = 0; i < Ccpu.size(); ++i) {
                max_abs_C = std::max(max_abs_C, std::abs(Ccpu[i] - t.Cfact[i]));
            }
            double max_abs_S = 0.0;
            if (t.nupd > 0) {
                for (size_t i = 0; i < Scpu.size(); ++i) {
                    max_abs_S = std::max(max_abs_S, std::abs(Scpu[i] - t.S[i]));
                }
            }

            std::cerr << "[DBG] snode=" << t.k
                      << " nsrow=" << t.nsrow << " nscol=" << t.nscol << " nupd=" << t.nupd
                      << " max_abs_C=" << max_abs_C
                      << " max_abs_S=" << max_abs_S
                      << " (set ICHOL_CUDA_SERIALIZE=1 to force sync copies)\n";
        }
    }

    // zero upper in L11 for determinism
    for (int j = 0; j < t.nscol; ++j) {
        for (int i = 0; i < j; ++i) {
            t.Cfact[(size_t)i + (size_t)j * (size_t)t.nsrow] = 0.0;
        }
    }

    // symmetrize S from lower
    if (t.nupd > 0) {
        const int m = t.nupd;
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < j; ++i) {
                t.S[(size_t)i + (size_t)j * (size_t)m] =
                    t.S[(size_t)j + (size_t)i * (size_t)m];
            }
        }
    }

    std::copy(t.Cfact.begin(), t.Cfact.end(), x.begin() + (size_t)t.px0);

    UpdatePack& uk = up[(size_t)t.k];
    uk.nupd = t.nupd;
    uk.idx = std::move(t.upd_idx);
    uk.S = std::move(t.S);

    return true;
}

// ------------------ exported entrypoints ------------------
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
    // out.sym set after schedule
    // out.x allocated after schedule

    std::string why;
    if (!cuda_runtime_usable(&why)) {
        out.ok = false;
        std::cerr << "[CUDA] unavailable: " << why << "\n";
        return out;
    }

    // Ensure the supernode DAG schedule (parent/children/levels/relpos) is available.
    // Some callers may only populate (perm,sym) for numeric; GPU numeric needs children/levels
    // to merge child updates into parent fronts.
    symbolic::SupernodalLLPlan plan_work = plan;
    const int ncols = A.num_cols;
    const int nsuper = (int)plan_work.sym.super.size() - 1;
    if ((int)plan_work.children.size() != nsuper ||
        (int)plan_work.parent.size() != nsuper ||
        (int)plan_work.level.size() != nsuper ||
        plan_work.buckets.empty())
    {
        fill_schedule_from_sym(plan_work, ncols);
    }
    else if (plan_work.child_relpos.empty())
    {
        fill_child_relpos_from_sym(plan_work, ncols);
    }

    out.sym = plan_work.sym;
    out.x.assign((size_t)plan_work.sym.px.back(), 0.0);


    // parent from children
    std::vector<int> parent((size_t)nsuper, -1);
    for (int p = 0; p < nsuper; ++p) {
        for (int c : plan_work.children[(size_t)p]) parent[(size_t)c] = p;
    }
    auto levels = build_levels_from_parent(parent);

    auto& ctx = cuda_ctx();
    const int S = std::max(1, opt.streams);
    ctx.init(opt.device, S);

    std::vector<UpdatePack> up((size_t)nsuper);
    std::vector<int> g2p((size_t)A.num_cols, -1);

    out.threads_used = S;
    out.thread_work.assign((size_t)S, 0);

    // IMPORTANT: process leaves -> root (reverse depth order).
    for (int li = (int)levels.size() - 1; li >= 0; --li) {
        const auto& lvl = levels[(size_t)li];
        if (!out.ok) break;

        if (opt.print_schedule) {
            std::cerr << "[GPU schedule] level=" << li << " snodes=" << lvl.size() << "\n";
            const int shown = std::min((int)lvl.size(), std::max(0, opt.schedule_print_limit));
            for (int i = 0; i < shown; ++i) {
                const int k = lvl[(size_t)i];
                std::cerr << "  k=" << k << " -> stream=" << (i % S) << "\n";
            }
        }

        std::vector<PendingGpuTask> pend;
        pend.reserve(lvl.size());

        int rr = 0;
        for (int k : lvl) {
            if (!out.ok) break;

            int nsrow = 0, nscol = 0, nupd = 0;
            int scol = 0, pi0 = 0, px0 = 0;
            std::vector<double> C;
            std::vector<double> Sinit;
            std::vector<int> upd_idx;

            assemble_front_C_and_F22(k, A, plan_work, g2p, nsrow, nscol, nupd, scol, pi0, px0, C, Sinit, upd_idx);

            // merge child updates into S only (F22)
            merge_children_updates_into_S(k, plan_work, up, g2p, nscol, nupd, Sinit);

            const int sid = rr % S;
            rr++;
            out.thread_work[(size_t)sid]++;

            // CPU fallback for small nscol
            if (opt.cpu_fallback_max_n > 0 && nscol <= opt.cpu_fallback_max_n) {
                bool ok = cpu_factor_and_update(nsrow, nscol, C, nupd, Sinit);
                if (!ok) {
                    out.ok = false;
                    out.fail_snode = k;
                    out.fail_col_in_snode = -1;
                    break;
                }

                std::copy(C.begin(), C.end(), out.x.begin() + (size_t)px0);

                UpdatePack& uk = up[(size_t)k];
                uk.nupd = nupd;
                uk.idx = std::move(upd_idx);
                uk.S = std::move(Sinit);
                if (uk.nupd > 0) {
                    const int m = uk.nupd;
                    for (int j = 0; j < m; ++j)
                        for (int i = 0; i < j; ++i)
                            uk.S[(size_t)i + (size_t)j * (size_t)m] =
                                uk.S[(size_t)j + (size_t)i * (size_t)m];
                }
                continue;
            }

            // GPU enqueue
            try {
                PendingGpuTask t;
                t.k = k;
                t.sid = sid;
                t.nsrow = nsrow;
                t.nscol = nscol;
                t.nupd  = nupd;
                t.px0   = px0;
                t.info  = 0;
                t.C_in = std::move(C);
                t.S = std::move(Sinit);
                t.S_in = t.S; // debug snapshot (cheap for small, acceptable for debug)
                t.upd_idx = std::move(upd_idx);

                bool ok = gpu_factor_and_update_enqueue(ctx, sid, nsrow, nscol, t.C_in, nupd,
                                                       t.Cfact, t.S, &t.info);
                if (!ok) {
                    out.ok = false;
                    out.fail_snode = k;
                    out.fail_col_in_snode = -1;
                    break;
                }

                pend.push_back(std::move(t));
            } catch (const std::exception& e) {
                std::cerr << "[CUDA] exception: " << e.what() << "\n";
                out.ok = false;
                out.fail_snode = k;
                out.fail_col_in_snode = -1;
                break;
            }
        }

        // barrier for this level
        if (!pend.empty()) {
            ctx.sync_all();

            for (auto& t : pend) {
                if (!finalize_gpu_task(t, out.x, up)) {
                    out.ok = false;
                    out.fail_snode = t.k;
                    out.fail_col_in_snode = -1;
                    break;
                }
            }
        }
    }

    return out;
}

} // namespace ichol::numeric
