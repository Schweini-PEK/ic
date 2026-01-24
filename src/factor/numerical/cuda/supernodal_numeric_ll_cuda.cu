// ============================================================
// FILE: src/factor/numerical/cuda/supernodal_numeric_ll_cuda.cu
// ============================================================
#include "factor/numerical/supernodal_numeric_ll.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../../symbolic/snode_schedule.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <dlfcn.h>
#include <link.h>

#include "factor/symbolic/supernodal_ll_plan.hpp"

namespace ichol::symbolic
{
    struct SupernodalLLPlan;
}

namespace ichol::numeric {
    static std::string loaded_lib_path(const char* so)
    {
        void* h = dlopen(so, RTLD_NOW);
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
        std::string libcuda = loaded_lib_path("libcuda.so.1");
        std::cerr << "[CUDA] libcuda.so.1 loaded from: " << libcuda << "\n";
        if (libcuda.find("stubs") != std::string::npos) {
            if (why) *why = "Loaded CUDA stub libcuda from: " + libcuda;
            return false;
        }
        int driver = 0, runtime = 0;
        // debug: which libcuda are we loading?
        std::string libcuda_path = dl_path_of_loaded("libcuda.so.1");
        if (why) {
            *why = std::string("libcuda.so.1 loaded from: ") + libcuda_path;
        }

        // 如果加载到 stubs，直接判定不可用（这就是你 “No CUDA device available” 的常见根因）
        if (libcuda_path.find("stubs") != std::string::npos) {
            if (why) {
                *why = "Loaded CUDA stub library: " + libcuda_path +
                       " (remove /usr/local/cuda/lib64/stubs from LD_LIBRARY_PATH / rpath)";
            }
            return false;
        }
        cudaError_t e1 = cudaDriverGetVersion(&driver);
        cudaError_t e2 = cudaRuntimeGetVersion(&runtime);

        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            if (why) *why = "cudaDriverGetVersion/cudaRuntimeGetVersion failed";
            return false;
        }

        // runtime 版本通常要求 driver >= runtime 对应的最小驱动版本
        // 这里用最保守判定：driver < runtime => 直接认为不可用
        if (driver < runtime) {
            if (why) {
                *why = "CUDA driver too old: driver=" + std::to_string(driver) +
                       " runtime=" + std::to_string(runtime);
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
        std::vector<int> idx;
        std::vector<double> S; // host dense symmetric, full stored col-major (ld=nupd)
    };

    template <class T>
    struct CscView {
        const ichol::matrix::CscMatrix<T>& A;
        int cb(int j) const { return A.col_ptr[j]; }
        int ce(int j) const { return A.col_ptr[j + 1]; }
        int row(int p) const { return A.row_ind[p]; }
        T   val(int p) const { return A.values[p]; }
    };


    // -------------- CUDA context (reuse buffers) --------------
    struct CudaCtx {
        bool inited = false;
        cublasHandle_t cublas = nullptr;
        cusolverDnHandle_t cusolver = nullptr;
        cudaStream_t stream = nullptr;

        double* dC = nullptr;      // nsrow x nscol
        double* dS = nullptr;      // nupd x nupd (full)
        int*    dinfo = nullptr;
        void*   dwork = nullptr;
        int     lwork = 0;

        int cap_nsrow = 0;
        int cap_nscol = 0;
        int cap_nupd  = 0;

        void init()
        {
            if (inited) return;
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
            init();
            if (nsrow <= cap_nsrow && nscol <= cap_nscol && dC) return;
            if (dC) CUDA_CHECK(cudaFree(dC));
            cap_nsrow = std::max(cap_nsrow, nsrow);
            cap_nscol = std::max(cap_nscol, nscol);
            CUDA_CHECK(cudaMalloc((void**)&dC, sizeof(double) * (size_t)cap_nsrow * (size_t)cap_nscol));
        }

        void ensure_S(int nupd)
        {
            init();
            if (nupd <= cap_nupd && dS) return;
            if (dS) CUDA_CHECK(cudaFree(dS));
            cap_nupd = std::max(cap_nupd, nupd);
            CUDA_CHECK(cudaMalloc((void**)&dS, sizeof(double) * (size_t)cap_nupd * (size_t)cap_nupd));
        }

        void ensure_potrf_workspace(int n, int lda)
        {
            init();
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

            // Free device buffers
            if (dC) {
                CUDA_CHECK(cudaFree(dC));
                dC = nullptr;
            }
            if (dS) {
                CUDA_CHECK(cudaFree(dS));
                dS = nullptr;
            }
            if (dwork) {
                CUDA_CHECK(cudaFree(dwork));
                dwork = nullptr;
                lwork = 0;
            }
            if (dinfo) {
                CUDA_CHECK(cudaFree(dinfo));
                dinfo = nullptr;
            }

            // Destroy handles/stream
            if (cusolver) {
                CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
                cusolver = nullptr;
            }
            if (cublas) {
                CUBLAS_CHECK(cublasDestroy(cublas));
                cublas = nullptr;
            }
            if (stream) {
                CUDA_CHECK(cudaStreamDestroy(stream));
                stream = nullptr;
            }

            inited = false;
            cap_nsrow = 0;
            cap_nscol = 0;
            cap_nupd  = 0;
        }
    };

    static CudaCtx& cuda_ctx()
    {
        static CudaCtx ctx;
        return ctx;
    }

    void supernodal_cuda_shutdown()
    {
        cuda_ctx().shutdown();
    }

    // GPU: POTRF on L11, TRSM for L21, SYRK for update S
    static bool gpu_factor_and_update(
        int nsrow, int nscol,
        const std::vector<double>& C_host, // nsrow x nscol
        int nupd,
        std::vector<double>& C_out_host,   // nsrow x nscol (L block)
        std::vector<double>& S_out_host)   // nupd x nupd (full)
    {
        auto& ctx = cuda_ctx();
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
            // A21 starts at row nscol, col 0 in dC
            double* dA21 = ctx.dC + (size_t)nscol;
            // Right-side solve with L11^T
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

            // We need initial S = F22 (provided by host in S_out_host already)
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

            // Copy S back
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

        // Deterministic: zero upper of L11 in C_out_host
        for (int j = 0; j < nscol; ++j)
            for (int i = 0; i < j; ++i)
                C_out_host[(size_t)i + (size_t)j * (size_t)nsrow] = 0.0;

        return true;
    }

    static void compute_one_supernode_cpu_assemble_then_gpu_factor(
        int k,
        const ichol::matrix::CscMatrix<double>& A,
        const symbolic::SuperSym& sym,
        const std::vector<std::vector<int>>& children,
        std::vector<UpdatePack>& up,
        std::vector<double>& x,
        numeric::SuperNumeric& status,
        std::vector<int>& g2p)
    {
        if (!status.ok) return;

        CscView<double> Ac{A};

        const int scol  = sym.super[(size_t)k];
        const int ecol  = sym.super[(size_t)k + 1];
        const int nscol = ecol - scol;

        const int pi0   = sym.pi[(size_t)k];
        const int pi1   = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;

        const int px0   = sym.px[(size_t)k];

        // assemble symmetric front F (host)
        std::vector<double> F((size_t)nsrow * (size_t)nsrow, 0.0);

        std::fill(g2p.begin(), g2p.end(), -1);
        for (int t = 0; t < nsrow; ++t) {
            int r = sym.s[(size_t)(pi0 + t)];
            g2p[(size_t)r] = t;
        }

        // scatter A lower (assume stype=-1): only i>=j
        for (int j = scol; j < ecol; ++j) {
            int jpos = g2p[(size_t)j];
            if (jpos < 0) continue;

            for (int p = Ac.cb(j); p < Ac.ce(j); ++p) {
                int i = Ac.row(p);
                if (i < j) continue;
                int ipos = g2p[(size_t)i];
                if (ipos < 0) continue;

                double v = (double)Ac.val(p);
                HCM(F, nsrow, ipos, jpos) += v;
                if (ipos != jpos) HCM(F, nsrow, jpos, ipos) += v;
            }
        }

        // add children updates
        for (int c : children[(size_t)k]) {
            const UpdatePack& uc = up[(size_t)c];
            const int m = uc.nupd;
            if (m <= 0) continue;

            std::vector<int> pos((size_t)m, -1);
            for (int a = 0; a < m; ++a) pos[(size_t)a] = g2p[(size_t)uc.idx[(size_t)a]];

            for (int j = 0; j < m; ++j) {
                int pj = pos[(size_t)j];
                if (pj < 0) continue;
                for (int i = j; i < m; ++i) {
                    int pi = pos[(size_t)i];
                    if (pi < 0) continue;
                    double v = HCMc(uc.S, m, i, j);
                    HCM(F, nsrow, pi, pj) += v;
                    if (pi != pj) HCM(F, nsrow, pj, pi) += v;
                }
            }
        }

        // C = F(:, 0:nscol-1)
        std::vector<double> C((size_t)nsrow * (size_t)nscol, 0.0);
        for (int j = 0; j < nscol; ++j)
            for (int i = 0; i < nsrow; ++i)
                HCM(C, nsrow, i, j) = HCMc(F, nsrow, i, j);

        const int nupd = nsrow - nscol;

        // init S = F22
        std::vector<double> S;
        if (nupd > 0) {
            S.assign((size_t)nupd * (size_t)nupd, 0.0);
            for (int j = 0; j < nupd; ++j) {
                for (int i = j; i < nupd; ++i) {
                    double v = HCMc(F, nsrow, nscol + i, nscol + j);
                    S[(size_t)i + (size_t)j * (size_t)nupd] = v;
                    S[(size_t)j + (size_t)i * (size_t)nupd] = v;
                }
            }
        }

        std::vector<double> Cfact;
        bool ok = gpu_factor_and_update(nsrow, nscol, C, nupd, Cfact, S);
        if (!ok) {
            status.ok = false;
            status.fail_snode = k;
            status.fail_col_in_snode = -1;
            return;
        }

        // write block to x
        std::copy(Cfact.begin(), Cfact.end(), x.begin() + (size_t)px0);

        // store update pack
        UpdatePack& uk = up[(size_t)k];
        uk.nupd = nupd;
        uk.idx.clear();
        uk.S.clear();

        if (nupd > 0) {
            uk.idx.resize((size_t)nupd);
            for (int t = 0; t < nupd; ++t)
                uk.idx[(size_t)t] = sym.s[(size_t)(pi0 + nscol + t)];
            uk.S = std::move(S);
        }
    }

    numeric::SuperNumeric factorize_supernodal_ll_cuda(
        const ichol::matrix::CscMatrix<double>& A,
        const symbolic::SupernodalLLPlan& plan)
    {
        numeric::SuperNumeric out;

        out.ok = true;
        out.fail_snode = -1;
        out.fail_col_in_snode = -1;
        out.sym = plan.sym;
        out.x.assign((size_t)plan.sym.px.back(), 0.0);
        // ---- CUDA usable check (must be AFTER 'out' is defined) ----

        std::string why;

        if (!cuda_runtime_usable(&why)) {
            out.ok = false;
            std::cerr << "[CUDA] unavailable: " << why << "\n";
            return out; // let test GTEST_SKIP()
        }
        const int n = A.num_cols;
        const int nsuper = (int)plan.sym.super.size() - 1;

        // All symbolic scheduling info should come from 'plan' (no symbolic recomputation here).
        // For legacy callers that only pass SuperSym, plan.children may be empty.
        std::vector<std::vector<int>> children_fallback;
        const std::vector<std::vector<int>>* children_ptr = &plan.children;
        if (plan.children.empty()) {
            auto col2s  = symbolic::build_col2snode(plan.sym.super, n);
            auto parent = symbolic::build_snode_parent_from_rowlist(plan.sym, col2s);
            children_fallback = symbolic::build_children(parent);
            children_ptr = &children_fallback;
        }
        const auto& children = *children_ptr;

        // 按 CHOLMOD 的树，这个 case 是链：顺序执行即可
        //（如果未来换 ordering 树变宽，你可以做 level-set + 多 stream，但这版先把 GPU 数值核跑通）
        std::vector<UpdatePack> up((size_t)nsuper);

        std::vector<int> g2p((size_t)n, -1);
        for (int k = 0; k < nsuper; ++k) {
            compute_one_supernode_cpu_assemble_then_gpu_factor(
                k, A, plan.sym, children, up, out.x, out, g2p);
            if (!out.ok) break;
        }

        return out;
    }

} // namespace ichol::symbolic
