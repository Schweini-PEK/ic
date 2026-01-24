#include "factor/numerical/supernodal_numeric_ll.hpp"
#include "factor/symbolic/super_sym.hpp"
#include "../symbolic/snode_schedule.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <vector>

#if defined(ICHOL_USE_OPENMP)
#include <omp.h>
#endif

#if defined(ICHOL_USE_BLAS)
extern "C"
{
    void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);

    void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
                const int *m, const int *n,
                const double *alpha, const double *a, const int *lda,
                double *b, const int *ldb);

    void dsyrk_(const char *uplo, const char *trans,
                const int *n, const int *k,
                const double *alpha, const double *a, const int *lda,
                const double *beta, double *c, const int *ldc);
}
#endif

namespace ichol::numeric
{
    static inline double &CM(std::vector<double> &a, int ld, int r, int c)
    {
        return a[(size_t)r + (size_t)c * (size_t)ld];
    }
    static inline double CMc(const std::vector<double> &a, int ld, int r, int c)
    {
        return a[(size_t)r + (size_t)c * (size_t)ld];
    }

    struct UpdatePack
    {
        int nupd = 0;
        std::vector<int> idx;  // global indices of update rows
        std::vector<double> S; // dense symmetric (full stored), column-major, ld=nupd
    };

    template <class T>
    struct CscView
    {
        const ichol::matrix::CscMatrix<T> &A;
        int cb(int j) const { return A.col_ptr[j]; }
        int ce(int j) const { return A.col_ptr[j + 1]; }
        int row(int p) const { return A.row_ind[p]; }
        T val(int p) const { return A.values[p]; }
    };

#if defined(ICHOL_USE_BLAS)
    static bool chol_ll_potrf_trsm(double *C, int ld, int nsrow, int nscol, int &fail_col)
    {
        fail_col = -1;

        // POTRF on L11 (lower)
        char uplo = 'L';
        int n = nscol;
        int lda = ld;
        int info = 0;
        dpotrf_(&uplo, &n, C, &lda, &info);
        if (info != 0)
        {
            fail_col = std::max(0, info - 1);
            return false;
        }

        // TRSM: L21 = A21 * inv(L11^T)
        const int nupd = nsrow - nscol;
        if (nupd > 0)
        {
            char side = 'R', upl = 'L', trans = 'T', diag = 'N';
            int m = nupd;
            int k = nscol;
            double alpha = 1.0;
            double *A21 = C + (size_t)nscol; // row offset nscol, col 0
            dtrsm_(&side, &upl, &trans, &diag, &m, &k, &alpha, C, &lda, A21, &lda);
        }

        // make deterministic (optional): zero upper of L11
        for (int j = 0; j < nscol; ++j)
            for (int i = 0; i < j; ++i)
                C[i + j * ld] = 0.0;

        return true;
    }
#else
    static bool chol_ll_naive(double *C, int ld, int nsrow, int nscol, int &fail_col)
    {
        fail_col = -1;
        for (int j = 0; j < nscol; ++j)
        {
            double d = C[j + j * ld];
            for (int k = 0; k < j; ++k)
            {
                double ljk = C[j + k * ld];
                d -= ljk * ljk;
            }
            if (!(d > 0.0))
            {
                fail_col = j;
                return false;
            }
            double ljj = std::sqrt(d);
            C[j + j * ld] = ljj;

            for (int i = j + 1; i < nsrow; ++i)
            {
                double v = C[i + j * ld];
                for (int k = 0; k < j; ++k)
                    v -= C[i + k * ld] * C[j + k * ld];
                C[i + j * ld] = v / ljj;
            }
        }
        // deterministic
        for (int j = 0; j < nscol; ++j)
            for (int i = 0; i < j; ++i)
                C[i + j * ld] = 0.0;
        return true;
    }
#endif

    static void compute_one_supernode(
        int k,
        const ichol::matrix::CscMatrix<double> &A,
        const symbolic::SuperSym &sym,
        const std::vector<std::vector<int>> &children,
        std::vector<UpdatePack> &up,
        std::vector<double> &x,
        numeric::SuperNumeric &status,
        std::vector<int> &g2p) // per-thread scratch: size n, init -1
    {
        if (!status.ok)
            return;

        CscView<double> Ac{A};

        const int scol = sym.super[(size_t)k];
        const int ecol = sym.super[(size_t)k + 1];
        const int nscol = ecol - scol;

        const int pi0 = sym.pi[(size_t)k];
        const int pi1 = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;

        const int px0 = sym.px[(size_t)k];

        // assemble symmetric front F (nsrow x nsrow)
        std::vector<double> F((size_t)nsrow * (size_t)nsrow, 0.0);

        std::fill(g2p.begin(), g2p.end(), -1);
        for (int t = 0; t < nsrow; ++t)
        {
            const int r = sym.s[(size_t)(pi0 + t)];
            g2p[(size_t)r] = t;
        }

        // scatter A (assume stype=-1 lower stored): only i>=j
        for (int j = scol; j < ecol; ++j)
        {
            const int jpos = g2p[(size_t)j];
            if (jpos < 0)
                continue;

            for (int p = Ac.cb(j); p < Ac.ce(j); ++p)
            {
                const int i = Ac.row(p);
                if (i < j)
                    continue;
                const int ipos = g2p[(size_t)i];
                if (ipos < 0)
                    continue;

                const double v = (double)Ac.val(p);
                CM(F, nsrow, ipos, jpos) += v;
                if (ipos != jpos)
                    CM(F, nsrow, jpos, ipos) += v;
            }
        }

        // add children updates (scatter-add)
        for (int c : children[(size_t)k])
        {
            const UpdatePack &uc = up[(size_t)c];
            const int m = uc.nupd;
            if (m <= 0)
                continue;

            std::vector<int> pos((size_t)m, -1);
            for (int a = 0; a < m; ++a)
                pos[(size_t)a] = g2p[(size_t)uc.idx[(size_t)a]];

            for (int j = 0; j < m; ++j)
            {
                const int pj = pos[(size_t)j];
                if (pj < 0)
                    continue;
                for (int i = j; i < m; ++i)
                {
                    const int pi = pos[(size_t)i];
                    if (pi < 0)
                        continue;
                    const double v = CMc(uc.S, m, i, j);
                    CM(F, nsrow, pi, pj) += v;
                    if (pi != pj)
                        CM(F, nsrow, pj, pi) += v;
                }
            }
        }

        // C = F(:,0:nscol-1) (nsrow x nscol)
        std::vector<double> C((size_t)nsrow * (size_t)nscol, 0.0);
        for (int j = 0; j < nscol; ++j)
            for (int i = 0; i < nsrow; ++i)
                CM(C, nsrow, i, j) = CMc(F, nsrow, i, j);

        int fail_col = -1;

#if defined(ICHOL_USE_BLAS)
        if (!chol_ll_potrf_trsm(C.data(), nsrow, nsrow, nscol, fail_col))
        {
#else
        if (!chol_ll_naive(C.data(), nsrow, nsrow, nscol, fail_col))
        {
#endif
            status.ok = false;
            status.fail_snode = k;
            status.fail_col_in_snode = fail_col;
            return;
        }

        // write block to x (unique range per k)
        std::copy(C.begin(), C.end(), x.begin() + (size_t)px0);

        // update: S = F22 - L21*L21^T
        const int nupd = nsrow - nscol;
        UpdatePack &uk = up[(size_t)k];
        uk.nupd = nupd;
        uk.idx.clear();
        uk.S.clear();

        if (nupd <= 0)
            return;

        uk.idx.resize((size_t)nupd);
        for (int t = 0; t < nupd; ++t)
            uk.idx[(size_t)t] = sym.s[(size_t)(pi0 + nscol + t)];

        uk.S.assign((size_t)nupd * (size_t)nupd, 0.0);

        // init from F22 (full)
        for (int j = 0; j < nupd; ++j)
        {
            for (int i = j; i < nupd; ++i)
            {
                const double v = CMc(F, nsrow, nscol + i, nscol + j);
                uk.S[(size_t)i + (size_t)j * (size_t)nupd] = v;
                uk.S[(size_t)j + (size_t)i * (size_t)nupd] = v;
            }
        }

#if defined(ICHOL_USE_BLAS)
        // dsyrk updates LOWER only: S = (-1)*L21*L21^T + 1*S
        {
            char uplo = 'L', trans = 'N';
            int N = nupd;
            int K = nscol;
            double alpha = -1.0;
            double beta = 1.0;
            const double *L21 = C.data() + (size_t)nscol; // (row offset nscol)
            int lda = nsrow;
            int ldc = nupd;

            dsyrk_(&uplo, &trans, &N, &K, &alpha, L21, &lda, &beta, uk.S.data(), &ldc);

            // symmetrize upper from lower
            for (int j = 0; j < nupd; ++j)
                for (int i = 0; i < j; ++i)
                    uk.S[(size_t)i + (size_t)j * (size_t)nupd] = uk.S[(size_t)j + (size_t)i * (size_t)nupd];
        }
#else
        // fallback: subtract dot products
        for (int j = 0; j < nupd; ++j)
        {
            for (int i = j; i < nupd; ++i)
            {
                double dot = 0.0;
                const int ri = nscol + i;
                const int rj = nscol + j;
                for (int t = 0; t < nscol; ++t)
                    dot += CMc(C, nsrow, ri, t) * CMc(C, nsrow, rj, t);
                double v = uk.S[(size_t)i + (size_t)j * (size_t)nupd] - dot;
                uk.S[(size_t)i + (size_t)j * (size_t)nupd] = v;
                uk.S[(size_t)j + (size_t)i * (size_t)nupd] = v;
            }
        }
#endif
    }

    numeric::SuperNumeric factorize_supernodal_ll(
        const ichol::matrix::CscMatrix<double> &A,
        const symbolic::SupernodalLLPlan &plan)
    {
        numeric::SuperNumeric out;
        out.ok = true;
        out.fail_snode = -1;
        out.fail_col_in_snode = -1;
        out.sym = plan.sym;
        out.x.assign((size_t)plan.sym.px.back(), 0.0);

        const int n = A.num_cols;
        const int nsuper = (int)plan.sym.super.size() - 1;

        // All symbolic scheduling info must come from 'plan' (no symbolic recomputation here).
        const auto &children = plan.children;

        // plan.buckets is expected to be provided by the symbolic phase.
        // If it is empty (e.g., legacy callers), fall back to a single sequential bucket.
        std::vector<std::vector<int>> buckets_fallback;
        const std::vector<std::vector<int>> *buckets_ptr = &plan.buckets;
        if (plan.buckets.empty())
        {
            buckets_fallback.resize(1);
            buckets_fallback[0].resize((size_t)nsuper);
            for (int k = 0; k < nsuper; ++k)
                buckets_fallback[0][(size_t)k] = k;
            buckets_ptr = &buckets_fallback;
        }
        const auto &buckets = *buckets_ptr;

        // ---- 关键：up 必须在这里声明（在并行 region 外），否则并行段引用不到 ----
        std::vector<UpdatePack> up((size_t)nsuper);

        // ---- level buckets: same level can be processed in parallel ----
        const int maxL = (int)buckets.size() - 1;

#if defined(ICHOL_USE_OPENMP)
        const int max_threads = omp_get_max_threads();
        std::vector<std::atomic<int>> work_atomic((size_t)std::max(1, max_threads));
        for (auto &a : work_atomic)
            a.store(0);

        std::atomic<int> threads_used_atomic{1};

#pragma omp parallel
        {
            // 记录本次 parallel region 实际线程数
#pragma omp single
            {
                threads_used_atomic.store(omp_get_num_threads());
            }

            std::vector<int> g2p((size_t)n, -1);

            for (int L = 0; L <= maxL; ++L)
            {
                auto &nodes = buckets[(size_t)L];

#pragma omp for schedule(dynamic, 1)
                for (int ii = 0; ii < (int)nodes.size(); ++ii)
                {
                    const int k = nodes[(size_t)ii];
                    compute_one_supernode(k, A, plan.sym, children, up, out.x, out, g2p);

                    const int tid = omp_get_thread_num();
                    work_atomic[(size_t)tid].fetch_add(1);
                }

#pragma omp barrier
            }
        }

        out.threads_used = threads_used_atomic.load();
        out.thread_work.assign((size_t)out.threads_used, 0);
        for (int t = 0; t < out.threads_used && t < (int)work_atomic.size(); ++t)
        {
            out.thread_work[(size_t)t] = work_atomic[(size_t)t].load();
        }
#else
#endif

        return out;
    }

}