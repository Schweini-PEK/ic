#include "factor/numerical/detail/supernodal_numeric_ll_internal.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>


#if defined(ICHOL_USE_BLAS)
extern "C" {
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);

void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
            const int* m, const int* n,
            const double* alpha, const double* a, const int* lda,
            double* b, const int* ldb);

void dsyrk_(const char* uplo, const char* trans,
            const int* n, const int* k,
            const double* alpha, const double* a, const int* lda,
            const double* beta, double* c, const int* ldc);
}
#endif

namespace ichol::numeric::detail {

#if defined(ICHOL_USE_BLAS)
static bool chol_ll_potrf_trsm(double* F, int ld, int nsrow, int nscol, int& fail_col)
{
    fail_col = -1;

    // POTRF on leading nscol x nscol (lower)
    char uplo = 'L';
    int n = nscol;
    int lda = ld;
    int info = 0;
    dpotrf_(&uplo, &n, F, &lda, &info);
    if (info != 0) { fail_col = std::max(0, info - 1); return false; }

    // TRSM: L21 = A21 * inv(L11^T)
    const int nupd = nsrow - nscol;
    if (nupd > 0) {
        char side='R', upl='L', trans='T', diag='N';
        int m = nupd;
        int k = nscol;
        double alpha = 1.0;
        double* A21 = F + (size_t)nscol; // row offset nscol, col 0
        dtrsm_(&side, &upl, &trans, &diag, &m, &k, &alpha, F, &lda, A21, &lda);
    }

    // deterministic: zero upper of L11
    for (int j = 0; j < nscol; ++j)
        for (int i = 0; i < j; ++i)
            CM(F, ld, i, j) = 0.0;

    return true;
}
#else
static bool chol_ll_naive(double* F, int ld, int nsrow, int nscol, int& fail_col)
{
    fail_col = -1;
    for (int j = 0; j < nscol; ++j) {
        double d = CM(F, ld, j, j);
        for (int k = 0; k < j; ++k) {
            double ljk = CMc(F, ld, j, k);
            d -= ljk * ljk;
        }
        if (!(d > 0.0)) { fail_col = j; return false; }
        double ljj = std::sqrt(d);
        CM(F, ld, j, j) = ljj;

        for (int i = j + 1; i < nsrow; ++i) {
            double v = CMc(F, ld, i, j);
            for (int k = 0; k < j; ++k) v -= CMc(F, ld, i, k) * CMc(F, ld, j, k);
            CM(F, ld, i, j) = v / ljj;
        }
    }
    // deterministic
    for (int j = 0; j < nscol; ++j)
        for (int i = 0; i < j; ++i)
            CM(F, ld, i, j) = 0.0;
    return true;
}
#endif

void compute_one_supernode_cpu(
    int k,
    const ichol::matrix::CscMatrix<double>& A,
    const symbolic::SuperSym& sym,
    const std::vector<std::vector<int>>& children,
    const std::vector<std::vector<std::vector<int>>>& child_relpos,
    std::vector<UpdatePack>& up,
    std::vector<double>& x,
    std::atomic<bool>& ok,
    std::atomic<int>& fail_snode,
    std::atomic<int>& fail_col_in_snode,
    SupernodalWorkspace& ws)
{
    if (!ok.load(std::memory_order_relaxed)) return;

    CscView<double> Ac{A};

    const int scol  = sym.super[(size_t)k];
    const int ecol  = sym.super[(size_t)k + 1];
    const int nscol = ecol - scol;

    const int pi0   = sym.pi[(size_t)k];
    const int pi1   = sym.pi[(size_t)k + 1];
    const int nsrow = pi1 - pi0;

    const int px0   = sym.px[(size_t)k];

    // Assemble symmetric dense front F (nsrow x nsrow), stored in ws.F.
    ws.reset_front(nsrow);
    ws.build_mapping_from_s(sym.s, pi0, nsrow);

    double* F = ws.F.data();
    const int ld = nsrow;

    // Scatter A (assume lower stored, stype=-1): use only i>=j and mirror to keep symmetry
    for (int j = scol; j < ecol; ++j) {
        const int jpos = ws.g2l[(size_t)j];
        if (jpos < 0) continue;

        for (int p = Ac.cb(j); p < Ac.ce(j); ++p) {
            const int i = Ac.row(p);
            if (i < j) continue;
            const int ipos = ws.g2l[(size_t)i];
            if (ipos < 0) continue;

            const double v = (double)Ac.val(p);
            CM(F, ld, ipos, jpos) += v;
            if (ipos != jpos) CM(F, ld, jpos, ipos) += v;
        }
    }

    // Add children updates (scatter-add) using precomputed relpos (CHOLMOD-style).
    const auto& ch = children[(size_t)k];
    const auto& ch_rel = child_relpos[(size_t)k];
    for (size_t ci = 0; ci < ch.size(); ++ci) {
        const int c = ch[ci];
        const UpdatePack& uc = up[(size_t)c];
        const int m = uc.nupd;
        if (m <= 0) continue;

        // relpos maps child's update rows directly into parent's rowlist positions.
        const std::vector<int>& rel = ch_rel[ci];

        const double* S = uc.S.data();
        const int sld = m;

        for (int j = 0; j < m; ++j) {
            const int pj = rel[(size_t)j];
            if (pj < 0) continue;
            for (int i = j; i < m; ++i) {
                const int pi = rel[(size_t)i];
                if (pi < 0) continue;
                const double v = CMc(S, sld, i, j);
                CM(F, ld, pi, pj) += v;
                if (pi != pj) CM(F, ld, pj, pi) += v;
            }
        }
    }

    int fail_col = -1;

#if defined(ICHOL_USE_BLAS)
    const bool ok_local = chol_ll_potrf_trsm(F, ld, nsrow, nscol, fail_col);
#else
    const bool ok_local = chol_ll_naive(F, ld, nsrow, nscol, fail_col);
#endif

    if (!ok_local) {
        bool expected = true;
        if (ok.compare_exchange_strong(expected, false, std::memory_order_relaxed)) {
            fail_snode.store(k, std::memory_order_relaxed);
            fail_col_in_snode.store(fail_col, std::memory_order_relaxed);
        }
        ws.clear_mapping();
        return;
    }

    // Write L block to x (CHOLMOD block layout: nsrow x nscol, column-major, ld=nsrow)
    double* xk = x.data() + (size_t)px0;
    for (int j = 0; j < nscol; ++j) {
        std::copy(F + (size_t)j * (size_t)ld,
                  F + (size_t)(j + 1) * (size_t)ld,
                  xk + (size_t)j * (size_t)ld);
    }

    // Form update: S = F22 - L21*L21^T, store as dense symmetric (full)
    const int nupd = nsrow - nscol;
    UpdatePack& uk = up[(size_t)k];
    uk.nupd = nupd;
    uk.idx.clear();
    uk.S.clear();

    if (nupd > 0) {
        uk.idx.resize((size_t)nupd);
        for (int t = 0; t < nupd; ++t) uk.idx[(size_t)t] = sym.s[(size_t)(pi0 + nscol + t)];

        uk.S.assign((size_t)nupd * (size_t)nupd, 0.0);

        // Initialize from F22
        for (int j = 0; j < nupd; ++j) {
            for (int i = j; i < nupd; ++i) {
                const double v = CMc(F, ld, nscol + i, nscol + j);
                CM(uk.S.data(), nupd, i, j) = v;
                CM(uk.S.data(), nupd, j, i) = v;
            }
        }

#if defined(ICHOL_USE_BLAS)
        {
            char uplo='L', trans='N';
            int N = nupd;
            int K = nscol;
            double alpha = -1.0;
            double beta  = 1.0;
            const double* L21 = F + (size_t)nscol; // row offset nscol
            int lda = ld;
            int ldc = nupd;

            dsyrk_(&uplo, &trans, &N, &K, &alpha, L21, &lda, &beta, uk.S.data(), &ldc);

            // Symmetrize
            for (int j = 0; j < nupd; ++j)
                for (int i = 0; i < j; ++i)
                    CM(uk.S.data(), nupd, i, j) = CMc(uk.S.data(), nupd, j, i);
        }
#else
        for (int j = 0; j < nupd; ++j) {
            for (int i = j; i < nupd; ++i) {
                double dot = 0.0;
                const int ri = nscol + i;
                const int rj = nscol + j;
                for (int t = 0; t < nscol; ++t) dot += CMc(F, ld, ri, t) * CMc(F, ld, rj, t);
                const double v = CMc(uk.S.data(), nupd, i, j) - dot;
                CM(uk.S.data(), nupd, i, j) = v;
                CM(uk.S.data(), nupd, j, i) = v;
            }
        }
#endif
    }

    ws.clear_mapping();
}
} // namespace ichol::numeric::detail
