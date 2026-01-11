
#include "supernodal_numeric_ll.hpp"

#include <cmath>
#include <algorithm>

namespace ichol::symbolic {

static inline double& CM(std::vector<double>& a, int ld, int r, int c) {
    return a[(size_t)r + (size_t)c * (size_t)ld];
}
static inline double CMc(const std::vector<double>& a, int ld, int r, int c) {
    return a[(size_t)r + (size_t)c * (size_t)ld];
}

static inline double row_dot_colmajor(const double* B, int ld, int ra, int rb, int ncol)
{
    double s = 0.0;
    for (int c = 0; c < ncol; ++c) s += B[ra + c * ld] * B[rb + c * ld];
    return s;
}



// In-place LL factorization on C(:,0:nscol-1) where C is nsrow x nscol (col-major, ld=nsrow).
// The leading nscol x nscol block is factorized; below-diagonal block becomes L21.
// Returns false if not SPD.
static bool chol_dense_ll_inplace(double* C, int ld, int nsrow, int nscol, int& fail_col)
{
    fail_col = -1;
    for (int j = 0; j < nscol; ++j) {
        double d = C[j + j * ld];
        for (int k = 0; k < j; ++k) {
            double ljk = C[j + k * ld];
            d -= ljk * ljk;
        }
        if (!(d > 0.0)) {
            fail_col = j;
            return false;
        }
        double ljj = std::sqrt(d);
        C[j + j * ld] = ljj;

        for (int i = j + 1; i < nsrow; ++i) {
            double v = C[i + j * ld];
            for (int k = 0; k < j; ++k) {
                v -= C[i + k * ld] * C[j + k * ld];
            }
            C[i + j * ld] = v / ljj;
        }
    }
    return true;
}

static std::vector<int> build_col2snode(const std::vector<int>& super, int ncols)
{
    std::vector<int> col2s((size_t)ncols, -1);
    const int nsuper = (int)super.size() - 1;
    for (int k = 0; k < nsuper; ++k) {
        for (int c = super[(size_t)k]; c < super[(size_t)k + 1]; ++c) col2s[(size_t)c] = k;
    }
    return col2s;
}

// CHOLMOD-style parent from rowlist: parent col is first update row (pi+nscol)
static std::vector<int> build_snode_parent_from_rowlist(const SuperSym& sym, const std::vector<int>& col2snode)
{
    const int nsuper = (int)sym.super.size() - 1;
    std::vector<int> parent((size_t)nsuper, -1);

    for (int k = 0; k < nsuper; ++k) {
        const int nscol = sym.super[(size_t)k + 1] - sym.super[(size_t)k];
        const int pi0 = sym.pi[(size_t)k];
        const int pi1 = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;

        if (nsrow <= nscol) { parent[(size_t)k] = -1; continue; }

        int t = pi0 + nscol;
        int ps = -1;
        while (t < pi1) {
            int pcol = sym.s[(size_t)t];
            ps = col2snode[(size_t)pcol];
            if (ps != k) break;
            ++t;
        }
        parent[(size_t)k] = (t < pi1) ? ps : -1;
    }
    return parent;
}

static std::vector<std::vector<int>> build_children(const std::vector<int>& parent)
{
    const int nsuper = (int)parent.size();
    std::vector<std::vector<int>> ch((size_t)nsuper);
    for (int k = 0; k < nsuper; ++k) {
        int p = parent[(size_t)k];
        if (p >= 0) ch[(size_t)p].push_back(k);
    }
    return ch;
}

static void postorder_dfs(int u, const std::vector<std::vector<int>>& ch, std::vector<int>& out)
{
    for (int v : ch[(size_t)u]) postorder_dfs(v, ch, out);
    out.push_back(u);
}

static std::vector<int> postorder(const std::vector<int>& parent)
{
    const int nsuper = (int)parent.size();
    auto ch = build_children(parent);
    std::vector<int> out;
    out.reserve((size_t)nsuper);
    for (int k = 0; k < nsuper; ++k) if (parent[(size_t)k] < 0) postorder_dfs(k, ch, out);
    return out;
}

struct UpdatePack {
    int nupd = 0;
    std::vector<int> idx;       // global indices of update rows
    std::vector<double> S;      // dense symmetric update, full stored
};

template <class T>
struct CscAccess {
    const ichol::matrix::CscMatrix<T>& A;
    int col_begin(int j) const { return A.col_ptr[j]; }
    int col_end(int j) const { return A.col_ptr[j+1]; }
    int row(int p) const { return A.row_ind[p]; }
    T   val(int p) const { return A.values[p]; }
};

SuperNumeric factorize_supernodal_ll(
    const ichol::matrix::CscMatrix<double>& A,
    const SuperSym& sym)
{
    SuperNumeric out;
    out.sym = sym;
    out.x.assign((size_t)sym.px.back(), 0.0);

    const int n = A.num_cols;
    const int nsuper = (int)sym.super.size() - 1;

    auto col2s = build_col2snode(sym.super, n);
    auto parent = build_snode_parent_from_rowlist(sym, col2s);
    auto children = build_children(parent);
    auto ord = postorder(parent);

    std::vector<int> g2p((size_t)n, -1);
    std::vector<UpdatePack> up((size_t)nsuper);

    CscAccess<double> Ac{A};

    for (int kk = 0; kk < (int)ord.size(); ++kk) {
        const int k = ord[(size_t)kk];

        const int scol  = sym.super[(size_t)k];
        const int ecol  = sym.super[(size_t)k + 1];
        const int nscol = ecol - scol;

        const int pi0   = sym.pi[(size_t)k];
        const int pi1   = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;

        const int px0   = sym.px[(size_t)k];

        // -------- assemble F (nsrow x nsrow) --------
        std::vector<double> F((size_t)nsrow * (size_t)nsrow, 0.0);

        std::fill(g2p.begin(), g2p.end(), -1);
        for (int t = 0; t < nsrow; ++t) {
            int r = sym.s[(size_t)(pi0 + t)];
            g2p[(size_t)r] = t;
        }

        // scatter A (lower) into symmetric F
        for (int j = scol; j < ecol; ++j) {
            int jpos = g2p[(size_t)j];
            if (jpos < 0) continue;
            for (int p = Ac.col_begin(j); p < Ac.col_end(j); ++p) {
                int i = Ac.row(p);
                if (i < j) continue;
                int ipos = g2p[(size_t)i];
                if (ipos < 0) continue;
                double v = (double)Ac.val(p);
                CM(F, nsrow, ipos, jpos) += v;
                if (ipos != jpos) CM(F, nsrow, jpos, ipos) += v;
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
                    double v = CMc(uc.S, m, i, j);
                    CM(F, nsrow, pi, pj) += v;
                    if (pi != pj) CM(F, nsrow, pj, pi) += v;
                }
            }
        }

        // -------- factor: C = F(:,0:nscol-1) --------
        std::vector<double> C((size_t)nsrow * (size_t)nscol, 0.0);
        for (int j = 0; j < nscol; ++j) {
            for (int i = 0; i < nsrow; ++i) {
                CM(C, nsrow, i, j) = CMc(F, nsrow, i, j);
            }
        }

        int fail_col = -1;
        if (!chol_dense_ll_inplace(C.data(), nsrow, nsrow, nscol, fail_col)) {
            out.ok = false;
            out.fail_snode = k;
            out.fail_col_in_snode = fail_col;
            return out;
        }

        // write block to x
        std::copy(C.begin(), C.end(), out.x.begin() + (size_t)px0);

        // -------- compute update S = F22 - L21*L21^T --------
        const int nupd = nsrow - nscol;
        UpdatePack& uk = up[(size_t)k];
        uk.nupd = nupd;
        uk.idx.clear();
        uk.S.clear();

        if (nupd > 0) {
            uk.idx.resize((size_t)nupd);
            for (int t = 0; t < nupd; ++t) uk.idx[(size_t)t] = sym.s[(size_t)(pi0 + nscol + t)];
            uk.S.assign((size_t)nupd * (size_t)nupd, 0.0);

            // init from F22
            for (int j = 0; j < nupd; ++j) {
                for (int i = j; i < nupd; ++i) {
                    double v = CMc(F, nsrow, nscol + i, nscol + j);
                    CM(uk.S, nupd, i, j) = v;
                    CM(uk.S, nupd, j, i) = v;
                }
            }

            // subtract L21*L21^T (L21 rows in C: nscol+i)
            for (int j = 0; j < nupd; ++j) {
                for (int i = j; i < nupd; ++i) {
                    double dot = 0.0;
                    int ri = nscol + i;
                    int rj = nscol + j;
                    for (int t = 0; t < nscol; ++t) dot += CMc(C, nsrow, ri, t) * CMc(C, nsrow, rj, t);
                    double v = CMc(uk.S, nupd, i, j) - dot;
                    CM(uk.S, nupd, i, j) = v;
                    CM(uk.S, nupd, j, i) = v;
                }
            }
        }
    }

    return out;
}

} // namespace ichol::symbolic
