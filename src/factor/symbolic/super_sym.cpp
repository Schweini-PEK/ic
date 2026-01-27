#include "super_sym.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <filesystem>
#include <stdexcept>

#include "ichol/matrix_formats.hpp"
#include "factor/symbolic/detail/symbolic_plan.hpp"

namespace {

// Build symmetric upper adjacency for ereach:
//   adj[k] = all i < k s.t. A(i,k) != 0 (treating A as symmetric pattern).
// We build it as CSR-like compressed arrays to avoid vector<vector<int>> overhead.
struct UpperAdj {
    std::vector<int> ptr;  // size n+1
    std::vector<int> ind;  // size nnz
};

static UpperAdj build_upper_adj_from_csc(const ichol::matrix::CscMatrix<double>& A)
{
    const int n = A.num_cols;
    std::vector<int> cnt((size_t)n, 0);

    // Count strict-upper adjacency entries:
    // for each nonzero (i,j), map to (v=min(i,j), u=max(i,j)), store v in column u.
    for (int j = 0; j < n; ++j) {
        const int p0 = A.col_ptr[(size_t)j];
        const int p1 = A.col_ptr[(size_t)j + 1];
        for (int p = p0; p < p1; ++p) {
            const int i = A.row_ind[(size_t)p];
            if (i == j) continue;
            const int u = (i > j) ? i : j;
            ++cnt[(size_t)u];
        }
    }

    UpperAdj adj;
    adj.ptr.assign((size_t)n + 1, 0);
    for (int k = 0; k < n; ++k) adj.ptr[(size_t)k + 1] = adj.ptr[(size_t)k] + cnt[(size_t)k];
    adj.ind.assign((size_t)adj.ptr[(size_t)n], 0);

    std::vector<int> next = adj.ptr;
    for (int j = 0; j < n; ++j) {
        const int p0 = A.col_ptr[(size_t)j];
        const int p1 = A.col_ptr[(size_t)j + 1];
        for (int p = p0; p < p1; ++p) {
            const int i = A.row_ind[(size_t)p];
            if (i == j) continue;
            const int u = (i > j) ? i : j;
            const int v = (i > j) ? j : i;
            adj.ind[(size_t)next[(size_t)u]++] = v;
        }
    }

    // Deduplicate per column without sorting (linear-time, stable w.r.t. scan order).
    std::vector<int> mark((size_t)n, -1);
    std::vector<int> new_ptr((size_t)n + 1, 0);

    int nnz = 0;
    new_ptr[0] = 0;
    for (int k = 0; k < n; ++k) {
        const int a = adj.ptr[(size_t)k];
        const int b = adj.ptr[(size_t)k + 1];
        for (int p = a; p < b; ++p) {
            const int v = adj.ind[(size_t)p];
            if ((unsigned)v >= (unsigned)n) continue;
            if (mark[(size_t)v] == k) continue;
            mark[(size_t)v] = k;
            adj.ind[(size_t)nnz++] = v;
        }
        new_ptr[(size_t)k + 1] = nnz;
    }

    adj.ptr.swap(new_ptr);
    adj.ind.resize((size_t)nnz);
    return adj;
}


// A lightweight ereach for symmetric Cholesky pattern.
// Returns 'top' such that stack[top..n-1] are the reached nodes (all < k),
// with duplicates removed via the 'w' stamp array.
static int ereach_upper_adj(int k,
                            const UpperAdj& adj,
                            const std::vector<int>& parent,
                            std::vector<int>& stack,
                            std::vector<int>& w,
                            int stamp)
{
    const int n = (int)parent.size();
    int top = n;

    const int a = adj.ptr[(size_t)k];
    const int b = adj.ptr[(size_t)k + 1];
    for (int p = a; p < b; ++p) {
        int i = adj.ind[(size_t)p];
        // Follow parent chain while staying strictly below k.
        while (i != -1 && i < k && w[(size_t)i] != stamp) {
            stack[(size_t)--top] = i;
            w[(size_t)i] = stamp;
            i = parent[(size_t)i];
        }
    }
    return top;
}

// Variant of ereach that takes adjacency arrays directly (no struct copies).
// Implemented with raw pointers to minimize bounds checks and iterator overhead.
static inline int ereach_upper_adj_vec(int k,
                                      const std::vector<int>& adj_ptr,
                                      const std::vector<int>& adj_ind,
                                      const std::vector<int>& parent,
                                      std::vector<int>& stack,
                                      std::vector<int>& w,
                                      int stamp)
{
    const int n = (int)parent.size();
    int top = n;

    const int* __restrict Ap = adj_ptr.data();
    const int* __restrict Ai = adj_ind.data();
    const int* __restrict Par = parent.data();
    int* __restrict Stk = stack.data();
    int* __restrict W = w.data();

    const int a = Ap[k];
    const int b = Ap[k + 1];
    for (int p = a; p < b; ++p) {
        int i = Ai[p];
        // Follow parent chain while staying strictly below k.
        while (i != -1 && i < k && W[i] != stamp) {
            Stk[--top] = i;
            W[i] = stamp;
            i = Par[i];
        }
    }
    return top;
}

} // anonymous namespace

namespace ichol::symbolic
{
    ichol::symbolic::SuperSym build_super_sym(
        const std::vector<std::pair<int,int>>& snodes,
        const std::vector<std::vector<int>>& snode_rows)
    {
        if (snodes.empty()) {
            throw std::runtime_error("build_super_sym: snodes is empty");
        }
        if ((int)snode_rows.size() != (int)snodes.size()) {
            throw std::runtime_error("build_super_sym: snode_rows.size != snodes.size");
        }

        const int nsuper = (int)snodes.size();

        SuperSym sym;
        sym.super.resize((size_t)nsuper + 1);
        sym.pi.resize((size_t)nsuper + 1);
        sym.px.resize((size_t)nsuper + 1);

        // super
        for (int k = 0; k < nsuper; ++k) {
            // 对于第 k 个 supernode：
            //  - pivot 行固定为 [scol, ecol)
            //  - update 行来自 snode_rows 的其余元素（去掉 pivots 后的集合）
            // 这里会把 update 行排序 + 去重，并检查最小 update 行必须 >= ecol。

            sym.super[(size_t)k] = snodes[(size_t)k].first;
        }
        sym.super[(size_t)nsuper] = snodes.back().second;

        // build s and pi
        sym.pi[0] = 0;
        sym.s.clear();
        sym.s.reserve(1024);

        for (int k = 0; k < nsuper; ++k) {
            // 对于第 k 个 supernode：
            //  - pivot 行固定为 [scol, ecol)
            //  - update 行来自 snode_rows 的其余元素（去掉 pivots 后的集合）
            // 这里会把 update 行排序 + 去重，并检查最小 update 行必须 >= ecol。

            const int scol  = snodes[(size_t)k].first;
            const int ecol  = snodes[(size_t)k].second;
            const int nscol = ecol - scol;

            if (nscol <= 0) {
                throw std::runtime_error("build_super_sym: invalid supernode width (nscol<=0)");
            }

            const auto& rows_in = snode_rows[(size_t)k];

            // 1) pivot rows first: scol..ecol-1
            const int base = (int)sym.s.size();
            sym.s.resize((size_t)base + (size_t)nscol);
            for (int j = 0; j < nscol; ++j) {
                sym.s[(size_t)base + (size_t)j] = scol + j;
            }

            // 2) append update rows = rows_in \ pivots
            std::vector<int> upd;
            upd.reserve(rows_in.size());

            for (int r : rows_in) {
                if (r >= scol && r < ecol) continue; // skip pivots
                upd.push_back(r);
            }
            std::sort(upd.begin(), upd.end());
            upd.erase(std::unique(upd.begin(), upd.end()), upd.end());
            if (!upd.empty() && upd.front() < ecol) {
                throw std::runtime_error("build_super_sym: update row < ecol; rowlist not CHOLMOD-compatible");
            }

            sym.s.insert(sym.s.end(), upd.begin(), upd.end());

            sym.pi[(size_t)k + 1] = (int)sym.s.size();
        }

        // px（数值阶段稠密块偏移）
        // 对应 CHOLMOD 中每个 supernode 的 Lx 片段大小：nsrow * nscol。

        sym.px[0] = 0;
        for (int k = 0; k < nsuper; ++k) {
            // 对于第 k 个 supernode：
            //  - pivot 行固定为 [scol, ecol)
            //  - update 行来自 snode_rows 的其余元素（去掉 pivots 后的集合）
            // 这里会把 update 行排序 + 去重，并检查最小 update 行必须 >= ecol。

            const int scol  = sym.super[(size_t)k];
            const int ecol  = sym.super[(size_t)k + 1];
            const int nscol = ecol - scol;

            const int nsrow = sym.pi[(size_t)k + 1] - sym.pi[(size_t)k];
            sym.px[(size_t)k + 1] = sym.px[(size_t)k] + nsrow * nscol;
        }

        return sym;
    }

        SuperSym build_super_sym_direct(
        const ichol::matrix::CscMatrix<double>& A,
        const ETree& etree,
        const std::vector<std::pair<int,int>>& snodes)
    {
        const int n = A.num_cols;
        if (n <= 0) throw std::runtime_error("build_super_sym_direct: empty matrix");
        if ((int)etree.parent.size() != n) throw std::runtime_error("build_super_sym_direct: etree.parent size mismatch");
        if (snodes.empty()) throw std::runtime_error("build_super_sym_direct: snodes is empty");

        const int nsuper = (int)snodes.size();

        // super boundaries
        SuperSym sym;
        sym.super.resize((size_t)nsuper + 1);
        for (int k = 0; k < nsuper; ++k) sym.super[(size_t)k] = snodes[(size_t)k].first;
        sym.super[(size_t)nsuper] = snodes.back().second;

        // Cache supernode end columns to reduce loads in the hot loop.
        std::vector<int> super_ecol((size_t)nsuper);
        for (int s = 0; s < nsuper; ++s) super_ecol[(size_t)s] = sym.super[(size_t)s + 1];

        // col -> snode mapping (avoid header dependency)
        std::vector<int> col2s((size_t)n, -1);
        for (int s = 0; s < nsuper; ++s) {
            const int scol = sym.super[(size_t)s];
            const int ecol = super_ecol[(size_t)s];
            for (int c = scol; c < ecol; ++c) col2s[(size_t)c] = s;
        }

        // Build (or reuse cached) strict-upper adjacency for ereach.
        UpperAdj adj_tmp;
        const std::vector<int>* adj_ptr = &etree.upper_ptr;
        const std::vector<int>* adj_ind = &etree.upper_ind;
        if (adj_ptr->size() != (size_t)n + 1) {
            adj_tmp = build_upper_adj_from_csc(A);
            adj_ptr = &adj_tmp.ptr;
            adj_ind = &adj_tmp.ind;
        }

        // Accumulate update rows per supernode. We append rows in increasing order (k increasing),
        // and ensure each (row k, snode s) pair is appended at most once using a per-row stamp.
        std::vector<std::vector<int>> upd((size_t)nsuper);
        for (auto& v : upd) v.reserve(64);

        std::vector<int> stack((size_t)n);
        std::vector<int> w((size_t)n, -1);
        std::vector<int> seen_snode((size_t)nsuper, -1);

        for (int k = 0; k < n; ++k) {
            const int top = ereach_upper_adj_vec(k, *adj_ptr, *adj_ind, etree.parent, stack, w, /*stamp=*/k);

            // For this row k, mark which snodes are hit by any L(k,j), j<k.
            for (int t = top; t < n; ++t) {
                const int j = stack[(size_t)t];
                const int s = col2s[(size_t)j];
                if (s < 0) continue;

                // Only keep update rows (must be >= ecol of that supernode).
                const int ecol = super_ecol[(size_t)s];
                if (k < ecol) continue;

                if (seen_snode[(size_t)s] != k) {
                    seen_snode[(size_t)s] = k;
                    upd[(size_t)s].push_back(k);
                }
            }
        }

        // Build packed rowlists: pivots first, then update rows.
        sym.pi.resize((size_t)nsuper + 1);
        sym.px.resize((size_t)nsuper + 1);

        size_t total_s = 0;
        for (int s = 0; s < nsuper; ++s) {
            const int scol = sym.super[(size_t)s];
            const int ecol = super_ecol[(size_t)s];
            total_s += (size_t)(ecol - scol);
            total_s += upd[(size_t)s].size();
        }
        sym.s.clear();
        sym.s.reserve(total_s);

        sym.pi[0] = 0;
        for (int s = 0; s < nsuper; ++s) {
            const int scol = sym.super[(size_t)s];
            const int ecol = super_ecol[(size_t)s];
            const int nscol = ecol - scol;
            if (nscol <= 0) throw std::runtime_error("build_super_sym_direct: invalid supernode width");

            // pivots
            for (int r = scol; r < ecol; ++r) sym.s.push_back(r);

            // updates (already unique and in increasing order by construction)
            const auto& u = upd[(size_t)s];
            if (!u.empty() && u.front() < ecol) {
                throw std::runtime_error("build_super_sym_direct: update row < ecol; rowlist not CHOLMOD-compatible");
            }
            sym.s.insert(sym.s.end(), u.begin(), u.end());

            sym.pi[(size_t)s + 1] = (int)sym.s.size();
        }

        // px prefix sums
        sym.px[0] = 0;
        for (int s = 0; s < nsuper; ++s) {
            const int nscol = sym.super[(size_t)s + 1] - sym.super[(size_t)s];
            const int nsrow = sym.pi[(size_t)s + 1] - sym.pi[(size_t)s];
            sym.px[(size_t)s + 1] = sym.px[(size_t)s] + nsrow * nscol;
        }

        return sym;
    }


}
