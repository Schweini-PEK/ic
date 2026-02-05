#pragma once

#include <vector>
#include <algorithm>

#include "ichol/options.hpp"
#include "ichol/matrix_formats.hpp"

#include "factor/symbolic/detail/symbolic_plan.hpp"  // Permutation
#include "factor/symbolic/super_sym.hpp"             // SuperSym

namespace ichol::symbolic {

// ----------------------------------------------------------------------------
// Supernode schedule helpers (derived from CHOLMOD-style SuperSym rowlists)
// ----------------------------------------------------------------------------

// col2snode[c] = supernode id that owns column c
inline std::vector<int> build_col2snode(const std::vector<int>& super, int ncols)
{
    std::vector<int> col2s((size_t)ncols, -1);
    const int nsuper = (int)super.size() - 1;
    for (int k = 0; k < nsuper; ++k) {
        for (int c = super[(size_t)k]; c < super[(size_t)k + 1]; ++c) {
            col2s[(size_t)c] = k;
        }
    }
    return col2s;
}

// CHOLMOD-style parent rule: first update row's column determines parent supernode.
inline std::vector<int> build_snode_parent_from_rowlist(const SuperSym& sym,
                                                        const std::vector<int>& col2snode)
{
    const int nsuper = (int)sym.super.size() - 1;
    std::vector<int> parent((size_t)nsuper, -1);

    for (int k = 0; k < nsuper; ++k) {
        const int nscol = sym.super[(size_t)k + 1] - sym.super[(size_t)k];
        const int pi0   = sym.pi[(size_t)k];
        const int pi1   = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;

        if (nsrow <= nscol) { parent[(size_t)k] = -1; continue; }

        int t = pi0 + nscol;
        int ps = -1;
        while (t < pi1) {
            const int col = sym.s[(size_t)t];
            ps = col2snode[(size_t)col];
            if (ps != k) break;
            ++t;
        }
        parent[(size_t)k] = (t < pi1) ? ps : -1;
    }
    return parent;
}

inline std::vector<std::vector<int>> build_children(const std::vector<int>& parent)
{
    const int nsuper = (int)parent.size();
    std::vector<std::vector<int>> ch((size_t)nsuper);
    for (int k = 0; k < nsuper; ++k) {
        const int p = parent[(size_t)k];
        if (p >= 0) ch[(size_t)p].push_back(k);
    }
    return ch;
}

// leaf level = 0; parent level = 1 + max(child levels)
inline std::vector<int> compute_level_from_leaves(const std::vector<std::vector<int>>& children)
{
    const int nsuper = (int)children.size();
    std::vector<int> level((size_t)nsuper, 0);

    std::vector<int> parent((size_t)nsuper, -1);
    for (int p = 0; p < nsuper; ++p) {
        for (int c : children[(size_t)p]) parent[(size_t)c] = p;
    }

    std::vector<int> roots;
    for (int k = 0; k < nsuper; ++k) if (parent[(size_t)k] < 0) roots.push_back(k);

    std::vector<int> post;
    post.reserve((size_t)nsuper);

    auto dfs = [&](auto&& self, int u) -> void {
        for (int v : children[(size_t)u]) self(self, v);
        post.push_back(u);
    };
    for (int r : roots) dfs(dfs, r);

    for (int u : post) {
        int mx = -1;
        for (int v : children[(size_t)u]) mx = std::max(mx, level[(size_t)v]);
        level[(size_t)u] = (mx < 0) ? 0 : (mx + 1);
    }
    return level;
}

inline std::vector<std::vector<int>> bucket_by_level(const std::vector<int>& level)
{
    int maxl = 0;
    for (int v : level) maxl = std::max(maxl, v);
    std::vector<std::vector<int>> buckets((size_t)maxl + 1);
    for (int i = 0; i < (int)level.size(); ++i) buckets[(size_t)level[(size_t)i]].push_back(i);
    return buckets;
}

// ----------------------------------------------------------------------------
// Supernodal LL plan (consumed by CPU/GPU numeric)
// ----------------------------------------------------------------------------

struct SupernodalLLPlan {
    // Ordering used for symbolic (CHOLMOD convention: perm[new] = old).
    Permutation perm;

    // CHOLMOD-like packed supernodal symbolic data.
    SuperSym sym;

    // Supernode DAG scheduling.
    std::vector<int> parent;                // size = nsuper
    std::vector<std::vector<int>> children; // adjacency list

    // Precomputed child->parent rowlist positions (CHOLMOD-style relpos).
    //
    // child_relpos[p][ci][t] = position in parent p rowlist (0..nsrow_p-1) of
    // the t-th update row of child c=children[p][ci]. -1 if that global index
    // is not present in the parent front rowlist.
    //
    // This removes per-child/per-supernode g2p lookups and enables much cheaper
    // scatter-add of child updates in the numeric phase.
    std::vector<std::vector<std::vector<int>>> child_relpos;
    std::vector<int> level;                 // per-snode level
    std::vector<std::vector<int>> buckets;  // buckets[level] = snode ids

    // Convenience: front dimension per snode (nsrow) and max.
    std::vector<int> front_dim;
    int max_front_dim = 0;
};

// Build CHOLMOD-style relpos arrays:
// for each edge child->parent, map child's update row ids into parent's rowlist positions.
inline void fill_child_relpos_from_sym(SupernodalLLPlan& plan, int ncols)
{
    const int nsuper = (int)plan.sym.super.size() - 1;
    plan.child_relpos.assign((size_t)nsuper, {});

    // Marker arrays (CHOLMOD "mark" technique) to avoid hashing.
    std::vector<int> mark_pos((size_t)ncols, -1);
    std::vector<int> mark_stamp((size_t)ncols, 0);
    int cur = 1;

    for (int p = 0; p < nsuper; ++p) {
        const int pi0_p = plan.sym.pi[(size_t)p];
        const int pi1_p = plan.sym.pi[(size_t)p + 1];
        const int nsrow_p = pi1_p - pi0_p;

        // Mark all rows in the parent front rowlist with their local position.
        for (int t = 0; t < nsrow_p; ++t) {
            const int g = plan.sym.s[(size_t)(pi0_p + t)];
            mark_stamp[(size_t)g] = cur;
            mark_pos[(size_t)g] = t;
        }

        const auto& ch = plan.children[(size_t)p];
        plan.child_relpos[(size_t)p].resize(ch.size());

        for (size_t ci = 0; ci < ch.size(); ++ci) {
            const int c = ch[ci];
            const int scol_c = plan.sym.super[(size_t)c];
            const int ecol_c = plan.sym.super[(size_t)c + 1];
            const int nscol_c = ecol_c - scol_c;
            const int pi0_c = plan.sym.pi[(size_t)c];
            const int pi1_c = plan.sym.pi[(size_t)c + 1];
            const int nsrow_c = pi1_c - pi0_c;
            const int nupd_c = nsrow_c - nscol_c;

            auto& rel = plan.child_relpos[(size_t)p][ci];
            rel.assign((size_t)std::max(0, nupd_c), -1);

            for (int t = 0; t < nupd_c; ++t) {
                const int g = plan.sym.s[(size_t)(pi0_c + nscol_c + t)];
                rel[(size_t)t] = (mark_stamp[(size_t)g] == cur) ? mark_pos[(size_t)g] : -1;
            }
        }

        // Bump stamp; if it overflows, reset stamps (very unlikely).
        ++cur;
        if (cur == 0) {
            std::fill(mark_stamp.begin(), mark_stamp.end(), 0);
            cur = 1;
        }
    }
}

inline void fill_schedule_from_sym(SupernodalLLPlan& plan, int ncols)
{
    const auto col2s = build_col2snode(plan.sym.super, ncols);
    plan.parent   = build_snode_parent_from_rowlist(plan.sym, col2s);
    plan.children = build_children(plan.parent);
    plan.level    = compute_level_from_leaves(plan.children);
    plan.buckets  = bucket_by_level(plan.level);

    // Precompute relpos mapping for cheap child-update scatter in numeric.
    fill_child_relpos_from_sym(plan, ncols);

    const int nsuper = (int)plan.sym.super.size() - 1;
    plan.front_dim.assign((size_t)nsuper, 0);
    plan.max_front_dim = 0;
    for (int k = 0; k < nsuper; ++k) {
        const int nsrow = plan.sym.pi[(size_t)k + 1] - plan.sym.pi[(size_t)k];
        plan.front_dim[(size_t)k] = nsrow;
        plan.max_front_dim = std::max(plan.max_front_dim, nsrow);
    }
}

// Wrap a CHOLMOD-style SuperSym into a full plan (identity ordering).
// This is used by backward-compatible numeric entrypoints that still accept SuperSym.
inline SupernodalLLPlan ll_plan_from_sym(const SuperSym& sym, int ncols)
{
    SupernodalLLPlan plan;
    plan.sym = sym;
    plan.perm.perm.assign((size_t)ncols, 0);
    plan.perm.inv_perm.assign((size_t)ncols, 0);
    for (int i = 0; i < ncols; ++i) {
        plan.perm.perm[(size_t)i] = i;
        plan.perm.inv_perm[(size_t)i] = i;
    }
    fill_schedule_from_sym(plan, ncols);
    return plan;
}

// Supernodal LL symbolic via CHOLMOD.
// - A may be permuted in-place (A := P*A*P^T) when a non-identity ordering is used.
// - sn_options is kept for API compatibility; CHOLMOD controls supernode amalgamation.
//
// NOTE: We keep this API templated (like ic_analyze) even though CHOLMOD's
// numeric payload is typically double/single. The symbolic depends only on
// the pattern; we still permute A's values so the numeric phase can consume it.
template <typename T>
SupernodalLLPlan supernodal_ll_analyze_fast(matrix::CscMatrix<T>& A,
                                            const SuperNodeOptions& sn_options,
                                            const SymbolicOptions& sym_options);

template <typename T>
inline SupernodalLLPlan supernodal_ll_analyze_fast(matrix::CscMatrix<T>& A,
                                                   const SuperNodeOptions& sn_options)
{
    SymbolicOptions sym{};
    sym.ordering = Ordering::Identity;    return supernodal_ll_analyze_fast<T>(A, sn_options, sym);
}

} // namespace ichol::symbolic
