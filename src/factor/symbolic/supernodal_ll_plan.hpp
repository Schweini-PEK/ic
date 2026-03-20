#pragma once

#include <vector>
#include <algorithm>

#include "ichol/options.hpp"
#include "ichol/matrix_formats.hpp"

#include "factor/symbolic/detail/symbolic_plan.hpp" // Permutation

namespace
{
    /**
     * @brief Build a dense lookup from column id to owning supernode id.
     *
     * @param super Supernode boundary array of size `nsuper + 1`.
     * Each supernode `k` owns columns `[super[k], super[k+1])`.
     * @param ncols Global matrix column count.
     * @return Vector `col2s` of size `ncols`, where `col2s[c]` is the supernode id
     * owning column `c`, or `-1` if `c` is not covered by `super`.
     *
     * @details
     * This helper is used to convert rowlist entries (global column ids) back to
     * supernode ids while building the supernode DAG.
     *
     * @par Problems
     * - Severity: Low
     * - Issue: No explicit validation that each column is assigned exactly once.
     * - Suggested fix: Add optional debug assertions that `super` is monotone,
     *   covers `[0, ncols)`, and never overlaps.
     */
    inline std::vector<int> build_col2snode(const std::vector<int> &super, int ncols)
    {
        // Step 1: initialize all columns as "unassigned".
        std::vector<int> col2s((size_t)ncols, -1);
        const int nsuper = (int)super.size() - 1;

        // Step 2: stamp each supernode id into its column interval.
        for (int k = 0; k < nsuper; ++k)
        {
            for (int c = super[(size_t)k]; c < super[(size_t)k + 1]; ++c)
            {
                col2s[(size_t)c] = k;
            }
        }
        return col2s;
    }
}

namespace ichol::symbolic
{
    /**
     * @brief supernodal symbolic info struct.
     *
     * Given supernode k:
     *   - columns are [super[k], super[k+1]) with width nscol
     *   - rowlist is s[pi[k] .. pi[k+1]) with length nsrow
     *   - dense block storage for node k typically uses offsets px[k]..px[k+1)
     *     with size nsrow * nscol.
     *
     * Convention (matching CHOLMOD supernodal LL default):
     *   - rowlist begins with pivot rows: super[k]..super[k+1)-1 (nscol entries), the triangular part
     *   - followed by strictly increasing update rows >= super[k+1], the block that produces the Schur-complement
     */
    struct SuperSym
    {
        std::vector<int> super; // size = nsuper + 1, supernode column boundaries
        std::vector<int> pi;    // size = nsuper + 1, rowlist pointers into s
        std::vector<int> px;    // size = nsuper + 1, dense-block offsets (prefix sum of nsrow*nscol)
        std::vector<int> s;     // packed rowlists
    };

    /**
     * @brief Build supernode parent array using CHOLMOD rowlist semantics.
     *
     * @param sym Packed supernodal symbolic (`super`, `pi`, `px`, `s`) where each
     * supernode rowlist starts with pivot rows then update rows.
     * @param col2snode Column-to-supernode lookup, typically from
     * `build_col2snode(sym.super, ncols)`.
     * @return Parent array of size `nsuper`. `parent[k] == -1` means root.
     *
     * @details
     * CHOLMOD's parent rule for supernodal LL is:
     * - inspect node `k` rowlist after pivot rows;
     * - the first update row that belongs to a different supernode is parent.
     *
     * @par Problems
     * - Severity: Medium
     * - Issue: If rowlist data are malformed (out-of-range column ids), this can
     *   index `col2snode` out-of-bounds.
     * - Suggested fix: Add optional bounds checks in debug mode before indexing.
     */
    inline std::vector<int> build_snode_parent_from_rowlist(const SuperSym &sym,
                                                            const std::vector<int> &col2snode)
    {
        const int nsuper = (int)sym.super.size() - 1;
        std::vector<int> parent((size_t)nsuper, -1);

        for (int k = 0; k < nsuper; ++k)
        {
            // Step 1: decode dimensions for supernode k.
            const int nscol = sym.super[(size_t)k + 1] - sym.super[(size_t)k];
            const int pi0 = sym.pi[(size_t)k];
            const int pi1 = sym.pi[(size_t)k + 1];
            const int nsrow = pi1 - pi0;

            // Step 2: pivot rows only, no parent.
            if (nsrow <= nscol)
            {
                continue;
            }

            // Step 3: find first update row that maps to a different supernode.
            int t = pi0 + nscol;
            int ps = -1;
            while (t < pi1)
            {
                const int col = sym.s[(size_t)t];
                ps = col2snode[(size_t)col];
                if (ps != k)
                    break;
                ++t;
            }
            parent[(size_t)k] = (t < pi1) ? ps : -1;
        }
        return parent;
    }

    /**
     * @brief Build child adjacency lists from a parent array.
     *
     * @param parent Parent array, where `parent[k]` is parent of node `k` or `-1`.
     * @return `children[p]` contains all children whose parent is `p`.
     *
     * @par Problems
     * - Severity: Low
     * - Issue: Assumes parent ids are valid without checking.
     * - Suggested fix: Validate `parent[k] < nsuper` in debug builds.
     */
    inline std::vector<std::vector<int>> build_children(const std::vector<int> &parent)
    {
        // Step 1: allocate one child list per node.
        const int nsuper = (int)parent.size();
        std::vector<std::vector<int>> ch((size_t)nsuper);

        // Step 2: append each node to its parent's list.
        for (int k = 0; k < nsuper; ++k)
        {
            const int p = parent[(size_t)k];
            if (p >= 0)
                ch[(size_t)p].push_back(k);
        }
        return ch;
    }

    /**
     * @brief Compute bottom-up level index on the supernode DAG.
     *
     * @param children Child adjacency lists.
     * @return Level per node: leaves are level 0, parent is `1 + max(children)`.
     *
     * @details
     * This routine converts child lists to temporary parent array, finds roots,
     * computes postorder DFS, then propagates levels bottom-up.
     *
     * @par Problems
     * - Severity: Medium
     * - Issue: Recursive lambda DFS can overflow stack on very deep trees.
     * - Suggested fix: Replace with explicit iterative stack traversal.
     */
    inline std::vector<int> compute_level_from_leaves(const std::vector<std::vector<int>> &children)
    {
        // Step 1: prepare storage and derive parent array from children.
        const int nsuper = (int)children.size();
        std::vector<int> level((size_t)nsuper, 0);

        std::vector<int> parent((size_t)nsuper, -1);
        for (int p = 0; p < nsuper; ++p)
        {
            for (int c : children[(size_t)p])
                parent[(size_t)c] = p;
        }

        // Step 2: collect roots (nodes with no parent).
        std::vector<int> roots;
        for (int k = 0; k < nsuper; ++k)
            if (parent[(size_t)k] < 0)
                roots.push_back(k);

        // Step 3: postorder traversal for bottom-up DP.
        std::vector<int> post;
        post.reserve((size_t)nsuper);

        auto dfs = [&](auto &&self, int u) -> void
        {
            for (int v : children[(size_t)u])
                self(self, v);
            post.push_back(u);
        };
        for (int r : roots)
            dfs(dfs, r);

        // Step 4: dynamic program in postorder.
        for (int u : post)
        {
            int mx = -1;
            for (int v : children[(size_t)u])
                mx = std::max(mx, level[(size_t)v]);
            level[(size_t)u] = (mx < 0) ? 0 : (mx + 1);
        }
        return level;
    }

    /**
     * @brief Group nodes by level id.
     *
     * @param level Per-node level.
     * @return Buckets where `buckets[l]` contains node ids with level `l`.
     *
     * @par Problems
     * - Severity: Low
     * - Issue: Assumes non-negative level values.
     * - Suggested fix: Validate inputs and throw for invalid negative values.
     */
    inline std::vector<std::vector<int>> bucket_by_level(const std::vector<int> &level)
    {
        // Step 1: determine maximum level.
        int maxl = 0;
        for (int v : level)
            maxl = std::max(maxl, v);

        // Step 2: append node ids into their level buckets.
        std::vector<std::vector<int>> buckets((size_t)maxl + 1);
        for (int i = 0; i < (int)level.size(); ++i)
            buckets[(size_t)level[(size_t)i]].push_back(i);
        return buckets;
    }

    // ----------------------------------------------------------------------------
    // Supernodal LL plan (consumed by CPU/GPU numeric)
    // ----------------------------------------------------------------------------

    struct SupernodalLLPlan
    {
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
        std::vector<int> level;                // per-snode level
        std::vector<std::vector<int>> buckets; // buckets[level] = snode ids

        // Convenience: front dimension per snode (nsrow) and max.
        std::vector<int> front_dim;
        int max_front_dim = 0;
    };

    /**
     * @brief Precompute child-update to parent-rowlist positional mapping.
     *
     * @param plan Input/output supernodal plan. Requires valid `plan.sym` and
     * `plan.children`; fills `plan.child_relpos`.
     * @param ncols Total matrix columns, used to size marker arrays by global id.
     *
     * @details
     * For each parent `p` and each child `c` in `children[p]`, this computes:
     * - `child_relpos[p][ci][t] = local row position in parent front`
     * for the `t`-th update row of child `c`, or `-1` if absent.
     *
     * This avoids repeated per-edge hash/table lookups at numeric time.
     *
     * @par Problems
     * - Severity: Medium
     * - Issue: Memory overhead can be high for large DAGs due to triple nesting.
     * - Suggested fix: Consider compressed flat storage with one offset array.
     */
    inline void fill_child_relpos_from_sym(SupernodalLLPlan &plan, int ncols)
    {
        // Step 1: allocate top-level container and marker workspaces.
        const int nsuper = (int)plan.sym.super.size() - 1;
        plan.child_relpos.assign((size_t)nsuper, {});

        // Marker arrays (CHOLMOD "mark" technique) to avoid hashing.
        std::vector<int> mark_pos((size_t)ncols, -1);
        std::vector<int> mark_stamp((size_t)ncols, 0);
        int cur = 1;

        for (int p = 0; p < nsuper; ++p)
        {
            // Step 2: mark all parent rowlist entries with their local position.
            const int pi0_p = plan.sym.pi[(size_t)p];
            const int pi1_p = plan.sym.pi[(size_t)p + 1];
            const int nsrow_p = pi1_p - pi0_p;

            // Mark all rows in the parent front rowlist with their local position.
            for (int t = 0; t < nsrow_p; ++t)
            {
                const int g = plan.sym.s[(size_t)(pi0_p + t)];
                mark_stamp[(size_t)g] = cur;
                mark_pos[(size_t)g] = t;
            }

            const auto &ch = plan.children[(size_t)p];
            plan.child_relpos[(size_t)p].resize(ch.size());

            for (size_t ci = 0; ci < ch.size(); ++ci)
            {
                // Step 3: for one child edge, map each child update row into parent.
                const int c = ch[ci];
                const int scol_c = plan.sym.super[(size_t)c];
                const int ecol_c = plan.sym.super[(size_t)c + 1];
                const int nscol_c = ecol_c - scol_c;
                const int pi0_c = plan.sym.pi[(size_t)c];
                const int pi1_c = plan.sym.pi[(size_t)c + 1];
                const int nsrow_c = pi1_c - pi0_c;
                const int nupd_c = nsrow_c - nscol_c;

                auto &rel = plan.child_relpos[(size_t)p][ci];
                rel.assign((size_t)std::max(0, nupd_c), -1);

                for (int t = 0; t < nupd_c; ++t)
                {
                    const int g = plan.sym.s[(size_t)(pi0_c + nscol_c + t)];
                    rel[(size_t)t] = (mark_stamp[(size_t)g] == cur) ? mark_pos[(size_t)g] : -1;
                }
            }

            // Step 4: bump marker stamp; reset on overflow (rare).
            ++cur;
            if (cur == 0)
            {
                std::fill(mark_stamp.begin(), mark_stamp.end(), 0);
                cur = 1;
            }
        }
    }

    /**
     * @brief Build all numeric-consumed schedule metadata from `plan.sym`.
     *
     * @param plan Input/output plan. Requires valid `plan.sym`; fills parent/child
     * DAG, level buckets, child_relpos, front dimensions, and max front dimension.
     * @param ncols Matrix column count.
     *
     * @par Problems
     * - Severity: Low
     * - Issue: Recomputes several derived arrays unconditionally each call.
     * - Suggested fix: Add a small "is_built" cache flag for repeated invocations.
     */
    inline void fill_schedule_from_sym(SupernodalLLPlan &plan, int ncols)
    {
        // Step 1: derive supernode DAG and level schedule.
        const auto col2s = build_col2snode(plan.sym.super, ncols);
        plan.parent = build_snode_parent_from_rowlist(plan.sym, col2s);
        plan.children = build_children(plan.parent);
        plan.level = compute_level_from_leaves(plan.children);
        plan.buckets = bucket_by_level(plan.level);

        // Step 2: precompute child->parent positional maps used in numeric scatter.
        fill_child_relpos_from_sym(plan, ncols);

        // Step 3: precompute per-front dimension stats.
        const int nsuper = (int)plan.sym.super.size() - 1;
        plan.front_dim.assign((size_t)nsuper, 0);
        plan.max_front_dim = 0;
        for (int k = 0; k < nsuper; ++k)
        {
            const int nsrow = plan.sym.pi[(size_t)k + 1] - plan.sym.pi[(size_t)k];
            plan.front_dim[(size_t)k] = nsrow;
            plan.max_front_dim = std::max(plan.max_front_dim, nsrow);
        }
    }

    /**
     * @brief Wrap a bare `SuperSym` into a full `SupernodalLLPlan`.
     *
     * @param sym CHOLMOD-style packed supernodal symbolic.
     * @param ncols Matrix column count.
     * @return Full plan using identity permutation and derived schedule metadata.
     *
     * @details
     * This is a compatibility bridge for older numeric call sites that only pass
     * `SuperSym`. It guarantees the modern numeric path gets all schedule fields.
     *
     * @par Problems
     * - Severity: Low
     * - Issue: Always assumes identity permutation.
     * - Suggested fix: Optionally accept an explicit permutation when available.
     */
    inline SupernodalLLPlan ll_plan_from_sym(const SuperSym &sym, int ncols)
    {
        // Step 1: copy symbolic and create identity ordering vectors.
        SupernodalLLPlan plan;
        plan.sym = sym;
        plan.perm.perm.assign((size_t)ncols, 0);
        plan.perm.inv_perm.assign((size_t)ncols, 0);
        for (int i = 0; i < ncols; ++i)
        {
            plan.perm.perm[(size_t)i] = i;
            plan.perm.inv_perm[(size_t)i] = i;
        }

        // Step 2: derive DAG schedule metadata consumed by numeric.
        fill_schedule_from_sym(plan, ncols);
        return plan;
    }

    template <typename T>
SupernodalLLPlan supernodal_analyze(matrix::CscMatrix<T> &A,
                                            const SymbolicOptions &options);

} // namespace ichol::symbolic
