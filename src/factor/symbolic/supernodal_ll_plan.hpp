#pragma once

// supernodal_ll_plan.hpp
//
// supernodal LL（下三角 Cholesky/IC）符号阶段的一站式输出：SupernodalLLPlan。
//
// 该 plan 把“列级别”的 etree/pattern 进一步转换为：
//   - snodes（supernode 边界）
//   - snode_rows / SuperSym（CHOLMOD 风格 rowlist 打包）
//   - parent/children/level/buckets（supernode DAG 调度信息）
//
// 数值阶段应直接使用该 plan，避免重复构建 supernode/rowlist。

#include <vector>
#include "ichol/options.hpp"
#include "ichol/matrix_formats.hpp"

#include "factor/symbolic/symbolic.hpp"           // etree / pattern / supernode detection
#include "factor/symbolic/super_sym.hpp"          // SuperSym + build_super_sym
#include "factor/symbolic/snode_schedule.hpp"     // CHOLMOD-style parent/children/level buckets

namespace ichol::symbolic {

// A single, explicit "symbolic result" that the LL supernodal numeric kernels consume.
// The numeric phase should NOT rebuild any of these.
struct SupernodalLLPlan {
    // Keep these for debugging / inspection (optional for numeric, but cheap to store)
    ETree etree;
    FactorPattern factor_pattern;

    // CHOLMOD-like supernodal symbolic data (super, pi, px, s)
    SuperSym sym;

    // Supernodal elimination tree derived from the rowlist (CHOLMOD-style)
    std::vector<int> parent;                    // size = nsuper
    std::vector<std::vector<int>> children;     // adjacency list

    // Level buckets (height-from-leaves). Same level can be processed in parallel.
    std::vector<int> level;                     // size = nsuper
    std::vector<std::vector<int>> buckets;      // buckets[level] = snode ids

    // Optional debug / inspection
    std::vector<std::pair<int,int>> snodes;     // [start,end) per snode
    std::vector<std::vector<int>> snode_rows;   // union row-set per snode
};

// Build only the scheduling information from an already-built SuperSym.
// This is the exact symbolic work that used to be duplicated inside numeric.
inline SupernodalLLPlan ll_plan_from_sym(const SuperSym& sym, int ncols)
{
    SupernodalLLPlan plan;
    plan.sym = sym;

    auto col2s  = build_col2snode(sym.super, ncols);
    plan.parent = build_snode_parent_from_rowlist(sym, col2s);
    plan.children = build_children(plan.parent);

    plan.level   = compute_level_from_leaves(plan.children);
    plan.buckets = bucket_by_level(plan.level);
    return plan;
}

// Full supernodal symbolic phase for LL numeric factorization.
// This matches the symbolic logic the numeric code expects:
//   - detect supernodes (CHOLMOD relaxed default unless approximate is requested)
//   - compute each snode rowlist from the L pattern
//   - build SuperSym (super,pi,px,s) with pivot rows first, then update rows
//   - build parent/children/buckets from SuperSym rowlist (CHOLMOD-style)
inline SupernodalLLPlan supernodal_ll_analyze(const ichol::matrix::CscMatrix<double>& A,
                                             const SuperNodeOptions& sn_options)
{
    SupernodalLLPlan plan;

    // 1) etree + complete pattern (CSC-native)
    plan.etree = build_etree<double>(A);
    plan.factor_pattern = compute_complete_cholesky_pattern<double>(A, plan.etree);

    // 2) detect supernodes
    if (sn_options.approximate) {
        plan.snodes = detect_supernodes_approx(plan.factor_pattern, plan.etree, sn_options.overlap_threshold);
    } else {
    #ifdef ICHOL_SUPERNODES_FUNDAMENTAL
        plan.snodes = detect_supernodes_fundamental(plan.etree);
    #else
        plan.snodes = detect_supernodes(plan.factor_pattern, plan.etree);
    #endif
    }

    // 3) rowlists per snode + SuperSym
    plan.snode_rows = compute_snode_rows(plan.factor_pattern, plan.snodes);
    plan.sym = build_super_sym(plan.snodes, plan.snode_rows);

    // 4) schedule (CHOLMOD-style) from SuperSym
    auto sched = ll_plan_from_sym(plan.sym, A.num_cols);
    plan.parent   = std::move(sched.parent);
    plan.children = std::move(sched.children);
    plan.level    = std::move(sched.level);
    plan.buckets  = std::move(sched.buckets);

    return plan;
}

} // namespace ichol::numeric
