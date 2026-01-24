#pragma once
#include <vector>
#include <algorithm>
#include "factor/symbolic/super_sym.hpp"

// snode_schedule.hpp
//
// 该文件提供从 SuperSym(=CHOLMOD 风格的 supernodal rowlist 打包结果) 推导调度信息的工具函数：
//   1) col -> supernode id 的映射
//   2) supernode parent/children（类似 CHOLMOD 的 supernodal elimination tree）
//   3) level/buckets（从叶子向上的层级，便于按 level 并行执行不同 supernode）
//
// 说明：CHOLMOD 对 supernode 的 parent 选择规则是“该 supernode 的 rowlist 中第一个 update 行所对应的列所在的 supernode”。
// 这里的 build_snode_parent_from_rowlist 与之保持一致。

namespace ichol::symbolic {

/**
 * @brief 将列号映射到所属 supernode id。
 *
 * @param super  SuperSym.super（长度 nsuper+1），给出每个 supernode 的列边界
 * @param ncols  总列数（通常等于矩阵维度 n）
 * @return       col2snode[c] = supernode id
 */
inline std::vector<int> build_col2snode(const std::vector<int>& super, int ncols)
{
    std::vector<int> col2s((size_t)ncols, -1);
    const int nsuper = (int)super.size() - 1;
    for (int k = 0; k < nsuper; ++k) {
        for (int c = super[(size_t)k]; c < super[(size_t)k + 1]; ++c) col2s[(size_t)c] = k;
    }
    return col2s;
}

/**
 * @brief 构造 supernode 级别的 parent（CHOLMOD 规则）。
 *
 * 对第 k 个 supernode：
 *  - nscol = super[k+1]-super[k]
 *  - rowlist = s[pi[k]..pi[k+1])
 *  - 若 nsrow<=nscol，则无 update 行，parent=-1
 *  - 否则 parent 取 rowlist 中第一个 update 行对应列所在的 supernode
 *
 * 该规则与 CHOLMOD 的 cholmod_super_symbolic 默认管线一致，
 * 可保证调度 DAG 与 CHOLMOD 的 supernode 消去依赖一致。
 */
inline std::vector<int> build_snode_parent_from_rowlist(const symbolic::SuperSym& sym,
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
            int pcol = sym.s[(size_t)t];
            ps = col2snode[(size_t)pcol];
            if (ps != k) break;
            ++t;
        }
        parent[(size_t)k] = (t < pi1) ? ps : -1;
    }
    return parent;
}

/**
 * @brief 将 parent 指针数组转换为 children 邻接表。
 */
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

/**
 * @brief 从叶子向上计算 level（leaf=0; parent=1+max(child)）。
 *
 * 这个 level 可以用于简单的分层调度：同一层的 supernode 互不依赖，可并行。
 */
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

/**
 * @brief 按 level 分桶：buckets[l] = 所有 level==l 的 supernode id。
 */
inline std::vector<std::vector<int>> bucket_by_level(const std::vector<int>& level)
{
    int maxl = 0;
    for (int v : level) maxl = std::max(maxl, v);
    std::vector<std::vector<int>> buckets((size_t)maxl + 1);
    for (int i = 0; i < (int)level.size(); ++i) buckets[(size_t)level[(size_t)i]].push_back(i);
    return buckets;
}

} // namespace ichol::symbolic::detail
