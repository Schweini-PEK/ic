#pragma once

#include <vector>
#include <utility>

namespace ichol::matrix {
    template <typename T>
    struct CscMatrix;
}

namespace ichol::symbolic {
    struct ETree;
}

namespace ichol::symbolic
{
    /**
     *
     * 该结构用于把“每个 supernode 的列范围 + 行列表”编码成一组紧凑数组，方便数值分解阶段
     * 直接消费，而无需再次扫描稀疏结构。
     *
     * 记第 k 个 supernode 的列区间为 [super[k], super[k+1])，宽度 nscol = super[k+1]-super[k]。
     * 对应的行列表（rowlist）存放在 s[pi[k] .. pi[k+1])，长度 nsrow。
     *
     * 约定（与 CHOLMOD 默认 supernodal LL 管线一致）：
     *  - rowlist 的前 nscol 个条目是 pivot 行：super[k], super[k]+1, ..., super[k+1)-1
     *  - 后续条目为 update 行：严格递增、去重，并且必须满足 r >= super[k+1]
     *
     * px 是数值阶段用于定位每个 supernode 稠密块存储的前缀和：
     *  - block(k) 的元素数 = nsrow * nscol
     *  - px[k] 为第 k 个 supernode 稠密块在一维数组中的起始偏移（类似 CHOLMOD 的 Lx / Px 用法）
     */
    struct SuperSym
    {
        std::vector<int> super; // size = nsuper + 1, supernode 的列边界（起始列号）
        std::vector<int> pi;    // size = nsuper + 1, rowlist 指针：rowlist(k)=s[pi[k]..pi[k+1])
        std::vector<int> px;    // size = nsuper + 1, 数值块偏移前缀和：px[k+1]=px[k]+nsrow*nscol
        std::vector<int> s;     // size = pi.back(), 按 supernode 打包后的 rowlist
    };

    /**
     * @brief 从 snodes + snode_rows 构造 SuperSym（CHOLMOD 兼容的 rowlist 格式）。
     *
     * @param snodes      每个 supernode 的列区间 [start_col,end_col)
     * @param snode_rows  每个 supernode 的行集合（通常是该 supernode 中所有列的并集行模式）
     * @return            填充好的 SuperSym（包含 super/pi/px/s）
     */
    SuperSym build_super_sym(
        const std::vector<std::pair<int,int>>& snodes,
        const std::vector<std::vector<int>>& snode_rows);

    /**
     * @brief 直接从 A + etree + snodes 构造 SuperSym（避免先构造完整 FactorPattern + snode_rows）。
     *
     * 该函数的目标是匹配 CHOLMOD(supernodal LL) 的 rowlist 语义，同时减少符号阶段的
     * 中间结构开销（尤其是完整 L 的 column pattern）。
     *
     * 限制：目前仅用于 complete-Cholesky 的 supernodal LL 管线（无 IC(k) level）。
     */
    SuperSym build_super_sym_direct(
        const ichol::matrix::CscMatrix<double>& A,
        const ETree& etree,
        const std::vector<std::pair<int,int>>& snodes);
}
