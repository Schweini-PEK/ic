#pragma once

#include <atomic>
#include <vector>

#include "ichol/matrix/csc.hpp"
#include "factor/symbolic/symbolic.hpp" // symbolic::SuperSym

namespace ichol::numeric {

    struct SupernodalWorkspace; // forward

    namespace detail {

        // UpdatePack 在实现里定义，这里只 forward declare
        struct UpdatePack;

        // 新签名（带 child_relpos）：scheduler 已经在传这个
        // 先提供一个 inline 兼容重载：暂时忽略 child_relpos，转调旧实现，先把工程编过。
        // 下一步再把 kernels 里真正用上 child_relpos 做提速。
        inline void compute_one_supernode_cpu(
            int k,
            const matrix::CscMatrix<double>& A,
            const symbolic::SuperSym& sym,
            const std::vector<std::vector<int>>& children,
            const std::vector<std::vector<std::vector<int>>>& child_relpos,
            std::vector<UpdatePack>& up,
            std::vector<double>& x,
            std::atomic<bool>& ok,
            std::atomic<int>& fail_snode,
            std::atomic<int>& fail_col,
            SupernodalWorkspace& wk)
        {
            (void)child_relpos;
            compute_one_supernode_cpu(k, A, sym, children, up, x, ok, fail_snode, fail_col, wk);
        }

    } // namespace detail
} // namespace ichol::numeric