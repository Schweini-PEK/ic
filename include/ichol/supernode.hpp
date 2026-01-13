#ifndef INCHOL_SUPERNODE_HPP
#define INCHOL_SUPERNODE_HPP

#include <vector>
#include "matrix_formats.hpp"
#include "symbolic.hpp"

namespace ichol
{
    namespace core
    {
        /**
         * @brief 超级节点结构，用于存储稀疏矩阵分解中具有相似列模式的连续列集合（基于CSC矩阵）
         *
         * 超级节点是一组连续的列，这些列具有相同或相似的行索引模式，可用于优化
         * 稀疏矩阵分解（如不完全Cholesky）的计算效率和存储开销。
         */
        struct Supernode
        {
            int start_col;       // 超级节点起始列索引（包含）
            int end_col;         // 超级节点结束列索引（包含）
            int row_idx_start;   // 在全局行索引数组中的起始偏移
            int row_idx_count;   // 该超级节点包含的行索引数量
            int first_child;     // 第一个子超级节点索引（-1表示无）
            int next_sibling;    // 下一个同级超级节点索引（-1表示无）
        };

        /**
         * @brief 超级节点符号分解结果，包含所有超级节点的结构信息（基于CSC矩阵）
         */
        struct Supernode_Symbolic
        {
            int n;                          // 矩阵维度（行数/列数）
            std::vector<Supernode> nodes;   // 超级节点列表
            std::vector<int> row_indices;   // 全局行索引数组，供超级节点引用
            std::vector<int> col_to_snode;  // 列到所属超级节点的映射（size: n）
        };

        /**
         * @brief 从CSC矩阵和IC符号分解结果构建超级节点符号结构
         *
         * @param A 输入CSC矩阵（假设为对称正定矩阵的下三角部分）
         * @param ic_sym 已有的IC符号分解结果
         * @return 构建的超级节点符号结构
         */
        Supernode_Symbolic build_supernode_symbolic(
            const ichol::CscMatrix<double>& A,
            const ichol::core::IC_Symbolic& ic_sym);

        /**
         * @brief 检查两个连续列是否具有相同的行模式（用于超级节点合并）
         *
         * @param col_ptr 矩阵列指针（CSC格式）
         * @param row_ind 矩阵行索引（CSC格式）
         * @param i 第i列
         * @param j 第j列（假设j = i+1）
         * @return 若行模式相同则返回true，否则返回false
         */
        bool check_col_pattern_equal(
            const std::vector<int>& col_ptr,
            const std::vector<int>& row_ind,
            int i,
            int j);
    } // namespace core
} // namespace ichol

#endif // INCHOL_SUPERNODE_HPP