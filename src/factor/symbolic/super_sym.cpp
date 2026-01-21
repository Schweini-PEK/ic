#include "super_sym.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <filesystem>
#include <stdexcept>

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
}
