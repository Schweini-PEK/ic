#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "factor/symbolic/super_sym.hpp"
#include "ichol/matrix_formats.hpp"

namespace ichol::supernodal
{
    inline void build_lower_csc_pattern_from_csr(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        std::vector<int> &col_ptr,
        std::vector<int> &row_ind)
    {
        std::vector<int> ignored_map;
        matrix::csr_to_csc_pattern_only(n, row_ptr, col_ind, col_ptr, row_ind, ignored_map);
    }

    inline void build_lower_csc_pattern_from_csr(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        std::vector<int> &col_ptr,
        std::vector<int> &row_ind,
        std::vector<int> &csc_to_csr_map)
    {
        matrix::csr_to_csc_pattern_only(n, row_ptr, col_ind, col_ptr, row_ind, csc_to_csr_map);
    }

    inline bool are_consecutive_supernode_columns(
        int j,
        const std::vector<int> &col_ptr,
        const std::vector<int> &row_ind)
    {
        const int p0 = col_ptr[(size_t)j];
        const int p1 = col_ptr[(size_t)j + 1];
        const int q0 = col_ptr[(size_t)j + 1];
        const int q1 = col_ptr[(size_t)j + 2];

        const int len_j = p1 - p0;
        const int len_j1 = q1 - q0;
        if (len_j != len_j1 + 1)
            return false;
        if (len_j <= 0 || len_j1 <= 0)
            return false;
        if (row_ind[(size_t)p0] != j || row_ind[(size_t)q0] != j + 1)
            return false;

        for (int t = 1; t < len_j; ++t)
        {
            if (row_ind[(size_t)(p0 + t)] != row_ind[(size_t)(q0 + t - 1)])
                return false;
        }
        return true;
    }

    inline std::vector<int> detect_supernode_boundaries_from_csr_l(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind)
    {
        std::vector<int> col_ptr, row_ind;
        build_lower_csc_pattern_from_csr(n, row_ptr, col_ind, col_ptr, row_ind);

        std::vector<int> super;
        super.reserve((size_t)n + 1);
        super.push_back(0);

        int j = 0;
        while (j < n)
        {
            int end = j + 1;
            while (end < n && are_consecutive_supernode_columns(end - 1, col_ptr, row_ind))
                ++end;
            super.push_back(end);
            j = end;
        }

        return super;
    }

    inline symbolic::SuperSym build_super_sym_from_csr_l(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<int> &super)
    {
        std::vector<int> col_ptr, row_ind;
        build_lower_csc_pattern_from_csr(n, row_ptr, col_ind, col_ptr, row_ind);

        symbolic::SuperSym sym;
        sym.super = super;
        sym.pi.reserve(super.size());
        sym.px.reserve(super.size());
        sym.pi.push_back(0);
        sym.px.push_back(0);

        for (int s = 0; s + 1 < (int)super.size(); ++s)
        {
            const int c0 = super[(size_t)s];
            const int c1 = super[(size_t)s + 1];
            const int nscol = c1 - c0;
            const int p0 = col_ptr[(size_t)c0];
            const int p1 = col_ptr[(size_t)c0 + 1];
            const int nsrow = p1 - p0;

            sym.s.insert(sym.s.end(), row_ind.begin() + p0, row_ind.begin() + p1);
            sym.pi.push_back((int)sym.s.size());
            sym.px.push_back(sym.px.back() + nsrow * nscol);
        }

        return sym;
    }

    struct RelaxedSupernodeCandidatePattern
    {
        std::vector<int> rowlist;
        int actual_nnz = 0;
        int dense_slots = 0;
        double density = 1.0;
    };

    inline RelaxedSupernodeCandidatePattern build_relaxed_candidate_pattern_from_csc(
        int n,
        int c0,
        int c1,
        const std::vector<int> &col_ptr,
        const std::vector<int> &row_ind)
    {
        RelaxedSupernodeCandidatePattern out;
        const int nscol = c1 - c0;
        out.rowlist.reserve((size_t)nscol);
        for (int c = c0; c < c1; ++c)
            out.rowlist.push_back(c);

        std::vector<int> mark((size_t)n, 0);
        for (int c = c0; c < c1; ++c)
        {
            out.actual_nnz += col_ptr[(size_t)c + 1] - col_ptr[(size_t)c];
            for (int p = col_ptr[(size_t)c]; p < col_ptr[(size_t)c + 1]; ++p)
            {
                const int r = row_ind[(size_t)p];
                if (r >= c1)
                    mark[(size_t)r] = 1;
            }
        }

        for (int r = c1; r < n; ++r)
        {
            if (mark[(size_t)r] != 0)
                out.rowlist.push_back(r);
        }

        if (out.rowlist.size() > (size_t)(std::numeric_limits<int>::max() / std::max(1, nscol)))
            throw std::runtime_error("build_relaxed_candidate_pattern_from_csc: dense panel too large");

        out.dense_slots = static_cast<int>(out.rowlist.size()) * nscol;
        out.density = (out.dense_slots > 0)
                          ? static_cast<double>(out.actual_nnz) / static_cast<double>(out.dense_slots)
                          : 1.0;
        return out;
    }

    inline symbolic::SuperSym build_relaxed_super_sym_from_csr_l(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        int max_width,
        double relaxed_extra)
    {
        if (n < 0 || (int)row_ptr.size() != n + 1)
            throw std::runtime_error("build_relaxed_super_sym_from_csr_l: invalid CSR row_ptr size");
        if (max_width <= 0)
            throw std::runtime_error("build_relaxed_super_sym_from_csr_l: max_width must be positive");

        std::vector<int> col_ptr, row_ind;
        build_lower_csc_pattern_from_csr(n, row_ptr, col_ind, col_ptr, row_ind);

        const auto strict_super = detect_supernode_boundaries_from_csr_l(n, row_ptr, col_ind);
        const int strict_count = static_cast<int>(strict_super.size()) - 1;
        const double min_density = std::max(0.0, 1.0 - relaxed_extra);

        symbolic::SuperSym sym;
        sym.super.push_back(0);
        sym.pi.push_back(0);
        sym.px.push_back(0);

        int start_snode = 0;
        while (start_snode < strict_count)
        {
            int best_end_snode = start_snode + 1;
            auto best = build_relaxed_candidate_pattern_from_csc(
                n, strict_super[(size_t)start_snode], strict_super[(size_t)best_end_snode],
                col_ptr, row_ind);

            for (int end_snode = start_snode + 2; end_snode <= strict_count; ++end_snode)
            {
                const int c0 = strict_super[(size_t)start_snode];
                const int c1 = strict_super[(size_t)end_snode];
                if (c1 - c0 > max_width)
                    break;

                auto candidate = build_relaxed_candidate_pattern_from_csc(n, c0, c1, col_ptr, row_ind);
                if (candidate.density + 1e-12 < min_density)
                    break;

                best_end_snode = end_snode;
                best = std::move(candidate);
            }

            const int c1 = strict_super[(size_t)best_end_snode];
            const int nscol = c1 - strict_super[(size_t)start_snode];
            const int nsrow = static_cast<int>(best.rowlist.size());
            if (nsrow < nscol)
                throw std::runtime_error("build_relaxed_super_sym_from_csr_l: invalid relaxed panel dimensions");
            if (nsrow > 0 && nscol > std::numeric_limits<int>::max() / nsrow)
                throw std::runtime_error("build_relaxed_super_sym_from_csr_l: packed panel too large");

            sym.super.push_back(c1);
            sym.s.insert(sym.s.end(), best.rowlist.begin(), best.rowlist.end());
            sym.pi.push_back(static_cast<int>(sym.s.size()));
            sym.px.push_back(sym.px.back() + nsrow * nscol);

            start_snode = best_end_snode;
        }

        return sym;
    }

    inline int count_structural_entries_from_sym(const symbolic::SuperSym &sym)
    {
        if (sym.super.empty())
            return 0;

        int nnz = 0;
        const int nsuper = (int)sym.super.size() - 1;
        for (int k = 0; k < nsuper; ++k)
        {
            const int nscol = sym.super[(size_t)k + 1] - sym.super[(size_t)k];
            const int nsrow = sym.pi[(size_t)k + 1] - sym.pi[(size_t)k];
            for (int j = 0; j < nscol; ++j)
                nnz += (nsrow - j);
        }
        return nnz;
    }

    template <typename OutT, typename InT>
    inline OutT cast_packed_value(const InT &v)
    {
        if constexpr (std::is_same_v<OutT, __half>)
            return __float2half(static_cast<float>(v));
        else if constexpr (std::is_same_v<InT, __half>)
            return static_cast<OutT>(__half2float(v));
        else
            return static_cast<OutT>(v);
    }

    template <typename OutT, typename InT>
    inline std::vector<OutT> pack_supernode_values_from_csr_l(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<InT> &values,
        const symbolic::SuperSym &sym)
    {
        if ((int)row_ptr.size() != n + 1)
            throw std::runtime_error("pack_supernode_values_from_csr_l: invalid CSR row_ptr size");
        if (col_ind.size() != values.size())
            throw std::runtime_error("pack_supernode_values_from_csr_l: col/value size mismatch");
        if (sym.super.size() != sym.pi.size() || sym.super.size() != sym.px.size())
            throw std::runtime_error("pack_supernode_values_from_csr_l: inconsistent SuperSym pointer sizes");
        if (!sym.super.empty() && (sym.super.front() != 0 || sym.pi.front() != 0 || sym.px.front() != 0))
            throw std::runtime_error("pack_supernode_values_from_csr_l: invalid SuperSym prefix pointers");

        std::vector<int> col_ptr, row_ind, csc_to_csr_map;
        build_lower_csc_pattern_from_csr(n, row_ptr, col_ind, col_ptr, row_ind, csc_to_csr_map);

        std::vector<OutT> packed(sym.px.empty() ? 0u : (size_t)sym.px.back(), OutT{});
        std::vector<int> row_pos((size_t)n, -1);
        const int nsuper = (int)sym.super.size() - 1;

        for (int k = 0; k < nsuper; ++k)
        {
            const int c0 = sym.super[(size_t)k];
            const int c1 = sym.super[(size_t)k + 1];
            const int nscol = c1 - c0;
            const int pi0 = sym.pi[(size_t)k];
            const int pi1 = sym.pi[(size_t)k + 1];
            const int nsrow = pi1 - pi0;
            const int px0 = sym.px[(size_t)k];

            if (nscol <= 0 || nsrow < nscol)
                throw std::runtime_error("pack_supernode_values_from_csr_l: invalid supernode dimensions");

            for (int local_row = 0; local_row < nsrow; ++local_row)
                row_pos[(size_t)sym.s[(size_t)(pi0 + local_row)]] = local_row;

            for (int j = 0; j < nscol; ++j)
            {
                const int gcol = c0 + j;
                const int cp0 = col_ptr[(size_t)gcol];
                const int cp1 = col_ptr[(size_t)gcol + 1];

                for (int p = cp0; p < cp1; ++p)
                {
                    const int grow = row_ind[(size_t)p];
                    const int local_row = row_pos[(size_t)grow];
                    if (local_row < 0)
                        throw std::runtime_error("pack_supernode_values_from_csr_l: rowlist mismatch while packing supernode values");
                    if (local_row < j)
                        throw std::runtime_error("pack_supernode_values_from_csr_l: upper-triangular entry found in lower packed panel");

                    const int csr_pos = csc_to_csr_map[(size_t)p];
                    packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row] =
                        cast_packed_value<OutT>(values[(size_t)csr_pos]);
                }
            }

            for (int local_row = 0; local_row < nsrow; ++local_row)
                row_pos[(size_t)sym.s[(size_t)(pi0 + local_row)]] = -1;
        }

        return packed;
    }

    template <typename ValueT>
    inline matrix::CooMatrix<ValueT> expand_supernode_values_to_coo(
        const symbolic::SuperSym &sym,
        const std::vector<ValueT> &packed)
    {
        if (sym.super.size() != sym.pi.size() || sym.super.size() != sym.px.size())
            throw std::runtime_error("expand_supernode_values_to_coo: inconsistent SuperSym pointer sizes");
        if (!sym.px.empty() && (size_t)sym.px.back() != packed.size())
            throw std::runtime_error("expand_supernode_values_to_coo: packed value size does not match SuperSym");

        matrix::CooMatrix<ValueT> coo;
        coo.num_rows = sym.super.empty() ? 0 : sym.super.back();
        coo.num_cols = coo.num_rows;
        coo.nnz = count_structural_entries_from_sym(sym);
        coo.row_ind.reserve((size_t)coo.nnz);
        coo.col_ind.reserve((size_t)coo.nnz);
        coo.values.reserve((size_t)coo.nnz);

        const int nsuper = (int)sym.super.size() - 1;
        for (int k = 0; k < nsuper; ++k)
        {
            const int c0 = sym.super[(size_t)k];
            const int nscol = sym.super[(size_t)k + 1] - c0;
            const int pi0 = sym.pi[(size_t)k];
            const int nsrow = sym.pi[(size_t)k + 1] - pi0;
            const int px0 = sym.px[(size_t)k];

            for (int j = 0; j < nscol; ++j)
            {
                const int gcol = c0 + j;
                for (int local_row = j; local_row < nsrow; ++local_row)
                {
                    coo.row_ind.push_back(sym.s[(size_t)(pi0 + local_row)]);
                    coo.col_ind.push_back(gcol);
                    coo.values.push_back(packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row]);
                }
            }
        }

        return coo;
    }
} // namespace ichol::supernodal
