#pragma once

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "factor/symbolic/supernodal_ll_plan.hpp"
#include "factor/symbolic/super_sym.hpp"

namespace ichol::supernodal
{
    template <typename ValueT>
    inline void validate_cpu_solve_value_type()
    {
        static_assert(std::is_floating_point_v<ValueT>,
                      "CPU reference supernodal solve currently supports float/double only.");
    }

    template <typename ValueT>
    inline void build_csr_transpose_diag_last(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<ValueT> &values,
        std::vector<int> &row_ptr_t,
        std::vector<int> &col_ind_t,
        std::vector<ValueT> &values_t)
    {
        validate_cpu_solve_value_type<ValueT>();

        if ((int)row_ptr.size() != n + 1)
            throw std::runtime_error("build_csr_transpose_diag_last: invalid row_ptr size");
        if (col_ind.size() != values.size())
            throw std::runtime_error("build_csr_transpose_diag_last: col/value size mismatch");

        const int nnz = static_cast<int>(values.size());
        row_ptr_t.assign((size_t)n + 1, 0);
        col_ind_t.resize((size_t)nnz);
        values_t.resize((size_t)nnz);

        for (int i = 0; i < n; ++i)
        {
            const int s = row_ptr[(size_t)i];
            const int e = row_ptr[(size_t)i + 1];
            if (e <= s)
                throw std::runtime_error("build_csr_transpose_diag_last: empty row is not allowed");
            const int end = e - 1;

            for (int p = s; p < end; ++p)
            {
                const int j = col_ind[(size_t)p];
                ++row_ptr_t[(size_t)j + 1];
            }

            ++row_ptr_t[(size_t)i + 1];
        }

        for (int r = 0; r < n; ++r)
            row_ptr_t[(size_t)r + 1] += row_ptr_t[(size_t)r];

        std::vector<int> next = row_ptr_t;
        for (int i = 0; i < n; ++i)
        {
            const int s = row_ptr[(size_t)i];
            const int e = row_ptr[(size_t)i + 1];
            const int end = e - 1;

            for (int p = s; p < end; ++p)
            {
                const int j = col_ind[(size_t)p];
                const int dst = next[(size_t)j]++;
                col_ind_t[(size_t)dst] = i;
                values_t[(size_t)dst] = values[(size_t)p];
            }

            const int diag_dst = row_ptr_t[(size_t)i + 1] - 1;
            col_ind_t[(size_t)diag_dst] = i;
            values_t[(size_t)diag_dst] = values[(size_t)end];
        }
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_lower_scalar_csr_diag_last(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<ValueT> &values,
        const std::vector<ValueT> &b)
    {
        validate_cpu_solve_value_type<ValueT>();

        if ((int)b.size() != n)
            throw std::runtime_error("solve_lower_scalar_csr_diag_last: rhs size mismatch");

        std::vector<ValueT> x((size_t)n, ValueT{});
        for (int i = 0; i < n; ++i)
        {
            const int s = row_ptr[(size_t)i];
            const int e = row_ptr[(size_t)i + 1];
            if (e <= s)
                throw std::runtime_error("solve_lower_scalar_csr_diag_last: empty row is not allowed");

            ValueT rhs = b[(size_t)i];
            for (int p = s; p < e - 1; ++p)
                rhs -= values[(size_t)p] * x[(size_t)col_ind[(size_t)p]];

            const ValueT diag = values[(size_t)e - 1];
            if (diag == ValueT{})
                throw std::runtime_error("solve_lower_scalar_csr_diag_last: zero diagonal");

            x[(size_t)i] = rhs / diag;
        }
        return x;
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_upper_scalar_csr_diag_last(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<ValueT> &values,
        const std::vector<ValueT> &b)
    {
        validate_cpu_solve_value_type<ValueT>();

        if ((int)b.size() != n)
            throw std::runtime_error("solve_upper_scalar_csr_diag_last: rhs size mismatch");

        std::vector<ValueT> x((size_t)n, ValueT{});
        for (int i = n - 1; i >= 0; --i)
        {
            const int s = row_ptr[(size_t)i];
            const int e = row_ptr[(size_t)i + 1];
            if (e <= s)
                throw std::runtime_error("solve_upper_scalar_csr_diag_last: empty row is not allowed");

            ValueT rhs = b[(size_t)i];
            for (int p = s; p < e - 1; ++p)
                rhs -= values[(size_t)p] * x[(size_t)col_ind[(size_t)p]];

            const ValueT diag = values[(size_t)e - 1];
            if (diag == ValueT{})
                throw std::runtime_error("solve_upper_scalar_csr_diag_last: zero diagonal");

            x[(size_t)i] = rhs / diag;
        }
        return x;
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_lower_transpose_scalar_csr_diag_last(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<ValueT> &values,
        const std::vector<ValueT> &b)
    {
        validate_cpu_solve_value_type<ValueT>();

        std::vector<int> row_ptr_t, col_ind_t;
        std::vector<ValueT> values_t;
        build_csr_transpose_diag_last(n, row_ptr, col_ind, values, row_ptr_t, col_ind_t, values_t);
        return solve_upper_scalar_csr_diag_last(n, row_ptr_t, col_ind_t, values_t, b);
    }

    template <typename ValueT>
    inline void validate_supernodal_solve_inputs(
        const symbolic::SuperSym &sym,
        const std::vector<ValueT> &packed,
        const std::vector<ValueT> &b)
    {
        validate_cpu_solve_value_type<ValueT>();

        if (sym.super.size() != sym.pi.size() || sym.super.size() != sym.px.size())
            throw std::runtime_error("validate_supernodal_solve_inputs: inconsistent SuperSym pointer sizes");
        if (!sym.super.empty() && (sym.super.front() != 0 || sym.pi.front() != 0 || sym.px.front() != 0))
            throw std::runtime_error("validate_supernodal_solve_inputs: invalid SuperSym prefix pointers");
        if (!sym.px.empty() && (size_t)sym.px.back() != packed.size())
            throw std::runtime_error("validate_supernodal_solve_inputs: packed value size mismatch");

        const int n = sym.super.empty() ? 0 : sym.super.back();
        if ((int)b.size() != n)
            throw std::runtime_error("validate_supernodal_solve_inputs: rhs size mismatch");

        const int nsuper = static_cast<int>(sym.super.size()) - 1;
        for (int k = 0; k < nsuper; ++k)
        {
            const int c0 = sym.super[(size_t)k];
            const int c1 = sym.super[(size_t)k + 1];
            const int nscol = c1 - c0;
            const int pi0 = sym.pi[(size_t)k];
            const int pi1 = sym.pi[(size_t)k + 1];
            const int nsrow = pi1 - pi0;
            if (nscol <= 0 || nsrow < nscol)
                throw std::runtime_error("validate_supernodal_solve_inputs: invalid supernode dimensions");

            for (int t = 0; t < nscol; ++t)
            {
                if (sym.s[(size_t)(pi0 + t)] != c0 + t)
                    throw std::runtime_error("validate_supernodal_solve_inputs: pivot rowlist prefix mismatch");
            }
        }
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_lower_supernodal(
        const symbolic::SuperSym &sym,
        const std::vector<ValueT> &packed,
        const std::vector<ValueT> &b)
    {
        validate_supernodal_solve_inputs(sym, packed, b);

        const int n = sym.super.empty() ? 0 : sym.super.back();
        std::vector<ValueT> x((size_t)n, ValueT{});
        std::vector<ValueT> work = b;

        const int nsuper = static_cast<int>(sym.super.size()) - 1;
        for (int k = 0; k < nsuper; ++k)
        {
            const int c0 = sym.super[(size_t)k];
            const int c1 = sym.super[(size_t)k + 1];
            const int nscol = c1 - c0;
            const int pi0 = sym.pi[(size_t)k];
            const int pi1 = sym.pi[(size_t)k + 1];
            const int nsrow = pi1 - pi0;
            const int px0 = sym.px[(size_t)k];

            for (int j = 0; j < nscol; ++j)
            {
                ValueT rhs = work[(size_t)(c0 + j)];
                for (int i = 0; i < j; ++i)
                    rhs -= packed[(size_t)px0 + (size_t)i * (size_t)nsrow + (size_t)j] * x[(size_t)(c0 + i)];

                const ValueT diag = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)j];
                if (diag == ValueT{})
                    throw std::runtime_error("solve_lower_supernodal: zero diagonal");

                x[(size_t)(c0 + j)] = rhs / diag;
            }

            for (int local_row = nscol; local_row < nsrow; ++local_row)
            {
                ValueT update = ValueT{};
                for (int j = 0; j < nscol; ++j)
                    update += packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row] * x[(size_t)(c0 + j)];

                const int grow = sym.s[(size_t)(pi0 + local_row)];
                work[(size_t)grow] -= update;
            }
        }

        return x;
    }

    template <typename ValueT>
    inline void validate_supernodal_plan_solve_inputs(
        const symbolic::SupernodalLLPlan &plan,
        const std::vector<ValueT> &packed,
        const std::vector<ValueT> &b)
    {
        validate_supernodal_solve_inputs(plan.sym, packed, b);

        const int nsuper = static_cast<int>(plan.sym.super.size()) - 1;
        if ((int)plan.level.size() != nsuper)
            throw std::runtime_error("validate_supernodal_plan_solve_inputs: level size mismatch");

        std::vector<int> seen((size_t)nsuper, 0);
        for (size_t lvl = 0; lvl < plan.buckets.size(); ++lvl)
        {
            for (int k : plan.buckets[lvl])
            {
                if (k < 0 || k >= nsuper)
                    throw std::runtime_error("validate_supernodal_plan_solve_inputs: bucket entry out of range");
                if (plan.level[(size_t)k] != (int)lvl)
                    throw std::runtime_error("validate_supernodal_plan_solve_inputs: bucket/level mismatch");
                if (++seen[(size_t)k] != 1)
                    throw std::runtime_error("validate_supernodal_plan_solve_inputs: duplicate supernode in buckets");
            }
        }

        for (int k = 0; k < nsuper; ++k)
        {
            if (seen[(size_t)k] != 1)
                throw std::runtime_error("validate_supernodal_plan_solve_inputs: missing supernode in buckets");
        }
    }

    inline std::vector<std::vector<int>> build_forward_solve_buckets_from_sym(
        const symbolic::SuperSym &sym)
    {
        if (sym.super.empty())
            return {};

        const int nsuper = static_cast<int>(sym.super.size()) - 1;
        const int ncols = sym.super.back();
        const auto col2snode = symbolic::build_col2snode(sym.super, ncols);

        std::vector<std::vector<int>> children((size_t)nsuper);
        std::vector<int> indegree((size_t)nsuper, 0);
        std::vector<int> marked((size_t)nsuper, -1);

        for (int k = 0; k < nsuper; ++k)
        {
            const int nscol = sym.super[(size_t)k + 1] - sym.super[(size_t)k];
            const int pi0 = sym.pi[(size_t)k];
            const int pi1 = sym.pi[(size_t)k + 1];

            for (int local_row = nscol; local_row < pi1 - pi0; ++local_row)
            {
                const int grow = sym.s[(size_t)(pi0 + local_row)];
                if (grow < 0 || grow >= ncols)
                    throw std::runtime_error("build_forward_solve_buckets_from_sym: row index out of range");

                const int dst = col2snode[(size_t)grow];
                if (dst < 0 || dst >= nsuper)
                    throw std::runtime_error("build_forward_solve_buckets_from_sym: row owner out of range");
                if (dst == k || marked[(size_t)dst] == k)
                    continue;

                marked[(size_t)dst] = k;
                children[(size_t)k].push_back(dst);
                ++indegree[(size_t)dst];
            }
        }

        std::vector<int> frontier;
        frontier.reserve((size_t)nsuper);
        for (int k = 0; k < nsuper; ++k)
        {
            if (indegree[(size_t)k] == 0)
                frontier.push_back(k);
        }

        std::vector<std::vector<int>> buckets;
        int visited = 0;
        while (!frontier.empty())
        {
            std::sort(frontier.begin(), frontier.end());
            buckets.push_back(frontier);
            visited += static_cast<int>(frontier.size());

            std::vector<int> next;
            for (int k : frontier)
            {
                for (int dst : children[(size_t)k])
                {
                    --indegree[(size_t)dst];
                    if (indegree[(size_t)dst] == 0)
                        next.push_back(dst);
                }
            }
            frontier = std::move(next);
        }

        if (visited != nsuper)
            throw std::runtime_error("build_forward_solve_buckets_from_sym: cycle in supernode solve dependency graph");

        return buckets;
    }

    inline void validate_forward_solve_buckets(
        const symbolic::SuperSym &sym,
        const std::vector<std::vector<int>> &buckets)
    {
        const int nsuper = sym.super.empty() ? 0 : static_cast<int>(sym.super.size()) - 1;
        std::vector<int> seen((size_t)nsuper, 0);

        for (const auto &bucket : buckets)
        {
            for (int k : bucket)
            {
                if (k < 0 || k >= nsuper)
                    throw std::runtime_error("validate_forward_solve_buckets: bucket entry out of range");
                if (++seen[(size_t)k] != 1)
                    throw std::runtime_error("validate_forward_solve_buckets: duplicate supernode in buckets");
            }
        }

        for (int k = 0; k < nsuper; ++k)
        {
            if (seen[(size_t)k] != 1)
                throw std::runtime_error("validate_forward_solve_buckets: missing supernode in buckets");
        }
    }

    template <typename ValueT>
    inline void solve_lower_supernode_in_place(
        const symbolic::SuperSym &sym,
        const std::vector<ValueT> &packed,
        int k,
        std::vector<ValueT> &x,
        std::vector<ValueT> &work)
    {
        const int c0 = sym.super[(size_t)k];
        const int c1 = sym.super[(size_t)k + 1];
        const int nscol = c1 - c0;
        const int pi0 = sym.pi[(size_t)k];
        const int pi1 = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;
        const int px0 = sym.px[(size_t)k];

        for (int j = 0; j < nscol; ++j)
        {
            ValueT rhs = work[(size_t)(c0 + j)];
            for (int i = 0; i < j; ++i)
                rhs -= packed[(size_t)px0 + (size_t)i * (size_t)nsrow + (size_t)j] * x[(size_t)(c0 + i)];

            const ValueT diag = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)j];
            if (diag == ValueT{})
                throw std::runtime_error("solve_lower_supernode_in_place: zero diagonal");

            x[(size_t)(c0 + j)] = rhs / diag;
        }

        for (int local_row = nscol; local_row < nsrow; ++local_row)
        {
            ValueT update = ValueT{};
            for (int j = 0; j < nscol; ++j)
                update += packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row] * x[(size_t)(c0 + j)];

            const int grow = sym.s[(size_t)(pi0 + local_row)];
            work[(size_t)grow] -= update;
        }
    }

    template <typename ValueT>
    inline void solve_lower_transpose_supernode_in_place(
        const symbolic::SuperSym &sym,
        const std::vector<ValueT> &packed,
        int k,
        const std::vector<ValueT> &b,
        std::vector<ValueT> &x)
    {
        const int c0 = sym.super[(size_t)k];
        const int c1 = sym.super[(size_t)k + 1];
        const int nscol = c1 - c0;
        const int pi0 = sym.pi[(size_t)k];
        const int pi1 = sym.pi[(size_t)k + 1];
        const int nsrow = pi1 - pi0;
        const int px0 = sym.px[(size_t)k];

        std::vector<ValueT> rhs((size_t)nscol, ValueT{});
        for (int j = 0; j < nscol; ++j)
            rhs[(size_t)j] = b[(size_t)(c0 + j)];

        for (int local_row = nscol; local_row < nsrow; ++local_row)
        {
            const int grow = sym.s[(size_t)(pi0 + local_row)];
            const ValueT x_update = x[(size_t)grow];
            for (int j = 0; j < nscol; ++j)
                rhs[(size_t)j] -= packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row] * x_update;
        }

        for (int j = nscol - 1; j >= 0; --j)
        {
            ValueT s = rhs[(size_t)j];
            for (int i = j + 1; i < nscol; ++i)
                s -= packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)i] * x[(size_t)(c0 + i)];

            const ValueT diag = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)j];
            if (diag == ValueT{})
                throw std::runtime_error("solve_lower_transpose_supernode_in_place: zero diagonal");

            x[(size_t)(c0 + j)] = s / diag;
        }
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_lower_supernodal_bucketed(
        const symbolic::SupernodalLLPlan &plan,
        const std::vector<ValueT> &packed,
        const std::vector<ValueT> &b,
        const std::vector<std::vector<int>> &buckets)
    {
        validate_supernodal_plan_solve_inputs(plan, packed, b);

        const auto &sym = plan.sym;
        validate_forward_solve_buckets(sym, buckets);
        const int n = sym.super.empty() ? 0 : sym.super.back();
        std::vector<ValueT> x((size_t)n, ValueT{});
        std::vector<ValueT> work = b;

        // The solve schedule is derived from the full rowlist dependencies.
        // plan.buckets is a parent-tree schedule and can miss extra IC edges.
        for (const auto &bucket : buckets)
        {
            for (int k : bucket)
                solve_lower_supernode_in_place(sym, packed, k, x, work);
        }

        return x;
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_lower_supernodal_bucketed(
        const symbolic::SupernodalLLPlan &plan,
        const std::vector<ValueT> &packed,
        const std::vector<ValueT> &b)
    {
        const auto buckets = build_forward_solve_buckets_from_sym(plan.sym);
        return solve_lower_supernodal_bucketed(plan, packed, b, buckets);
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_lower_transpose_supernodal(
        const symbolic::SuperSym &sym,
        const std::vector<ValueT> &packed,
        const std::vector<ValueT> &b)
    {
        validate_supernodal_solve_inputs(sym, packed, b);

        const int n = sym.super.empty() ? 0 : sym.super.back();
        std::vector<ValueT> x((size_t)n, ValueT{});

        const int nsuper = static_cast<int>(sym.super.size()) - 1;
        for (int k = nsuper - 1; k >= 0; --k)
        {
            const int c0 = sym.super[(size_t)k];
            const int c1 = sym.super[(size_t)k + 1];
            const int nscol = c1 - c0;
            const int pi0 = sym.pi[(size_t)k];
            const int pi1 = sym.pi[(size_t)k + 1];
            const int nsrow = pi1 - pi0;
            const int px0 = sym.px[(size_t)k];

            std::vector<ValueT> rhs((size_t)nscol, ValueT{});
            for (int j = 0; j < nscol; ++j)
                rhs[(size_t)j] = b[(size_t)(c0 + j)];

            for (int local_row = nscol; local_row < nsrow; ++local_row)
            {
                const int grow = sym.s[(size_t)(pi0 + local_row)];
                const ValueT x_update = x[(size_t)grow];
                for (int j = 0; j < nscol; ++j)
                    rhs[(size_t)j] -= packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row] * x_update;
            }

            for (int j = nscol - 1; j >= 0; --j)
            {
                ValueT s = rhs[(size_t)j];
                for (int i = j + 1; i < nscol; ++i)
                    s -= packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)i] * x[(size_t)(c0 + i)];

                const ValueT diag = packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)j];
                if (diag == ValueT{})
                    throw std::runtime_error("solve_lower_transpose_supernodal: zero diagonal");

                x[(size_t)(c0 + j)] = s / diag;
            }
        }

        return x;
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_lower_transpose_supernodal_bucketed(
        const symbolic::SupernodalLLPlan &plan,
        const std::vector<ValueT> &packed,
        const std::vector<ValueT> &b,
        const std::vector<std::vector<int>> &buckets)
    {
        validate_supernodal_plan_solve_inputs(plan, packed, b);

        const auto &sym = plan.sym;
        validate_forward_solve_buckets(sym, buckets);
        const int n = sym.super.empty() ? 0 : sym.super.back();
        std::vector<ValueT> x((size_t)n, ValueT{});

        // Reverse of the forward schedule: L^T dependencies flow from
        // parent/update rows back to their child supernodes.
        for (int lvl = static_cast<int>(buckets.size()) - 1; lvl >= 0; --lvl)
        {
            const auto &bucket = buckets[(size_t)lvl];
            for (int k : bucket)
                solve_lower_transpose_supernode_in_place(sym, packed, k, b, x);
        }

        return x;
    }

    template <typename ValueT>
    inline std::vector<ValueT> solve_lower_transpose_supernodal_bucketed(
        const symbolic::SupernodalLLPlan &plan,
        const std::vector<ValueT> &packed,
        const std::vector<ValueT> &b)
    {
        const auto buckets = build_forward_solve_buckets_from_sym(plan.sym);
        return solve_lower_transpose_supernodal_bucketed(plan, packed, b, buckets);
    }
} // namespace ichol::supernodal
