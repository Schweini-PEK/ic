#include <gtest/gtest.h>
#include <petscsys.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/util/cast.hpp"
#include "factor/numerical/detail/numeric_plan.hpp"
#include "factor/numerical/factorize.hpp"
#include "factor/supernodal_solve.hpp"
#include "factor/supernodal_solve_cuda.cuh"
#include "factor/supernodal_storage.hpp"
#include "factor/symbolic/symbolic.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/mtx_read.hpp"
#include "ichol/options.hpp"
#include "ichol/pcg.hpp"
#include "ichol/preconditioner.hpp"

namespace
{
    template <typename To, typename From>
    __global__ void experiment_cast_vec(int n, const From *__restrict__ in, To *__restrict__ out)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            out[i] = static_cast<To>(in[i]);
    }

    struct CandidatePattern
    {
        std::vector<int> rowlist;
        int actual_nnz = 0;
        int dense_slots = 0;
        double density = 1.0;
    };

    CandidatePattern build_candidate_pattern(
        int n,
        int c0,
        int c1,
        const std::vector<int> &col_ptr,
        const std::vector<int> &row_ind)
    {
        CandidatePattern out;
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

        out.dense_slots = static_cast<int>(out.rowlist.size()) * nscol;
        out.density = (out.dense_slots > 0)
                          ? static_cast<double>(out.actual_nnz) / static_cast<double>(out.dense_slots)
                          : 1.0;
        return out;
    }

    ichol::symbolic::SuperSym build_relaxed_sym_from_csr_l(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        int max_width,
        double relaxed_extra)
    {
        std::vector<int> col_ptr, row_ind;
        ichol::supernodal::build_lower_csc_pattern_from_csr(n, row_ptr, col_ind, col_ptr, row_ind);

        const auto strict_super = ichol::supernodal::detect_supernode_boundaries_from_csr_l(n, row_ptr, col_ind);
        const int strict_count = static_cast<int>(strict_super.size()) - 1;
        const double min_density = std::max(0.0, 1.0 - relaxed_extra);

        ichol::symbolic::SuperSym sym;
        sym.super.push_back(0);
        sym.pi.push_back(0);
        sym.px.push_back(0);

        int start_snode = 0;
        while (start_snode < strict_count)
        {
            int best_end_snode = start_snode + 1;
            CandidatePattern best = build_candidate_pattern(
                n, strict_super[(size_t)start_snode], strict_super[(size_t)best_end_snode],
                col_ptr, row_ind);

            for (int end_snode = start_snode + 2; end_snode <= strict_count; ++end_snode)
            {
                const int c0 = strict_super[(size_t)start_snode];
                const int c1 = strict_super[(size_t)end_snode];
                if (c1 - c0 > max_width)
                    break;

                CandidatePattern cand = build_candidate_pattern(n, c0, c1, col_ptr, row_ind);
                if (cand.density + 1e-12 < min_density)
                    break;

                best_end_snode = end_snode;
                best = std::move(cand);
            }

            const int c0 = strict_super[(size_t)start_snode];
            const int c1 = strict_super[(size_t)best_end_snode];
            const int nscol = c1 - c0;
            const int nsrow = static_cast<int>(best.rowlist.size());

            sym.super.push_back(c1);
            sym.s.insert(sym.s.end(), best.rowlist.begin(), best.rowlist.end());
            sym.pi.push_back(static_cast<int>(sym.s.size()));
            sym.px.push_back(sym.px.back() + nsrow * nscol);

            start_snode = best_end_snode;
        }

        return sym;
    }

    template <typename OutT, typename InT>
    std::vector<OutT> pack_relaxed_supernode_values_from_csr_l(
        int n,
        const std::vector<int> &row_ptr,
        const std::vector<int> &col_ind,
        const std::vector<InT> &values,
        const ichol::symbolic::SuperSym &sym)
    {
        std::vector<int> col_ptr, row_ind, csc_to_csr_map;
        ichol::supernodal::build_lower_csc_pattern_from_csr(n, row_ptr, col_ind, col_ptr, row_ind, csc_to_csr_map);

        std::vector<OutT> packed(sym.px.empty() ? 0u : (size_t)sym.px.back(), OutT{});
        std::vector<int> row_pos((size_t)n, -1);

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

            for (int local_row = 0; local_row < nsrow; ++local_row)
                row_pos[(size_t)sym.s[(size_t)(pi0 + local_row)]] = local_row;

            for (int j = 0; j < nscol; ++j)
            {
                const int gcol = c0 + j;
                for (int p = col_ptr[(size_t)gcol]; p < col_ptr[(size_t)gcol + 1]; ++p)
                {
                    const int grow = row_ind[(size_t)p];
                    const int local_row = row_pos[(size_t)grow];
                    if (local_row < 0)
                        throw std::runtime_error("pack_relaxed_supernode_values_from_csr_l: row missing from rowlist");
                    if (local_row < j)
                        continue;

                    const int csr_pos = csc_to_csr_map[(size_t)p];
                    packed[(size_t)px0 + (size_t)j * (size_t)nsrow + (size_t)local_row] =
                        static_cast<OutT>(values[(size_t)csr_pos]);
                }
            }

            for (int local_row = 0; local_row < nsrow; ++local_row)
                row_pos[(size_t)sym.s[(size_t)(pi0 + local_row)]] = -1;
        }

        return packed;
    }

    struct Variant
    {
        std::string name;
        double relaxed_extra = 0.0;
        int max_width = 1;
    };

    struct ShapeMetrics
    {
        double avg_width = 0.0;
        double width1_ratio = 0.0;
        double width_ge4_ratio = 0.0;
        double packed_over_nnz = 0.0;
        int solve_dependency_levels = 0;
    };

    ShapeMetrics compute_shape_metrics(
        const ichol::symbolic::SuperSym &sym,
        const std::vector<std::vector<int>> &buckets,
        int nnz_l)
    {
        ShapeMetrics m;
        const int nsuper = static_cast<int>(sym.super.size()) - 1;
        const int n = sym.super.empty() ? 0 : sym.super.back();
        if (nsuper <= 0)
            return m;

        int width1 = 0;
        int width_ge4 = 0;
        for (int k = 0; k < nsuper; ++k)
        {
            const int width = sym.super[(size_t)k + 1] - sym.super[(size_t)k];
            if (width == 1)
                ++width1;
            if (width >= 4)
                ++width_ge4;
        }

        m.avg_width = static_cast<double>(n) / static_cast<double>(nsuper);
        m.width1_ratio = static_cast<double>(width1) / static_cast<double>(nsuper);
        m.width_ge4_ratio = static_cast<double>(width_ge4) / static_cast<double>(nsuper);
        m.packed_over_nnz = (nnz_l > 0) ? static_cast<double>(sym.px.back()) / static_cast<double>(nnz_l) : 0.0;
        m.solve_dependency_levels = static_cast<int>(buckets.size());
        return m;
    }

    template <typename SolveT>
    struct ExperimentPrecondContext
    {
        int n = 0;
        int num_levels = 0;
        int max_bucket_size = 0;
        std::vector<int> bucket_ptr;
        std::vector<int> bucket_nodes;

        ichol::supernodal::cuda_reference::DeviceArray<int> d_super;
        ichol::supernodal::cuda_reference::DeviceArray<int> d_pi;
        ichol::supernodal::cuda_reference::DeviceArray<int> d_px;
        ichol::supernodal::cuda_reference::DeviceArray<int> d_s;
        ichol::supernodal::cuda_reference::DeviceArray<int> d_bucket_ptr;
        ichol::supernodal::cuda_reference::DeviceArray<int> d_bucket_nodes;
        ichol::supernodal::cuda_reference::DeviceArray<int> d_status;
        ichol::supernodal::cuda_reference::DeviceArray<SolveT> d_packed;
        ichol::supernodal::cuda_reference::DeviceArray<SolveT> d_r_work;
        ichol::supernodal::cuda_reference::DeviceArray<SolveT> d_w_work;
        ichol::supernodal::cuda_reference::DeviceArray<SolveT> d_z_work;

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        int h_status = 0;
        double solve_total_ms = 0.0;
        int solve_timed_iters = 0;

        ~ExperimentPrecondContext()
        {
            if (start != nullptr)
                cudaEventDestroy(start);
            if (stop != nullptr)
                cudaEventDestroy(stop);
        }

        void init(
            const ichol::symbolic::SuperSym &sym,
            const std::vector<std::vector<int>> &buckets,
            const std::vector<SolveT> &packed)
        {
            n = sym.super.empty() ? 0 : sym.super.back();
            ichol::supernodal::cuda_reference::flatten_buckets(buckets, bucket_ptr, bucket_nodes);
            num_levels = static_cast<int>(bucket_ptr.size()) - 1;
            max_bucket_size = 0;
            for (size_t level = 0; level + 1 < bucket_ptr.size(); ++level)
                max_bucket_size = std::max(max_bucket_size, bucket_ptr[level + 1] - bucket_ptr[level]);

            d_super.copy_from_host(sym.super, 0);
            d_pi.copy_from_host(sym.pi, 0);
            d_px.copy_from_host(sym.px, 0);
            d_s.copy_from_host(sym.s, 0);
            d_bucket_ptr.copy_from_host(bucket_ptr, 0);
            d_bucket_nodes.copy_from_host(bucket_nodes, 0);
            d_packed.copy_from_host(packed, 0);
            d_r_work.alloc((size_t)n);
            d_w_work.alloc((size_t)n);
            d_z_work.alloc((size_t)n);
            d_status.alloc(1);
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }
    };

    template <typename SolveT>
    void apply_experiment_supernode_precond(
        void *raw_ctx,
        const double *d_r,
        double *d_z,
        int n,
        cudaStream_t stream)
    {
        auto *ctx = static_cast<ExperimentPrecondContext<SolveT> *>(raw_ctx);
        if (ctx == nullptr || ctx->n != n)
            throw std::runtime_error("apply_experiment_supernode_precond: invalid context");

        experiment_cast_vec<SolveT, double><<<(n + 255) / 256, 256, 0, stream>>>(n, d_r, ctx->d_r_work.get());

        cudaEventRecord(ctx->start, stream);
        const bool forward_persistent = ichol::supernodal::cuda_reference::solve_lower_device_persistent<SolveT>(
            n, ctx->num_levels, ctx->max_bucket_size,
            ctx->d_bucket_ptr.get(), ctx->d_bucket_nodes.get(),
            ctx->d_super.get(), ctx->d_pi.get(), ctx->d_px.get(), ctx->d_s.get(),
            ctx->d_packed.get(),
            ctx->d_r_work.get(), ctx->d_r_work.get(), ctx->d_w_work.get(), ctx->d_status.get(),
            stream, true);
        if (!forward_persistent)
        {
            ichol::supernodal::cuda_reference::solve_lower_device<SolveT>(
                n, ctx->bucket_ptr, ctx->d_bucket_nodes.get(),
                ctx->d_super.get(), ctx->d_pi.get(), ctx->d_px.get(), ctx->d_s.get(),
                ctx->d_packed.get(),
                ctx->d_r_work.get(), ctx->d_r_work.get(), ctx->d_w_work.get(), ctx->d_status.get(),
                stream, true);
        }

        const bool backward_persistent = ichol::supernodal::cuda_reference::solve_lower_transpose_device_persistent<SolveT>(
            n, ctx->num_levels, ctx->max_bucket_size,
            ctx->d_bucket_ptr.get(), ctx->d_bucket_nodes.get(),
            ctx->d_super.get(), ctx->d_pi.get(), ctx->d_px.get(), ctx->d_s.get(),
            ctx->d_packed.get(),
            ctx->d_w_work.get(), ctx->d_z_work.get(), ctx->d_status.get(),
            stream, false);
        if (!backward_persistent)
        {
            ichol::supernodal::cuda_reference::solve_lower_transpose_device<SolveT>(
                n, ctx->bucket_ptr, ctx->d_bucket_nodes.get(),
                ctx->d_super.get(), ctx->d_pi.get(), ctx->d_px.get(), ctx->d_s.get(),
                ctx->d_packed.get(),
                ctx->d_w_work.get(), ctx->d_z_work.get(), ctx->d_status.get(),
                stream, false);
        }

        cudaMemcpyAsync(&ctx->h_status, ctx->d_status.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ctx->stop, stream);
        cudaEventSynchronize(ctx->stop);
        if (ctx->h_status != 0)
            throw std::runtime_error("apply_experiment_supernode_precond: zero diagonal on device");

        float iter_ms = 0.0f;
        cudaEventElapsedTime(&iter_ms, ctx->start, ctx->stop);
        ctx->solve_total_ms += iter_ms;
        ++ctx->solve_timed_iters;

        experiment_cast_vec<double, SolveT><<<(n + 255) / 256, 256, 0, stream>>>(n, ctx->d_z_work.get(), d_z);
    }

    void print_result_header()
    {
        std::cout << "amalgamation,max_width,avg_supernode_width,width1_ratio,width_ge4_ratio,"
                  << "packed_dense_slots_over_nnzL,solve_dependency_levels,avg_supernode_solve_ms,"
                  << "pcg_time_s,iterations,finalRes\n";
    }
} // namespace

TEST(SupernodeAmalgamationExperiment, SweepRelaxedWidth)
{
    const char *run = std::getenv("ICHOL_RUN_SUPERNODE_AMALGAMATION_EXPERIMENT");
    if (run == nullptr || std::string(run) != "1")
        GTEST_SKIP() << "Set ICHOL_RUN_SUPERNODE_AMALGAMATION_EXPERIMENT=1 to run this experiment.";

    const char *matrix_env = std::getenv("ICHOL_SUPERNODE_EXPERIMENT_MATRIX");
    const std::string matrix_path = (matrix_env != nullptr)
                                        ? std::string(matrix_env)
                                        : std::string("./data/matrices/bcsstk14/bcsstk14.mtx");

    using SolveT = float;
    auto A = ichol::io::mtx_to_csr<double>(matrix_path, false);
    const int n = A.num_rows;

    ichol::SymbolicOptions sym_options;
    sym_options.ordering = ichol::Ordering::RCM;
    sym_options.level_k = -1;

    ichol::IncompleteCholeskyOptions ic_options;
    ic_options.scaling = ichol::Scaling::UnitSqrtDiag;
    ic_options.pivot_shift_strategy = ichol::PivotShiftStrategy::Static;
    ic_options.static_shift = 1e-6;
    ic_options.lfil = 20;
    ic_options.drop_tol = 1e-4;

    auto sym_plan = ichol::symbolic::ic_analyze<double>(A, sym_options);
    ichol::numeric::NumericPlan num_plan;
    auto L = ichol::numeric::incomplete_cholesky_preconditioner<double>(A, sym_plan, num_plan, ic_options);

    const auto &D = num_plan.prescaling.D;
    std::vector<double> b((size_t)n, 1.0);
    std::vector<double> b_perm = ichol::symbolic::apply_permutation_vec(b, sym_plan.perm);
    std::vector<double> b_tilde((size_t)n);
    for (int i = 0; i < n; ++i)
        b_tilde[(size_t)i] = b_perm[(size_t)i] / D[(size_t)i];

    std::vector<SolveT> L_values_pcg;
    L_values_pcg.reserve(L.values.size());
    for (double v : L.values)
        L_values_pcg.push_back(ichol::util::cast_fp_type<SolveT>(v));

    const std::vector<Variant> variants = {
        {"off", 0.0, 4},
        {"off", 0.0, 8},
        {"relaxed_0.1", 0.1, 4},
        {"relaxed_0.1", 0.1, 8},
        {"relaxed_0.2", 0.2, 4},
        {"relaxed_0.2", 0.2, 8},
        {"relaxed_0.3", 0.3, 4},
        {"relaxed_0.3", 0.3, 8},
    };

    std::cout << "[supernode-amalgamation-experiment] matrix=" << matrix_path
              << " ordering=RCM precision_pcg=float\n";
    print_result_header();

    ichol::supernodal::cuda_reference::DeviceArray<double> d_warm_rhs;
    ichol::supernodal::cuda_reference::DeviceArray<double> d_warm_z;
    d_warm_rhs.copy_from_host(b_tilde, 0);
    d_warm_z.alloc((size_t)n);

    bool pcg_warmed = false;
    for (const auto &variant : variants)
    {
        ichol::symbolic::SuperSym sym;
        if (variant.name == "off")
        {
            const auto strict_super = ichol::supernodal::detect_supernode_boundaries_from_csr_l(n, L.row_ptr, L.col_ind);
            sym = ichol::supernodal::build_super_sym_from_csr_l(n, L.row_ptr, L.col_ind, strict_super);
        }
        else
        {
            sym = build_relaxed_sym_from_csr_l(
                n, L.row_ptr, L.col_ind, variant.max_width, variant.relaxed_extra);
        }

        const auto buckets = ichol::supernodal::build_forward_solve_buckets_from_sym(sym);
        const auto metrics = compute_shape_metrics(sym, buckets, static_cast<int>(L.values.size()));
        const auto packed = pack_relaxed_supernode_values_from_csr_l<SolveT>(
            n, L.row_ptr, L.col_ind, L.values, sym);

        ExperimentPrecondContext<SolveT> ctx;
        ctx.init(sym, buckets, packed);
        apply_experiment_supernode_precond<SolveT>(&ctx, d_warm_rhs.get(), d_warm_z.get(), n, 0);
        ctx.solve_total_ms = 0.0;
        ctx.solve_timed_iters = 0;

        ichol::precond::PrecondApply precond{
            &apply_experiment_supernode_precond<SolveT>,
            &ctx};

        ichol::solver::PCGParams params;
        params.maxits = 1000;
        params.tol = 1e-10;
        params.custom_precond = &precond;

        if (!pcg_warmed)
        {
            std::vector<double> y_warm;
            (void)ichol::solver::pcg<SolveT>(
                A.row_ptr, A.col_ind, A.values,
                L.row_ptr, L.col_ind, L_values_pcg,
                b_tilde, y_warm, D, params);
            cudaDeviceSynchronize();
            ctx.solve_total_ms = 0.0;
            ctx.solve_timed_iters = 0;
            pcg_warmed = true;
        }

        std::vector<double> y;
        const auto pcg_start = std::chrono::steady_clock::now();
        const auto result = ichol::solver::pcg<SolveT>(
            A.row_ptr, A.col_ind, A.values,
            L.row_ptr, L.col_ind, L_values_pcg,
            b_tilde, y, D, params);
        const auto pcg_end = std::chrono::steady_clock::now();
        const double pcg_time_s = std::chrono::duration<double>(pcg_end - pcg_start).count();
        const double avg_solve_ms = (ctx.solve_timed_iters > 0)
                                        ? ctx.solve_total_ms / static_cast<double>(ctx.solve_timed_iters)
                                        : 0.0;

        std::cout << variant.name << ","
                  << variant.max_width << ","
                  << std::setprecision(8) << metrics.avg_width << ","
                  << metrics.width1_ratio << ","
                  << metrics.width_ge4_ratio << ","
                  << metrics.packed_over_nnz << ","
                  << metrics.solve_dependency_levels << ","
                  << avg_solve_ms << ","
                  << pcg_time_s << ","
                  << result.iterations << ","
                  << result.finalRes << "\n";
    }
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr)
        return ierr;

    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();

    ierr = PetscFinalize();
    if (ierr)
        return ierr;
    return rc;
}
