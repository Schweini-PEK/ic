#include "ichol/subdomain_preconditioner_gpu.hpp"

#include <algorithm>
#include <future>
#include <stdexcept>
#include <vector>

#include "backends/CUDA/subdomain_preconditioner_backend.hpp"

namespace ichol::precond
{
    struct SubdomainPreconditionerContext
    {
        SubdomainPreconditionerKind kind = SubdomainPreconditionerKind::PSAI;
        void *impl = nullptr;
    };

    std::vector<SubdomainRegion> partition_subdomains(const GridShape &global_shape, const SubdomainSize &sub_size)
    {
        if (global_shape.w <= 0 || global_shape.h <= 0 || global_shape.d <= 0)
            throw std::runtime_error("partition_subdomains: invalid global shape");
        if (sub_size.w <= 0 || sub_size.h <= 0 || sub_size.d <= 0)
            throw std::runtime_error("partition_subdomains: invalid subdomain size");

        std::vector<SubdomainRegion> regions;
        for (int z = 0; z < global_shape.d; z += sub_size.d)
        {
            for (int y = 0; y < global_shape.h; y += sub_size.h)
            {
                for (int x = 0; x < global_shape.w; x += sub_size.w)
                {
                    regions.push_back({x, std::min(x + sub_size.w, global_shape.w),
                                       y, std::min(y + sub_size.h, global_shape.h),
                                       z, std::min(z + sub_size.d, global_shape.d)});
                }
            }
        }
        return regions;
    }

    static SubdomainPreconditionerContext *create_context_impl(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options)
    {
        auto *ctx = new SubdomainPreconditionerContext();
        try
        {
            ctx->kind = options.kind;
            switch (options.kind)
            {
            case SubdomainPreconditionerKind::ExactCholesky:
                ctx->impl = detail::create_subdomain_exact_cholesky_context(A, global_shape, region, options);
                break;
            case SubdomainPreconditionerKind::IncompleteCholesky:
                ctx->impl = detail::create_subdomain_incomplete_cholesky_context(A, global_shape, region, options);
                break;
            case SubdomainPreconditionerKind::PSAI:
                ctx->impl = detail::create_subdomain_psai_context(A, global_shape, region, options);
                break;
            default:
                throw std::runtime_error("create_subdomain_preconditioner_context: unsupported preconditioner kind");
            }
            return ctx;
        }
        catch (...)
        {
            delete ctx;
            throw;
        }
    }

    SubdomainPreconditionerContext *create_subdomain_preconditioner_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options)
    {
        return create_context_impl(A, global_shape, region, options);
    }

    std::vector<SubdomainPreconditionerContext *> create_subdomain_preconditioner_contexts_parallel(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const std::vector<SubdomainRegion> &regions,
        const SubdomainPreconditionerOptions &options)
    {
        std::vector<std::future<SubdomainPreconditionerContext *>> futures;
        futures.reserve(regions.size());
        for (const auto &region : regions)
        {
            futures.emplace_back(std::async(std::launch::async, [&A, global_shape, region, options]()
                                            { return create_context_impl(A, global_shape, region, options); }));
        }

        std::vector<SubdomainPreconditionerContext *> contexts;
        contexts.reserve(regions.size());
        try
        {
            for (auto &future : futures)
                contexts.push_back(future.get());
        }
        catch (...)
        {
            for (auto *ctx : contexts)
                destroy_subdomain_preconditioner_context(ctx);
            throw;
        }
        return contexts;
    }

    void apply_subdomain_preconditioner(
        void *vctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream)
    {
        auto *ctx = reinterpret_cast<SubdomainPreconditionerContext *>(vctx);
        switch (ctx->kind)
        {
        case SubdomainPreconditionerKind::ExactCholesky:
            detail::apply_subdomain_exact_cholesky(ctx->impl, d_r, d_z, N, prec, stream);
            break;
        case SubdomainPreconditionerKind::IncompleteCholesky:
            detail::apply_subdomain_incomplete_cholesky(ctx->impl, d_r, d_z, N, prec, stream);
            break;
        case SubdomainPreconditionerKind::PSAI:
            detail::apply_subdomain_psai(ctx->impl, d_r, d_z, N, prec, stream);
            break;
        default:
            throw std::runtime_error("apply_subdomain_preconditioner: unsupported preconditioner kind");
        }
    }

    void destroy_subdomain_preconditioner_context(SubdomainPreconditionerContext *ctx)
    {
        if (!ctx)
            return;
        switch (ctx->kind)
        {
        case SubdomainPreconditionerKind::ExactCholesky:
            detail::destroy_subdomain_exact_cholesky_context(ctx->impl);
            break;
        case SubdomainPreconditionerKind::IncompleteCholesky:
            detail::destroy_subdomain_incomplete_cholesky_context(ctx->impl);
            break;
        case SubdomainPreconditionerKind::PSAI:
            detail::destroy_subdomain_psai_context(ctx->impl);
            break;
        default:
            break;
        }
        delete ctx;
    }
} // namespace ichol::precond
