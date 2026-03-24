#include "ichol/subdomain_preconditioner_gpu.hpp"

#include <algorithm>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/CUDA/subdomain_preconditioner_backend.hpp"

namespace ichol::precond
{
    struct SubdomainPreconditionerContext
    {
        SubdomainPreconditionerKind kind = SubdomainPreconditionerKind::SPAI;
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
            case SubdomainPreconditionerKind::FSAI:
                ctx->impl = detail::create_subdomain_fsai_context(A, global_shape, region, options);
                break;
            case SubdomainPreconditionerKind::SPAI:
                ctx->impl = detail::create_subdomain_spai_context(A, global_shape, region, options);
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
        // SPAI contexts are created via the same async path as other preconditioners.
        // The petsc_mutex inside create_subdomain_spai_context serialises PETSc
        // calls while GPU operations (upload, cuSPARSE setup) run in parallel.
        int device = 0;
        const cudaError_t device_status = cudaGetDevice(&device);
        if (device_status != cudaSuccess)
            throw std::runtime_error(std::string("create_subdomain_preconditioner_contexts_parallel: cudaGetDevice failed: ") +
                                     cudaGetErrorString(device_status));

        std::vector<std::future<SubdomainPreconditionerContext *>> futures;
        futures.reserve(regions.size());
        for (std::size_t region_index = 0; region_index < regions.size(); ++region_index)
        {
            const auto region = regions[region_index];
            auto region_options = options;
            region_options.debug_subdomain_index = static_cast<int>(region_index);
            futures.emplace_back(std::async(std::launch::async, [&A, global_shape, region, region_options, device]()
                                            {
                                                const cudaError_t set_device_status = cudaSetDevice(device);
                                                if (set_device_status != cudaSuccess)
                                                {
                                                    throw std::runtime_error(std::string("create_subdomain_preconditioner_contexts_parallel: cudaSetDevice failed: ") +
                                                                             cudaGetErrorString(set_device_status));
                                                }
                                                return create_context_impl(A, global_shape, region, region_options);
                                            }));
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
        case SubdomainPreconditionerKind::FSAI:
            detail::apply_subdomain_fsai(ctx->impl, d_r, d_z, N, prec, stream);
            break;
        case SubdomainPreconditionerKind::SPAI:
            detail::apply_subdomain_spai(ctx->impl, d_r, d_z, N, prec, stream);
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
        case SubdomainPreconditionerKind::FSAI:
            detail::destroy_subdomain_fsai_context(ctx->impl);
            break;
        case SubdomainPreconditionerKind::SPAI:
            detail::destroy_subdomain_spai_context(ctx->impl);
            break;
        default:
            break;
        }
        delete ctx;
    }
} // namespace ichol::precond
