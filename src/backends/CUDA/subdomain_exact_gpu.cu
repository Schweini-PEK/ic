#include "ichol/subdomain_exact_gpu.hpp"

#include <algorithm>
#include <stdexcept>

#include "backends/CUDA/subdomain_precond_impl.hpp"

namespace ichol::precond
{
    struct SubdomainSpSVContext
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

    SubdomainSpSVContext *create_subdomain_spsv_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options)
    {
        auto *ctx = new SubdomainSpSVContext();
        try
        {
            ctx->kind = options.kind;
            switch (options.kind)
            {
            case SubdomainPreconditionerKind::PSAI:
                ctx->impl = detail::create_subdomain_psai_context(A, global_shape, region, options);
                break;
            case SubdomainPreconditionerKind::ExactCholesky:
            case SubdomainPreconditionerKind::IncompleteCholesky:
                ctx->impl = detail::create_subdomain_ic_context(A, global_shape, region, options);
                break;
            default:
                throw std::runtime_error("create_subdomain_spsv_context: unsupported preconditioner kind");
            }
            return ctx;
        }
        catch (...)
        {
            delete ctx;
            throw;
        }
    }

    void apply_subdomain_exact_spsv(
        void *vctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream)
    {
        auto *ctx = reinterpret_cast<SubdomainSpSVContext *>(vctx);
        switch (ctx->kind)
        {
        case SubdomainPreconditionerKind::PSAI:
            detail::apply_subdomain_psai(ctx->impl, d_r, d_z, N, prec, stream);
            break;
        case SubdomainPreconditionerKind::ExactCholesky:
        case SubdomainPreconditionerKind::IncompleteCholesky:
            detail::apply_subdomain_ic(ctx->impl, d_r, d_z, N, prec, stream);
            break;
        default:
            throw std::runtime_error("apply_subdomain_exact_spsv: unsupported preconditioner kind");
        }
    }

    void destroy_subdomain_spsv_context(SubdomainSpSVContext *ctx)
    {
        if (!ctx)
            return;
        switch (ctx->kind)
        {
        case SubdomainPreconditionerKind::PSAI:
            detail::destroy_subdomain_psai_context(ctx->impl);
            break;
        case SubdomainPreconditionerKind::ExactCholesky:
        case SubdomainPreconditionerKind::IncompleteCholesky:
            detail::destroy_subdomain_ic_context(ctx->impl);
            break;
        default:
            break;
        }
        delete ctx;
    }
} // namespace ichol::precond
