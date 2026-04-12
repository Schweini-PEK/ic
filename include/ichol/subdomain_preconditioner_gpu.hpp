#pragma once

#ifndef ICHOL_SUBDOMAIN_PRECONDITIONER_GPU_HPP
#define ICHOL_SUBDOMAIN_PRECONDITIONER_GPU_HPP

#include <vector>

#include "ichol/cuda_compat.hpp"
#include "ichol/matrix_formats.hpp"
#include "ichol/subdomain_preconditioner_common.hpp"

namespace ichol::precond
{
    struct SubdomainPreconditionerContext;
    using SubdomainSpSVContext = SubdomainPreconditionerContext;

    SubdomainPreconditionerContext *create_subdomain_preconditioner_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options = {});

    std::vector<SubdomainPreconditionerContext *> create_subdomain_preconditioner_contexts_parallel(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const std::vector<SubdomainRegion> &regions,
        const SubdomainPreconditionerOptions &options = {});

    void destroy_subdomain_preconditioner_context(SubdomainPreconditionerContext *ctx);

    void apply_subdomain_preconditioner(
        void *ctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream);

    inline SubdomainSpSVContext *create_subdomain_spsv_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options = {})
    {
        return create_subdomain_preconditioner_context(A, global_shape, region, options);
    }

    inline void destroy_subdomain_spsv_context(SubdomainSpSVContext *ctx)
    {
        destroy_subdomain_preconditioner_context(ctx);
    }

    inline void apply_subdomain_exact_spsv(
        void *ctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream)
    {
        apply_subdomain_preconditioner(ctx, d_r, d_z, N, prec, stream);
    }
} // namespace ichol::precond

#endif // ICHOL_SUBDOMAIN_PRECONDITIONER_GPU_HPP
