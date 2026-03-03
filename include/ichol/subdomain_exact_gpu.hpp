#pragma once

#ifndef ICHOL_SUBDOMAIN_EXACT_GPU_HPP
#define ICHOL_SUBDOMAIN_EXACT_GPU_HPP

#include <cuda_runtime.h>
#include <vector>

#include "ichol/matrix_formats.hpp"
#include "ichol/options.hpp"

namespace ichol::precond
{
    struct GridShape
    {
        int w = 0;
        int h = 0;
        int d = 1;
    };

    struct SubdomainSize
    {
        int w = 0;
        int h = 0;
        int d = 1;
    };

    struct SubdomainRegion
    {
        int x0 = 0;
        int x1 = 0;
        int y0 = 0;
        int y1 = 0;
        int z0 = 0;
        int z1 = 1;
    };

    enum class SubdomainPreconditionerKind
    {
        PSAI,
        ExactCholesky,
        IncompleteCholesky
    };

    struct SubdomainPreconditionerOptions
    {
        SubdomainPreconditionerKind kind = SubdomainPreconditionerKind::PSAI;
        int psai_radius = 1;
        int ic_level_k = 0;
    };

    std::vector<SubdomainRegion> partition_subdomains(
        const GridShape &global_shape,
        const SubdomainSize &subdomain_size);

    struct SubdomainSpSVContext;

    SubdomainSpSVContext *create_subdomain_spsv_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options = {});

    void destroy_subdomain_spsv_context(SubdomainSpSVContext *ctx);

    void apply_subdomain_exact_spsv(
        void *ctx,
        const double *d_r,
        double *d_z,
        int N,
        cudaStream_t stream);
} // namespace ichol::precond

#endif // ICHOL_SUBDOMAIN_EXACT_GPU_HPP
