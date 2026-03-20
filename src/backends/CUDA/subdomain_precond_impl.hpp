#pragma once

#include <cuda_runtime.h>

#include "ichol/matrix_formats.hpp"
#include "ichol/subdomain_exact_gpu.hpp"

namespace ichol::precond::detail
{
    void *create_subdomain_psai_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options);

    void apply_subdomain_psai(
        void *ctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream);

    void destroy_subdomain_psai_context(void *ctx);

    void *create_subdomain_ic_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options);

    void apply_subdomain_ic(
        void *ctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream);

    void destroy_subdomain_ic_context(void *ctx);
} // namespace ichol::precond::detail
