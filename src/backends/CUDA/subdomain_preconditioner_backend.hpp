#pragma once

#include <cuda_runtime.h>

#include "ichol/matrix_formats.hpp"
#include "ichol/subdomain_preconditioner_gpu.hpp"

namespace ichol::precond::detail
{
    void *create_subdomain_exact_cholesky_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options);

    void apply_subdomain_exact_cholesky(
        void *ctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream);

    void destroy_subdomain_exact_cholesky_context(void *ctx);

    void *create_subdomain_incomplete_cholesky_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options);

    void apply_subdomain_incomplete_cholesky(
        void *ctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream);

    void destroy_subdomain_incomplete_cholesky_context(void *ctx);

    void *create_subdomain_spai_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options);

    void apply_subdomain_spai(
        void *ctx,
        const void *d_r,
        void *d_z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream);

    void destroy_subdomain_spai_context(void *ctx);
} // namespace ichol::precond::detail
