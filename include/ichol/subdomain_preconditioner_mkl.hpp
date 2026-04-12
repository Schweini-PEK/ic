#pragma once

#ifndef ICHOL_SUBDOMAIN_PRECONDITIONER_MKL_HPP
#define ICHOL_SUBDOMAIN_PRECONDITIONER_MKL_HPP

#include <vector>

#include "ichol/preconditioner.hpp"
#include "ichol/subdomain_preconditioner_common.hpp"

namespace ichol::precond
{
    struct SubdomainIncompleteCholeskyMklContext;

    SubdomainIncompleteCholeskyMklContext *create_subdomain_incomplete_cholesky_mkl_context(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const SubdomainRegion &region,
        const SubdomainPreconditionerOptions &options = {});

    std::vector<SubdomainIncompleteCholeskyMklContext *> create_subdomain_incomplete_cholesky_mkl_contexts_parallel(
        const ichol::matrix::CsrMatrix<double> &A,
        const GridShape &global_shape,
        const std::vector<SubdomainRegion> &regions,
        const SubdomainPreconditionerOptions &options = {});

    void destroy_subdomain_incomplete_cholesky_mkl_context(SubdomainIncompleteCholeskyMklContext *ctx);

    void apply_subdomain_incomplete_cholesky_mkl(
        void *ctx,
        const void *r,
        void *z,
        int N,
        ichol::solver::ComputePrecision prec,
        cudaStream_t stream);
} // namespace ichol::precond

#endif // ICHOL_SUBDOMAIN_PRECONDITIONER_MKL_HPP
