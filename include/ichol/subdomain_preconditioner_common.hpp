#pragma once

#ifndef ICHOL_SUBDOMAIN_PRECONDITIONER_COMMON_HPP
#define ICHOL_SUBDOMAIN_PRECONDITIONER_COMMON_HPP

#include <vector>

#include "ichol/options.hpp"
#include "ichol/precision.hpp"

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
        SPAI,
        ExactCholesky,
        IncompleteCholesky,
        FSAI
    };

    struct SubdomainPreconditionerOptions
    {
        SubdomainPreconditionerKind kind = SubdomainPreconditionerKind::SPAI;
        int spai_radius = 1;
        int ic_level_k = 0;
        int fsai_level_k = 0;
        ichol::solver::ComputePrecision precision = ichol::solver::ComputePrecision::FP64;
        int debug_subdomain_index = -1;

        double spai_epsilon = 0.4;
        int spai_nbsteps = 5;
        int spai_max_cols = 5;
        int spai_maxnew = 5;
        bool spai_symmetric = false;
    };

    std::vector<SubdomainRegion> partition_subdomains(
        const GridShape &global_shape,
        const SubdomainSize &subdomain_size);
} // namespace ichol::precond

#endif // ICHOL_SUBDOMAIN_PRECONDITIONER_COMMON_HPP
