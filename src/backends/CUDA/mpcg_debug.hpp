#pragma once

#include <vector>

#include "ichol/pcg.hpp"
#include "ichol/preconditioner.hpp"

namespace ichol::solver::debug
{
    struct ZnewBuildOptions
    {
        ComputePrecision prec_precond = ComputePrecision::FP64;
        bool serial = false;
        bool sync_after_each_apply = false;
    };

    std::vector<double> build_mpcg_znew_columns(
        const std::vector<ichol::precond::PrecondApply> &preconds,
        const std::vector<double> &h_r,
        const ZnewBuildOptions &options = {});
} // namespace ichol::solver::debug
