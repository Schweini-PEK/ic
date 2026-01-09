//include/ichol/options.hpp
#pragma once

#include <string>

namespace ichol
{
    enum class Ordering
    {
        Identity,
        AMD,
        NestedDissection
    };

    enum class Scaling
    {
        None,
        UnitSqrtDiag,
        UnitColNorm
    };

    enum class PivotShiftStrategy
    {
        None,
        MachineEpsilon,
        Static,
        Dynamic
    };

    enum class FactorizationAlgorithm
    {
        PARICT,
        ICKDT
    };

    struct SymbolicOptions
    {
        Ordering ordering = Ordering::Identity;
        bool use_etree = true;
        // IC(k)
        int level_k = -1; // -1 means complete Cholesky
    };

    struct SuperNodeOptions
    {
        int min_supernode_size = 16;
        int max_supernode_size = 128;
        bool relaxed = false;
        bool approximate = false;
        double overlap_threshold = 0.8;
    };

    struct IncompleteCholeskyOptions
    {
        Scaling scaling = Scaling::UnitSqrtDiag;
        PivotShiftStrategy pivot_shift_strategy = PivotShiftStrategy::MachineEpsilon;
        FactorizationAlgorithm algorithm = FactorizationAlgorithm::ICKDT;
        int max_restarts = 5;

        double static_shift = 1e-6; // For static shift strategy
        double pivot_tol = 0.0; // For pivot check
        double shift_growth = 2.0; // For shift increase on restart

        // For ICKDT
        int lfil = 20;
        double drop_tol = 1e-4;
    };

    struct PCGOptions
    {
        int max_iterations = 1000;
        double relative_tolerance = 1e-6;
        double absolute_tolerance = 1e-10;
        bool verbose = false;
    };
} // namespace ichol