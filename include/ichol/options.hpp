#pragma once

namespace ichol
{
    enum class Ordering {
        Identity,
        AMD,
        NestedDissection
    };

    struct SymbolicOptions {
        Ordering ordering = Ordering::AMD;
        bool use_etree = true;

        // IC(k)
        int level_k = -1; // -1 means complete Cholesky
    };

    struct SuperNodeOptions {
        int min_supernode_size = 16;
        int max_supernode_size = 128;
        bool relaxed = false;
    };

    struct PCGOptions {
        int max_iterations = 1000;
        double relative_tolerance = 1e-6;
        double absolute_tolerance = 1e-10;
        bool verbose = false;
    };
} // namespace ichol