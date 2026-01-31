#pragma once

#include <vector>

namespace ichol::symbolic
{
    /**
     * @brief CHOLMOD-style packed supernodal symbolic.
     *
     * This is the minimal symbolic data the (CPU/GPU) supernodal numeric phase needs.
     * For supernode k:
     *   - columns are [super[k], super[k+1]) with width nscol
     *   - rowlist is s[pi[k] .. pi[k+1]) with length nsrow
     *   - dense block storage for node k typically uses offsets px[k]..px[k+1)
     *     with size nsrow * nscol.
     *
     * Convention (matching CHOLMOD supernodal LL default):
     *   - rowlist begins with pivot rows: super[k]..super[k+1)-1 (nscol entries)
     *   - followed by strictly increasing update rows >= super[k+1].
     */
    struct SuperSym
    {
        std::vector<int> super; // size = nsuper + 1, supernode column boundaries
        std::vector<int> pi;    // size = nsuper + 1, rowlist pointers into s
        std::vector<int> px;    // size = nsuper + 1, dense-block offsets (prefix sum of nsrow*nscol)
        std::vector<int> s;     // packed rowlists
    };
} // namespace ichol::symbolic
